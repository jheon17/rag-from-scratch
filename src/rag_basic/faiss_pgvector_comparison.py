"""동일한 DB 저장 Vector로 FAISS와 pgvector exact search를 비교한다."""

import math

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import CHUNK_OVERLAP, CHUNK_SIZE, PDF_PATH
from rag_basic.embedding import MODEL_NAME
from rag_basic.evaluation import EVAL_CASES
from rag_basic.pgvector_retrieval import (
    create_query_embedding,
    get_database_config,
    search_pgvector,
)
from rag_basic.vector_search import build_index


TOP_K = 5
EMBEDDING_DIMENSION = 384
ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-5
NORMALIZATION_TOLERANCE = 1e-4


def load_stored_documents(
    conn: psycopg.Connection,
) -> tuple[list[dict], np.ndarray]:
    """baseline metadata에 해당하는 Chunk와 저장 Embedding을 읽는다."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT chunk_id, page_number, content, embedding
            FROM rag_chunks
            WHERE document_name = %s
              AND embedding_model = %s
              AND chunk_size = %s
              AND chunk_overlap = %s
            ORDER BY chunk_id;
            """,
            (PDF_PATH.name, MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP),
        )
        rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(
            "baseline Document Embedding이 없습니다. "
            "pgvector_ingest 실행 여부를 확인해 주세요."
        )

    documents = [
        {
            "chunk_id": int(chunk_id),
            "page_number": int(page_number),
            "text": content,
        }
        for chunk_id, page_number, content, _ in rows
    ]
    embeddings = np.ascontiguousarray(
        [embedding.to_numpy() for *_, embedding in rows],
        dtype=np.float32,
    )
    return documents, embeddings


def validate_document_embeddings(
    documents: list[dict], embeddings: np.ndarray
) -> np.ndarray:
    """DB Vector의 shape, 값과 L2 정규화 상태를 검증한다."""
    if embeddings.ndim != 2:
        raise ValueError(f"Document Embedding이 2차원이 아닙니다: {embeddings.shape}")
    if embeddings.shape[0] != len(documents):
        raise ValueError("Document 수와 Embedding row 수가 일치하지 않습니다.")
    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            "Document Embedding dimension이 예상과 다릅니다: "
            f"{embeddings.shape[1]}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Document Embedding에 NaN 또는 infinity가 있습니다.")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(
        norms,
        1.0,
        rtol=0.0,
        atol=NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(
            "저장된 Document Embedding의 L2 norm이 1과 명백히 다릅니다. "
            "Vector를 임의로 재정규화하지 않고 비교를 중단합니다."
        )
    return norms


def validate_query_embedding(query_embedding: np.ndarray) -> float:
    """한 번 생성한 Query Vector의 형식과 정규화를 검증한다."""
    if query_embedding.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            f"Query Embedding shape가 올바르지 않습니다: {query_embedding.shape}"
        )
    if query_embedding.dtype != np.float32:
        raise ValueError(
            f"Query Embedding dtype이 float32가 아닙니다: {query_embedding.dtype}"
        )
    if not query_embedding.flags.c_contiguous:
        raise ValueError("Query Embedding이 contiguous array가 아닙니다.")
    if not np.isfinite(query_embedding).all():
        raise ValueError("Query Embedding에 NaN 또는 infinity가 있습니다.")

    norm = float(np.linalg.norm(query_embedding))
    if not math.isclose(
        norm, 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE
    ):
        raise ValueError(
            f"Query Embedding의 L2 norm이 1과 명백히 다릅니다: {norm:.8f}"
        )
    return norm


def search_faiss(
    index, documents: list[dict], query_embedding: np.ndarray
) -> list[dict]:
    """동일 Query Vector로 FAISS IndexFlatIP Top-5를 검색한다."""
    query_matrix = np.ascontiguousarray(
        query_embedding.reshape(1, -1), dtype=np.float32
    )
    scores, positions = index.search(query_matrix, TOP_K)

    results = []
    for rank, (score, position) in enumerate(
        zip(scores[0], positions[0]), start=1
    ):
        document = documents[int(position)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": document["chunk_id"],
                "page_number": document["page_number"],
                "text": document["text"],
            }
        )
    return results


def evaluate_gold(results: list[dict], expected_chunk_ids: list[int]) -> dict:
    """Top-5에서 첫 gold evidence의 순위와 RR을 계산한다."""
    first_gold_rank = next(
        (
            result["rank"]
            for result in results
            if result["chunk_id"] in expected_chunk_ids
        ),
        None,
    )
    return {
        "hit": first_gold_rank is not None,
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": (
            1.0 / first_gold_rank if first_gold_rank is not None else 0.0
        ),
    }


def compare_results(faiss_results: list[dict], pgvector_results: list[dict]) -> dict:
    """Top-5 순서·집합·순위와 같은 rank의 score 차이를 비교한다."""
    faiss_ids = [result["chunk_id"] for result in faiss_results]
    pgvector_ids = [result["chunk_id"] for result in pgvector_results]
    faiss_ranks = {result["chunk_id"]: result["rank"] for result in faiss_results}
    pgvector_ranks = {
        result["chunk_id"]: result["rank"] for result in pgvector_results
    }

    rank_differences = {
        chunk_id: faiss_ranks[chunk_id] - pgvector_ranks[chunk_id]
        for chunk_id in sorted(faiss_ranks.keys() & pgvector_ranks.keys())
    }
    aligned_score_differences = [
        abs(faiss_result["score"] - pgvector_result["score"])
        for faiss_result, pgvector_result in zip(
            faiss_results, pgvector_results
        )
        if faiss_result["chunk_id"] == pgvector_result["chunk_id"]
    ]
    aligned_scores_close = bool(aligned_score_differences) and all(
        math.isclose(
            faiss_result["score"],
            pgvector_result["score"],
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        )
        for faiss_result, pgvector_result in zip(
            faiss_results, pgvector_results
        )
        if faiss_result["chunk_id"] == pgvector_result["chunk_id"]
    )

    return {
        "exact_order_match": faiss_ids == pgvector_ids,
        "set_match": set(faiss_ids) == set(pgvector_ids),
        "rank_differences": rank_differences,
        "score_differences": aligned_score_differences,
        "scores_close": aligned_scores_close,
    }


def print_case_result(
    case: dict,
    query_norm: float,
    faiss_results: list[dict],
    pgvector_results: list[dict],
    comparison: dict,
    faiss_gold: dict,
    pgvector_gold: dict,
) -> None:
    """한 Case의 두 backend 결과와 검증값을 출력한다."""
    print("\n" + "=" * 88)
    print(f"Case: {case['name']}")
    print(f"Query: {case['query']}")
    print(f"expected_chunk_ids: {case['expected_chunk_ids']}")
    print(f"Query Embedding norm: {query_norm:.8f}")

    print("\nFAISS:")
    print("rank | chunk_id | score")
    for result in faiss_results:
        print(
            f"{result['rank']:>4} | {result['chunk_id']:>8} | "
            f"{result['score']:.8f}"
        )

    print("\npgvector:")
    print("rank | chunk_id | score")
    for result in pgvector_results:
        print(
            f"{result['rank']:>4} | {result['chunk_id']:>8} | "
            f"{result['score']:.8f}"
        )

    print("\nrank별 비교:")
    print(
        "rank | FAISS chunk | pgvector chunk | FAISS score | "
        "pgvector score | abs difference"
    )
    for faiss_result, pgvector_result in zip(
        faiss_results, pgvector_results
    ):
        if faiss_result["chunk_id"] == pgvector_result["chunk_id"]:
            difference_text = (
                f"{abs(faiss_result['score'] - pgvector_result['score']):.10f}"
            )
        else:
            difference_text = "N/A (chunk_id 다름)"
        print(
            f"{faiss_result['rank']:>4} | {faiss_result['chunk_id']:>11} | "
            f"{pgvector_result['chunk_id']:>14} | "
            f"{faiss_result['score']:.8f} | "
            f"{pgvector_result['score']:.8f} | {difference_text}"
        )

    differences = comparison["score_differences"]
    maximum_difference = max(differences) if differences else float("nan")
    average_difference = (
        sum(differences) / len(differences) if differences else float("nan")
    )
    print(f"\nTop-{TOP_K} exact order match: {comparison['exact_order_match']}")
    print(f"Top-{TOP_K} set match: {comparison['set_match']}")
    print(f"공통 Chunk rank 차이 (FAISS - pgvector): {comparison['rank_differences']}")
    print(f"maximum absolute score difference: {maximum_difference:.10f}")
    print(f"average absolute score difference: {average_difference:.10f}")
    print(
        "score가 부동소수점 오차 범위에서 일치하는가: "
        f"{comparison['scores_close']}"
    )
    print(
        "FAISS gold: "
        f"Hit@{TOP_K}={faiss_gold['hit']}, "
        f"first gold rank={faiss_gold['first_gold_rank']}, "
        f"RR={faiss_gold['reciprocal_rank']:.4f}"
    )
    print(
        "pgvector gold: "
        f"Hit@{TOP_K}={pgvector_gold['hit']}, "
        f"first gold rank={pgvector_gold['first_gold_rank']}, "
        f"RR={pgvector_gold['reciprocal_rank']:.4f}"
    )


def print_summary(case_results: list[dict], document_count: int) -> None:
    """6개 in-domain Case의 backend별 Retrieval 결과를 요약한다."""
    case_count = len(case_results)
    exact_order_count = sum(
        result["comparison"]["exact_order_match"] for result in case_results
    )
    set_match_count = sum(
        result["comparison"]["set_match"] for result in case_results
    )
    faiss_hit_count = sum(result["faiss_gold"]["hit"] for result in case_results)
    pgvector_hit_count = sum(
        result["pgvector_gold"]["hit"] for result in case_results
    )
    faiss_mrr = sum(
        result["faiss_gold"]["reciprocal_rank"] for result in case_results
    ) / case_count
    pgvector_mrr = sum(
        result["pgvector_gold"]["reciprocal_rank"] for result in case_results
    ) / case_count
    score_differences = [
        difference
        for result in case_results
        for difference in result["comparison"]["score_differences"]
    ]
    maximum_difference = max(score_differences) if score_differences else float("nan")
    average_difference = (
        sum(score_differences) / len(score_differences)
        if score_differences
        else float("nan")
    )

    print("\n" + "=" * 88)
    print("=== FAISS vs pgvector Summary ===")
    print(f"\nCases: {case_count}")
    print(f"Top-K: {TOP_K}")
    print(f"\nExact Top-{TOP_K} order match: {exact_order_count}/{case_count}")
    print(f"Top-{TOP_K} set match: {set_match_count}/{case_count}")
    print("\nFAISS:")
    print(f"Hit@{TOP_K}: {faiss_hit_count}/{case_count}")
    print(f"MRR: {faiss_mrr:.4f}")
    print("\npgvector:")
    print(f"Hit@{TOP_K}: {pgvector_hit_count}/{case_count}")
    print(f"MRR: {pgvector_mrr:.4f}")
    print("\nScore difference:")
    print(f"max: {maximum_difference:.10f}")
    print(f"average: {average_difference:.10f}")
    print("\nSearch:")
    print("FAISS = IndexFlatIP exact search")
    print("pgvector = cosine distance exact search")

    all_scores_close = all(
        result["comparison"]["scores_close"] for result in case_results
    )
    if exact_order_count == case_count and all_scores_close:
        print(
            "\n해석: 현재 baseline 조건에서는 동일한 저장 Vector와 Query "
            "Vector를 사용했을 때 두 backend의 Top-5 순서가 같았고, "
            "score도 부동소수점 오차 범위에서 일치했습니다."
        )
    else:
        print(
            "\n해석: 현재 baseline 조건에서 backend 사이에 순위 또는 score "
            "차이가 관찰되었습니다. dtype, 정규화, 동점 정렬을 추가로 "
            "확인해야 합니다."
        )
    print(
        f"이 결과는 현재 {document_count}개 baseline dataset, 동일 E5 "
        "Vector, cosine 기준 exact search, Top-K=5 조건에만 해당합니다."
    )


def main() -> None:
    try:
        database_config = get_database_config()
        in_domain_cases = [
            case for case in EVAL_CASES if case["type"] == "in_domain"
        ]
        if not in_domain_cases:
            raise ValueError("in-domain 평가 Case가 없습니다.")

        with psycopg.connect(**database_config) as conn:
            register_vector(conn)
            documents, document_embeddings = load_stored_documents(conn)
            document_norms = validate_document_embeddings(
                documents, document_embeddings
            )
            # 정규화된 Vector에서는 inner product를 cosine similarity로 쓸 수 있다.
            index = build_index(document_embeddings)

            print("DB Connection 성공: True")
            print(f"DB Document row count: {len(documents)}")
            print(f"Document Embedding shape: {document_embeddings.shape}")
            print(f"Document Embedding dtype: {document_embeddings.dtype}")
            print(f"Document norm min: {document_norms.min():.8f}")
            print(f"Document norm max: {document_norms.max():.8f}")
            print(f"Document norm average: {document_norms.mean():.8f}")
            print(f"FAISS index Vector count: {index.ntotal}")

            embedding_model = SentenceTransformer(MODEL_NAME)
            case_results = []
            for case in in_domain_cases:
                # Case당 정확히 한 번 만든 Query Vector를 양쪽 검색에 공유한다.
                query_embedding = create_query_embedding(
                    embedding_model, case["query"]
                )
                query_embedding = np.ascontiguousarray(
                    query_embedding, dtype=np.float32
                )
                query_norm = validate_query_embedding(query_embedding)

                faiss_results = search_faiss(
                    index, documents, query_embedding
                )
                pgvector_results = search_pgvector(
                    conn, query_embedding, top_k=TOP_K
                )
                comparison = compare_results(
                    faiss_results, pgvector_results
                )
                faiss_gold = evaluate_gold(
                    faiss_results, case["expected_chunk_ids"]
                )
                pgvector_gold = evaluate_gold(
                    pgvector_results, case["expected_chunk_ids"]
                )
                print_case_result(
                    case,
                    query_norm,
                    faiss_results,
                    pgvector_results,
                    comparison,
                    faiss_gold,
                    pgvector_gold,
                )
                case_results.append(
                    {
                        "comparison": comparison,
                        "faiss_gold": faiss_gold,
                        "pgvector_gold": pgvector_gold,
                    }
                )

        print_summary(case_results, len(documents))
    except (OSError, ValueError, RuntimeError, psycopg.Error) as error:
        print(f"비교 실패: {error}")


if __name__ == "__main__":
    main()
