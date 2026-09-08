"""PostgreSQL pgvector에 저장된 Chunk를 cosine distance로 검색한다."""

import math
import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import CHUNK_OVERLAP, CHUNK_SIZE, PDF_PATH
from rag_basic.embedding import MODEL_NAME, embed_texts
from rag_basic.evaluation import EVAL_CASES
from rag_basic.retrieval import build_context


TOP_K = 5
EMBEDDING_DIMENSION = 384
PREVIEW_LENGTH = 120
CASE_NAMES = (
    "copyright_in_domain",
    "creative_contribution_copyright",
    "france_out_of_domain",
)
REQUIRED_ENV_VARS = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")


def get_database_config() -> dict:
    """환경변수에서 DB 연결 설정을 읽고 필수 항목을 검증한다."""
    missing_vars = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing_vars:
        raise ValueError(
            "필수 DB 환경변수가 설정되지 않았습니다: "
            + ", ".join(missing_vars)
        )

    return {
        "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
    }


def count_baseline_rows(conn: psycopg.Connection) -> int:
    """현재 baseline metadata와 일치하는 저장 행 수를 반환한다."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM rag_chunks
            WHERE document_name = %s
              AND embedding_model = %s
              AND chunk_size = %s
              AND chunk_overlap = %s;
            """,
            (PDF_PATH.name, MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP),
        )
        row_count = int(cursor.fetchone()[0])

    if row_count == 0:
        raise RuntimeError(
            "baseline dataset이 없습니다. pgvector_ingest를 먼저 실행해 주세요."
        )
    return row_count


def create_query_embedding(
    model: SentenceTransformer, query: str
) -> np.ndarray:
    """E5 query prefix로 질문 하나를 384차원 Vector로 변환한다."""
    embeddings = embed_texts(model, [query], "query")
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    if embeddings.shape != (1, EMBEDDING_DIMENSION):
        raise ValueError(
            "Query Embedding shape가 DB VECTOR(384)와 일치하지 않습니다: "
            f"{embeddings.shape}"
        )
    return embeddings[0]


def search_pgvector(
    conn: psycopg.Connection,
    query_embedding: np.ndarray,
    top_k: int = TOP_K,
    document_name: str = PDF_PATH.name,
) -> list[dict]:
    """지정 문서의 metadata 범위에서 cosine distance가 작은 Chunk를 검색한다."""
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if query_embedding.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            "검색 Vector가 384차원이 아닙니다: "
            f"{query_embedding.shape}"
        )

    with conn.cursor() as cursor:
        cursor.execute(
            """
            WITH query_vector AS (
                SELECT %s::vector AS embedding
            )
            SELECT
                rag_chunks.chunk_id,
                rag_chunks.page_number,
                rag_chunks.content,
                rag_chunks.embedding <=> query_vector.embedding
                    AS cosine_distance
            FROM rag_chunks
            CROSS JOIN query_vector
            WHERE rag_chunks.document_name = %s
              AND rag_chunks.embedding_model = %s
              AND rag_chunks.chunk_size = %s
              AND rag_chunks.chunk_overlap = %s
            ORDER BY rag_chunks.embedding <=> query_vector.embedding
            LIMIT %s;
            """,
            (
                query_embedding,
                document_name,
                MODEL_NAME,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
                top_k,
            ),
        )
        rows = cursor.fetchall()

    results = []
    for rank, (chunk_id, page_number, content, cosine_distance) in enumerate(
        rows, start=1
    ):
        distance = float(cosine_distance)
        # 기존 retrieve() 구조와 맞추기 위해 cosine similarity를 score로 사용한다.
        results.append(
            {
                "rank": rank,
                "score": 1.0 - distance,
                "cosine_distance": distance,
                "chunk_id": int(chunk_id),
                "page_number": int(page_number),
                "text": content,
            }
        )
    return results


def validate_results(results: list[dict], top_k: int) -> dict:
    """검색 개수, 순서, score 관계와 Context 생성을 검증한다."""
    ranks = [result["rank"] for result in results]
    distances = [result["cosine_distance"] for result in results]
    similarities = [result["score"] for result in results]
    context = build_context(results)

    return {
        "result_count_matches": len(results) == top_k,
        "ranks_match": ranks == list(range(1, top_k + 1)),
        "distance_ordered": distances == sorted(distances),
        "similarity_ordered": similarities == sorted(
            similarities, reverse=True
        ),
        "score_matches_distance": all(
            math.isclose(
                result["score"],
                1.0 - result["cosine_distance"],
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for result in results
        ),
        "context_created": bool(context),
        "context_length": len(context),
    }


def print_case_result(
    case: dict, results: list[dict], validation: dict
) -> None:
    """한 Case의 SQL 검색 결과와 검증 정보를 출력한다."""
    print("\n" + "=" * 80)
    print(f"Case: {case['name']}")
    print(f"Query: {case['query']}")
    print("Query Embedding shape: (1, 384)")
    print("\nrank | chunk_id | page | cosine distance | cosine similarity")

    for result in results:
        print(
            f"{result['rank']:>4} | {result['chunk_id']:>8} | "
            f"{result['page_number']:>4} | "
            f"{result['cosine_distance']:.4f} | {result['score']:.4f}"
        )
        print(f"  text={result['text'][:PREVIEW_LENGTH]}...")

    retrieved_chunk_ids = [result["chunk_id"] for result in results]
    print(f"\nTop-{TOP_K} chunk_id: {retrieved_chunk_ids}")

    if case["type"] == "in_domain":
        first_gold_rank = next(
            (
                result["rank"]
                for result in results
                if result["chunk_id"] in case["expected_chunk_ids"]
            ),
            None,
        )
        print(f"expected_chunk_ids: {case['expected_chunk_ids']}")
        print(f"gold Hit@{TOP_K}: {first_gold_rank is not None}")
        print(f"first gold rank: {first_gold_rank}")
    else:
        print(
            "OOD 질문에도 가장 가까운 Top-5가 반환됩니다. 이번 단계에서는 "
            "Similarity Threshold를 적용하지 않습니다."
        )

    print(f"Context 글자 수: {validation['context_length']}")
    print("검증 결과:")
    print(f"  정확히 Top-{TOP_K}를 반환했는가: {validation['result_count_matches']}")
    print(f"  rank가 1~{TOP_K}인가: {validation['ranks_match']}")
    print(f"  cosine distance가 오름차순인가: {validation['distance_ordered']}")
    print(f"  cosine similarity가 내림차순인가: {validation['similarity_ordered']}")
    print(f"  score ≈ 1 - distance인가: {validation['score_matches_distance']}")
    print(f"  build_context 생성 성공: {validation['context_created']}")


def select_cases() -> list[dict]:
    """EVAL_CASES에서 이번 검증에 사용할 세 Case를 순서대로 선택한다."""
    cases_by_name = {case["name"]: case for case in EVAL_CASES}
    missing_names = [name for name in CASE_NAMES if name not in cases_by_name]
    if missing_names:
        raise ValueError(f"평가 Case를 찾지 못했습니다: {missing_names}")
    return [cases_by_name[name] for name in CASE_NAMES]


def main() -> None:
    try:
        database_config = get_database_config()
        model = SentenceTransformer(MODEL_NAME)
        cases = select_cases()

        with psycopg.connect(**database_config) as conn:
            register_vector(conn)
            row_count = count_baseline_rows(conn)
            print("DB Connection 성공: True")
            print(f"baseline dataset row count: {row_count}")

            for case in cases:
                query_embedding = create_query_embedding(model, case["query"])
                results = search_pgvector(
                    conn, query_embedding, top_k=TOP_K
                )
                validation = validate_results(results, TOP_K)
                print_case_result(case, results, validation)

        print(
            "\n현재는 Vector index가 없는 exact search입니다. 데이터가 작기 "
            "때문에 검색 정확성과 SQL 동작 확인을 우선합니다."
        )
    except (OSError, ValueError, RuntimeError, psycopg.Error) as error:
        print(f"검색 실패: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
