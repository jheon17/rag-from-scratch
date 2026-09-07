"""FAISS Top-5 후보를 CrossEncoder로 재정렬하고 gold 순위를 비교한다."""

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME, embed_texts
from rag_basic.evaluation import EVAL_CASES
from rag_basic.retrieval import retrieve
from rag_basic.vector_search import build_index


TOP_K = 5
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def rerank_results(
    query: str,
    results: list[dict],
    reranker: CrossEncoder,
) -> list[dict]:
    """동일한 FAISS 후보에 reranker score를 부여하고 새 순서로 반환한다."""
    pairs = [(query, result["text"]) for result in results]
    reranker_scores = reranker.predict(pairs, show_progress_bar=False)

    scored_results = []
    for result, reranker_score in zip(results, reranker_scores):
        scored_results.append(
            {
                "chunk_id": int(result["chunk_id"]),
                "original_rank": int(result["rank"]),
                "original_score": float(result["score"]),
                "rerank_score": float(reranker_score),
                "page_number": int(result["page_number"]),
                "text": result["text"],
            }
        )

    sorted_results = sorted(
        scored_results,
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return [
        {**result, "rerank_rank": rerank_rank}
        for rerank_rank, result in enumerate(sorted_results, start=1)
    ]


def evaluate_ranking(
    results: list[dict],
    expected_chunk_ids: list[int],
    rank_key: str,
) -> dict:
    """gold evidence의 포함 여부, 최초 순위, RR과 Top-1 여부를 계산한다."""
    first_gold_rank = next(
        (
            result[rank_key]
            for result in results
            if result["chunk_id"] in expected_chunk_ids
        ),
        None,
    )

    return {
        "hit_at_5": first_gold_rank is not None,
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": (
            1.0 / first_gold_rank if first_gold_rank is not None else 0.0
        ),
        "top_1_gold": first_gold_rank == 1,
    }


def print_case_result(
    case: dict,
    baseline_results: list[dict],
    reranked_results: list[dict],
    baseline_evaluation: dict,
    reranked_evaluation: dict,
) -> None:
    """한 Case의 원본 순위, 재정렬 순위와 평가 결과를 출력한다."""
    print("\n" + "=" * 80)
    print(f"Case name: {case['name']}")
    print(f"Query: {case['query']}")
    print(f"Gold evidence: {case['expected_chunk_ids']}")

    print("\nFAISS 원본:")
    print("rank | chunk_id | similarity score")
    for result in baseline_results:
        print(
            f"{result['rank']:>4} | {result['chunk_id']:>8} | "
            f"{result['score']:.4f}"
        )

    print("\nReranking 후:")
    print("rerank rank | chunk_id | original rank | reranker score")
    for result in reranked_results:
        print(
            f"{result['rerank_rank']:>11} | {result['chunk_id']:>8} | "
            f"{result['original_rank']:>13} | {result['rerank_score']:.4f}"
        )

    print(
        "\nBaseline first gold rank: "
        f"{baseline_evaluation['first_gold_rank']}"
    )
    print(
        "Reranked first gold rank: "
        f"{reranked_evaluation['first_gold_rank']}"
    )
    print(f"Baseline RR: {baseline_evaluation['reciprocal_rank']:.4f}")
    print(f"Reranked RR: {reranked_evaluation['reciprocal_rank']:.4f}")
    print(f"Baseline top1 gold: {baseline_evaluation['top_1_gold']}")
    print(f"Reranked top1 gold: {reranked_evaluation['top_1_gold']}")


def compare_rank_change(baseline_rank: int | None, reranked_rank: int | None) -> str:
    """동일 후보 집합 안에서 gold의 최초 순위 변화를 분류한다."""
    if baseline_rank == reranked_rank:
        return "same"
    if baseline_rank is None:
        return "improved"
    if reranked_rank is None:
        return "worsened"
    if reranked_rank < baseline_rank:
        return "improved"
    return "worsened"


def print_summary(case_results: list[dict]) -> None:
    """전체 Case의 baseline과 reranked 지표 및 순위 변화를 요약한다."""
    case_count = len(case_results)
    baseline_hit_count = sum(
        item["baseline"]["hit_at_5"] for item in case_results
    )
    reranked_hit_count = sum(
        item["reranked"]["hit_at_5"] for item in case_results
    )
    baseline_mrr = sum(
        item["baseline"]["reciprocal_rank"] for item in case_results
    ) / case_count
    reranked_mrr = sum(
        item["reranked"]["reciprocal_rank"] for item in case_results
    ) / case_count
    baseline_top_1_count = sum(
        item["baseline"]["top_1_gold"] for item in case_results
    )
    reranked_top_1_count = sum(
        item["reranked"]["top_1_gold"] for item in case_results
    )

    rank_changes = {"improved": 0, "same": 0, "worsened": 0}
    for item in case_results:
        change = compare_rank_change(
            item["baseline"]["first_gold_rank"],
            item["reranked"]["first_gold_rank"],
        )
        rank_changes[change] += 1

    print("\n" + "=" * 80)
    print("전체 요약")
    print("\nBaseline")
    print(f"Hit@5: {baseline_hit_count}/{case_count}")
    print(f"MRR: {baseline_mrr:.4f}")
    print(f"Top-1 gold: {baseline_top_1_count}/{case_count}")
    print("\nReranked")
    # 후보 집합이 같으므로 Reranking은 Hit@5가 아니라 gold 순위를 바꾼다.
    print(f"Hit@5: {reranked_hit_count}/{case_count}")
    print(f"MRR: {reranked_mrr:.4f}")
    print(f"Top-1 gold: {reranked_top_1_count}/{case_count}")
    print("\nRank 변화")
    print(f"개선: {rank_changes['improved']}")
    print(f"동일: {rank_changes['same']}")
    print(f"악화: {rank_changes['worsened']}")

    print("\nCase별 비교")
    print("Case | Baseline gold rank | Reranked gold rank | Baseline RR | Reranked RR")
    print("-" * 110)
    for item in case_results:
        print(
            f"{item['name']} | "
            f"{item['baseline']['first_gold_rank']} | "
            f"{item['reranked']['first_gold_rank']} | "
            f"{item['baseline']['reciprocal_rank']:.4f} | "
            f"{item['reranked']['reciprocal_rank']:.4f}"
        )


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    in_domain_cases = [
        case for case in EVAL_CASES if case["type"] == "in_domain"
    ]

    embedding_model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(
        embedding_model, chunk_texts, "passage"
    )
    chunk_embeddings = np.ascontiguousarray(
        chunk_embeddings, dtype=np.float32
    )
    index = build_index(chunk_embeddings)

    # FAISS는 Vector로 후보를 빠르게 찾는 1차 검색이다. CrossEncoder는 질문과
    # 후보 Chunk를 함께 읽어 더 세밀한 점수를 계산하므로, 계산량을 줄이기 위해
    # 전체 162개가 아닌 FAISS Top-5 후보에만 2차 정렬을 적용한다.
    case_results = []
    for case in in_domain_cases:
        baseline_results = retrieve(
            case["query"], embedding_model, index, chunks, TOP_K
        )
        reranked_results = rerank_results(
            case["query"], baseline_results, reranker
        )
        baseline_evaluation = evaluate_ranking(
            baseline_results, case["expected_chunk_ids"], "rank"
        )
        reranked_evaluation = evaluate_ranking(
            reranked_results, case["expected_chunk_ids"], "rerank_rank"
        )

        print_case_result(
            case,
            baseline_results,
            reranked_results,
            baseline_evaluation,
            reranked_evaluation,
        )
        case_results.append(
            {
                "name": case["name"],
                "baseline": baseline_evaluation,
                "reranked": reranked_evaluation,
            }
        )

    print_summary(case_results)


if __name__ == "__main__":
    main()
