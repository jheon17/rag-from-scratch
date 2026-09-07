"""Top-5 검색 결과에 Similarity Threshold를 적용해 변화를 비교한다."""

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME, embed_texts
from rag_basic.evaluation import EVAL_CASES
from rag_basic.retrieval import build_context, retrieve
from rag_basic.vector_search import build_index


TOP_K = 5

# 일반적인 최적값이 아니라 현재 모델과 평가 Case의 score 분포를 보기 위한 값이다.
THRESHOLD_VALUES = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]


def print_original_results(case: dict, results: list[dict]) -> None:
    """Threshold 적용 전 Case별 Top-5 결과와 gold score를 출력한다."""
    chunk_ids = [result["chunk_id"] for result in results]
    scores = [round(result["score"], 4) for result in results]

    print("\n" + "=" * 80)
    print(f"Case name: {case['name']}")
    print(f"Type: {case['type']}")
    print(f"Query: {case['query']}")
    print(f"Top-{TOP_K} chunk_id: {chunk_ids}")
    print(f"Top-{TOP_K} similarity score: {scores}")

    if case["type"] == "in_domain":
        gold_results = [
            result
            for result in results
            if result["chunk_id"] in case["expected_chunk_ids"]
        ]
        gold_scores = [
            {
                "chunk_id": result["chunk_id"],
                "rank": result["rank"],
                "score": round(result["score"], 4),
            }
            for result in gold_results
        ]

        print(f"expected_chunk_ids: {case['expected_chunk_ids']}")
        if gold_scores:
            print(f"검색된 gold evidence score: {gold_scores}")
            print(
                "가장 높은 gold evidence score: "
                f"{max(result['score'] for result in gold_results):.4f}"
            )
        else:
            print(f"gold evidence가 Top-{TOP_K}에 검색되지 않았습니다.")


def evaluate_in_domain(
    filtered_results: list[dict], expected_chunk_ids: list[int]
) -> dict:
    """Threshold 이후 gold evidence 보존 여부와 원래 순위 기반 RR을 계산한다."""
    first_gold_rank = next(
        (
            result["rank"]
            for result in filtered_results
            if result["chunk_id"] in expected_chunk_ids
        ),
        None,
    )

    return {
        "surviving_count": len(filtered_results),
        "gold_retained": first_gold_rank is not None,
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": (
            1.0 / first_gold_rank if first_gold_rank is not None else 0.0
        ),
    }


def context_length(filtered_results: list[dict]) -> int:
    """남은 검색 결과가 없으면 0, 있으면 생성된 Context 길이를 반환한다."""
    if not filtered_results:
        return 0
    return len(build_context(filtered_results))


def run_threshold(
    threshold: float,
    retrieved_cases: list[dict],
) -> dict:
    """저장된 Top-5 결과에 하나의 Threshold를 적용하고 지표를 집계한다."""
    in_domain_evaluations = []
    out_of_domain_evaluations = []

    print("\n" + "#" * 80)
    print(f"Threshold = {threshold:.2f}")

    for item in retrieved_cases:
        case = item["case"]
        filtered_results = [
            result
            for result in item["results"]
            if result["score"] >= threshold
        ]
        filtered_chunk_ids = [
            result["chunk_id"] for result in filtered_results
        ]
        filtered_context_length = context_length(filtered_results)

        print(f"\nCase: {case['name']}")
        print(f"남은 chunk_id: {filtered_chunk_ids}")
        print(f"surviving chunks: {len(filtered_results)}")
        print(f"Context 글자 수: {filtered_context_length}")

        if case["type"] == "in_domain":
            evaluation = evaluate_in_domain(
                filtered_results, case["expected_chunk_ids"]
            )
            evaluation["context_length"] = filtered_context_length
            in_domain_evaluations.append(evaluation)

            print(f"gold evidence retained: {evaluation['gold_retained']}")
            print(f"first gold rank: {evaluation['first_gold_rank']}")
            print(
                f"Reciprocal Rank: {evaluation['reciprocal_rank']:.4f}"
            )
        else:
            rejected = not filtered_results
            out_of_domain_evaluations.append({"rejected": rejected})
            print(f"rejected: {rejected}")

    retained_count = sum(
        evaluation["gold_retained"]
        for evaluation in in_domain_evaluations
    )
    mean_reciprocal_rank = sum(
        evaluation["reciprocal_rank"]
        for evaluation in in_domain_evaluations
    ) / len(in_domain_evaluations)
    mean_surviving_count = sum(
        evaluation["surviving_count"]
        for evaluation in in_domain_evaluations
    ) / len(in_domain_evaluations)
    mean_context_length = sum(
        evaluation["context_length"]
        for evaluation in in_domain_evaluations
    ) / len(in_domain_evaluations)
    rejected_count = sum(
        evaluation["rejected"]
        for evaluation in out_of_domain_evaluations
    )

    summary = {
        "threshold": threshold,
        "in_domain_count": len(in_domain_evaluations),
        "retained_count": retained_count,
        "mrr": mean_reciprocal_rank,
        "mean_surviving_count": mean_surviving_count,
        "mean_context_length": mean_context_length,
        "out_of_domain_count": len(out_of_domain_evaluations),
        "rejected_count": rejected_count,
    }

    print(f"\nThreshold = {threshold:.2f} 요약")
    print(f"in-domain Case 수: {summary['in_domain_count']}")
    print(
        "in-domain gold retained: "
        f"{retained_count}/{summary['in_domain_count']}"
    )
    print(f"MRR: {mean_reciprocal_rank:.4f}")
    print(f"평균 surviving chunks: {mean_surviving_count:.2f}")
    print(f"평균 Context 글자 수: {mean_context_length:.1f}")
    print(f"out-of-domain Case 수: {summary['out_of_domain_count']}")
    print(
        "out-of-domain rejected: "
        f"{rejected_count}/{summary['out_of_domain_count']}"
    )

    return summary


def print_comparison(summaries: list[dict]) -> None:
    """Threshold별 주요 결과를 일반 문자열 표로 출력한다."""
    print("\n" + "=" * 80)
    print(
        "Threshold | In-domain gold retained | MRR | OOD rejected | "
        "평균 surviving chunks | 평균 Context 글자 수"
    )
    print("-" * 105)
    for summary in summaries:
        print(
            f"{summary['threshold']:.2f} | "
            f"{summary['retained_count']}/{summary['in_domain_count']} | "
            f"{summary['mrr']:.4f} | "
            f"{summary['rejected_count']}/{summary['out_of_domain_count']} | "
            f"{summary['mean_surviving_count']:.2f} | "
            f"{summary['mean_context_length']:.1f}"
        )

    print(
        "Similarity score의 절대값은 Embedding 모델과 데이터에 따라 달라질 수 "
        "있으므로, 여기서 사용한 값을 다른 RAG 시스템에 그대로 적용할 수는 "
        "없습니다."
    )


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    model = SentenceTransformer(MODEL_NAME)
    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(model, chunk_texts, "passage")
    chunk_embeddings = np.ascontiguousarray(
        chunk_embeddings, dtype=np.float32
    )
    index = build_index(chunk_embeddings)

    retrieved_cases = []
    for case in EVAL_CASES:
        # Case당 한 번만 검색하고 모든 Threshold에서 동일 결과를 재사용한다.
        results = retrieve(case["query"], model, index, chunks, TOP_K)
        retrieved_cases.append({"case": case, "results": results})
        print_original_results(case, results)

    summaries = [
        run_threshold(threshold, retrieved_cases)
        for threshold in THRESHOLD_VALUES
    ]
    print_comparison(summaries)


if __name__ == "__main__":
    main()
