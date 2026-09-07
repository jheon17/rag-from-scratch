"""동일한 in-domain 평가 Case로 Top-K별 Retrieval 결과를 비교한다."""

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME, embed_texts
from rag_basic.evaluation import EVAL_CASES
from rag_basic.retrieval import build_context, retrieve
from rag_basic.vector_search import build_index


TOP_K_VALUES = [1, 3, 5, 10]


def evaluate_retrieval(
    results: list[dict], expected_chunk_ids: list[int]
) -> dict:
    """검색 결과에서 gold evidence의 포함 여부와 최초 순위를 계산한다."""
    retrieved_chunk_ids = [result["chunk_id"] for result in results]
    first_gold_rank = next(
        (
            result["rank"]
            for result in results
            if result["chunk_id"] in expected_chunk_ids
        ),
        None,
    )

    return {
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "hit": first_gold_rank is not None,
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": (
            1.0 / first_gold_rank if first_gold_rank is not None else 0.0
        ),
    }


def print_case_result(
    top_k: int,
    case: dict,
    evaluation: dict,
    context_length: int,
) -> None:
    """한 Case의 Top-K 검색 결과와 Retrieval 지표를 출력한다."""
    print(f"\nCase: {case['name']}")
    print(f"검색 chunk_id: {evaluation['retrieved_chunk_ids']}")
    print(f"gold evidence: {case['expected_chunk_ids']}")
    print(f"first gold rank: {evaluation['first_gold_rank']}")
    print(f"Hit@{top_k}: {evaluation['hit']}")
    print(f"Reciprocal Rank: {evaluation['reciprocal_rank']:.4f}")
    print(f"Context 글자 수: {context_length}")


def run_top_k(
    top_k: int,
    cases: list[dict],
    model: SentenceTransformer,
    index,
    chunks: list[dict],
) -> dict:
    """하나의 K 값으로 모든 in-domain Case를 검색하고 요약한다."""
    hit_count = 0
    reciprocal_ranks = []
    context_lengths = []

    print("\n" + "=" * 80)
    print(f"Top-K: {top_k}")

    for case in cases:
        results = retrieve(case["query"], model, index, chunks, top_k)
        context = build_context(results)
        evaluation = evaluate_retrieval(results, case["expected_chunk_ids"])

        hit_count += int(evaluation["hit"])
        reciprocal_ranks.append(evaluation["reciprocal_rank"])
        context_lengths.append(len(context))
        print_case_result(top_k, case, evaluation, len(context))

    mean_reciprocal_rank = sum(reciprocal_ranks) / len(cases)
    mean_context_length = sum(context_lengths) / len(cases)

    print(f"\nTop-K = {top_k} 요약")
    print(f"in-domain Case 수: {len(cases)}")
    print(f"Hit@{top_k}: {hit_count}/{len(cases)}")
    print(f"MRR: {mean_reciprocal_rank:.4f}")
    print(f"평균 Context 글자 수: {mean_context_length:.1f}")

    return {
        "top_k": top_k,
        "hit_count": hit_count,
        "case_count": len(cases),
        "mrr": mean_reciprocal_rank,
        "mean_context_length": mean_context_length,
    }


def print_comparison(summaries: list[dict]) -> None:
    """Top-K별 요약 수치를 간단한 표 형태로 출력한다."""
    print("\n" + "=" * 80)
    print("K | Hit 통과 | MRR | 평균 Context 글자 수")
    print("-" * 48)
    for summary in summaries:
        print(
            f"{summary['top_k']:>2} | "
            f"{summary['hit_count']}/{summary['case_count']} | "
            f"{summary['mrr']:.4f} | "
            f"{summary['mean_context_length']:.1f}"
        )

    print(
        "현재 결과는 6개의 in-domain baseline Case만 사용하므로 "
        "전체 Retrieval 성능을 대표하지 않습니다."
    )


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    in_domain_cases = [
        case for case in EVAL_CASES if case["type"] == "in_domain"
    ]

    model = SentenceTransformer(MODEL_NAME)
    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(model, chunk_texts, "passage")
    chunk_embeddings = np.ascontiguousarray(
        chunk_embeddings, dtype=np.float32
    )
    index = build_index(chunk_embeddings)

    summaries = []
    for top_k in TOP_K_VALUES:
        summary = run_top_k(
            top_k, in_domain_cases, model, index, chunks
        )
        summaries.append(summary)

    print_comparison(summaries)


if __name__ == "__main__":
    main()
