"""Chunk 크기별 통계와 page-level Retrieval 결과를 비교한다."""

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME, embed_texts
from rag_basic.evaluation import EVAL_CASES
from rag_basic.retrieval import build_context, retrieve
from rag_basic.vector_search import build_index


CHUNK_CONFIGS = [(300, 60), (500, 100), (800, 160)]
TOP_K = 5
PREVIEW_LENGTH = 120


def prepare_cases_with_gold_pages(
    cases: list[dict], baseline_chunks: list[dict]
) -> list[dict]:
    """baseline gold chunk ID를 Chunk 크기와 무관한 page 집합으로 변환한다."""
    chunk_pages = {
        chunk["chunk_id"]: chunk["page_number"] for chunk in baseline_chunks
    }
    prepared_cases = []

    for case in cases:
        missing_chunk_ids = [
            chunk_id
            for chunk_id in case["expected_chunk_ids"]
            if chunk_id not in chunk_pages
        ]
        if missing_chunk_ids:
            raise ValueError(
                "baseline Chunk에서 gold chunk_id를 찾지 못했습니다: "
                f"{missing_chunk_ids}"
            )

        gold_pages = {
            chunk_pages[chunk_id] for chunk_id in case["expected_chunk_ids"]
        }
        prepared_cases.append({"case": case, "gold_pages": gold_pages})

    return prepared_cases


def evaluate_page_retrieval(
    results: list[dict], gold_pages: set[int]
) -> dict:
    """검색된 페이지에서 첫 gold page 순위와 page-level RR을 계산한다."""
    first_gold_page_rank = next(
        (
            result["rank"]
            for result in results
            if result["page_number"] in gold_pages
        ),
        None,
    )

    return {
        "page_hit": first_gold_page_rank is not None,
        "first_gold_page_rank": first_gold_page_rank,
        "reciprocal_rank": (
            1.0 / first_gold_page_rank
            if first_gold_page_rank is not None
            else 0.0
        ),
    }


def print_case_result(
    case: dict,
    gold_pages: set[int],
    results: list[dict],
    evaluation: dict,
    context_length: int,
) -> None:
    """한 질문의 page-level 평가와 Top-5 text preview를 출력한다."""
    retrieved_pages = [result["page_number"] for result in results]
    retrieved_chunk_ids = [result["chunk_id"] for result in results]

    print(f"\nCase: {case['name']}")
    print(f"Gold pages: {sorted(gold_pages)}")
    print(f"Retrieved pages: {retrieved_pages}")
    print(f"Retrieved chunk_ids: {retrieved_chunk_ids}")
    print(f"First gold page rank: {evaluation['first_gold_page_rank']}")
    print(f"Page Hit@{TOP_K}: {evaluation['page_hit']}")
    print(f"Page RR: {evaluation['reciprocal_rank']:.4f}")
    print(f"Context 글자 수: {context_length}")
    print("Top-5 Chunk preview:")

    for result in results:
        preview = result["text"][:PREVIEW_LENGTH]
        print(
            f"  rank={result['rank']}, chunk_id={result['chunk_id']}, "
            f"page={result['page_number']}, score={result['score']:.4f}"
        )
        print(f"    {preview}...")


def run_config(
    pages: list[dict],
    prepared_cases: list[dict],
    model: SentenceTransformer,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """하나의 Chunk 설정으로 index를 만들고 6개 질문을 평가한다."""
    chunks = create_chunks(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunk_lengths = [len(chunk["text"]) for chunk in chunks]
    chunk_embeddings = embed_texts(
        model, [chunk["text"] for chunk in chunks], "passage"
    )
    chunk_embeddings = np.ascontiguousarray(
        chunk_embeddings, dtype=np.float32
    )
    index = build_index(chunk_embeddings)

    print("\n" + "=" * 80)
    print(f"Chunk config: {chunk_size} / {chunk_overlap}")
    print(f"전체 Chunk 수: {len(chunks)}")
    print(f"최소 Chunk 길이: {min(chunk_lengths)}")
    print(f"최대 Chunk 길이: {max(chunk_lengths)}")
    print(f"평균 Chunk 길이: {sum(chunk_lengths) / len(chunks):.1f}")

    evaluations = []
    context_lengths = []
    for item in prepared_cases:
        case = item["case"]
        results = retrieve(case["query"], model, index, chunks, TOP_K)
        context = build_context(results)
        evaluation = evaluate_page_retrieval(results, item["gold_pages"])

        evaluations.append(evaluation)
        context_lengths.append(len(context))
        print_case_result(
            case,
            item["gold_pages"],
            results,
            evaluation,
            len(context),
        )

    case_count = len(prepared_cases)
    page_hit_count = sum(
        evaluation["page_hit"] for evaluation in evaluations
    )
    page_mrr = sum(
        evaluation["reciprocal_rank"] for evaluation in evaluations
    ) / case_count
    average_chunk_length = sum(chunk_lengths) / len(chunks)
    average_context_length = sum(context_lengths) / case_count

    print(f"\nChunk Size: {chunk_size}")
    print(f"Overlap: {chunk_overlap}")
    print(f"전체 Chunk 수: {len(chunks)}")
    print(f"평균 Chunk 길이: {average_chunk_length:.1f}")
    print(f"Page Hit@{TOP_K}: {page_hit_count}/{case_count}")
    print(f"Page-level MRR: {page_mrr:.4f}")
    print(f"평균 Context 글자 수: {average_context_length:.1f}")

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks),
        "average_chunk_length": average_chunk_length,
        "page_hit_count": page_hit_count,
        "case_count": case_count,
        "page_mrr": page_mrr,
        "average_context_length": average_context_length,
    }


def print_comparison(summaries: list[dict]) -> None:
    """Chunk 설정별 통계와 page-level Retrieval 결과를 출력한다."""
    print("\n" + "=" * 80)
    print(
        "Chunk Size | Overlap | Chunk Count | Avg Chunk Length | "
        "Page Hit@5 | Page MRR | Avg Context Length"
    )
    print("-" * 110)
    for summary in summaries:
        print(
            f"{summary['chunk_size']} | "
            f"{summary['chunk_overlap']} | "
            f"{summary['chunk_count']} | "
            f"{summary['average_chunk_length']:.1f} | "
            f"{summary['page_hit_count']}/{summary['case_count']} | "
            f"{summary['page_mrr']:.4f} | "
            f"{summary['average_context_length']:.1f}"
        )

    print(
        "이번 실험은 Chunk Size 변경으로 기존 expected_chunk_ids를 그대로 "
        "사용할 수 없어, baseline gold chunk가 위치한 page_number를 기준으로 "
        "비교했습니다."
    )
    print(
        "Page-level Hit는 Chunk-level gold evidence 평가보다 느슨합니다. 같은 "
        "페이지의 관련 없는 Chunk가 검색될 수도 있으므로 Page Hit가 정확한 "
        "evidence retrieval을 의미하지는 않습니다."
    )


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    pages, _ = load_pages(PDF_PATH)
    in_domain_cases = [
        case for case in EVAL_CASES if case["type"] == "in_domain"
    ]
    baseline_chunks = create_chunks(pages)
    prepared_cases = prepare_cases_with_gold_pages(
        in_domain_cases, baseline_chunks
    )
    model = SentenceTransformer(MODEL_NAME)

    summaries = [
        run_config(
            pages,
            prepared_cases,
            model,
            chunk_size,
            chunk_overlap,
        )
        for chunk_size, chunk_overlap in CHUNK_CONFIGS
    ]
    print_comparison(summaries)


if __name__ == "__main__":
    main()
