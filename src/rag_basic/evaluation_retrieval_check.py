"""사람이 선정한 잠정 gold Chunk가 Top-5에 검색되는지 확인한다."""

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME, embed_texts
from rag_basic.retrieval import TOP_K, retrieve
from rag_basic.vector_search import build_index


CANDIDATE_CASES = [
    {
        "name": "creative_contribution_copyright",
        "query": "생성형 AI 결과물에 이용자가 창작적 표현을 추가하면 저작권을 인정받을 수 있나요?",
        "candidate_gold_chunk_id": 33,
    },
    {
        "name": "ai_assignment_submission",
        "query": "생성형 AI가 만든 결과물을 그대로 과제로 제출해도 되나요?",
        "candidate_gold_chunk_id": 68,
    },
    {
        "name": "midjourney_contest_controversy",
        "query": "미드저니를 사용한 작품이 미술대회에서 논란이 된 이유는 무엇인가요?",
        "candidate_gold_chunk_id": 65,
    },
    {
        "name": "fake_news_damage_report",
        "query": "생성형 AI로 만든 가짜 뉴스 피해는 어디에 신고하거나 상담할 수 있나요?",
        "candidate_gold_chunk_id": 104,
    },
    {
        "name": "generative_ai_work_benefits",
        "query": "생성형 AI를 업무에 활용하면 어떤 장점이 있나요?",
        "candidate_gold_chunk_id": 137,
    },
]
RESULT_SEPARATOR = "-" * 80
CASE_SEPARATOR = "=" * 80


def find_gold_rank(results: list[dict], candidate_gold_chunk_id: int) -> int | None:
    """검색 결과에서 잠정 gold Chunk가 처음 등장한 rank를 찾는다."""
    return next(
        (
            result["rank"]
            for result in results
            if result["chunk_id"] == candidate_gold_chunk_id
        ),
        None,
    )


def print_case(case: dict, results: list[dict], gold_rank: int | None) -> None:
    """한 후보 질문의 잠정 gold 확인 결과와 Top-5 전체 text를 출력한다."""
    print(CASE_SEPARATOR)
    print(f"case name: {case['name']}")
    print(f"query: {case['query']}")
    print(f"candidate_gold_chunk_id: {case['candidate_gold_chunk_id']}")
    print(f"gold chunk가 Top-{TOP_K} 안에 포함됐는가: {gold_rank is not None}")
    print(f"gold rank: {gold_rank if gold_rank is not None else '검색되지 않음'}")

    for result in results:
        print(RESULT_SEPARATOR)
        print(f"rank: {result['rank']}")
        print(f"score: {result['score']:.4f}")
        print(f"chunk_id: {result['chunk_id']}")
        print(f"page_number: {result['page_number']}")
        print("text:")
        print(result["text"])


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)

    model = SentenceTransformer(MODEL_NAME)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(model, chunk_texts, "passage")
    chunk_embeddings = np.ascontiguousarray(chunk_embeddings, dtype=np.float32)
    index = build_index(chunk_embeddings)

    gold_ranks = {}
    for case in CANDIDATE_CASES:
        results = retrieve(case["query"], model, index, chunks, TOP_K)
        gold_rank = find_gold_rank(
            results, case["candidate_gold_chunk_id"]
        )
        gold_ranks[case["name"]] = gold_rank
        print_case(case, results, gold_rank)

    print(CASE_SEPARATOR)
    print("요약:")
    print(f"전체 후보 질문 수: {len(CANDIDATE_CASES)}")
    print(
        f"gold chunk가 Top-{TOP_K}에 포함된 질문 수: "
        f"{sum(rank is not None for rank in gold_ranks.values())}"
    )
    print("질문별 gold rank:")
    for case_name, gold_rank in gold_ranks.items():
        rank_text = gold_rank if gold_rank is not None else "검색되지 않음"
        print(f"  {case_name}: {rank_text}")
    print(
        "잠정 gold 값은 자동으로 수정하지 않습니다. "
        "검색 결과 본문을 사람이 검토한 뒤 다음 단계에서 확정해야 합니다."
    )


if __name__ == "__main__":
    main()
