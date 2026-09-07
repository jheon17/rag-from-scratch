"""문서 전 범위에서 평가셋 검토용 대표 Chunk를 출력한다."""

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages


PAGE_RANGES = [
    (1, 15),
    (16, 30),
    (31, 45),
    (46, 60),
    (61, 74),
]
MIN_TEXT_LENGTH = 200
CANDIDATES_PER_RANGE = 3
SEPARATOR = "=" * 80


def select_representative_chunks(
    chunks: list[dict], start_page: int, end_page: int
) -> list[dict]:
    """페이지 중복을 피하며 길이가 충분한 대표 Chunk를 선택한다."""
    eligible_chunks = [
        chunk
        for chunk in chunks
        if start_page <= chunk["page_number"] <= end_page
        and len(chunk["text"]) >= MIN_TEXT_LENGTH
    ]

    # 이 기준은 평가 품질을 최적화하는 알고리즘이 아니라,
    # 사람이 문서 내용을 빠르게 훑기 위한 단순한 탐색 규칙이다.
    longest_chunk_by_page = {}
    for chunk in eligible_chunks:
        page_number = chunk["page_number"]
        current = longest_chunk_by_page.get(page_number)
        if current is None or len(chunk["text"]) > len(current["text"]):
            longest_chunk_by_page[page_number] = chunk

    page_representatives = list(longest_chunk_by_page.values())
    page_representatives.sort(
        key=lambda chunk: (
            -len(chunk["text"]),
            chunk["page_number"],
            chunk["chunk_id"],
        )
    )
    return page_representatives[:CANDIDATES_PER_RANGE]


def print_candidate(range_label: str, chunk: dict) -> None:
    """후보 Chunk의 metadata와 전체 text를 출력한다."""
    print(SEPARATOR)
    print(f"range: {range_label}")
    print(f"chunk_id: {chunk['chunk_id']}")
    print(f"page_number: {chunk['page_number']}")
    print(f"text_length: {len(chunk['text'])}")
    print("text:")
    print(chunk["text"])


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)
    candidate_counts = {}
    total_candidate_count = 0

    for start_page, end_page in PAGE_RANGES:
        range_label = f"{start_page}~{end_page}"
        candidates = select_representative_chunks(
            chunks, start_page, end_page
        )
        candidate_counts[range_label] = len(candidates)
        total_candidate_count += len(candidates)

        for chunk in candidates:
            print_candidate(range_label, chunk)

    if total_candidate_count:
        print(SEPARATOR)

    print("요약:")
    print(f"전체 Chunk 수: {len(chunks)}")
    print(f"출력된 후보 Chunk 수: {total_candidate_count}")
    print("page range별 후보 수:")
    for range_label, count in candidate_counts.items():
        print(f"  {range_label}: {count}")
    print(
        "평가 질문과 expected_chunk_ids는 자동 생성하지 않습니다. "
        "출력된 Chunk를 사람이 읽고 직접 결정해야 합니다."
    )


if __name__ == "__main__":
    main()
