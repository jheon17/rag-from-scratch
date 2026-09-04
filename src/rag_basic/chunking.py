"""PDF 텍스트를 페이지별로 나누어 단순한 글자 수 기준 Chunk를 만든다."""

from pathlib import Path

from pypdf import PdfReader


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PDF_PATH = DATA_DIR / "ai_ethics_guide.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
PREVIEW_LENGTH = 120


def clean_text(text: str | None) -> str:
    """추출된 텍스트에서 제어문자와 불필요한 공백을 제거한다."""
    if not text:
        return ""

    printable_text = "".join(
        char for char in text if char.isprintable() or char.isspace()
    )
    return " ".join(printable_text.split())


def load_pages(pdf_path: Path) -> tuple[list[dict], list[int]]:
    """PDF에서 텍스트가 있는 페이지와 빈 페이지 번호를 읽는다."""
    reader = PdfReader(pdf_path)
    pages: list[dict] = []
    empty_pages: list[int] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text())

        if text:
            pages.append({"page_number": page_number, "text": text})
        else:
            empty_pages.append(page_number)

    return pages, empty_pages


def create_chunks(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """페이지 경계를 유지하면서 글자 수를 기준으로 Chunk를 만든다."""
    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap은 0 이상이고 chunk_size보다 작아야 합니다.")

    chunks: list[dict] = []
    step = chunk_size - chunk_overlap

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            chunk_text = text[start : start + chunk_size]
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "page_number": page["page_number"],
                    "text": chunk_text,
                }
            )

            if start + chunk_size >= len(text):
                break
            start += step

    return chunks


def print_validation(chunks: list[dict], empty_pages: list[int]) -> None:
    """생성된 Chunk의 길이, 일부 내용, overlap과 빈 페이지를 확인한다."""
    chunk_lengths = [len(chunk["text"]) for chunk in chunks]

    print(f"전체 Chunk 개수: {len(chunks)}")
    print("Chunk별 글자 수:")
    for start in range(0, len(chunks), 10):
        group = chunks[start : start + 10]
        summary = ", ".join(
            f"{chunk['chunk_id']}번={len(chunk['text'])}자" for chunk in group
        )
        print(f"  {summary}")

    print(f"최소 Chunk 길이: {min(chunk_lengths)}자")
    print(f"최대 Chunk 길이: {max(chunk_lengths)}자")
    print(f"평균 Chunk 길이: {sum(chunk_lengths) / len(chunk_lengths):.1f}자")

    print("\n첫 번째 3개 Chunk:")
    for chunk in chunks[:3]:
        preview = chunk["text"][:PREVIEW_LENGTH]
        print(
            f"  chunk_id={chunk['chunk_id']}, "
            f"page_number={chunk['page_number']}, text={preview}..."
        )

    overlap_pairs = []
    failed_pairs = []
    for previous, current in zip(chunks, chunks[1:]):
        if previous["page_number"] != current["page_number"]:
            continue

        overlap_pairs.append((previous, current))
        if previous["text"][-CHUNK_OVERLAP:] != current["text"][:CHUNK_OVERLAP]:
            failed_pairs.append((previous["chunk_id"], current["chunk_id"]))

    print(f"\n같은 페이지의 인접 Chunk 쌍: {len(overlap_pairs)}개")
    if failed_pairs:
        print(f"100자 overlap 불일치: {failed_pairs}")
    else:
        print("100자 overlap 일치: 모든 인접 Chunk 쌍에서 확인")

    if overlap_pairs:
        previous, current = overlap_pairs[0]
        sample = current["text"][:CHUNK_OVERLAP]
        print(
            f"  예시: {previous['chunk_id']}번 → {current['chunk_id']}번 "
            f"overlap({len(sample)}자)={sample}"
        )

    chunk_page_numbers = {chunk["page_number"] for chunk in chunks}
    chunks_from_empty_pages = sorted(chunk_page_numbers.intersection(empty_pages))
    print(f"\n텍스트가 없는 페이지: {empty_pages}")
    if chunks_from_empty_pages:
        print(f"빈 페이지에서 생성된 Chunk 발견: {chunks_from_empty_pages}")
    else:
        print("빈 페이지의 Chunk 미생성: 확인")


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    pages, empty_pages = load_pages(PDF_PATH)
    chunks = create_chunks(pages)

    print(f"PDF 파일: {PDF_PATH.name}")
    print(f"청킹 기준: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print_validation(chunks, empty_pages)


if __name__ == "__main__":
    main()
