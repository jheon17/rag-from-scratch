"""Load PDF files from data/ and preview text extracted page by page."""

from pathlib import Path

from pypdf import PdfReader


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PREVIEW_LENGTH = 300


def main() -> None:
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"PDF 파일을 찾지 못했습니다: {DATA_DIR}")
        return

    print(f"PDF 파일 {len(pdf_paths)}개를 찾았습니다: {DATA_DIR}")

    for pdf_path in pdf_paths:
        print(f"\n파일: {pdf_path.name}")
        reader = PdfReader(pdf_path)
        print(f"전체 페이지 수: {len(reader.pages)}")

        empty_pages: list[int] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            printable_text = (
                "".join(char for char in text if char.isprintable() or char.isspace())
                if text
                else ""
            )
            cleaned_text = " ".join(printable_text.split())

            if not cleaned_text:
                empty_pages.append(page_number)
                print(f"[페이지 {page_number}] 추출된 텍스트가 없습니다.")
                continue

            preview = cleaned_text[:PREVIEW_LENGTH]
            if len(cleaned_text) > PREVIEW_LENGTH:
                preview += "..."
            print(f"[페이지 {page_number}] {preview}")

        if empty_pages:
            page_numbers = ", ".join(map(str, empty_pages))
            print(f"텍스트가 없는 페이지: {page_numbers}")
        else:
            print("텍스트가 없는 페이지: 없음")


if __name__ == "__main__":
    main()
