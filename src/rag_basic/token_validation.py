"""E5 tokenizer를 기준으로 PDF Chunk의 토큰 수를 검증한다."""

from transformers import AutoTokenizer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME


TOKEN_LIMIT = 512


def count_tokens(tokenizer, text: str) -> int:
    """passage prefix와 특수 토큰을 포함한 전체 토큰 수를 계산한다."""
    input_ids = tokenizer.encode(
        f"passage: {text}",
        add_special_tokens=True,
        truncation=False,
    )
    return len(input_ids)


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)

    token_counts = [count_tokens(tokenizer, chunk["text"]) for chunk in chunks]
    max_token_count = max(token_counts)
    average_token_count = sum(token_counts) / len(token_counts)
    over_limit_chunks = [
        chunk
        for chunk, token_count in zip(chunks, token_counts)
        if token_count > TOKEN_LIMIT
    ]
    max_token_chunks = [
        chunk
        for chunk, token_count in zip(chunks, token_counts)
        if token_count == max_token_count
    ]

    max_chunk_locations = ", ".join(
        f"chunk_id={chunk['chunk_id']}, page_number={chunk['page_number']}"
        for chunk in max_token_chunks
    )
    over_limit_chunk_ids = [chunk["chunk_id"] for chunk in over_limit_chunks]

    print(f"Tokenizer: {MODEL_NAME}")
    print("Token 계산 범위: passage prefix와 특수 토큰 포함")
    print(f"전체 Chunk 개수: {len(chunks)}")
    print(f"최대 token 수: {max_token_count}")
    print(f"평균 token 수: {average_token_count:.1f}")
    print(f"{TOKEN_LIMIT} token 초과 Chunk 개수: {len(over_limit_chunks)}")
    print(f"최대 token Chunk: {max_chunk_locations}")
    print(f"{TOKEN_LIMIT} token 초과 chunk_id 목록: {over_limit_chunk_ids}")


if __name__ == "__main__":
    main()
