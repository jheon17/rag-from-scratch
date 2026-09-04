"""FAISS 검색 결과를 Chunk metadata와 연결하고 Context를 만든다."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME, embed_texts
from rag_basic.vector_search import build_index


QUERY = "생성형 AI가 만든 이미지의 저작권은 누구에게 있나요?"
TOP_K = 5
PREVIEW_LENGTH = 150
CONTEXT_PREVIEW_LENGTH = 500
EXPECTED_FAISS_INDICES = [27, 38, 2, 92, 34]
REQUIRED_KEYS = {
    "rank",
    "score",
    "faiss_index",
    "chunk_id",
    "page_number",
    "text",
}


def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """질문과 가까운 Chunk를 검색해 text와 metadata를 함께 반환한다."""
    if top_k <= 0 or top_k > index.ntotal:
        raise ValueError("top_k는 1 이상이고 등록된 Vector 수 이하여야 합니다.")

    query_embedding = embed_texts(model, [query], "query")
    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
    scores, faiss_indices = index.search(query_embedding, top_k)

    results: list[dict] = []
    for rank, (score, faiss_index) in enumerate(
        zip(scores[0], faiss_indices[0]), start=1
    ):
        chunk = chunks[int(faiss_index)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "faiss_index": int(faiss_index),
                "chunk_id": int(chunk["chunk_id"]),
                "page_number": int(chunk["page_number"]),
                "text": chunk["text"],
            }
        )

    return results


def build_context(results: list[dict]) -> str:
    """검색 결과의 rank 순서대로 전체 Chunk text를 하나로 합친다."""
    sources = []
    for result in sorted(results, key=lambda result: result["rank"]):
        source = (
            f"[Source {result['rank']} | page={result['page_number']} | "
            f"chunk_id={result['chunk_id']}]\n"
            f"{result['text']}"
        )
        sources.append(source)

    return "\n\n".join(sources)


def print_results(results: list[dict]) -> None:
    """검색 결과의 metadata와 text 앞부분만 출력한다."""
    for result in results:
        preview = result["text"][:PREVIEW_LENGTH]
        print(
            f"  rank={result['rank']}, score={result['score']:.4f}, "
            f"faiss_index={result['faiss_index']}, "
            f"chunk_id={result['chunk_id']}, "
            f"page_number={result['page_number']}"
        )
        print(f"    text={preview}...")


def print_validation(results: list[dict], context: str) -> None:
    """Retrieval 결과와 Context가 요구한 구조를 만족하는지 확인한다."""
    ranks = [result["rank"] for result in results]
    faiss_indices = [result["faiss_index"] for result in results]
    required_keys_exist = all(REQUIRED_KEYS.issubset(result) for result in results)
    all_texts_in_context = all(result["text"] in context for result in results)

    source_positions = [
        context.find(
            f"[Source {result['rank']} | page={result['page_number']} | "
            f"chunk_id={result['chunk_id']}]"
        )
        for result in results
    ]
    source_order_is_preserved = source_positions == sorted(source_positions) and all(
        position >= 0 for position in source_positions
    )

    print("\n검증 결과:")
    print(f"  Retrieval 결과가 정확히 {TOP_K}개인가: {len(results) == TOP_K}")
    print(f"  rank가 1~{TOP_K} 순서인가: {ranks == list(range(1, TOP_K + 1))}")
    print(f"  모든 결과에 필요한 key가 존재하는가: {required_keys_exist}")
    print(f"  이전 Vector Search index와 동일한가: {faiss_indices == EXPECTED_FAISS_INDICES}")
    print(f"  Context에 모든 Chunk text가 포함되는가: {all_texts_in_context}")
    print(f"  Context의 Source 순서가 rank와 동일한가: {source_order_is_preserved}")


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

    results = retrieve(QUERY, model, index, chunks, TOP_K)
    context = build_context(results)
    reversed_results = list(reversed(results))
    reversed_context = build_context(reversed_results)
    reversed_context_source_order = [
        line.removeprefix("[").split(" |", maxsplit=1)[0]
        for line in reversed_context.splitlines()
        if line.startswith("[Source ")
    ]
    expected_source_order = [f"Source {rank}" for rank in range(1, TOP_K + 1)]
    faiss_indices = [result["faiss_index"] for result in results]

    print(f"검색 질문: {QUERY}")
    print(f"Retrieval 결과 개수: {len(results)}")
    print_results(results)
    print(f"\n검색된 FAISS index 목록: {faiss_indices}")
    print(f"생성된 Context 전체 글자 수: {len(context)}")
    print(f"Context 앞부분:\n{context[:CONTEXT_PREVIEW_LENGTH]}...")
    print_validation(results, context)
    print(f"  역순 입력 Context의 Source 순서: {reversed_context_source_order}")
    print(
        "  역순 입력도 Source 1~5 순서로 배치되는가: "
        f"{reversed_context_source_order == expected_source_order}"
    )


if __name__ == "__main__":
    main()
