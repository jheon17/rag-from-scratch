"""정규화된 Chunk Vector를 FAISS로 검색하고 NumPy 결과와 비교한다."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME, embed_texts


QUERY = "생성형 AI가 만든 이미지의 저작권은 누구에게 있나요?"
TOP_K = 5
PREVIEW_LENGTH = 150


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """float32 Chunk Vector를 IndexFlatIP에 등록한다."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def print_search_results(
    chunks: list[dict], scores: np.ndarray, faiss_indices: np.ndarray
) -> None:
    """FAISS 검색 결과와 연결된 Chunk metadata 일부를 출력한다."""
    print(f"\nFAISS Top-{len(faiss_indices[0])} 검색 결과:")
    for rank, (score, faiss_index) in enumerate(
        zip(scores[0], faiss_indices[0]), start=1
    ):
        chunk = chunks[int(faiss_index)]
        preview = chunk["text"][:PREVIEW_LENGTH]
        print(
            f"  순위={rank}, similarity={score:.4f}, "
            f"FAISS index={faiss_index}, chunk_id={chunk['chunk_id']}, "
            f"page_number={chunk['page_number']}"
        )
        print(f"    text={preview}...")


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)

    model = SentenceTransformer(MODEL_NAME)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(model, chunk_texts, "passage")
    query_embedding = embed_texts(model, [QUERY], "query")

    # FAISS의 기본 입력 형식에 맞게 연속된 float32 배열임을 보장한다.
    chunk_embeddings = np.ascontiguousarray(chunk_embeddings, dtype=np.float32)
    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

    index = build_index(chunk_embeddings)
    scores, faiss_indices = index.search(query_embedding, TOP_K)

    numpy_similarities = chunk_embeddings @ query_embedding[0]
    numpy_indices = np.argsort(-numpy_similarities)[:TOP_K]
    numpy_top_scores = numpy_similarities[numpy_indices]
    index_order_matches = np.array_equal(faiss_indices[0], numpy_indices)
    similarity_scores_match = np.allclose(
        scores[0], numpy_top_scores, rtol=0.0, atol=1e-6
    )

    print(f"검색 질문: {QUERY}")
    print(f"전체 Chunk 개수: {len(chunks)}")
    print(f"Embedding shape: {chunk_embeddings.shape}")
    print(f"Embedding dtype: {chunk_embeddings.dtype}")
    print(f"FAISS index Vector 수: {index.ntotal}")
    print(f"Query Embedding shape: {query_embedding.shape}")
    print_search_results(chunks, scores, faiss_indices)
    result_count = len(faiss_indices[0])
    print(f"\nNumPy Top-{result_count} index: {numpy_indices.tolist()}")
    print(f"FAISS Top-{result_count} index: {faiss_indices[0].tolist()}")
    print(f"FAISS와 NumPy의 index 순서가 동일한가: {index_order_matches}")
    print(f"FAISS와 NumPy의 similarity 점수가 동일한가: {similarity_scores_match}")

    if index_order_matches:
        print("검증 결과: index 순서까지 동일")
    elif similarity_scores_match:
        print(
            "검증 결과: index 순서는 다르지만 similarity 점수는 동일 "
            "(동점 Vector의 정렬 순서 차이일 수 있음)"
        )
    else:
        print(
            "검증 결과: similarity 점수까지 다름 "
            "(실제 검색 결과 불일치 가능성이 있으므로 추가 확인 필요)"
        )


if __name__ == "__main__":
    main()
