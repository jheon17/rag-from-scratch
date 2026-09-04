"""문장과 PDF Chunk를 E5 모델로 Embedding하고 결과를 검증한다."""

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages


MODEL_NAME = "intfloat/multilingual-e5-small"


def embed_texts(
    model: SentenceTransformer, texts: list[str], prefix: str
) -> np.ndarray:
    """각 문자열에 E5 입력 prefix를 붙여 정규화된 Vector로 변환한다."""
    prefixed_texts = [f"{prefix}: {text}" for text in texts]
    return model.encode(
        prefixed_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def run_sentence_experiment(model: SentenceTransformer) -> None:
    """의미가 가까운 문장과 먼 문장의 cosine similarity를 비교한다."""
    sentence_a = "개인정보를 안전하게 보호해야 합니다."
    sentence_b = "개인 데이터를 안전하게 관리해야 합니다."
    sentence_c = "축구 경기에서 공격수가 골을 넣었습니다."

    query_embedding = embed_texts(model, [sentence_a], "query")[0]
    passage_embeddings = embed_texts(
        model, [sentence_b, sentence_c], "passage"
    )

    # Vector를 길이 1로 정규화했으므로 내적이 cosine similarity와 같다.
    similarity_ab = float(query_embedding @ passage_embeddings[0])
    similarity_ac = float(query_embedding @ passage_embeddings[1])

    print("문장 의미 유사도 실험:")
    print(f"  similarity(A, B): {similarity_ab:.4f}")
    print(f"  similarity(A, C): {similarity_ac:.4f}")
    print(f"  A와 B가 더 유사한가: {similarity_ab > similarity_ac}")


def validate_chunk_embeddings(chunks: list[dict], embeddings: np.ndarray) -> None:
    """Chunk Embedding의 개수, 형태와 숫자 상태를 확인한다."""
    vector_norms = np.linalg.norm(embeddings, axis=1)
    has_nan = bool(np.isnan(embeddings).any())
    has_infinity = bool(np.isinf(embeddings).any())
    has_zero_vector = bool(np.isclose(vector_norms, 0.0).any())
    is_normalized = bool(np.allclose(vector_norms, 1.0, atol=1e-5))

    print("\nChunk Embedding 검증:")
    print(f"  전체 Chunk 개수: {len(chunks)}")
    print(f"  생성된 Embedding 개수: {len(embeddings)}")
    print(f"  Embedding 배열 shape: {embeddings.shape}")
    print(f"  Vector 하나의 차원 수: {embeddings.shape[1]}")
    print(f"  첫 번째 Vector의 앞 10개 숫자: {embeddings[0][:10]}")
    print(f"  NaN 포함 여부: {has_nan}")
    print(f"  무한대 포함 여부: {has_infinity}")
    print(f"  영벡터 포함 여부: {has_zero_vector}")
    print(f"  모든 Vector의 길이가 약 1인가: {is_normalized}")


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    model = SentenceTransformer(MODEL_NAME)
    print(f"Embedding 모델: {MODEL_NAME}")
    print(f"사용 device: {model.device.type}")

    run_sentence_experiment(model)

    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(model, chunk_texts, "passage")
    validate_chunk_embeddings(chunks, chunk_embeddings)


if __name__ == "__main__":
    main()
