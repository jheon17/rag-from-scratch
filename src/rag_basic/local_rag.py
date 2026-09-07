"""기존 Retrieval Context를 Ollama Local LLM에 전달해 답변을 생성한다."""

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME, embed_texts
from rag_basic.local_llm import (
    LOCAL_MODEL_NAME,
    LocalLLMError,
    generate_local_answer,
)
from rag_basic.retrieval import TOP_K, build_context, retrieve
from rag_basic.vector_search import build_index


NORMAL_QUERY = "생성형 AI가 만든 이미지의 저작권은 누구에게 있나요?"
OUTSIDE_QUERY = "프랑스의 수도는 어디인가요?"
NO_ANSWER = "제공된 문서에서 확인할 수 없습니다."


def build_local_rag_prompt(query: str, context: str) -> str:
    """질문과 검색 Context를 grounding 규칙이 포함된 Prompt로 만든다."""
    return f"""당신은 제공된 문서를 근거로만 답하는 도우미입니다.
다음 규칙을 반드시 지키세요.
- 제공된 Context만 근거로 질문에 답합니다.
- Context에 없는 내용을 일반 지식으로 보충하거나 추측하지 않습니다.
- 답변 근거가 부족하면 정확히 '{NO_ANSWER}'라고만 답합니다.
- 답변에 사용한 근거를 [Source 1] 형태로 표시합니다.
- 간결한 한국어로 답합니다.

질문:
{query}

Context:
{context}"""


def run_question(
    query: str,
    embedding_model: SentenceTransformer,
    index,
    chunks: list[dict],
) -> str:
    """질문을 검색하고 Context와 Prompt를 만들어 Local LLM을 호출한다."""
    results = retrieve(query, embedding_model, index, chunks, TOP_K)
    context = build_context(results)
    prompt = build_local_rag_prompt(query, context)
    answer = generate_local_answer(prompt)

    faiss_indices = [result["faiss_index"] for result in results]
    chunk_ids = [result["chunk_id"] for result in results]

    print(f"질문: {query}")
    print(f"Retrieval 결과의 FAISS index 목록: {faiss_indices}")
    print(f"Retrieval 결과의 chunk_id 목록: {chunk_ids}")
    print(f"Context 글자 수: {len(context)}")
    print(f"Local LLM 모델명: {LOCAL_MODEL_NAME}")
    print(f"생성된 답변: {answer}\n")

    return answer


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        print("data 폴더에 PDF 파일이 있는지 확인하세요.")
        return

    embedding_model = SentenceTransformer(MODEL_NAME)
    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(embedding_model, chunk_texts, "passage")
    chunk_embeddings = np.ascontiguousarray(chunk_embeddings, dtype=np.float32)
    index = build_index(chunk_embeddings)

    try:
        normal_answer = run_question(
            NORMAL_QUERY, embedding_model, index, chunks
        )
        outside_answer = run_question(
            OUTSIDE_QUERY, embedding_model, index, chunks
        )
    except LocalLLMError as error:
        print(f"오류: {error}")
        print("Ollama 서버가 실행 중인지 확인하세요.")
        print("ollama list로 qwen3:8b 모델이 설치됐는지 확인하세요.")
        raise SystemExit(1) from None

    print("검증 결과:")
    print(f"  정상 질문의 답변이 비어 있지 않은가: {bool(normal_answer)}")
    print(f"  정상 질문의 답변에 [Source가 포함되는가: {'[Source' in normal_answer}")
    print(
        "  문서 밖 질문이 지정된 문장과 정확히 일치하는가: "
        f"{outside_answer == NO_ANSWER}"
    )


if __name__ == "__main__":
    main()
