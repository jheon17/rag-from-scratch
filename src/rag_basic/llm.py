"""검색된 Context만 근거로 Responses API에서 간결한 답변을 생성한다."""

import os

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME as EMBEDDING_MODEL_NAME
from rag_basic.embedding import embed_texts
from rag_basic.retrieval import TOP_K, build_context, retrieve
from rag_basic.vector_search import build_index


LLM_MODEL_NAME = "gpt-5.6-luna"
NORMAL_QUERY = "생성형 AI가 만든 이미지의 저작권은 누구에게 있나요?"
OUTSIDE_QUERY = "프랑스의 수도는 어디인가요?"
NO_ANSWER = "제공된 문서에서 확인할 수 없습니다."


def generate_answer(query: str, context: str, client: OpenAI, model: str) -> str:
    """이미 검색된 Context만 사용해 질문의 답변을 생성한다."""
    instructions = f"""당신은 제공된 문서를 근거로만 답하는 도우미입니다.
다음 규칙을 반드시 지키세요.
- 제공된 Context만 근거로 질문에 답합니다.
- Context에 없는 내용을 일반 지식으로 보충하거나 추측하지 않습니다.
- 답변 근거가 부족하면 정확히 '{NO_ANSWER}'라고만 답합니다.
- 답변에 근거로 사용한 Source 번호를 [Source 1] 형태로 표시합니다.
- 간결한 한국어로 답합니다."""

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=f"질문:\n{query}\n\nContext:\n{context}",
        max_output_tokens=300,
    )
    return response.output_text.strip()


def run_question(
    query: str,
    embedding_model: SentenceTransformer,
    index,
    chunks: list[dict],
    client: OpenAI,
) -> str:
    """질문을 검색하고 Context를 만든 뒤 LLM 답변을 출력한다."""
    results = retrieve(query, embedding_model, index, chunks, TOP_K)
    context = build_context(results)
    faiss_indices = [result["faiss_index"] for result in results]
    answer = generate_answer(query, context, client, LLM_MODEL_NAME)

    print(f"질문: {query}")
    print(f"Retrieval 결과의 FAISS index 목록: {faiss_indices}")
    print(f"Context 글자 수: {len(context)}")
    print(f"LLM 모델명: {LLM_MODEL_NAME}")
    print(f"생성된 답변: {answer}\n")

    return answer


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        print('설정 방법: export OPENAI_API_KEY="발급받은_API_Key"')
        print("환경변수를 설정한 뒤 다시 실행해 주세요.")
        return

    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    client = OpenAI()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(embedding_model, chunk_texts, "passage")
    chunk_embeddings = np.ascontiguousarray(chunk_embeddings, dtype=np.float32)
    index = build_index(chunk_embeddings)

    normal_answer = run_question(
        NORMAL_QUERY, embedding_model, index, chunks, client
    )
    outside_answer = run_question(
        OUTSIDE_QUERY, embedding_model, index, chunks, client
    )

    print("검증 결과:")
    print(f"  정상 질문의 답변이 비어 있지 않은가: {bool(normal_answer)}")
    print(f"  정상 질문의 답변에 [Source가 포함되는가: {'[Source' in normal_answer}")
    print(
        "  문서 밖 질문이 지정된 문장과 정확히 일치하는가: "
        f"{outside_answer == NO_ANSWER}"
    )


if __name__ == "__main__":
    main()
