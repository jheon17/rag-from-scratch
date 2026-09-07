"""Python 표준 라이브러리로 Ollama Local API를 호출한다."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/chat"
LOCAL_MODEL_NAME = "qwen3:8b"
TEST_PROMPT = "RAG에서 Retrieval이 필요한 이유를 두 문장으로 설명해줘."


class LocalLLMError(Exception):
    """Ollama 호출 또는 응답 처리에 실패했음을 나타낸다."""


def generate_local_answer(prompt: str) -> str:
    """Ollama /api/chat에 질문을 보내 최종 답변 문자열만 반환한다."""
    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
    }
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise LocalLLMError(
            f"Ollama가 HTTP 오류를 반환했습니다: {error.code} {error.reason}"
        ) from error
    except URLError as error:
        raise LocalLLMError(
            f"Ollama 서버에 연결할 수 없습니다: {error.reason}"
        ) from error
    except (TimeoutError, json.JSONDecodeError) as error:
        raise LocalLLMError(
            "Ollama 응답이 늦거나 올바른 JSON 형식이 아닙니다."
        ) from error

    try:
        answer = response_data["message"]["content"]
    except (KeyError, TypeError) as error:
        raise LocalLLMError(
            "Ollama 응답에서 message.content를 찾을 수 없습니다."
        ) from error

    if not isinstance(answer, str):
        raise LocalLLMError("Ollama의 message.content가 문자열이 아닙니다.")

    return answer.strip()


def main() -> None:
    try:
        answer = generate_local_answer(TEST_PROMPT)
    except LocalLLMError as error:
        print(f"오류: {error}")
        print("Ollama 서버가 실행 중인지 확인하세요.")
        print("ollama list로 qwen3:8b 모델이 설치됐는지 확인하세요.")
        raise SystemExit(1) from None

    print(f"Local model: {LOCAL_MODEL_NAME}")
    print(f"질문: {TEST_PROMPT}")
    print(f"답변: {answer}")


if __name__ == "__main__":
    main()
