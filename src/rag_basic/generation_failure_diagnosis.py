"""특정 Generation 실패 Case의 실제 입력과 출력을 관찰한다."""

from rag_basic.api_evaluation import APIConnectionError, call_query_api
from rag_basic.evaluation import EVAL_CASES
from rag_basic.local_rag import NO_ANSWER, build_local_rag_prompt


CASE_NAME = "ai_assignment_submission"
QUERY_TERMS = ("과제", "제출", "그대로", "생성형 AI")


def get_target_case() -> dict:
    """기존 평가 Case에서 진단 대상을 이름으로 찾는다."""
    return next(case for case in EVAL_CASES if case["name"] == CASE_NAME)


def print_source(result: dict, expected_chunk_ids: list[int]) -> None:
    """검색 Source의 metadata, 전체 text와 단순 문자열 포함 여부를 출력한다."""
    source_number = result["rank"]
    print("\n" + "-" * 88)
    print(f"Source {source_number}")
    print(f"rank: {result['rank']}")
    print(f"chunk_id: {result['chunk_id']}")
    print(f"page_number: {result['page_number']}")
    print(f"GOLD: {result['chunk_id'] in expected_chunk_ids}")
    print(f"score: {result['score']}")
    print(f"cosine_distance: {result['cosine_distance']}")
    print("Query 관련 표현 포함 여부(단순 literal 확인):")
    for term in QUERY_TERMS:
        print(f"contains '{term}': {term in result['text']}")
    print("text:")
    print(result["text"])


def print_structural_diagnosis(
    results: list[dict], answer: str, expected_chunk_ids: list[int]
) -> None:
    """응답에서 직접 확인할 수 있는 구조적 사실만 출력한다."""
    first_gold_rank = next(
        (
            result["rank"]
            for result in results
            if result["chunk_id"] in expected_chunk_ids
        ),
        None,
    )
    source_1_is_gold = bool(results) and (
        results[0]["chunk_id"] in expected_chunk_ids
    )

    print("\n" + "=" * 88)
    print("구조 진단")
    print(f"gold chunk가 Top-5에 존재하는가: {first_gold_rank is not None}")
    print(f"첫 gold rank: {first_gold_rank}")
    print(f"Source 1이 gold인가: {source_1_is_gold}")
    print(f"answer에 Source 1 citation이 있는가: {'[Source 1]' in answer}")
    print(f"answer에 NO_ANSWER 문구가 포함되는가: {NO_ANSWER in answer}")
    print(f"answer 전체가 NO_ANSWER와 정확히 같은가: {answer == NO_ANSWER}")


def main() -> None:
    """실제 /query 응답과 동일한 Prompt를 한 Case에 대해 출력한다."""
    case = get_target_case()

    try:
        status, body = call_query_api(case)
    except APIConnectionError as error:
        print(f"오류: {error}")
        print("FastAPI 서버가 127.0.0.1:8000에서 실행 중인지 확인하세요.")
        raise SystemExit(1) from None

    print(f"Case name: {case['name']}")
    print(f"Query: {case['query']}")
    print(f"HTTP status: {status}")

    if status != 200:
        print(f"Response detail: {body.get('detail', '확인할 수 없음')}")
        raise SystemExit(1)

    results = body["results"]
    context = body["context"]
    answer = body["answer"]
    prompt = build_local_rag_prompt(body["query"], context)

    print(f"document_name: {body['document_name']}")
    print(f"top_k: {body['top_k']}")
    print(f"llm_model: {body['llm_model']}")
    print(f"answer: {answer}")
    print(f"context length: {len(context)}")
    print(f"prompt length: {len(prompt)}")

    print("\n" + "=" * 88)
    print("Top-5 전체 Source")
    for result in results:
        print_source(result, case["expected_chunk_ids"])

    print("\n===== CONTEXT START =====")
    print(context)
    print("===== CONTEXT END =====")

    print("\n===== PROMPT START =====")
    print(prompt)
    print("===== PROMPT END =====")

    print_structural_diagnosis(
        results,
        answer,
        case["expected_chunk_ids"],
    )


if __name__ == "__main__":
    main()
