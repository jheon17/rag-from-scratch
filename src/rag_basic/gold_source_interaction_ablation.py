"""gold Source 조합에 따른 Local Generation 동작을 관찰한다."""

from rag_basic.api_evaluation import (
    APIConnectionError,
    call_query_api,
    extract_citations,
)
from rag_basic.evaluation import EVAL_CASES
from rag_basic.local_llm import (
    LOCAL_MODEL_NAME,
    LOCAL_SEED,
    LOCAL_TEMPERATURE,
    LocalLLMError,
    generate_local_answer,
)
from rag_basic.local_rag import NO_ANSWER, build_local_rag_prompt
from rag_basic.retrieval import build_context


CASE_NAME = "ai_assignment_submission"
REPEAT_COUNT = 3
EXPECTED_TOP_5_CHUNK_IDS = [69, 68, 76, 70, 75]
DIFFERENCE_PREVIEW_RADIUS = 40


def get_target_case() -> dict:
    """기존 평가 Case에서 실험 대상을 이름으로 찾는다."""
    return next(case for case in EVAL_CASES if case["name"] == CASE_NAME)


def find_first_difference(left: str, right: str) -> int | None:
    """두 문자열이 처음 달라지는 위치를 반환한다."""
    for index, (left_character, right_character) in enumerate(zip(left, right)):
        if left_character != right_character:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def print_difference(name: str, left: str, right: str) -> None:
    """문자열 차이가 있으면 첫 위치와 주변 일부를 출력한다."""
    difference_index = find_first_difference(left, right)
    if difference_index is None:
        return

    start = max(0, difference_index - DIFFERENCE_PREVIEW_RADIUS)
    end = difference_index + DIFFERENCE_PREVIEW_RADIUS
    print(f"first {name} difference index: {difference_index}")
    print(f"HTTP-derived preview: {left[start:end]!r}")
    print(f"rebuilt preview: {right[start:end]!r}")


def select_sources(results: list[dict]) -> dict[int, dict]:
    """실제 Retrieval 결과에서 original rank 1, 2, 4를 선택한다."""
    results_by_rank = {result["rank"]: result for result in results}
    required_ranks = (1, 2, 4)
    missing_ranks = [rank for rank in required_ranks if rank not in results_by_rank]
    if missing_ranks:
        raise ValueError(f"Retrieval 결과에 필요한 rank가 없습니다: {missing_ranks}")
    return {rank: results_by_rank[rank] for rank in required_ranks}


def create_conditions(sources: dict[int, dict]) -> list[tuple[str, list[dict]]]:
    """선택한 Source로 네 가지 Context 조건을 만든다."""
    return [
        ("SOURCE_1_ONLY", [sources[1]]),
        ("SOURCE_1_PLUS_2", [sources[1], sources[2]]),
        ("SOURCE_1_PLUS_4", [sources[1], sources[4]]),
        (
            "SOURCE_1_PLUS_2_PLUS_4",
            [sources[1], sources[2], sources[4]],
        ),
    ]


def run_condition(
    name: str,
    results: list[dict],
    query: str,
    expected_chunk_ids: list[int],
) -> dict:
    """한 Source 조합을 세 번 생성하고 관찰 결과를 출력한다."""
    context = build_context(results)
    prompt = build_local_rag_prompt(query, context)
    source_numbers = [result["rank"] for result in results]
    context_source_numbers = set(source_numbers)
    chunk_ids = [result["chunk_id"] for result in results]
    gold_status = [
        result["chunk_id"] in expected_chunk_ids for result in results
    ]

    print("\n" + "=" * 88)
    print(f"Condition name: {name}")
    print(f"Source numbers: {source_numbers}")
    print(f"chunk_ids: {chunk_ids}")
    print(f"gold 여부: {gold_status}")
    print(f"context length: {len(context)}")
    print(f"prompt length: {len(prompt)}")

    runs = []
    for run_number in range(1, REPEAT_COUNT + 1):
        answer = generate_local_answer(prompt)
        citations = extract_citations(answer)
        citations_valid = bool(citations) and all(
            citation in context_source_numbers for citation in citations
        )
        run = {
            "answer": answer,
            "citations": citations,
            "contains_no_answer": NO_ANSWER in answer,
            "exact_no_answer": answer == NO_ANSWER,
            "citation_present": bool(citations),
            "citations_valid": citations_valid,
        }
        runs.append(run)

        print(f"\nRun number: {run_number}")
        print("Answer:")
        print(answer)
        print(f"Citation 번호: {citations}")
        print(f"NO_ANSWER phrase 포함 여부: {run['contains_no_answer']}")
        print(f"Exact NO_ANSWER 여부: {run['exact_no_answer']}")
        print(f"Citation present: {run['citation_present']}")
        print(f"Citation valid for current Context: {citations_valid}")

    summary = {
        "name": name,
        "unique_answer_count": len({run["answer"] for run in runs}),
        "no_answer_count": sum(run["contains_no_answer"] for run in runs),
        "exact_no_answer_count": sum(run["exact_no_answer"] for run in runs),
        "citation_present_count": sum(run["citation_present"] for run in runs),
        "citation_valid_count": sum(run["citations_valid"] for run in runs),
    }
    print("\nCondition Summary:")
    print(f"Unique answers: {summary['unique_answer_count']}")
    print(f"NO_ANSWER phrase: {summary['no_answer_count']}/{REPEAT_COUNT}")
    print(f"Exact NO_ANSWER: {summary['exact_no_answer_count']}/{REPEAT_COUNT}")
    print(f"Citation present: {summary['citation_present_count']}/{REPEAT_COUNT}")
    print(f"Citation valid: {summary['citation_valid_count']}/{REPEAT_COUNT}")
    return summary


def print_comparison(summaries: list[dict]) -> None:
    """네 조건과 Source 추가 전후의 refusal 관찰값을 출력한다."""
    print("\n" + "=" * 88)
    print("네 Condition 비교 Summary")
    summaries_by_name = {summary["name"]: summary for summary in summaries}
    for summary in summaries:
        print(
            f"{summary['name']}: "
            f"NO_ANSWER phrase {summary['no_answer_count']}/{REPEAT_COUNT}, "
            f"Exact NO_ANSWER {summary['exact_no_answer_count']}/{REPEAT_COUNT}, "
            f"Unique answers {summary['unique_answer_count']}, "
            f"Citation valid {summary['citation_valid_count']}/{REPEAT_COUNT}"
        )

    source_1_count = summaries_by_name["SOURCE_1_ONLY"]["no_answer_count"]
    comparisons = [
        ("Source 2 추가", "SOURCE_1_PLUS_2"),
        ("Source 4 추가", "SOURCE_1_PLUS_4"),
        ("Source 2와 Source 4 모두 추가", "SOURCE_1_PLUS_2_PLUS_4"),
    ]
    print("\nSource 추가 전후 refusal behavior:")
    for label, condition_name in comparisons:
        condition_count = summaries_by_name[condition_name]["no_answer_count"]
        if condition_count != source_1_count:
            observation = "refusal behavior 차이가 관찰됨"
        else:
            observation = "refusal behavior 차이가 관찰되지 않음"
        print(f"{label}: {observation}")


def main() -> None:
    """실제 Retrieval에서 gold Source 조합의 Generation을 비교한다."""
    case = get_target_case()

    print(f"Local model: {LOCAL_MODEL_NAME}")
    print(f"temperature: {LOCAL_TEMPERATURE}")
    print(f"seed: {LOCAL_SEED}")

    try:
        status, body = call_query_api(case)
    except APIConnectionError as error:
        print(f"오류: {error}")
        print("FastAPI 서버가 127.0.0.1:8000에서 실행 중인지 확인하세요.")
        raise SystemExit(1) from None

    print("\n최초 HTTP /query")
    print(f"HTTP status: {status}")
    if status != 200:
        print(f"Response detail: {body.get('detail', '확인할 수 없음')}")
        raise SystemExit(1)

    results = body["results"]
    chunk_ids = [result["chunk_id"] for result in results]
    print(f"Query: {body['query']}")
    print(f"Top-5 chunk_id: {chunk_ids}")
    print(f"기존 Top-5와 동일한가: {chunk_ids == EXPECTED_TOP_5_CHUNK_IDS}")
    print("실제 HTTP Top-5:")
    for result in results:
        is_gold = result["chunk_id"] in case["expected_chunk_ids"]
        print(
            f"rank={result['rank']}, chunk_id={result['chunk_id']}, "
            f"gold={is_gold}"
        )
    print("현재 HTTP answer:")
    print(body["answer"])

    rebuilt_full_context = build_context(results)
    http_context = body["context"]
    print("\nContext consistency")
    print(f"HTTP context length: {len(http_context)}")
    print(f"rebuilt context length: {len(rebuilt_full_context)}")
    print(f"HTTP context == rebuilt context: {http_context == rebuilt_full_context}")
    print_difference("context", http_context, rebuilt_full_context)

    http_prompt = build_local_rag_prompt(body["query"], http_context)
    rebuilt_prompt = build_local_rag_prompt(
        body["query"], rebuilt_full_context
    )
    print("\nPrompt consistency")
    print(f"HTTP-derived prompt length: {len(http_prompt)}")
    print(f"rebuilt prompt length: {len(rebuilt_prompt)}")
    print(f"HTTP-derived prompt == rebuilt prompt: {http_prompt == rebuilt_prompt}")
    print_difference("prompt", http_prompt, rebuilt_prompt)

    if http_context == rebuilt_full_context and http_prompt == rebuilt_prompt:
        print(
            "입력 Context/Prompt 문자열 차이로는 이전 answer 차이를 설명할 수 없음"
        )

    try:
        sources = select_sources(results)
        print("\n선택한 실제 Source:")
        for rank, result in sources.items():
            is_gold = result["chunk_id"] in case["expected_chunk_ids"]
            print(f"Source {rank}: chunk_id={result['chunk_id']}, gold={is_gold}")

        summaries = [
            run_condition(
                name,
                condition_results,
                case["query"],
                case["expected_chunk_ids"],
            )
            for name, condition_results in create_conditions(sources)
        ]
    except (LocalLLMError, ValueError) as error:
        print(f"오류: {error}")
        if isinstance(error, LocalLLMError):
            print("Ollama 서버와 qwen3:8b 모델 상태를 확인하세요.")
        raise SystemExit(1) from None

    print_comparison(summaries)


if __name__ == "__main__":
    main()
