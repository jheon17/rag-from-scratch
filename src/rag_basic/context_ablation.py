"""동일한 Retrieval 결과에서 Context 구성만 바꿔 Generation을 관찰한다."""

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


def get_target_case() -> dict:
    """기존 평가 Case에서 ablation 대상을 이름으로 찾는다."""
    return next(case for case in EVAL_CASES if case["name"] == CASE_NAME)


def create_conditions(
    results: list[dict], case: dict
) -> list[tuple[str, list[dict]]]:
    """동일한 검색 결과에서 비교할 세 가지 Context 조건을 만든다."""
    source_1_only = results[:1]
    gold_only = [
        result
        for result in results
        if result["chunk_id"] in case["expected_chunk_ids"]
    ]
    return [
        ("SOURCE_1_ONLY", source_1_only),
        ("GOLD_ONLY", gold_only),
        ("FULL_TOP_5", results),
    ]


def run_condition(
    name: str,
    results: list[dict],
    query: str,
) -> dict:
    """한 Context 조건을 세 번 생성하고 관찰 결과를 출력한다."""
    context = build_context(results)
    prompt = build_local_rag_prompt(query, context)
    chunk_ids = [result["chunk_id"] for result in results]
    source_numbers = [result["rank"] for result in results]
    context_source_numbers = set(source_numbers)

    print("\n" + "=" * 88)
    print(f"Condition name: {name}")
    print(f"chunk_ids: {chunk_ids}")
    print(f"original ranks: {source_numbers}")
    print(f"context source numbers: {source_numbers}")
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
        print(f"exact NO_ANSWER 여부: {run['exact_no_answer']}")
        print(f"Citation 존재: {run['citation_present']}")
        print(f"Citation이 Context Source 번호에 유효: {citations_valid}")

    summary = {
        "name": name,
        "runs": runs,
        "unique_answer_count": len({run["answer"] for run in runs}),
        "no_answer_count": sum(run["contains_no_answer"] for run in runs),
        "exact_no_answer_count": sum(run["exact_no_answer"] for run in runs),
        "citation_present_count": sum(run["citation_present"] for run in runs),
        "citation_valid_count": sum(run["citations_valid"] for run in runs),
    }

    print("\nCondition 요약:")
    print(f"Unique answer count: {summary['unique_answer_count']}")
    print(f"NO_ANSWER phrase 포함: {summary['no_answer_count']}/{REPEAT_COUNT}")
    print(f"Exact NO_ANSWER: {summary['exact_no_answer_count']}/{REPEAT_COUNT}")
    print(f"Citation present: {summary['citation_present_count']}/{REPEAT_COUNT}")
    print(
        "Citation valid for context: "
        f"{summary['citation_valid_count']}/{REPEAT_COUNT}"
    )
    return summary


def print_comparison(summaries: list[dict]) -> None:
    """세 조건의 refusal 관찰값을 나란히 출력한다."""
    print("\n" + "=" * 88)
    print("세 Condition 비교 요약")
    for summary in summaries:
        print(
            f"{summary['name']}: "
            f"NO_ANSWER phrase {summary['no_answer_count']}/{REPEAT_COUNT}, "
            f"exact NO_ANSWER {summary['exact_no_answer_count']}/{REPEAT_COUNT}, "
            f"Unique answers {summary['unique_answer_count']}"
        )

    refusal_counts = {summary["no_answer_count"] for summary in summaries}
    if len(refusal_counts) > 1:
        print("Context 구성에 따라 refusal behavior 차이가 관찰됨")
    else:
        print("이번 ablation에서는 Context 축소만으로 refusal behavior가 바뀌지 않음")

    print("\n사람이 판단해야 할 핵심 관찰:")
    print("- Source 1 only에서 직접 답하는가?")
    print("- Gold only에서 직접 답하는가?")
    print("- Full Top-5에서만 refusal하는가?")


def main() -> None:
    """실제 Top-5를 한 번 확보하고 세 Context 조건을 반복 비교한다."""
    case = get_target_case()

    print(f"Local model: {LOCAL_MODEL_NAME}")
    print(f"temperature: {LOCAL_TEMPERATURE}")
    print(f"seed: {LOCAL_SEED}")
    print("HTTP /query는 실제 Retrieval Top-5를 확보하는 데 사용합니다.")
    print("Direct Local Generation은 동일 결과에서 Context 영향만 분리합니다.")

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
    http_answer = body["answer"]
    print(f"Query: {body['query']}")
    print(f"Top-5 chunk_id: {chunk_ids}")
    print(f"기존 Top-5와 동일한가: {chunk_ids == EXPECTED_TOP_5_CHUNK_IDS}")
    print("Actual API answer:")
    print(http_answer)

    try:
        summaries = [
            run_condition(name, condition_results, case["query"])
            for name, condition_results in create_conditions(results, case)
        ]
    except LocalLLMError as error:
        print(f"오류: {error}")
        print("Ollama 서버가 실행 중인지 확인하세요.")
        print("ollama list로 qwen3:8b 모델 설치 여부를 확인하세요.")
        raise SystemExit(1) from None

    full_top_5_summary = next(
        summary for summary in summaries if summary["name"] == "FULL_TOP_5"
    )
    matches = [
        run["answer"] == http_answer for run in full_top_5_summary["runs"]
    ]
    print("\nFULL_TOP_5 direct answer와 HTTP answer 비교:")
    for run_number, matches_http in enumerate(matches, start=1):
        print(f"Run {run_number} exact match: {matches_http}")
    print(f"모든 direct answer가 HTTP answer와 일치: {all(matches)}")

    print_comparison(summaries)


if __name__ == "__main__":
    main()
