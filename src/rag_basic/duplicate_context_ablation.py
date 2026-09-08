"""Source 4의 중복 부분과 새 부분을 나눠 Generation을 관찰한다."""

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
from rag_basic.source4_content_diagnosis import longest_suffix_prefix_overlap


CASE_NAME = "ai_assignment_submission"
REPEAT_COUNT = 3
PREVIOUS_ORIGINAL_REFUSAL_COUNT = 3


def get_target_case() -> dict:
    """기존 평가 Case에서 실험 대상을 이름으로 찾는다."""
    return next(case for case in EVAL_CASES if case["name"] == CASE_NAME)


def select_source(results: list[dict], rank: int) -> dict:
    """실제 HTTP Retrieval 결과에서 original rank에 해당하는 Source를 찾는다."""
    source = next((result for result in results if result["rank"] == rank), None)
    if source is None:
        raise ValueError(f"Retrieval 결과에 Source {rank}가 없습니다.")
    return source


def split_source_4(source_1: dict, source_4: dict) -> tuple[str, str]:
    """Source 4를 Source 1과 겹치는 부분과 새 부분으로 분리한다."""
    source_1_text = source_1["text"]
    source_4_text = source_4["text"]
    overlap_length = longest_suffix_prefix_overlap(
        source_1_text,
        source_4_text,
    )
    overlap_text = source_4_text[:overlap_length]
    new_portion_text = source_4_text[overlap_length:]

    print("\n실제 overlap 계산:")
    print(f"Source 1 length: {len(source_1_text)}")
    print(f"Source 4 length: {len(source_4_text)}")
    print(f"longest exact overlap: {overlap_length}")
    print(f"overlap length: {len(overlap_text)}")
    print(f"new portion length: {len(new_portion_text)}")
    print("===== SOURCE 4 OVERLAP PORTION START =====")
    print(overlap_text)
    print("===== SOURCE 4 OVERLAP PORTION END =====")
    print("===== SOURCE 4 NEW PORTION START =====")
    print(new_portion_text)
    print("===== SOURCE 4 NEW PORTION END =====")

    if overlap_length == 0:
        exact_match = overlap_text == ""
    else:
        exact_match = source_1_text[-overlap_length:] == overlap_text
    print(f"Source 1 tail == Source 4 overlap portion: {exact_match}")
    if not exact_match:
        raise ValueError("계산한 overlap portion이 Source 1 tail과 다릅니다.")

    return overlap_text, new_portion_text


def create_conditions(
    source_1: dict,
    source_4: dict,
    overlap_text: str,
    new_portion_text: str,
) -> list[dict]:
    """원본 두 조건과 메모리에서 만든 synthetic 두 조건을 구성한다."""
    overlap_only_source_4 = dict(source_4)
    overlap_only_source_4["text"] = overlap_text
    new_only_source_4 = dict(source_4)
    new_only_source_4["text"] = new_portion_text

    return [
        {
            "name": "SOURCE_1_ONLY",
            "results": [source_1],
            "synthetic": False,
            "meaning": "baseline",
        },
        {
            "name": "SOURCE_1_PLUS_4_ORIGINAL",
            "results": [source_1, source_4],
            "synthetic": False,
            "meaning": "이전 refusal behavior 재현 여부",
        },
        {
            "name": "SOURCE_1_PLUS_4_OVERLAP_ONLY",
            "results": [source_1, overlap_only_source_4],
            "synthetic": True,
            "meaning": "near-duplicate overlap portion만 추가한 비교",
        },
        {
            "name": "SOURCE_1_PLUS_4_NEW_ONLY",
            "results": [source_1, new_only_source_4],
            "synthetic": True,
            "meaning": "Source 4의 새 부분만 추가한 비교",
        },
    ]


def run_condition(condition: dict, query: str) -> dict:
    """한 Context 조건을 세 번 실행하고 관찰 지표를 출력한다."""
    name = condition["name"]
    results = condition["results"]
    context = build_context(results)
    prompt = build_local_rag_prompt(query, context)
    source_numbers = [result["rank"] for result in results]
    context_source_numbers = set(source_numbers)
    chunk_ids = [result["chunk_id"] for result in results]
    text_lengths = [len(result["text"]) for result in results]

    print("\n" + "=" * 88)
    print(f"Condition name: {name}")
    print(f"의미: {condition['meaning']}")
    print(f"Synthetic diagnostic Context: {condition['synthetic']}")
    print(f"Source numbers: {source_numbers}")
    print(f"chunk_ids: {chunk_ids}")
    print(f"text lengths: {text_lengths}")
    print(f"context length: {len(context)}")
    print(f"prompt length: {len(prompt)}")
    if condition["synthetic"]:
        print(f"Synthetic Source 4 text length: {len(results[1]['text'])}")
        print("Synthetic Source 4 text 전체:")
        print(results[1]["text"])

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
    """네 조건의 refusal count와 이전 ORIGINAL 관찰을 비교한다."""
    print("\n" + "=" * 88)
    print("네 Condition 비교표")
    print("Condition | NO_ANSWER | Exact | Unique | Citation valid")
    for summary in summaries:
        print(
            f"{summary['name']} | "
            f"{summary['no_answer_count']}/{REPEAT_COUNT} | "
            f"{summary['exact_no_answer_count']}/{REPEAT_COUNT} | "
            f"{summary['unique_answer_count']} | "
            f"{summary['citation_valid_count']}/{REPEAT_COUNT}"
        )

    summaries_by_name = {summary["name"]: summary for summary in summaries}
    original_count = summaries_by_name[
        "SOURCE_1_PLUS_4_ORIGINAL"
    ]["no_answer_count"]
    print("\n이전 SOURCE_1_PLUS_4 패턴과 비교:")
    print(
        "Previous SOURCE_1_PLUS_4 refusal pattern: "
        f"{PREVIOUS_ORIGINAL_REFUSAL_COUNT}/{REPEAT_COUNT}"
    )
    print(
        "Current ORIGINAL refusal pattern: "
        f"{original_count}/{REPEAT_COUNT}"
    )
    print(
        "same observed pattern: "
        f"{original_count == PREVIOUS_ORIGINAL_REFUSAL_COUNT}"
    )

    print("\n관찰 문장:")
    for condition_name in (
        "SOURCE_1_PLUS_4_ORIGINAL",
        "SOURCE_1_PLUS_4_OVERLAP_ONLY",
        "SOURCE_1_PLUS_4_NEW_ONLY",
    ):
        count = summaries_by_name[condition_name]["no_answer_count"]
        behavior = "관찰됨" if count else "관찰되지 않음"
        print(f"{condition_name}: refusal behavior {behavior} ({count}/3)")

    print("\n사람이 판단해야 할 핵심:")
    print("- Original Source 4가 이전 refusal을 재현하는가?")
    print("- Overlap-only portion에서도 refusal이 관찰되는가?")
    print("- New-only portion에서도 refusal이 관찰되는가?")


def main() -> None:
    """실제 Source 4를 두 부분으로 나눠 네 Context 조건을 비교한다."""
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

    print(f"Query: {body['query']}")
    print("실제 HTTP Retrieval:")
    for result in body["results"]:
        is_gold = result["chunk_id"] in case["expected_chunk_ids"]
        print(
            f"rank={result['rank']}, chunk_id={result['chunk_id']}, "
            f"page_number={result['page_number']}, gold={is_gold}"
        )
    print("현재 HTTP answer:")
    print(body["answer"])

    try:
        source_1 = select_source(body["results"], 1)
        source_4 = select_source(body["results"], 4)
        print("\n선택한 Source:")
        for source in (source_1, source_4):
            print(
                f"rank={source['rank']}, chunk_id={source['chunk_id']}, "
                f"page_number={source['page_number']}, "
                f"text length={len(source['text'])}"
            )

        overlap_text, new_portion_text = split_source_4(source_1, source_4)
        conditions = create_conditions(
            source_1,
            source_4,
            overlap_text,
            new_portion_text,
        )
        summaries = [
            run_condition(condition, case["query"])
            for condition in conditions
        ]
    except (LocalLLMError, ValueError) as error:
        print(f"실험 실패: {error}")
        if isinstance(error, LocalLLMError):
            print("Ollama 서버와 qwen3:8b 모델 상태를 확인하세요.")
        raise SystemExit(1) from None

    print_comparison(summaries)


if __name__ == "__main__":
    main()
