"""동일 길이의 중복 text와 비중복 control text가 Generation에 미치는 영향을 관찰한다."""

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
REPEAT_COUNT = 5
QUERY_TERMS = ("과제", "제출", "그대로", "생성형 AI")


def get_target_case() -> dict:
    """기존 평가 Case에서 실험 대상을 이름으로 찾는다."""
    return next(case for case in EVAL_CASES if case["name"] == CASE_NAME)


def select_source(results: list[dict], rank: int) -> dict:
    """실제 HTTP Retrieval 결과에서 original rank의 Source를 찾는다."""
    source = next((result for result in results if result["rank"] == rank), None)
    if source is None:
        raise ValueError(f"Retrieval 결과에 Source {rank}가 없습니다.")
    return source


def find_occurrences(text: str, term: str) -> list[int]:
    """문자열에서 term이 등장하는 모든 시작 index를 반환한다."""
    indices = []
    start = 0
    while True:
        index = text.find(term, start)
        if index == -1:
            return indices
        indices.append(index)
        start = index + 1


def print_term_positions(label: str, text: str) -> None:
    """두 text의 내용 차이를 확인하도록 질문 관련 문자열 위치를 출력한다."""
    print(f"\n{label} query literal 위치:")
    for term in QUERY_TERMS:
        print(f"{term}: {find_occurrences(text, term)}")


def print_source_metadata(source_number: int, source: dict) -> None:
    """선택한 실제 Source의 핵심 metadata를 출력한다."""
    print(
        f"Source {source_number}: rank={source['rank']}, "
        f"chunk_id={source['chunk_id']}, "
        f"page_number={source['page_number']}, "
        f"text length={len(source['text'])}"
    )


def build_control_texts(
    source_1: dict,
    source_2: dict,
    source_4: dict,
) -> tuple[str, str]:
    """실제 Chunk에서 같은 길이의 duplicate와 deterministic control을 만든다."""
    source_1_text = source_1["text"]
    source_2_text = source_2["text"]
    source_4_text = source_4["text"]

    duplicate_length = longest_suffix_prefix_overlap(
        source_1_text,
        source_4_text,
    )
    print("\nSource 1 ↔ Source 4 duplicate 계산:")
    print(f"Source 1 length: {len(source_1_text)}")
    print(f"Source 4 length: {len(source_4_text)}")
    print(f"duplicate length: {duplicate_length}")
    if duplicate_length == 0:
        raise ValueError("Source 1과 Source 4 사이에서 exact overlap을 찾지 못했습니다.")

    duplicate_text = source_4_text[:duplicate_length]
    duplicate_matches = source_1_text[-duplicate_length:] == duplicate_text
    print(f"Source 1 suffix == duplicate text: {duplicate_matches}")
    if not duplicate_matches:
        raise ValueError("계산한 duplicate text가 Source 1 suffix와 다릅니다.")

    source_2_source_1_overlap = longest_suffix_prefix_overlap(
        source_2_text,
        source_1_text,
    )
    if source_2_source_1_overlap > 0:
        source_2_non_overlap_text = source_2_text[:-source_2_source_1_overlap]
    else:
        source_2_non_overlap_text = source_2_text
    control_text = source_2_non_overlap_text[:duplicate_length]

    print("\nSource 2 → Source 1 overlap 계산:")
    print(f"Source 2 length: {len(source_2_text)}")
    print(
        "Source 2 → Source 1 exact suffix-prefix overlap: "
        f"{source_2_source_1_overlap}"
    )
    print(f"Source 2 non-overlap length: {len(source_2_non_overlap_text)}")

    length_matched = len(duplicate_text) == len(control_text)
    control_in_source_1 = control_text in source_1_text
    print("\nControl validation:")
    print(f"duplicate length: {len(duplicate_text)}")
    print(f"control length: {len(control_text)}")
    print(f"length matched: {length_matched}")
    print(f"control text is exact substring of Source 1: {control_in_source_1}")
    if not length_matched:
        raise ValueError("Source 2의 non-overlap 영역이 control 길이보다 짧습니다.")
    if control_in_source_1:
        raise ValueError(
            "결정된 control text가 Source 1의 exact substring이어서 사용할 수 없습니다."
        )

    print("\n===== DUPLICATE TEXT START =====")
    print(duplicate_text)
    print("===== DUPLICATE TEXT END =====")
    print("\n===== LENGTH-MATCHED CONTROL TEXT START =====")
    print(control_text)
    print("===== LENGTH-MATCHED CONTROL TEXT END =====")
    print_term_positions("duplicate", duplicate_text)
    print_term_positions("control", control_text)
    print("위 위치는 semantic 판단이 아닌 단순 literal string 확인입니다.")

    return duplicate_text, control_text


def create_conditions(
    source_1: dict,
    source_4: dict,
    duplicate_text: str,
    control_text: str,
) -> list[dict]:
    """Source 4 metadata를 유지하고 text만 다른 세 조건을 만든다."""
    # B와 C에서 Source number/header/metadata/길이를 같게 유지하고 text만 비교한다.
    duplicate_source_4 = dict(source_4)
    duplicate_source_4["text"] = duplicate_text
    control_source_4 = dict(source_4)
    control_source_4["text"] = control_text

    duplicate_metadata = {
        key: value for key, value in duplicate_source_4.items() if key != "text"
    }
    control_metadata = {
        key: value for key, value in control_source_4.items() if key != "text"
    }
    metadata_identical = duplicate_metadata == control_metadata
    print("\nSynthetic Source 4 통제:")
    print(f"B/C synthetic Source metadata 동일 여부: {metadata_identical}")
    if not metadata_identical:
        raise ValueError("두 synthetic Source 4의 metadata가 다릅니다.")

    return [
        {
            "name": "SOURCE_1_ONLY",
            "results": [source_1],
            "synthetic": False,
        },
        {
            "name": "SOURCE_1_PLUS_DUPLICATE_100",
            "results": [source_1, duplicate_source_4],
            "synthetic": True,
        },
        {
            "name": "SOURCE_1_PLUS_CONTROL_100",
            "results": [source_1, control_source_4],
            "synthetic": True,
        },
    ]


def prepare_conditions(conditions: list[dict], query: str) -> None:
    """각 조건의 Context/Prompt를 만들고 B와 C의 길이 통제를 검증한다."""
    for condition in conditions:
        context = build_context(condition["results"])
        condition["context"] = context
        condition["prompt"] = build_local_rag_prompt(query, context)

    duplicate_condition = conditions[1]
    control_condition = conditions[2]
    duplicate_context_length = len(duplicate_condition["context"])
    control_context_length = len(control_condition["context"])
    duplicate_prompt_length = len(duplicate_condition["prompt"])
    control_prompt_length = len(control_condition["prompt"])

    print("\nLength-matched 입력 검증:")
    print(f"duplicate context length: {duplicate_context_length}")
    print(f"control context length: {control_context_length}")
    print(
        "duplicate/control Context length 동일 여부: "
        f"{duplicate_context_length == control_context_length}"
    )
    print(f"duplicate prompt length: {duplicate_prompt_length}")
    print(f"control prompt length: {control_prompt_length}")
    print(
        "duplicate/control Prompt length 동일 여부: "
        f"{duplicate_prompt_length == control_prompt_length}"
    )
    if duplicate_context_length != control_context_length:
        raise ValueError("duplicate와 control의 Context 길이가 다릅니다.")
    if duplicate_prompt_length != control_prompt_length:
        raise ValueError("duplicate와 control의 Prompt 길이가 다릅니다.")


def run_condition(condition: dict) -> dict:
    """한 Context 조건을 다섯 번 실행하고 관찰 지표를 출력한다."""
    results = condition["results"]
    source_numbers = [result["rank"] for result in results]
    chunk_ids = [result["chunk_id"] for result in results]
    text_lengths = [len(result["text"]) for result in results]
    valid_source_numbers = set(source_numbers)

    print("\n" + "=" * 88)
    print(f"Condition name: {condition['name']}")
    print(f"Synthetic 여부: {condition['synthetic']}")
    print(f"Source numbers: {source_numbers}")
    print(f"chunk_ids: {chunk_ids}")
    print(f"text lengths: {text_lengths}")
    print(f"context length: {len(condition['context'])}")
    print(f"prompt length: {len(condition['prompt'])}")
    if condition["synthetic"]:
        print("Synthetic Source 4 text 전체:")
        print(results[1]["text"])

    runs = []
    for run_number in range(1, REPEAT_COUNT + 1):
        answer = generate_local_answer(condition["prompt"])
        citations = extract_citations(answer)
        citation_valid = bool(citations) and all(
            citation in valid_source_numbers for citation in citations
        )
        run = {
            "answer": answer,
            "citations": citations,
            "contains_no_answer": NO_ANSWER in answer,
            "exact_no_answer": answer == NO_ANSWER,
            "citation_present": bool(citations),
            "citation_valid": citation_valid,
        }
        runs.append(run)

        print(f"\nRun number: {run_number}")
        print("Answer:")
        print(answer)
        print(f"Citation 번호: {citations}")
        print(f"NO_ANSWER phrase 포함 여부: {run['contains_no_answer']}")
        print(f"Exact NO_ANSWER 여부: {run['exact_no_answer']}")
        print(f"Citation present: {run['citation_present']}")
        print(f"Citation valid for current Context: {citation_valid}")

    summary = {
        "name": condition["name"],
        "unique_answer_count": len({run["answer"] for run in runs}),
        "no_answer_count": sum(run["contains_no_answer"] for run in runs),
        "exact_no_answer_count": sum(run["exact_no_answer"] for run in runs),
        "citation_present_count": sum(run["citation_present"] for run in runs),
        "citation_valid_count": sum(run["citation_valid"] for run in runs),
    }
    print("\nCondition Summary:")
    print(f"Unique answers: {summary['unique_answer_count']}")
    print(f"NO_ANSWER phrase: {summary['no_answer_count']}/{REPEAT_COUNT}")
    print(f"Exact NO_ANSWER: {summary['exact_no_answer_count']}/{REPEAT_COUNT}")
    print(f"Citation present: {summary['citation_present_count']}/{REPEAT_COUNT}")
    print(f"Citation valid: {summary['citation_valid_count']}/{REPEAT_COUNT}")
    return summary


def print_comparison(summaries: list[dict]) -> None:
    """세 조건의 관찰값과 length-matched 핵심 비교를 출력한다."""
    print("\n" + "=" * 88)
    print("세 Condition Summary")
    print("Condition | NO_ANSWER | Exact | Unique answers | Citation valid")
    for summary in summaries:
        print(
            f"{summary['name']} | "
            f"{summary['no_answer_count']}/{REPEAT_COUNT} | "
            f"{summary['exact_no_answer_count']}/{REPEAT_COUNT} | "
            f"{summary['unique_answer_count']} | "
            f"{summary['citation_valid_count']}/{REPEAT_COUNT}"
        )

    summaries_by_name = {summary["name"]: summary for summary in summaries}
    duplicate_count = summaries_by_name[
        "SOURCE_1_PLUS_DUPLICATE_100"
    ]["no_answer_count"]
    control_count = summaries_by_name[
        "SOURCE_1_PLUS_CONTROL_100"
    ]["no_answer_count"]
    print("\nLength-Matched 핵심 비교:")
    print(f"duplicate refusal count: {duplicate_count}/{REPEAT_COUNT}")
    print(f"control refusal count: {control_count}/{REPEAT_COUNT}")
    if duplicate_count > control_count:
        print(
            "동일 길이 control과 비교했을 때 duplicate 조건에서 더 많은 "
            "refusal behavior가 관찰됨"
        )
    else:
        print(
            "이번 length-matched control에서는 duplicate 여부만으로 뚜렷한 "
            "behavior 차이를 관찰하지 못함"
        )

    print("\n실험 한계:")
    print(
        "duplicate와 control은 길이와 Source metadata는 동일하지만 "
        "text의 semantic content까지 동일하게 통제된 것은 아니다."
    )
    print("따라서 이 결과는 완전한 causal proof가 아니라 진단용 ablation입니다.")


def main() -> None:
    """실제 Retrieval을 한 번 호출하고 세 조건을 각각 다섯 번 비교한다."""
    case = get_target_case()
    print(f"Local model: {LOCAL_MODEL_NAME}")
    print(f"temperature: {LOCAL_TEMPERATURE}")
    print(f"seed: {LOCAL_SEED}")
    print(f"REPEAT_COUNT: {REPEAT_COUNT}")

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
    print("실제 HTTP Retrieval Top-5:")
    for result in body["results"]:
        is_gold = result["chunk_id"] in case["expected_chunk_ids"]
        print(
            f"rank={result['rank']}, chunk_id={result['chunk_id']}, "
            f"page_number={result['page_number']}, gold={is_gold}"
        )
    print("HTTP response answer:")
    print(body["answer"])

    try:
        source_1 = select_source(body["results"], 1)
        source_2 = select_source(body["results"], 2)
        source_4 = select_source(body["results"], 4)
        print("\n선택한 실제 Source metadata:")
        print_source_metadata(1, source_1)
        print_source_metadata(2, source_2)
        print_source_metadata(4, source_4)

        duplicate_text, control_text = build_control_texts(
            source_1,
            source_2,
            source_4,
        )
        conditions = create_conditions(
            source_1,
            source_4,
            duplicate_text,
            control_text,
        )
        prepare_conditions(conditions, case["query"])
        summaries = [run_condition(condition) for condition in conditions]
    except (ValueError, LocalLLMError) as error:
        print(f"오류: {error}")
        if isinstance(error, LocalLLMError):
            print("Ollama 서버가 실행 중인지 확인하세요.")
            print("ollama list로 qwen3:8b 모델 설치 여부를 확인하세요.")
        raise SystemExit(1) from None

    print_comparison(summaries)


if __name__ == "__main__":
    main()
