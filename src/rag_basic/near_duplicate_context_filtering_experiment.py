"""실제 Top-5에서 near-duplicate Source를 제외한 Context의 Generation을 비교한다."""

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
NEAR_DUPLICATE_THRESHOLD = 0.90
REPEAT_COUNT = 5


def get_target_case() -> dict:
    """기존 평가 Case에서 실험 대상을 이름으로 찾는다."""
    return next(case for case in EVAL_CASES if case["name"] == CASE_NAME)


def calculate_candidate_coverage(kept: dict, candidate: dict) -> dict:
    """Kept Source가 후순위 candidate를 덮는 exact overlap 비율을 계산한다."""
    kept_text = kept["text"]
    candidate_text = candidate["text"]
    if not candidate_text:
        raise ValueError(
            f"candidate Source {candidate['rank']}의 text가 비어 있습니다."
        )

    forward_overlap = longest_suffix_prefix_overlap(kept_text, candidate_text)
    reverse_overlap = longest_suffix_prefix_overlap(candidate_text, kept_text)
    candidate_is_substring = candidate_text in kept_text
    overlap_length = max(forward_overlap, reverse_overlap)
    if candidate_is_substring:
        overlap_length = len(candidate_text)

    # 후순위 Source 자체가 상위 Source에 얼마나 덮이는지 보기 위해 candidate 길이로 나눈다.
    coverage = overlap_length / len(candidate_text)
    return {
        "kept_rank": kept["rank"],
        "kept_chunk_id": kept["chunk_id"],
        "candidate_rank": candidate["rank"],
        "candidate_chunk_id": candidate["chunk_id"],
        "forward_overlap": forward_overlap,
        "reverse_overlap": reverse_overlap,
        "candidate_is_substring": candidate_is_substring,
        "overlap_length": overlap_length,
        "candidate_length": len(candidate_text),
        "coverage": coverage,
    }


def print_pairwise_diagnostic(diagnostic: dict) -> None:
    """한 candidate와 kept Source 사이의 overlap 계산을 출력한다."""
    print("\n" + "-" * 88)
    print(
        f"candidate Source {diagnostic['candidate_rank']} "
        f"vs kept Source {diagnostic['kept_rank']}"
    )
    print(f"candidate chunk_id: {diagnostic['candidate_chunk_id']}")
    print(f"kept chunk_id: {diagnostic['kept_chunk_id']}")
    print(f"forward overlap: {diagnostic['forward_overlap']}")
    print(f"reverse overlap: {diagnostic['reverse_overlap']}")
    print(
        "candidate text is exact substring of kept text: "
        f"{diagnostic['candidate_is_substring']}"
    )
    print(f"overlap length: {diagnostic['overlap_length']}")
    print(f"candidate length: {diagnostic['candidate_length']}")
    print(f"coverage: {diagnostic['coverage']:.6f}")
    print(f"threshold: {NEAR_DUPLICATE_THRESHOLD:.2f}")


def filter_near_duplicates(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """상위 rank를 우선 보존하는 greedy diagnostic filter를 적용한다."""
    kept_results = []
    dropped_results = []

    print("\nPairwise near-duplicate diagnostic:")
    print(
        "0.90은 최적값이 아니라 candidate text가 상위 Source에 거의 전부 "
        "포함됐는지 확인하는 보수적인 진단 기준입니다."
    )
    for candidate in sorted(results, key=lambda result: result["rank"]):
        if not kept_results:
            kept_results.append(candidate)
            print(
                f"\ncandidate Source {candidate['rank']} "
                f"(chunk_id={candidate['chunk_id']}): KEEP (첫 Source)"
            )
            continue

        diagnostics = []
        for kept in kept_results:
            diagnostic = calculate_candidate_coverage(kept, candidate)
            diagnostics.append(diagnostic)
            print_pairwise_diagnostic(diagnostic)

        matched = max(diagnostics, key=lambda item: item["coverage"])
        if matched["coverage"] >= NEAR_DUPLICATE_THRESHOLD:
            dropped = dict(candidate)
            dropped["matched_kept_rank"] = matched["kept_rank"]
            dropped["matched_kept_chunk_id"] = matched["kept_chunk_id"]
            dropped["overlap_length"] = matched["overlap_length"]
            dropped["candidate_length"] = matched["candidate_length"]
            dropped["coverage"] = matched["coverage"]
            dropped_results.append(dropped)
            print(
                f"DROP as near-duplicate: Source {candidate['rank']} matched with "
                f"Source {matched['kept_rank']}, coverage={matched['coverage']:.6f}"
            )
        else:
            kept_results.append(candidate)
            print(
                f"KEEP: Source {candidate['rank']}, "
                f"maximum coverage={matched['coverage']:.6f}"
            )

    return kept_results, dropped_results


def print_filter_result(
    original_results: list[dict],
    filtered_results: list[dict],
    dropped_results: list[dict],
) -> None:
    """원본과 필터 결과 및 제거된 Source의 진단 근거를 출력한다."""
    print("\n" + "=" * 88)
    print("Filter 결과")
    print(f"Original ranks: {[result['rank'] for result in original_results]}")
    print(
        f"Original chunk_ids: {[result['chunk_id'] for result in original_results]}"
    )
    print(f"Filtered ranks: {[result['rank'] for result in filtered_results]}")
    print(
        f"Filtered chunk_ids: {[result['chunk_id'] for result in filtered_results]}"
    )
    print(f"Dropped ranks: {[result['rank'] for result in dropped_results]}")
    print(
        f"Dropped chunk_ids: {[result['chunk_id'] for result in dropped_results]}"
    )
    for dropped in dropped_results:
        print(
            f"dropped Source {dropped['rank']}: "
            f"matched kept Source={dropped['matched_kept_rank']}, "
            f"overlap length={dropped['overlap_length']}, "
            f"candidate length={dropped['candidate_length']}, "
            f"coverage={dropped['coverage']:.6f}"
        )


def print_gold_status(
    label: str,
    results: list[dict],
    expected_chunk_ids: list[int],
) -> bool:
    """Context 후보 목록에 남은 gold evidence 상태를 출력한다."""
    gold_chunk_ids = [
        result["chunk_id"]
        for result in results
        if result["chunk_id"] in expected_chunk_ids
    ]
    print(f"\n{label} Gold chunk_ids present: {gold_chunk_ids}")
    print(f"{label} Gold count: {len(gold_chunk_ids)}")
    print(f"{label} At least one gold remains: {bool(gold_chunk_ids)}")
    return bool(gold_chunk_ids)


def prepare_condition(name: str, results: list[dict], query: str) -> dict:
    """기존 builder로 한 Generation 조건의 Context와 Prompt를 만든다."""
    context = build_context(results)
    prompt = build_local_rag_prompt(query, context)
    return {
        "name": name,
        "results": results,
        "context": context,
        "prompt": prompt,
    }


def print_contexts(original: dict, filtered: dict) -> None:
    """두 조건의 크기와 Context 전체를 출력한다."""
    print("\nContext 비교:")
    print(f"Original Context length: {len(original['context'])}")
    print(f"Filtered Context length: {len(filtered['context'])}")
    print(f"Original Source count: {len(original['results'])}")
    print(f"Filtered Source count: {len(filtered['results'])}")
    print(f"Original Prompt length: {len(original['prompt'])}")
    print(f"Filtered Prompt length: {len(filtered['prompt'])}")
    print("\n===== ORIGINAL CONTEXT START =====")
    print(original["context"])
    print("===== ORIGINAL CONTEXT END =====")
    print("\n===== FILTERED CONTEXT START =====")
    print(filtered["context"])
    print("===== FILTERED CONTEXT END =====")


def run_condition(condition: dict) -> dict:
    """한 실제 Context 조건을 다섯 번 생성하고 관찰 지표를 집계한다."""
    results = condition["results"]
    source_ranks = [result["rank"] for result in results]
    valid_source_numbers = set(source_ranks)

    print("\n" + "=" * 88)
    print(f"Condition name: {condition['name']}")
    print(f"Source ranks: {source_ranks}")
    print(f"chunk_ids: {[result['chunk_id'] for result in results]}")
    print(f"Source count: {len(results)}")
    print(f"Context length: {len(condition['context'])}")
    print(f"Prompt length: {len(condition['prompt'])}")

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
        "source_count": len(results),
        "context_length": len(condition["context"]),
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
    """원본과 filtered Context의 Generation 관찰값을 비교한다."""
    print("\n" + "=" * 88)
    print("두 Condition Summary")
    print("Condition | NO_ANSWER | Exact | Unique answers | Citation valid")
    for summary in summaries:
        print(
            f"{summary['name']} | "
            f"{summary['no_answer_count']}/{REPEAT_COUNT} | "
            f"{summary['exact_no_answer_count']}/{REPEAT_COUNT} | "
            f"{summary['unique_answer_count']} | "
            f"{summary['citation_valid_count']}/{REPEAT_COUNT}"
        )

    original, filtered = summaries
    print("\n핵심 비교:")
    print(f"Original refusal count: {original['no_answer_count']}/{REPEAT_COUNT}")
    print(f"Filtered refusal count: {filtered['no_answer_count']}/{REPEAT_COUNT}")
    print(f"Original Source count: {original['source_count']}")
    print(f"Filtered Source count: {filtered['source_count']}")
    print(f"Original Context length: {original['context_length']}")
    print(f"Filtered Context length: {filtered['context_length']}")
    if filtered["no_answer_count"] < original["no_answer_count"]:
        print(
            "현재 Case와 실행환경에서는 near-duplicate Source를 Context에서 "
            "제외한 조건에서 refusal behavior가 감소하는 것이 관찰됨"
        )
    else:
        print(
            "이번 실제 Top-5 filtering 실험에서는 near-duplicate Source 제거만으로 "
            "refusal behavior 개선이 관찰되지 않음"
        )

    print("\n해석 범위:")
    print("Retrieval Top-5 결과는 그대로이고 LLM에 전달하는 Context selection만 바꿨습니다.")
    print("0.90 threshold를 최적값이나 Production 해결책으로 채택한 것이 아닙니다.")
    print("중복이 원인으로 증명된 것이 아니라 diagnostic practical experiment입니다.")
    print(
        "개선이 관찰되면 다음 단계는 동일 filtering rule을 전체 9개 Evaluation "
        "Case에 적용해 regression 여부를 확인하는 것입니다."
    )


def main() -> None:
    """실제 Top-5 원본과 near-duplicate filtered Context를 각각 다섯 번 비교한다."""
    case = get_target_case()
    print(f"Local model: {LOCAL_MODEL_NAME}")
    print(f"temperature: {LOCAL_TEMPERATURE}")
    print(f"seed: {LOCAL_SEED}")
    print(f"NEAR_DUPLICATE_THRESHOLD: {NEAR_DUPLICATE_THRESHOLD}")
    print(f"REPEAT_COUNT: {REPEAT_COUNT}")
    print("Retrieval Top-5는 변경하지 않고 Context selection만 진단적으로 비교합니다.")

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
    print(f"Query: {body['query']}")
    print(f"Retrieved result count: {len(results)}")
    print(f"Retrieved chunk_ids: {[result['chunk_id'] for result in results]}")
    print("실제 HTTP Retrieval Top-5:")
    for result in results:
        is_gold = result["chunk_id"] in case["expected_chunk_ids"]
        print(
            f"rank={result['rank']}, chunk_id={result['chunk_id']}, "
            f"page_number={result['page_number']}, score={result['score']:.6f}, "
            f"gold={is_gold}, text length={len(result['text'])}"
        )
    print("\n===== HTTP ORIGINAL ANSWER =====")
    print(body["answer"])
    print("===== END =====")

    try:
        filtered_results, dropped_results = filter_near_duplicates(results)
        print_filter_result(results, filtered_results, dropped_results)
        print_gold_status("Original", results, case["expected_chunk_ids"])
        filtered_has_gold = print_gold_status(
            "Filtered",
            filtered_results,
            case["expected_chunk_ids"],
        )
        if not filtered_has_gold:
            print("필터 적용 후 gold evidence가 남지 않아 Generation 실험을 중단합니다.")
            raise SystemExit(1)

        original_condition = prepare_condition(
            "ORIGINAL_TOP_5",
            results,
            case["query"],
        )
        filtered_condition = prepare_condition(
            "FILTERED_CONTEXT",
            filtered_results,
            case["query"],
        )
        print_contexts(original_condition, filtered_condition)
        summaries = [
            run_condition(original_condition),
            run_condition(filtered_condition),
        ]
    except (ValueError, LocalLLMError) as error:
        print(f"오류: {error}")
        if isinstance(error, LocalLLMError):
            print("Ollama 서버가 실행 중인지 확인하세요.")
            print("ollama list로 qwen3:8b 모델 설치 여부를 확인하세요.")
        raise SystemExit(1) from None

    print_comparison(summaries)


if __name__ == "__main__":
    main()
