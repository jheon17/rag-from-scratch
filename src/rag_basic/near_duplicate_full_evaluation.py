"""전체 평가 Case에서 near-duplicate Context filtering의 회귀 여부를 확인한다."""

from rag_basic.api_evaluation import (
    APIConnectionError,
    call_query_api,
    evaluate_retrieval,
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
from rag_basic.near_duplicate_context_filtering_experiment import (
    NEAR_DUPLICATE_THRESHOLD,
    filter_near_duplicates,
)
from rag_basic.retrieval import TOP_K, build_context


# 전체 9개 Case를 넓게 확인하면서 실행 간 변동도 관찰하기 위해 조건별 3회를 사용한다.
REPEAT_COUNT = 3


def run_generations(prompt: str, source_ranks: set[int], label: str) -> list[dict]:
    """동일 Prompt를 세 번 생성하고 답변과 자동 관찰값을 출력한다."""
    print(f"\n{label} direct Generation:")
    runs = []
    for run_number in range(1, REPEAT_COUNT + 1):
        answer = generate_local_answer(prompt)
        citations = extract_citations(answer)
        run = {
            "answer": answer,
            "citations": citations,
            "contains_no_answer": NO_ANSWER in answer,
            "exact_no_answer": answer == NO_ANSWER,
            "citation_present": bool(citations),
            "citation_valid": bool(citations)
            and all(citation in source_ranks for citation in citations),
        }
        runs.append(run)

        print(f"\nRun number: {run_number}")
        print("Answer:")
        print(answer)
        print(f"Citations: {citations}")
        print(f"NO_ANSWER phrase 포함: {run['contains_no_answer']}")
        print(f"Exact NO_ANSWER: {run['exact_no_answer']}")
        print(f"Citation present: {run['citation_present']}")
        print(f"Citation valid: {run['citation_valid']}")
    return runs


def summarize_runs(runs: list[dict]) -> dict:
    """여러 Generation run의 자동 관찰값을 집계한다."""
    return {
        "unique_answers": len({run["answer"] for run in runs}),
        "no_answer_count": sum(run["contains_no_answer"] for run in runs),
        "exact_refusal_count": sum(run["exact_no_answer"] for run in runs),
        "citation_present_count": sum(run["citation_present"] for run in runs),
        "citation_valid_count": sum(run["citation_valid"] for run in runs),
    }


def classify_case(case_type: str, original: dict, filtered: dict) -> str:
    """Case 유형에 맞는 refusal count 변화만으로 단순 분류한다."""
    if case_type == "in_domain":
        original_count = original["no_answer_count"]
        filtered_count = filtered["no_answer_count"]
        if filtered_count < original_count:
            return "IMPROVED"
        if filtered_count > original_count:
            return "REGRESSED"
        return "UNCHANGED"

    original_count = original["exact_refusal_count"]
    filtered_count = filtered["exact_refusal_count"]
    if filtered_count > original_count:
        return "IMPROVED"
    if filtered_count < original_count:
        return "REGRESSED"
    return "UNCHANGED"


def gold_chunk_ids(results: list[dict], expected_chunk_ids: list[int]) -> list[int]:
    """현재 결과에 포함된 gold chunk_id를 rank 순서로 반환한다."""
    return [
        result["chunk_id"]
        for result in results
        if result["chunk_id"] in expected_chunk_ids
    ]


def print_retrieval(case: dict, results: list[dict]) -> None:
    """한 Case의 실제 Top-5 Retrieval을 출력한다."""
    print(f"Retrieved chunk_ids: {[result['chunk_id'] for result in results]}")
    for result in results:
        gold = (
            result["chunk_id"] in case["expected_chunk_ids"]
            if case["type"] == "in_domain"
            else "N/A"
        )
        print(
            f"rank={result['rank']}, chunk_id={result['chunk_id']}, "
            f"page_number={result['page_number']}, score={result['score']:.6f}, "
            f"text length={len(result['text'])}, gold={gold}"
        )


def print_filter_details(
    results: list[dict],
    filtered_results: list[dict],
    dropped_results: list[dict],
) -> None:
    """한 Case의 원본·필터 목록과 제거 근거를 출력한다."""
    print(f"Original ranks: {[result['rank'] for result in results]}")
    print(f"Original chunk_ids: {[result['chunk_id'] for result in results]}")
    print(f"Filtered ranks: {[result['rank'] for result in filtered_results]}")
    print(
        f"Filtered chunk_ids: {[result['chunk_id'] for result in filtered_results]}"
    )
    print(f"Dropped ranks: {[result['rank'] for result in dropped_results]}")
    print(
        f"Dropped chunk_ids: {[result['chunk_id'] for result in dropped_results]}"
    )
    if not dropped_results:
        print("Dropped Source: none")
    for dropped in dropped_results:
        print(
            f"dropped rank={dropped['rank']}, chunk_id={dropped['chunk_id']}, "
            f"matched kept rank={dropped['matched_kept_rank']}, "
            f"matched kept chunk_id={dropped['matched_kept_chunk_id']}, "
            f"overlap length={dropped['overlap_length']}, "
            f"candidate length={dropped['candidate_length']}, "
            f"coverage={dropped['coverage']:.6f}"
        )


def evaluate_case(case: dict) -> dict:
    """한 Case의 actual Retrieval과 원본·필터 Context Generation을 평가한다."""
    print("\n" + "=" * 100)
    print(f"Case name: {case['name']}")
    print(f"Type: {case['type']}")
    print(f"Query: {case['query']}")

    status, body = call_query_api(case)
    print(f"HTTP status: {status}")
    if status != 200:
        print(f"Response detail: {body.get('detail', '확인할 수 없음')}")
        raise RuntimeError(f"{case['name']} API 호출이 실패했습니다: HTTP {status}")

    results = body["results"]
    print("\n실제 Retrieval Top-5:")
    print_retrieval(case, results)
    rebuilt_context = build_context(results)
    context_matches = body["context"] == rebuilt_context
    print(f"HTTP context == rebuilt context: {context_matches}")
    if not context_matches:
        raise RuntimeError(f"{case['name']}의 API Context가 rebuilt Context와 다릅니다.")

    retrieval = None
    if case["type"] == "in_domain":
        retrieval = evaluate_retrieval(results, case["expected_chunk_ids"])
        print(f"Hit@{TOP_K}: {retrieval['hit_at_k']}")
        print(f"first gold rank: {retrieval['first_gold_rank']}")
        print(f"Reciprocal Rank: {retrieval['reciprocal_rank']:.4f}")

    filtered_results, dropped_results = filter_near_duplicates(results)
    print("\nCase Filter 결과:")
    print_filter_details(results, filtered_results, dropped_results)
    original_chunk_ids = [result["chunk_id"] for result in results]
    filtered_chunk_ids = [result["chunk_id"] for result in filtered_results]
    filter_changed = filtered_chunk_ids != original_chunk_ids
    print(f"Filter changed Context candidates: {filter_changed}")

    original_gold = []
    filtered_gold = []
    all_gold_removed = False
    if case["type"] == "in_domain":
        original_gold = gold_chunk_ids(results, case["expected_chunk_ids"])
        filtered_gold = gold_chunk_ids(
            filtered_results, case["expected_chunk_ids"]
        )
        all_gold_removed = bool(original_gold) and not filtered_gold
        print(f"Original gold chunk_ids present: {original_gold}")
        print(f"Original gold count: {len(original_gold)}")
        print(f"Original at least one gold remains: {bool(original_gold)}")
        print(f"Filtered gold chunk_ids present: {filtered_gold}")
        print(f"Filtered gold count: {len(filtered_gold)}")
        print(f"Filtered at least one gold remains: {bool(filtered_gold)}")
        if all_gold_removed:
            print("WARNING: all gold evidence removed")

    original_context = body["context"]
    filtered_context = build_context(filtered_results)
    original_prompt = build_local_rag_prompt(case["query"], original_context)
    filtered_prompt = build_local_rag_prompt(case["query"], filtered_context)
    print(f"Original Source count: {len(results)}")
    print(f"Filtered Source count: {len(filtered_results)}")
    print(f"Original Context length: {len(original_context)}")
    print(f"Filtered Context length: {len(filtered_context)}")
    print(f"Original Prompt length: {len(original_prompt)}")
    print(f"Filtered Prompt length: {len(filtered_prompt)}")
    if not filter_changed:
        print(f"Original Context == Filtered Context: {original_context == filtered_context}")
        print(f"Original Prompt == Filtered Prompt: {original_prompt == filtered_prompt}")
        if original_context != filtered_context or original_prompt != filtered_prompt:
            raise RuntimeError("Filter 미변경 Case에서 Context 또는 Prompt가 달라졌습니다.")

    print("\n===== HTTP PRODUCTION ANSWER =====")
    print(body["answer"])
    print("===== END HTTP PRODUCTION ANSWER =====")

    original_ranks = {result["rank"] for result in results}
    filtered_ranks = {result["rank"] for result in filtered_results}
    original_runs = run_generations(
        original_prompt, original_ranks, "ORIGINAL_TOP_5"
    )
    if filter_changed:
        filtered_runs = run_generations(
            filtered_prompt, filtered_ranks, "FILTERED_CONTEXT"
        )
        runs_reused = False
    else:
        print("\nFilter made no input change.")
        print(
            "Filtered Generation reuses Original runs to avoid stochastic false differences."
        )
        filtered_runs = original_runs
        runs_reused = True

    original_summary = summarize_runs(original_runs)
    filtered_summary = summarize_runs(filtered_runs)
    classification = classify_case(
        case["type"], original_summary, filtered_summary
    )

    print("\nCase Summary:")
    print(f"Case: {case['name']}")
    print(f"Filter changed: {filter_changed}")
    print(f"Dropped chunks: {[result['chunk_id'] for result in dropped_results]}")
    if case["type"] == "in_domain":
        print(
            "Original undesired refusal: "
            f"{original_summary['no_answer_count']}/{REPEAT_COUNT}"
        )
        print(
            "Filtered undesired refusal: "
            f"{filtered_summary['no_answer_count']}/{REPEAT_COUNT}"
        )
        print(
            "Original citation present: "
            f"{original_summary['citation_present_count']}/{REPEAT_COUNT}"
        )
        print(
            "Filtered citation present: "
            f"{filtered_summary['citation_present_count']}/{REPEAT_COUNT}"
        )
        print(
            "Original citation valid: "
            f"{original_summary['citation_valid_count']}/{REPEAT_COUNT}"
        )
        print(
            "Filtered citation valid: "
            f"{filtered_summary['citation_valid_count']}/{REPEAT_COUNT}"
        )
    else:
        print(
            "Original exact refusal: "
            f"{original_summary['exact_refusal_count']}/{REPEAT_COUNT}"
        )
        print(
            "Filtered exact refusal: "
            f"{filtered_summary['exact_refusal_count']}/{REPEAT_COUNT}"
        )
        print(f"Runs reused: {runs_reused}")
    print(f"Refusal behavior classification: {classification}")
    print("이 분류는 refusal behavior만 기준으로 하며 semantic correctness를 뜻하지 않습니다.")

    return {
        "case": case,
        "retrieval": retrieval,
        "filter_changed": filter_changed,
        "dropped_results": dropped_results,
        "original_source_count": len(results),
        "filtered_source_count": len(filtered_results),
        "original_summary": original_summary,
        "filtered_summary": filtered_summary,
        "classification": classification,
        "all_gold_removed": all_gold_removed,
        "runs_reused": runs_reused,
    }


def names_by_classification(evaluations: list[dict], value: str) -> list[str]:
    """지정한 분류에 속한 Case 이름을 반환한다."""
    return [
        item["case"]["name"]
        for item in evaluations
        if item["classification"] == value
    ]


def print_aggregate_summary(evaluations: list[dict]) -> None:
    """Retrieval, filtering, Generation 및 regression 결과를 전체 집계한다."""
    in_domain = [
        item for item in evaluations if item["case"]["type"] == "in_domain"
    ]
    out_of_domain = [
        item for item in evaluations if item["case"]["type"] == "out_of_domain"
    ]

    hit_count = sum(item["retrieval"]["hit_at_k"] for item in in_domain)
    mrr = sum(
        item["retrieval"]["reciprocal_rank"] for item in in_domain
    ) / len(in_domain)
    print("\n" + "=" * 100)
    print("IN-DOMAIN RETRIEVAL SUMMARY")
    print(f"Hit@{TOP_K}: {hit_count}/{len(in_domain)}")
    print(f"MRR: {mrr:.4f}")
    print("이 Retrieval 지표는 filter 전후 공통인 actual Top-5 결과입니다.")

    changed = [item for item in evaluations if item["filter_changed"]]
    unchanged = [item for item in evaluations if not item["filter_changed"]]
    before_count = sum(item["original_source_count"] for item in evaluations)
    after_count = sum(item["filtered_source_count"] for item in evaluations)
    print("\nAGGREGATE FILTERING SUMMARY")
    print(f"Total cases: {len(evaluations)}")
    print(f"Cases where filter changed Context: {len(changed)}/{len(evaluations)}")
    print(f"Cases unchanged by filter: {len(unchanged)}/{len(evaluations)}")
    print(f"Total Sources before filtering: {before_count}")
    print(f"Total Sources after filtering: {after_count}")
    print(f"Total Sources dropped: {before_count - after_count}")
    print("Cases with drops:")
    if not changed:
        print("- none")
    for item in changed:
        details = ", ".join(
            f"rank={drop['rank']}/chunk={drop['chunk_id']}/coverage={drop['coverage']:.6f}"
            for drop in item["dropped_results"]
        )
        print(f"- {item['case']['name']}: {details}")
    all_gold_removed_cases = [
        item["case"]["name"] for item in in_domain if item["all_gold_removed"]
    ]
    print(f"Cases where all gold evidence was removed: {all_gold_removed_cases}")

    in_original_refusal = sum(
        item["original_summary"]["no_answer_count"] for item in in_domain
    )
    in_filtered_refusal = sum(
        item["filtered_summary"]["no_answer_count"] for item in in_domain
    )
    in_original_citation_present = sum(
        item["original_summary"]["citation_present_count"] for item in in_domain
    )
    in_filtered_citation_present = sum(
        item["filtered_summary"]["citation_present_count"] for item in in_domain
    )
    in_original_citation_valid = sum(
        item["original_summary"]["citation_valid_count"] for item in in_domain
    )
    in_filtered_citation_valid = sum(
        item["filtered_summary"]["citation_valid_count"] for item in in_domain
    )
    in_run_count = len(in_domain) * REPEAT_COUNT
    print("\nAGGREGATE IN-DOMAIN GENERATION SUMMARY")
    print(f"Original undesired refusal: {in_original_refusal}/{in_run_count}")
    print(f"Filtered undesired refusal: {in_filtered_refusal}/{in_run_count}")
    print(
        f"Original citation present: {in_original_citation_present}/{in_run_count}"
    )
    print(
        f"Filtered citation present: {in_filtered_citation_present}/{in_run_count}"
    )
    print(f"Original citation valid: {in_original_citation_valid}/{in_run_count}")
    print(f"Filtered citation valid: {in_filtered_citation_valid}/{in_run_count}")
    print(f"Improved cases: {names_by_classification(in_domain, 'IMPROVED')}")
    print(f"Regressed cases: {names_by_classification(in_domain, 'REGRESSED')}")
    print(f"Unchanged cases: {names_by_classification(in_domain, 'UNCHANGED')}")

    ood_original_refusal = sum(
        item["original_summary"]["exact_refusal_count"] for item in out_of_domain
    )
    ood_filtered_refusal = sum(
        item["filtered_summary"]["exact_refusal_count"] for item in out_of_domain
    )
    ood_run_count = len(out_of_domain) * REPEAT_COUNT
    print("\nAGGREGATE OOD SUMMARY")
    print(f"Original exact refusal: {ood_original_refusal}/{ood_run_count}")
    print(f"Filtered exact refusal: {ood_filtered_refusal}/{ood_run_count}")
    print(f"OOD regressed cases: {names_by_classification(out_of_domain, 'REGRESSED')}")
    print(f"OOD unchanged cases: {names_by_classification(out_of_domain, 'UNCHANGED')}")

    citation_regressed_cases = [
        item["case"]["name"]
        for item in in_domain
        if item["filtered_summary"]["citation_valid_count"]
        < item["original_summary"]["citation_valid_count"]
    ]
    print("\nRegression check")
    print(
        "1. In-domain refusal behavior regression 존재 여부: "
        f"{bool(names_by_classification(in_domain, 'REGRESSED'))}"
    )
    print(
        "2. OOD exact-refusal regression 존재 여부: "
        f"{bool(names_by_classification(out_of_domain, 'REGRESSED'))}"
    )
    print(
        "3. Filter가 모든 gold evidence를 제거한 Case 존재 여부: "
        f"{bool(all_gold_removed_cases)}"
    )
    print(
        "4. Citation validity regression 존재 여부: "
        f"{bool(citation_regressed_cases)}"
    )
    print(f"Citation validity regressed cases: {citation_regressed_cases}")
    print(
        "이 자동 지표는 답변의 semantic correctness를 증명하지 않습니다. "
        "In-domain 답변은 여전히 사람이 직접 검토해야 합니다."
    )
    print("Production 적용 여부는 이 출력만으로 자동 결정하지 않습니다.")


def main() -> None:
    """전체 9개 Case에서 원본과 filtered Context의 회귀 여부를 평가한다."""
    print(f"Local model: {LOCAL_MODEL_NAME}")
    print(f"temperature: {LOCAL_TEMPERATURE}")
    print(f"seed: {LOCAL_SEED}")
    print(f"NEAR_DUPLICATE_THRESHOLD: {NEAR_DUPLICATE_THRESHOLD}")
    print(f"REPEAT_COUNT: {REPEAT_COUNT}")
    print(f"Total Evaluation Cases: {len(EVAL_CASES)}")
    print(
        "동일 입력을 다시 호출해 stochastic false difference를 만들지 않도록, "
        "filter 미변경 Case는 Original runs를 재사용합니다."
    )

    evaluations = []
    try:
        for case in EVAL_CASES:
            evaluations.append(evaluate_case(case))
    except (APIConnectionError, LocalLLMError, RuntimeError, ValueError) as error:
        print(f"오류: {error}")
        if isinstance(error, APIConnectionError):
            print("FastAPI 서버가 127.0.0.1:8000에서 실행 중인지 확인하세요.")
        if isinstance(error, LocalLLMError):
            print("Ollama 서버와 qwen3:8b 모델 상태를 확인하세요.")
        raise SystemExit(1) from None

    print_aggregate_summary(evaluations)


if __name__ == "__main__":
    main()
