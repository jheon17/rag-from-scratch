"""동일한 RAG API 입력을 반복해 Retrieval과 Generation 변동을 확인한다."""

from rag_basic.api_evaluation import (
    APIConnectionError,
    call_query_api,
    extract_citations,
)
from rag_basic.evaluation import EVAL_CASES
from rag_basic.local_rag import NO_ANSWER


REPEAT_COUNT = 5
CASE_NAMES = (
    "copyright_in_domain",
    "ai_assignment_submission",
    "fake_news_damage_report",
)
CASE_ROLES = {
    "copyright_in_domain": "이전에 안정적으로 답변한 control Case",
    "ai_assignment_submission": (
        "Retrieval 성공 후 Generation이 부적절하게 refusal한 Case"
    ),
    "fake_news_damage_report": (
        "gold evidence가 있는데 Generation이 refusal한 Case"
    ),
}


def select_cases() -> list[dict]:
    """기존 EVAL_CASES에서 실험 대상 세 Case를 이름으로 선택한다."""
    cases_by_name = {case["name"]: case for case in EVAL_CASES}
    missing_names = [name for name in CASE_NAMES if name not in cases_by_name]
    if missing_names:
        raise ValueError(f"평가 Case를 찾지 못했습니다: {missing_names}")
    return [cases_by_name[name] for name in CASE_NAMES]


def run_case(case: dict) -> list[dict]:
    """한 Case를 실제 /query에 다섯 번 요청하고 각 응답을 기록한다."""
    runs = []
    print("\n" + "=" * 88)
    print(f"Case: {case['name']}")
    print(f"역할: {CASE_ROLES[case['name']]}")
    print(f"Query: {case['query']}")

    for run_number in range(1, REPEAT_COUNT + 1):
        status, body = call_query_api(case)
        results = body.get("results", []) if status == 200 else []
        chunk_ids = [result["chunk_id"] for result in results]
        answer = body.get("answer", "") if status == 200 else ""
        citations = extract_citations(answer)
        run = {
            "run_number": run_number,
            "status": status,
            "chunk_ids": chunk_ids,
            "context_length": len(body.get("context", "")),
            "answer": answer,
            "citations": citations,
            "citation_present": bool(citations),
            "exact_refusal": answer == NO_ANSWER,
            "detail": body.get("detail"),
        }
        runs.append(run)

        print(f"\nRun {run_number}")
        print(f"HTTP status: {status}")
        if status != 200:
            print(f"Response detail: {run['detail']}")
            continue
        print(f"Top-5 chunk_id: {chunk_ids}")
        print(f"Context length: {run['context_length']}")
        print(f"Answer: {answer}")
        print(f"Citation Source 번호: {citations}")
        print(f"Citation present: {run['citation_present']}")
        print(f"Exact refusal: {run['exact_refusal']}")

    return runs


def summarize_case(case: dict, runs: list[dict]) -> dict:
    """한 Case의 Retrieval 및 정확한 답변 문자열 variant를 집계한다."""
    successful_runs = [run for run in runs if run["status"] == 200]
    retrieval_variants = {
        tuple(run["chunk_ids"]) for run in successful_runs
    }
    answer_variants = {run["answer"] for run in successful_runs}
    summary = {
        "case_name": case["name"],
        "successful_runs": len(successful_runs),
        "retrieval_variant_count": len(retrieval_variants),
        "unique_answer_count": len(answer_variants),
        "exact_refusal_count": sum(
            run["exact_refusal"] for run in successful_runs
        ),
        "citation_present_count": sum(
            run["citation_present"] for run in successful_runs
        ),
    }

    print("\nCase Summary")
    print(f"Runs: {len(runs)}")
    print(f"HTTP success: {summary['successful_runs']}/{len(runs)}")
    print(f"Retrieval variants: {summary['retrieval_variant_count']}")
    print(f"Unique answer count: {summary['unique_answer_count']}")
    print(
        f"Exact refusal: {summary['exact_refusal_count']}/{len(runs)}"
    )
    print(
        f"Citation present: {summary['citation_present_count']}/{len(runs)}"
    )
    return summary


def print_overall_summary(summaries: list[dict]) -> None:
    """세 Case의 variant 수와 이번 반복에서 관찰한 현상을 요약한다."""
    print("\n" + "=" * 88)
    print("전체 요약")
    for summary in summaries:
        print(f"\n{summary['case_name']}")
        print(
            f"  Retrieval variants: "
            f"{summary['retrieval_variant_count']}"
        )
        print(f"  Unique answers: {summary['unique_answer_count']}")
        print(
            f"  Exact refusal: "
            f"{summary['exact_refusal_count']}/{REPEAT_COUNT}"
        )
        print(
            f"  Citation present: "
            f"{summary['citation_present_count']}/{REPEAT_COUNT}"
        )

    retrieval_changed = any(
        summary["retrieval_variant_count"] > 1 for summary in summaries
    )
    generation_changed = any(
        summary["unique_answer_count"] > 1 for summary in summaries
    )

    print("\n해석:")
    if retrieval_changed:
        print(
            "Retrieval Top-5 변동이 관찰되어 Answer 차이를 Generation만의 "
            "변동성이라고 단정할 수 없습니다."
        )
    elif generation_changed:
        print(
            "Retrieval Top-5는 동일하지만 서로 다른 Answer 문자열이 생성되어 "
            "Generation output 변동성이 관찰됐습니다."
        )
    else:
        print(
            "이번 5회 반복에서는 Retrieval Top-5와 Answer 문자열의 변동이 "
            "관찰되지 않았습니다."
        )
    print(
        "이 결과만으로 temperature, seed 또는 모델 자체가 원인이라고 "
        "판단하지 않습니다."
    )
    print(
        "Answer는 공백이나 의미를 정규화하지 않은 exact string 기준으로 "
        "비교했습니다."
    )


def main() -> None:
    """선택한 세 Case를 각각 다섯 번 실행하고 변동성을 출력한다."""
    summaries = []
    try:
        for case in select_cases():
            runs = run_case(case)
            summaries.append(summarize_case(case, runs))
    except APIConnectionError as error:
        print(f"변동성 실험을 실행할 수 없습니다: {error}")
        print("FastAPI 서버가 실행 중인지 확인하세요.")
        raise SystemExit(1) from None

    print_overall_summary(summaries)


if __name__ == "__main__":
    main()
