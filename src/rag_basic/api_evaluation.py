"""기존 평가 Case를 실제 FastAPI /query 응답으로 검증한다."""

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_basic.chunking import PDF_PATH
from rag_basic.evaluation import EVAL_CASES
from rag_basic.local_llm import LOCAL_MODEL_NAME
from rag_basic.local_rag import NO_ANSWER
from rag_basic.retrieval import TOP_K


API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
TEXT_PREVIEW_LENGTH = 300


class APIConnectionError(Exception):
    """FastAPI 서버에 연결할 수 없음을 나타낸다."""


def call_query_api(case: dict) -> tuple[int, dict]:
    """한 평가 질문을 /query에 보내고 HTTP status와 JSON을 반환한다."""
    payload = {
        "document_name": PDF_PATH.name,
        "query": case["query"],
        "top_k": TOP_K,
    }
    request = Request(
        f"{API_BASE_URL}/query",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body
    except HTTPError as error:
        try:
            body = json.loads(error.read().decode("utf-8"))
        except json.JSONDecodeError:
            body = {"detail": "JSON 형식이 아닌 오류 응답입니다."}
        return error.code, body
    except URLError as error:
        raise APIConnectionError(
            f"FastAPI 서버에 연결할 수 없습니다: {error.reason}"
        ) from error


def extract_citations(answer: str) -> list[int]:
    """답변에서 [Source N] 번호를 중복 없이 등장 순서대로 추출한다."""
    numbers = [int(number) for number in re.findall(r"\[Source (\d+)\]", answer)]
    return list(dict.fromkeys(numbers))


def evaluate_retrieval(
    results: list[dict], expected_chunk_ids: list[int]
) -> dict:
    """실제 API 검색 결과에서 Hit@K와 Reciprocal Rank를 계산한다."""
    first_gold_rank = next(
        (
            result["rank"]
            for result in results
            if result["chunk_id"] in expected_chunk_ids
        ),
        None,
    )
    return {
        "first_gold_rank": first_gold_rank,
        "hit_at_k": first_gold_rank is not None,
        "reciprocal_rank": (
            1.0 / first_gold_rank if first_gold_rank is not None else 0.0
        ),
    }


def evaluate_case(case: dict, status: int, body: dict) -> dict:
    """한 HTTP 응답의 공통 구조와 Case 유형별 지표를 평가한다."""
    if status != 200:
        return {
            "case": case,
            "status": status,
            "body": body,
            "basic": None,
            "retrieval": None,
            "generation": None,
        }

    results = body.get("results", [])
    answer = body.get("answer", "")
    citations = extract_citations(answer)
    basic = {
        "document_name_matches": body.get("document_name") == PDF_PATH.name,
        "query_matches": body.get("query") == case["query"],
        "top_k_matches": body.get("top_k") == TOP_K,
        "result_count_matches": len(results) == TOP_K,
        "ranks_match": [result.get("rank") for result in results]
        == list(range(1, TOP_K + 1)),
        "context_non_empty": bool(body.get("context")),
        "answer_non_empty": bool(answer),
        "llm_model_matches": body.get("llm_model") == LOCAL_MODEL_NAME,
    }

    retrieval = None
    if case["type"] == "in_domain":
        retrieval = evaluate_retrieval(results, case["expected_chunk_ids"])

    generation = {
        "answer_non_empty": bool(answer),
        "citations": citations,
        "citation_present": bool(citations),
        "citation_numbers_valid": bool(citations)
        and all(1 <= number <= len(results) for number in citations),
        "exact_refusal": answer == NO_ANSWER,
    }
    return {
        "case": case,
        "status": status,
        "body": body,
        "basic": basic,
        "retrieval": retrieval,
        "generation": generation,
    }


def print_cited_sources(results: list[dict], citations: list[int]) -> None:
    """Citation 번호에 대응하는 검색 결과 metadata와 text 일부를 출력한다."""
    for source_number in citations:
        if not 1 <= source_number <= len(results):
            print(f"  Source {source_number}: Retrieval 결과 범위를 벗어남")
            continue

        result = results[source_number - 1]
        preview = result["text"][:TEXT_PREVIEW_LENGTH].replace("\n", " ")
        print(
            f"  Source {source_number}: rank={result['rank']}, "
            f"chunk_id={result['chunk_id']}, "
            f"page_number={result['page_number']}"
        )
        print(f"    text preview: {preview}")


def print_case_result(evaluation: dict) -> None:
    """한 Case의 응답과 자동 평가 결과를 사람이 검토할 수 있게 출력한다."""
    case = evaluation["case"]
    status = evaluation["status"]
    body = evaluation["body"]

    print("\n" + "=" * 88)
    print(f"Case: {case['name']}")
    print(f"Type: {case['type']}")
    print(f"Query: {case['query']}")
    print(f"HTTP status: {status}")

    if status != 200:
        print(f"Response detail: {body.get('detail', '확인할 수 없음')}")
        return

    results = body["results"]
    chunk_ids = [result["chunk_id"] for result in results]
    print(f"Top-{TOP_K} chunk_id: {chunk_ids}")
    print(f"기본 Response 검증: {evaluation['basic']}")

    if case["type"] == "in_domain":
        retrieval = evaluation["retrieval"]
        generation = evaluation["generation"]
        print(f"expected_chunk_ids: {case['expected_chunk_ids']}")
        print(f"first gold rank: {retrieval['first_gold_rank']}")
        print(f"Hit@{TOP_K}: {retrieval['hit_at_k']}")
        print(f"Reciprocal Rank: {retrieval['reciprocal_rank']:.4f}")
        print(f"Answer: {body['answer']}")
        print(f"Citation Source 번호: {generation['citations']}")
        print(f"Citation 존재: {generation['citation_present']}")
        print(
            "Citation 번호 범위 유효: "
            f"{generation['citation_numbers_valid']}"
        )
        print("Cited Source 연결:")
        print_cited_sources(results, generation["citations"])
    else:
        generation = evaluation["generation"]
        print(f"Answer: {body['answer']}")
        print(f"Exact refusal: {generation['exact_refusal']}")


def print_summary(evaluations: list[dict]) -> None:
    """Retrieval 및 Generation 자동 평가 결과와 기존 baseline 비교를 출력한다."""
    in_domain = [
        item for item in evaluations if item["case"]["type"] == "in_domain"
    ]
    out_of_domain = [
        item
        for item in evaluations
        if item["case"]["type"] == "out_of_domain"
    ]
    successful_in_domain = [item for item in in_domain if item["status"] == 200]
    successful_out_of_domain = [
        item for item in out_of_domain if item["status"] == 200
    ]

    hit_count = sum(
        item["retrieval"]["hit_at_k"] for item in successful_in_domain
    )
    mrr = sum(
        item["retrieval"]["reciprocal_rank"] for item in successful_in_domain
    ) / len(in_domain)
    answer_non_empty_count = sum(
        item["generation"]["answer_non_empty"] for item in successful_in_domain
    )
    citation_present_count = sum(
        item["generation"]["citation_present"] for item in successful_in_domain
    )
    valid_citation_count = sum(
        item["generation"]["citation_numbers_valid"]
        for item in successful_in_domain
    )
    refusal_count = sum(
        item["generation"]["exact_refusal"]
        for item in successful_out_of_domain
    )

    print("\n" + "=" * 88)
    print("Retrieval Summary")
    print(f"Hit@{TOP_K}: {hit_count}/{len(in_domain)}")
    print(f"MRR: {mrr:.4f}")

    print("\nGeneration Summary")
    print(
        f"In-domain answer non-empty: "
        f"{answer_non_empty_count}/{len(in_domain)}"
    )
    print(
        f"In-domain citation present: "
        f"{citation_present_count}/{len(in_domain)}"
    )
    print(
        f"In-domain citation number valid: "
        f"{valid_citation_count}/{len(in_domain)}"
    )
    print(f"OOD exact refusal: {refusal_count}/{len(out_of_domain)}")

    print("\n기존 Python baseline과 비교")
    print("기존 Retrieval: Hit@5=6/6, MRR=0.8750")
    print(f"현재 API Retrieval: Hit@5={hit_count}/6, MRR={mrr:.4f}")
    print("기존 Generation: in-domain citation=6/6, OOD refusal=3/3")
    print(
        "현재 API Generation: "
        f"in-domain citation={valid_citation_count}/6, "
        f"OOD refusal={refusal_count}/3"
    )
    print(
        "Vector Search는 threshold 없이 nearest Top-5를 반환하므로 "
        "OOD 질문에도 Retrieval 결과가 존재할 수 있습니다."
    )
    print(
        "Citation 검증은 표기 존재와 번호 범위만 확인하며, answer correctness나 "
        "semantic/citation faithfulness 자동 평가가 아닙니다."
    )


def main() -> None:
    """9개 Case를 실제 /query로 요청하고 응답 기반 평가를 출력한다."""
    evaluations = []
    try:
        for case in EVAL_CASES:
            status, body = call_query_api(case)
            evaluation = evaluate_case(case, status, body)
            evaluations.append(evaluation)
            print_case_result(evaluation)
    except APIConnectionError as error:
        print(f"API 평가를 시작할 수 없습니다: {error}")
        print("FastAPI 서버가 실행 중인지 확인하세요.")
        raise SystemExit(1) from None

    print_summary(evaluations)


if __name__ == "__main__":
    main()
