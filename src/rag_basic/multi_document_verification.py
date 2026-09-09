"""두 실제 PDF의 document_name 기반 RAG 분리를 검증한다."""

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg

from rag_basic.api_evaluation import (
    API_BASE_URL,
    extract_citations,
    extract_context_source_numbers,
)
from rag_basic.chunking import PDF_PATH
from rag_basic.context_selection import NEAR_DUPLICATE_COVERAGE_THRESHOLD
from rag_basic.embedding import MODEL_NAME
from rag_basic.evaluation import EVAL_CASES
from rag_basic.local_llm import LOCAL_MODEL_NAME
from rag_basic.local_rag import NO_ANSWER
from rag_basic.pgvector_ingest import count_document_rows, get_database_config
from rag_basic.retrieval import TOP_K, build_context


SECOND_DOCUMENT_NAME = "nist.ai.100-1.pdf"
SECOND_PDF_PATH = Path("/home/q2103/다운로드/nist.ai.100-1.pdf")
NIST_QUERY = "AI RMF Core를 구성하는 네 가지 기능은 무엇인가요?"
NIST_PREVIEW_LENGTH = 300
EXPECTED_ASSIGNMENT_CHUNK_IDS = [69, 68, 76, 70, 75]
EXPECTED_ASSIGNMENT_CONTEXT_SOURCES = [1, 2, 3, 5]


class VerificationError(Exception):
    """Multi-document 검증을 계속할 수 없는 오류를 나타낸다."""


def find_eval_case(name: str) -> dict:
    """기존 평가 Case를 이름으로 찾아 반환한다."""
    case = next((case for case in EVAL_CASES if case["name"] == name), None)
    if case is None:
        raise VerificationError(f"평가 Case를 찾지 못했습니다: {name}")
    return case


def call_query_api(
    document_name: str,
    query: str,
    top_k: int = TOP_K,
) -> tuple[int, dict]:
    """임의 문서명과 질문으로 Production /query API를 호출한다."""
    payload = {
        "document_name": document_name,
        "query": query,
        "top_k": top_k,
    }
    request = Request(
        f"{API_BASE_URL}/query",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body
    except HTTPError as error:
        try:
            body = json.loads(error.read().decode("utf-8"))
        except json.JSONDecodeError:
            body = {"detail": "JSON 형식이 아닌 오류 응답입니다."}
        return error.code, body
    except URLError as error:
        raise VerificationError(
            f"FastAPI 서버에 연결할 수 없습니다: {error.reason}"
        ) from error


def result_belongs_to_document(
    conn: psycopg.Connection,
    document_name: str,
    result: dict,
) -> bool:
    """문서명, Chunk ID, 본문이 모두 같은 DB 행이 존재하는지 확인한다."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM rag_chunks
                WHERE document_name = %s
                  AND chunk_id = %s
                  AND content = %s
                  AND embedding_model = %s
            );
            """,
            (
                document_name,
                result["chunk_id"],
                result["text"],
                MODEL_NAME,
            ),
        )
        return bool(cursor.fetchone()[0])


def evaluate_response(
    conn: psycopg.Connection,
    requested_document_name: str,
    status: int,
    body: dict,
) -> dict:
    """한 API 응답의 문서 소속, Context, Citation을 검증한다."""
    if status != 200:
        raise VerificationError(
            f"/query가 HTTP {status}를 반환했습니다: {body.get('detail')}"
        )

    results = body.get("results", [])
    context = body.get("context", "")
    answer = body.get("answer", "")
    result_ranks = {int(result["rank"]) for result in results}
    context_source_numbers = extract_context_source_numbers(context)
    citations = extract_citations(answer)

    membership_checks = [
        result_belongs_to_document(conn, requested_document_name, result)
        for result in results
    ]
    all_results_belong = bool(results) and all(membership_checks)
    context_sources_valid = bool(context_source_numbers) and set(
        context_source_numbers
    ).issubset(result_ranks)
    context_results = [
        result
        for result in results
        if result["rank"] in context_source_numbers
    ]
    context_rebuild_matches = build_context(context_results) == context
    citation_context_valid = not citations or set(citations).issubset(
        context_source_numbers
    )

    return {
        "membership_checks": membership_checks,
        "all_results_belong": all_results_belong,
        "contamination_detected": not all_results_belong,
        "context_source_numbers": context_source_numbers,
        "context_sources_valid": context_sources_valid,
        "context_rebuild_matches": context_rebuild_matches,
        "citations": citations,
        "citation_context_valid": citation_context_valid,
        "contains_no_answer": NO_ANSWER in answer,
    }


def print_query_result(
    label: str,
    requested_document_name: str,
    query: str,
    status: int,
    body: dict,
    validation: dict,
    show_previews: bool = False,
) -> None:
    """한 Query의 응답과 검증 결과를 출력한다."""
    results = body["results"]
    context = body["context"]

    print("\n" + "=" * 88)
    print(f"Label: {label}")
    print(f"Requested document_name: {requested_document_name}")
    print(f"Query: {query}")
    print(f"HTTP status: {status}")
    print(f"Response document_name: {body.get('document_name')}")
    print(f"top_k: {body.get('top_k')}")
    print(f"Result count: {len(results)}")
    print(f"Result ranks: {[result['rank'] for result in results]}")
    print(f"Result chunk_ids: {[result['chunk_id'] for result in results]}")
    print(f"Result page_numbers: {[result['page_number'] for result in results]}")
    print(f"Result scores: {[round(result['score'], 4) for result in results]}")
    print(f"Context Source numbers: {validation['context_source_numbers']}")
    print(f"Context length: {len(context)}")
    print(f"Retrieval Source count: {len(results)}")
    print(f"Context Source count: {len(validation['context_source_numbers'])}")
    print(f"Answer: {body['answer']}")
    print(f"Citations: {validation['citations']}")
    print(f"Citation context-valid: {validation['citation_context_valid']}")
    print(
        "All Retrieval results belong to requested document: "
        f"{validation['all_results_belong']}"
    )
    print(
        "Cross-document Retrieval contamination detected: "
        f"{validation['contamination_detected']}"
    )
    print(f"Context Source numbers valid: {validation['context_sources_valid']}")
    print(f"Context rebuild matches: {validation['context_rebuild_matches']}")

    if show_previews:
        print("\nNIST Top-5 Source text preview:")
        for result in results:
            preview = result["text"][:NIST_PREVIEW_LENGTH].replace("\n", " ")
            print(
                f"  rank={result['rank']}, chunk_id={result['chunk_id']}, "
                f"page={result['page_number']}"
            )
            print(f"    text preview: {preview}")


def run_query(
    conn: psycopg.Connection,
    label: str,
    document_name: str,
    query: str,
    show_previews: bool = False,
) -> dict:
    """한 Query를 정확히 한 번 호출하고 검증 및 출력을 수행한다."""
    status, body = call_query_api(document_name, query)
    validation = evaluate_response(conn, document_name, status, body)
    print_query_result(
        label,
        document_name,
        query,
        status,
        body,
        validation,
        show_previews,
    )
    return {"status": status, "body": body, "validation": validation}


def validate_assignment_regression(query_result: dict) -> bool:
    """기존 과제 질문의 Production Context deduplication을 회귀 검증한다."""
    body = query_result["body"]
    validation = query_result["validation"]
    chunk_ids = [result["chunk_id"] for result in body["results"]]
    passed = all(
        (
            chunk_ids == EXPECTED_ASSIGNMENT_CHUNK_IDS,
            validation["context_source_numbers"]
            == EXPECTED_ASSIGNMENT_CONTEXT_SOURCES,
            not validation["contains_no_answer"],
            validation["citation_context_valid"],
        )
    )

    print("\n기존 assignment regression:")
    print(f"  results chunk_ids: {chunk_ids}")
    print(
        "  Context Source: "
        f"{validation['context_source_numbers']}"
    )
    print(f"  NO_ANSWER phrase 포함: {validation['contains_no_answer']}")
    print(
        "  Citation context-valid: "
        f"{validation['citation_context_valid']}"
    )
    print(f"  Existing assignment regression passed: {passed}")
    return passed


def main() -> None:
    """두 문서에 대한 네 Query를 실행하고 document isolation을 검증한다."""
    if not SECOND_PDF_PATH.exists():
        print(f"두 번째 PDF를 찾지 못했습니다: {SECOND_PDF_PATH}")
        raise SystemExit(1)

    assignment_case = find_eval_case("ai_assignment_submission")

    print(f"Embedding model: {MODEL_NAME}")
    print(f"Local model: {LOCAL_MODEL_NAME}")
    print(
        "Near-duplicate coverage threshold: "
        f"{NEAR_DUPLICATE_COVERAGE_THRESHOLD}"
    )

    try:
        with psycopg.connect(**get_database_config()) as conn:
            original_rows = count_document_rows(conn, PDF_PATH.name)
            second_rows = count_document_rows(conn, SECOND_DOCUMENT_NAME)
            print(f"Original document row count: {original_rows}")
            print(f"Second document row count: {second_rows}")

            if original_rows <= 0 or second_rows <= 0:
                raise VerificationError(
                    "두 문서가 모두 DB에 적재된 상태에서 실행해야 합니다."
                )

            query_results = [
                run_query(
                    conn,
                    "A. NIST document + NIST query",
                    SECOND_DOCUMENT_NAME,
                    NIST_QUERY,
                    show_previews=True,
                ),
                run_query(
                    conn,
                    "B. Original document + NIST query",
                    PDF_PATH.name,
                    NIST_QUERY,
                ),
                run_query(
                    conn,
                    "C. Original document + assignment query",
                    PDF_PATH.name,
                    assignment_case["query"],
                ),
                run_query(
                    conn,
                    "D. NIST document + assignment query",
                    SECOND_DOCUMENT_NAME,
                    assignment_case["query"],
                ),
            ]

            assignment_regression_passed = validate_assignment_regression(
                query_results[2]
            )
            if not assignment_regression_passed:
                raise VerificationError(
                    "기존 assignment regression 결과가 예상과 다릅니다."
                )

            a_texts = [
                item["text"] for item in query_results[0]["body"]["results"]
            ]
            b_texts = [
                item["text"] for item in query_results[1]["body"]["results"]
            ]
            c_texts = [
                item["text"] for item in query_results[2]["body"]["results"]
            ]
            d_texts = [
                item["text"] for item in query_results[3]["body"]["results"]
            ]
            all_membership = all(
                item["validation"]["all_results_belong"]
                for item in query_results
            )
            all_http_200 = all(item["status"] == 200 for item in query_results)
            all_context_rebuild = all(
                item["validation"]["context_rebuild_matches"]
                for item in query_results
            )

            print("\n" + "=" * 88)
            print("Document-name effect 비교")
            print(f"A vs B result texts identical: {a_texts == b_texts}")
            print(f"C vs D result texts identical: {c_texts == d_texts}")
            print(
                "Same query is scoped independently by document_name: "
                f"{all_membership}"
            )

            print("\n" + "=" * 88)
            print("Multi-Document Summary")
            print(f"Original document row count: {original_rows}")
            print(f"Second document row count: {second_rows}")
            print(
                "Both documents coexist in PostgreSQL: "
                f"{original_rows > 0 and second_rows > 0}"
            )
            print(f"All four queries HTTP 200: {all_http_200}")
            print(
                "All retrieval results scoped to requested document: "
                f"{all_membership}"
            )
            print(
                "Cross-document contamination detected: "
                f"{not all_membership}"
            )
            print(
                "All Context rebuild checks passed: "
                f"{all_context_rebuild}"
            )
            print(
                "Existing assignment regression passed: "
                f"{assignment_regression_passed}"
            )
            print(
                "이번 결과는 두 실제 PDF의 document_name 기반 Retrieval isolation과 "
                "Production Context selection 경로를 확인한 integration verification입니다."
            )
    except (VerificationError, ValueError, OSError, psycopg.Error) as error:
        print(f"Multi-document 검증 실패: {error}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
