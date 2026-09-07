"""수동으로 구성한 소규모 Case로 Retrieval과 Generation을 평가한다."""

import os
import re

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME as EMBEDDING_MODEL_NAME
from rag_basic.embedding import embed_texts
from rag_basic.llm import LLM_MODEL_NAME, NO_ANSWER, generate_answer
from rag_basic.retrieval import TOP_K, build_context, retrieve
from rag_basic.vector_search import build_index


EVAL_CASES = [
    {
        "name": "copyright_in_domain",
        "query": "생성형 AI가 만든 이미지의 저작권은 누구에게 있나요?",
        "type": "in_domain",
        "expected_chunk_ids": [28],
        "should_refuse": False,
    },
    {
        "name": "creative_contribution_copyright",
        "query": "생성형 AI 결과물에 이용자가 창작적 표현을 추가하면 저작권을 인정받을 수 있나요?",
        "type": "in_domain",
        "expected_chunk_ids": [33],
        "should_refuse": False,
    },
    {
        "name": "ai_assignment_submission",
        "query": "생성형 AI가 만든 결과물을 그대로 과제로 제출해도 되나요?",
        "type": "in_domain",
        "expected_chunk_ids": [68, 69, 70],
        "should_refuse": False,
    },
    {
        "name": "midjourney_contest_controversy",
        "query": "미드저니를 사용한 작품이 미술대회에서 논란이 된 이유는 무엇인가요?",
        "type": "in_domain",
        "expected_chunk_ids": [65, 74],
        "should_refuse": False,
    },
    {
        "name": "fake_news_damage_report",
        "query": "생성형 AI로 만든 가짜 뉴스 피해는 어디에 신고하거나 상담할 수 있나요?",
        "type": "in_domain",
        "expected_chunk_ids": [104, 105],
        "should_refuse": False,
    },
    {
        "name": "generative_ai_work_benefits",
        "query": "생성형 AI를 업무에 활용하면 어떤 장점이 있나요?",
        "type": "in_domain",
        "expected_chunk_ids": [137],
        "should_refuse": False,
    },
    {
        "name": "france_out_of_domain",
        "query": "프랑스의 수도는 어디인가요?",
        "type": "out_of_domain",
        "expected_chunk_ids": [],
        "should_refuse": True,
    },
    {
        "name": "solar_system_out_of_domain",
        "query": "태양계에서 가장 큰 행성은 무엇인가요?",
        "type": "out_of_domain",
        "expected_chunk_ids": [],
        "should_refuse": True,
    },
    {
        "name": "triangle_out_of_domain",
        "query": "삼각형의 내각의 합은 몇 도인가요?",
        "type": "out_of_domain",
        "expected_chunk_ids": [],
        "should_refuse": True,
    },
]


def extract_source_numbers(answer: str) -> list[int]:
    """답변의 [Source N] 표기에서 Source 번호를 추출한다."""
    return [int(number) for number in re.findall(r"\[Source (\d+)\]", answer)]


def evaluate_in_domain(
    results: list[dict], answer: str, expected_chunk_ids: list[int]
) -> dict:
    """in-domain Case의 검색 성공, 순위와 Source 표기를 평가한다."""
    retrieved_chunk_ids = [result["chunk_id"] for result in results]
    retrieval_hit_at_k = any(
        chunk_id in retrieved_chunk_ids for chunk_id in expected_chunk_ids
    )

    first_gold_rank = next(
        (
            result["rank"]
            for result in results
            if result["chunk_id"] in expected_chunk_ids
        ),
        None,
    )
    reciprocal_rank = 1.0 / first_gold_rank if first_gold_rank is not None else 0.0

    cited_source_numbers = extract_source_numbers(answer)
    has_source_citation = bool(cited_source_numbers)
    source_citations_are_valid = all(
        1 <= source_number <= len(results)
        for source_number in cited_source_numbers
    )

    return {
        "retrieval_hit_at_k": retrieval_hit_at_k,
        "reciprocal_rank": reciprocal_rank,
        "answer_non_empty": bool(answer),
        "unexpected_refusal": answer == NO_ANSWER,
        "cited_source_numbers": cited_source_numbers,
        "has_source_citation": has_source_citation,
        "source_citations_are_valid": source_citations_are_valid,
    }


def evaluate_out_of_domain(answer: str) -> dict:
    """out-of-domain Case가 지정된 문장으로 답변을 거절했는지 평가한다."""
    return {"refusal_correct": answer == NO_ANSWER}


def print_case_result(
    case: dict, results: list[dict], answer: str, evaluation: dict
) -> None:
    """한 평가 Case의 검색 결과, 답변과 지표를 출력한다."""
    chunk_ids = [result["chunk_id"] for result in results]
    scores = [round(result["score"], 4) for result in results]

    print(f"\nCase name: {case['name']}")
    print(f"Query: {case['query']}")
    print(f"검색된 chunk_id 목록: {chunk_ids}")
    print(f"검색 score: {scores}")
    print(f"생성된 답변: {answer}")
    print("평가 결과:")
    for metric_name, metric_value in evaluation.items():
        print(f"  {metric_name}: {metric_value}")


def print_summary(case_evaluations: list[dict]) -> None:
    """두 종류의 Case 수와 기본 평가 지표를 요약한다."""
    in_domain_evaluations = [
        item for item in case_evaluations if item["type"] == "in_domain"
    ]
    out_of_domain_evaluations = [
        item for item in case_evaluations if item["type"] == "out_of_domain"
    ]

    hit_count = sum(
        item["evaluation"]["retrieval_hit_at_k"]
        for item in in_domain_evaluations
    )
    mean_reciprocal_rank = (
        sum(
            item["evaluation"]["reciprocal_rank"]
            for item in in_domain_evaluations
        )
        / len(in_domain_evaluations)
        if in_domain_evaluations
        else 0.0
    )
    valid_citation_count = sum(
        item["evaluation"]["has_source_citation"]
        and item["evaluation"]["source_citations_are_valid"]
        for item in in_domain_evaluations
    )
    refusal_count = sum(
        item["evaluation"]["refusal_correct"]
        for item in out_of_domain_evaluations
    )

    print("\n전체 요약:")
    print(f"  전체 평가 Case 수: {len(case_evaluations)}")
    print(f"  in_domain Case 수: {len(in_domain_evaluations)}")
    print(f"  out_of_domain Case 수: {len(out_of_domain_evaluations)}")
    print(
        f"  in_domain Hit@{TOP_K} 통과 수: "
        f"{hit_count}/{len(in_domain_evaluations)}"
    )
    print(f"  in_domain 평균 Reciprocal Rank: {mean_reciprocal_rank:.4f}")
    print(
        "  정상 Source citation 검증 통과 수: "
        f"{valid_citation_count}/{len(in_domain_evaluations)}"
    )
    print(
        "  out_of_domain refusal 통과 수: "
        f"{refusal_count}/{len(out_of_domain_evaluations)}"
    )
    print(
        "현재 평가는 소규모 수동 구성 baseline Case를 사용하므로 "
        "전체 RAG 품질을 대표하지 않습니다."
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        print('설정 방법: export OPENAI_API_KEY="발급받은_API_Key"')
        print("환경변수를 설정한 뒤 다시 실행해 주세요.")
        return

    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return

    client = OpenAI()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(embedding_model, chunk_texts, "passage")
    chunk_embeddings = np.ascontiguousarray(chunk_embeddings, dtype=np.float32)
    index = build_index(chunk_embeddings)

    case_evaluations = []
    for case in EVAL_CASES:
        results = retrieve(
            case["query"], embedding_model, index, chunks, TOP_K
        )
        context = build_context(results)
        answer = generate_answer(
            case["query"], context, client, LLM_MODEL_NAME
        )

        if case["type"] == "in_domain":
            evaluation = evaluate_in_domain(
                results, answer, case["expected_chunk_ids"]
            )
        else:
            evaluation = evaluate_out_of_domain(answer)

        case_evaluations.append(
            {"type": case["type"], "evaluation": evaluation}
        )
        print_case_result(case, results, answer, evaluation)

    print_summary(case_evaluations)


if __name__ == "__main__":
    main()
