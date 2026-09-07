"""동일한 Retrieval Context로 OpenAI와 Local LLM 답변을 비교한다."""

import os

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import PDF_PATH, create_chunks, load_pages
from rag_basic.embedding import MODEL_NAME, embed_texts
from rag_basic.evaluation import (
    EVAL_CASES,
    evaluate_in_domain,
    evaluate_out_of_domain,
)
from rag_basic.llm import LLM_MODEL_NAME, generate_answer
from rag_basic.local_llm import (
    LOCAL_MODEL_NAME,
    LocalLLMError,
    generate_local_answer,
)
from rag_basic.local_rag import build_local_rag_prompt
from rag_basic.retrieval import TOP_K, build_context, retrieve
from rag_basic.vector_search import build_index


IN_DOMAIN_GENERATION_METRICS = [
    "answer_non_empty",
    "unexpected_refusal",
    "cited_source_numbers",
    "has_source_citation",
    "source_citations_are_valid",
]


def evaluate_answer(case: dict, results: list[dict], answer: str) -> dict:
    """기존 평가 함수를 사용해 한 모델의 답변을 평가한다."""
    if case["type"] == "in_domain":
        return evaluate_in_domain(
            results, answer, case["expected_chunk_ids"]
        )
    return evaluate_out_of_domain(answer)


def print_model_result(
    label: str, model_name: str, answer: str, evaluation: dict, case_type: str
) -> None:
    """모델 답변과 Generation 관련 평가 결과를 출력한다."""
    print(f"\n{label}:")
    print(f"  모델명: {model_name}")
    print(f"  생성 답변: {answer}")
    print("  평가 결과:")

    if case_type == "in_domain":
        for metric_name in IN_DOMAIN_GENERATION_METRICS:
            print(f"    {metric_name}: {evaluation[metric_name]}")
    else:
        print(f"    refusal_correct: {evaluation['refusal_correct']}")


def print_case_result(
    case: dict,
    results: list[dict],
    openai_answer: str,
    openai_evaluation: dict,
    local_answer: str,
    local_evaluation: dict,
) -> None:
    """공통 Retrieval과 두 모델의 Generation 결과를 함께 출력한다."""
    chunk_ids = [result["chunk_id"] for result in results]
    scores = [round(result["score"], 4) for result in results]

    print("\n" + "=" * 80)
    print(f"Case name: {case['name']}")
    print(f"Query: {case['query']}")
    print("공통 Retrieval (두 모델이 동일한 결과와 Context를 사용):")
    print(f"  chunk_id 목록: {chunk_ids}")
    print(f"  score 목록: {scores}")

    if case["type"] == "in_domain":
        print(f"  retrieval_hit_at_k: {openai_evaluation['retrieval_hit_at_k']}")
        print(f"  reciprocal_rank: {openai_evaluation['reciprocal_rank']}")

    print_model_result(
        "OpenAI",
        LLM_MODEL_NAME,
        openai_answer,
        openai_evaluation,
        case["type"],
    )
    print_model_result(
        "Local",
        LOCAL_MODEL_NAME,
        local_answer,
        local_evaluation,
        case["type"],
    )


def summarize_generation(records: list[dict], evaluation_key: str) -> dict:
    """한 모델의 in-domain 및 out-of-domain Generation 결과를 집계한다."""
    in_domain_records = [
        record for record in records if record["type"] == "in_domain"
    ]
    out_of_domain_records = [
        record for record in records if record["type"] == "out_of_domain"
    ]

    answer_non_empty_count = sum(
        record[evaluation_key]["answer_non_empty"]
        for record in in_domain_records
    )
    no_unexpected_refusal_count = sum(
        not record[evaluation_key]["unexpected_refusal"]
        for record in in_domain_records
    )
    valid_citation_count = sum(
        record[evaluation_key]["has_source_citation"]
        and record[evaluation_key]["source_citations_are_valid"]
        for record in in_domain_records
    )
    refusal_count = sum(
        record[evaluation_key]["refusal_correct"]
        for record in out_of_domain_records
    )

    return {
        "answer_non_empty_count": answer_non_empty_count,
        "no_unexpected_refusal_count": no_unexpected_refusal_count,
        "valid_citation_count": valid_citation_count,
        "refusal_count": refusal_count,
        "in_domain_count": len(in_domain_records),
        "out_of_domain_count": len(out_of_domain_records),
    }


def print_generation_summary(label: str, summary: dict) -> None:
    """한 모델의 Generation 평가 요약을 출력한다."""
    print(f"\n{label} Generation:")
    print(
        "  in-domain answer_non_empty 통과 수: "
        f"{summary['answer_non_empty_count']}/{summary['in_domain_count']}"
    )
    print(
        "  in-domain unexpected_refusal이 False인 수: "
        f"{summary['no_unexpected_refusal_count']}/{summary['in_domain_count']}"
    )
    print(
        "  정상 Source citation 검증 통과 수: "
        f"{summary['valid_citation_count']}/{summary['in_domain_count']}"
    )
    print(
        "  out-of-domain refusal 통과 수: "
        f"{summary['refusal_count']}/{summary['out_of_domain_count']}"
    )


def print_summary(records: list[dict]) -> None:
    """공통 Retrieval과 두 모델의 Generation 비교 결과를 요약한다."""
    in_domain_records = [
        record for record in records if record["type"] == "in_domain"
    ]
    hit_count = sum(
        record["retrieval_evaluation"]["retrieval_hit_at_k"]
        for record in in_domain_records
    )
    mean_reciprocal_rank = (
        sum(
            record["retrieval_evaluation"]["reciprocal_rank"]
            for record in in_domain_records
        )
        / len(in_domain_records)
        if in_domain_records
        else 0.0
    )

    print("\n" + "=" * 80)
    print("전체 비교 요약:")
    print("\n공통 Retrieval:")
    print(f"  in-domain Case 수: {len(in_domain_records)}")
    print(f"  Hit@{TOP_K} 통과 수: {hit_count}/{len(in_domain_records)}")
    print(f"  MRR: {mean_reciprocal_rank:.4f}")

    openai_summary = summarize_generation(records, "openai_evaluation")
    local_summary = summarize_generation(records, "local_evaluation")
    print_generation_summary("OpenAI", openai_summary)
    print_generation_summary(f"Local {LOCAL_MODEL_NAME}", local_summary)

    print(
        "\n현재 비교는 Source 표기의 존재와 유효성, 거절 여부 등 기본적인 "
        "자동 검증만 수행합니다."
    )
    print(
        "두 모델의 답변이 Context를 얼마나 정확히 반영했는지와 어느 답변이 "
        "더 좋은지는 별도의 수동 검토가 필요합니다."
    )
    print(
        "자동 faithfulness, answer correctness 및 LLM-as-a-Judge는 "
        "아직 사용하지 않습니다."
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
    embedding_model = SentenceTransformer(MODEL_NAME)
    pages, _ = load_pages(PDF_PATH)
    chunks = create_chunks(pages)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(embedding_model, chunk_texts, "passage")
    chunk_embeddings = np.ascontiguousarray(chunk_embeddings, dtype=np.float32)
    index = build_index(chunk_embeddings)

    records = []
    try:
        for case in EVAL_CASES:
            # Retrieval과 Context는 Case마다 한 번만 만들고 두 모델이 공유한다.
            results = retrieve(
                case["query"], embedding_model, index, chunks, TOP_K
            )
            context = build_context(results)

            openai_answer = generate_answer(
                case["query"], context, client, LLM_MODEL_NAME
            )
            local_prompt = build_local_rag_prompt(case["query"], context)
            local_answer = generate_local_answer(local_prompt)

            openai_evaluation = evaluate_answer(
                case, results, openai_answer
            )
            local_evaluation = evaluate_answer(case, results, local_answer)
            retrieval_evaluation = (
                {
                    "retrieval_hit_at_k": openai_evaluation[
                        "retrieval_hit_at_k"
                    ],
                    "reciprocal_rank": openai_evaluation["reciprocal_rank"],
                }
                if case["type"] == "in_domain"
                else None
            )

            records.append(
                {
                    "type": case["type"],
                    "retrieval_evaluation": retrieval_evaluation,
                    "openai_evaluation": openai_evaluation,
                    "local_evaluation": local_evaluation,
                }
            )
            print_case_result(
                case,
                results,
                openai_answer,
                openai_evaluation,
                local_answer,
                local_evaluation,
            )
    except LocalLLMError as error:
        print(f"오류: {error}")
        print("Ollama 서버가 실행 중인지 확인하세요.")
        print("ollama list로 qwen3:8b 모델이 설치됐는지 확인하세요.")
        raise SystemExit(1) from None

    print_summary(records)


if __name__ == "__main__":
    main()
