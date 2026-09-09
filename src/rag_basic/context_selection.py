"""LLM Context에 포함할 Retrieval 결과를 선택하는 순수 함수를 제공한다."""


# 현재 프로젝트의 regression evaluation에서 사용한 보수적인 기준이다.
# 여러 문서에 대한 최적 threshold라는 의미는 아니다.
NEAR_DUPLICATE_COVERAGE_THRESHOLD = 0.90


def longest_suffix_prefix_overlap(left: str, right: str) -> int:
    """left suffix와 right prefix가 같은 가장 긴 문자 수를 반환한다."""
    maximum_length = min(len(left), len(right))
    for overlap_length in range(maximum_length, 0, -1):
        if left[-overlap_length:] == right[:overlap_length]:
            return overlap_length
    return 0


def calculate_candidate_coverage(kept_text: str, candidate_text: str) -> float:
    """Candidate text가 이미 유지된 text에 덮이는 exact overlap 비율을 계산한다."""
    if not candidate_text:
        return 0.0
    if candidate_text in kept_text:
        return 1.0

    forward_overlap = longest_suffix_prefix_overlap(kept_text, candidate_text)
    reverse_overlap = longest_suffix_prefix_overlap(candidate_text, kept_text)
    overlap_length = max(forward_overlap, reverse_overlap)

    # 후순위 candidate 자체가 얼마나 덮였는지 보기 위해 candidate 길이로 나눈다.
    return overlap_length / len(candidate_text)


def select_context_results(
    results: list[dict],
    threshold: float = NEAR_DUPLICATE_COVERAGE_THRESHOLD,
) -> list[dict]:
    """상위 rank를 보존하며 near-duplicate인 후순위 결과를 제외한 새 목록을 반환한다."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold는 0.0 이상 1.0 이하여야 합니다.")

    selected_results: list[dict] = []
    for candidate in sorted(results, key=lambda result: result["rank"]):
        is_near_duplicate = any(
            calculate_candidate_coverage(kept["text"], candidate["text"])
            >= threshold
            for kept in selected_results
        )
        if not is_near_duplicate:
            selected_results.append(candidate)

    return selected_results
