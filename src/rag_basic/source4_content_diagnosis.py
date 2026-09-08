"""Source 4의 본문과 인접 Chunk 구조를 read-only로 관찰한다."""

import psycopg

from rag_basic.api_evaluation import APIConnectionError, call_query_api
from rag_basic.evaluation import EVAL_CASES
from rag_basic.pgvector_retrieval import get_database_config


CASE_NAME = "ai_assignment_submission"
SOURCE_RANKS = (1, 2, 4)
QUERY_TERMS = ("과제", "제출", "그대로", "생성형 AI")
DIFFERENCE_PREVIEW_RADIUS = 40
QUERY_WINDOW_RADIUS = 80
BOUNDARY_LENGTH = 200


def get_target_case() -> dict:
    """기존 평가 Case에서 진단 대상을 이름으로 찾는다."""
    return next(case for case in EVAL_CASES if case["name"] == CASE_NAME)


def find_first_difference(left: str, right: str) -> int | None:
    """두 문자열이 처음 달라지는 위치를 반환한다."""
    for index, (left_character, right_character) in enumerate(zip(left, right)):
        if left_character != right_character:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def print_text_difference(left: str, right: str) -> None:
    """두 text가 다르면 첫 차이 위치와 주변 일부를 출력한다."""
    difference_index = find_first_difference(left, right)
    if difference_index is None:
        return
    start = max(0, difference_index - DIFFERENCE_PREVIEW_RADIUS)
    end = difference_index + DIFFERENCE_PREVIEW_RADIUS
    print(f"first difference index: {difference_index}")
    print(f"HTTP preview: {left[start:end]!r}")
    print(f"DB preview: {right[start:end]!r}")


def select_sources(results: list[dict]) -> dict[int, dict]:
    """실제 HTTP Retrieval 결과에서 Source 1, 2, 4를 선택한다."""
    results_by_rank = {result["rank"]: result for result in results}
    missing_ranks = [rank for rank in SOURCE_RANKS if rank not in results_by_rank]
    if missing_ranks:
        raise ValueError(f"Retrieval 결과에 필요한 rank가 없습니다: {missing_ranks}")
    return {rank: results_by_rank[rank] for rank in SOURCE_RANKS}


def find_configuration(
    conn: psycopg.Connection,
    document_name: str,
    source_1: dict,
) -> dict:
    """Source 1의 HTTP text와 정확히 일치하는 DB configuration을 찾는다."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT DISTINCT embedding_model, chunk_size, chunk_overlap
            FROM rag_chunks
            WHERE document_name = %s
              AND chunk_id = %s
              AND content = %s;
            """,
            (document_name, source_1["chunk_id"], source_1["text"]),
        )
        rows = cursor.fetchall()

    if not rows:
        raise RuntimeError("Source 1과 정확히 일치하는 DB row가 없습니다.")
    if len(rows) != 1:
        raise RuntimeError(
            "Source 1과 일치하는 DB configuration이 여러 개라 결정할 수 없습니다: "
            f"{len(rows)}개"
        )

    embedding_model, chunk_size, chunk_overlap = rows[0]
    return {
        "embedding_model": str(embedding_model),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
    }


def fetch_chunks(
    conn: psycopg.Connection,
    document_name: str,
    configuration: dict,
    chunk_ids: list[int],
) -> dict[int, dict]:
    """동일 configuration에서 지정한 Chunk를 read-only로 조회한다."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT chunk_id, page_number, content
            FROM rag_chunks
            WHERE document_name = %s
              AND embedding_model = %s
              AND chunk_size = %s
              AND chunk_overlap = %s
              AND chunk_id = ANY(%s)
            ORDER BY chunk_id;
            """,
            (
                document_name,
                configuration["embedding_model"],
                configuration["chunk_size"],
                configuration["chunk_overlap"],
                chunk_ids,
            ),
        )
        rows = cursor.fetchall()

    return {
        int(chunk_id): {
            "chunk_id": int(chunk_id),
            "page_number": int(page_number),
            "content": content,
        }
        for chunk_id, page_number, content in rows
    }


def longest_suffix_prefix_overlap(left: str, right: str) -> int:
    """left suffix와 right prefix가 같은 가장 긴 길이를 구한다."""
    maximum_length = min(len(left), len(right))
    for overlap_length in range(maximum_length, 0, -1):
        if left[-overlap_length:] == right[:overlap_length]:
            return overlap_length
    return 0


def configured_overlap_matches(left: dict, right: dict, overlap: int) -> bool:
    """설정된 길이만큼 두 Chunk가 정확히 겹치는지 확인한다."""
    if overlap == 0:
        return True
    if len(left["content"]) < overlap or len(right["content"]) < overlap:
        return False
    return left["content"][-overlap:] == right["content"][:overlap]


def print_source(
    source_number: int,
    source: dict,
    expected_chunk_ids: list[int],
) -> None:
    """HTTP Source의 metadata와 전체 text를 출력한다."""
    print("\n" + "=" * 88)
    print(f"Source {source_number}")
    print(f"chunk_id: {source['chunk_id']}")
    print(f"page_number: {source['page_number']}")
    print(f"gold: {source['chunk_id'] in expected_chunk_ids}")
    print(f"text length: {len(source['text'])}")
    print("전체 text:")
    print(source["text"])


def print_db_chunk(chunk: dict) -> None:
    """DB Chunk의 metadata와 전체 content를 구분선 안에 출력한다."""
    chunk_id = chunk["chunk_id"]
    print(f"\n===== CHUNK {chunk_id} START =====")
    print(f"page_number: {chunk['page_number']}")
    print(f"text length: {len(chunk['content'])}")
    print(chunk["content"])
    print(f"===== CHUNK {chunk_id} END =====")


def print_overlap(left: dict, right: dict, overlap: int) -> dict:
    """인접 Chunk pair의 configured 및 longest exact overlap을 출력한다."""
    exact_match = configured_overlap_matches(left, right, overlap)
    longest_overlap = longest_suffix_prefix_overlap(
        left["content"], right["content"]
    )
    print("\n" + "-" * 88)
    print(f"{left['chunk_id']} -> {right['chunk_id']}")
    print(f"same page: {left['page_number'] == right['page_number']}")
    print(f"configured overlap: {overlap}")
    print(f"left 마지막 {overlap}자: {left['content'][-overlap:]!r}")
    print(f"right 처음 {overlap}자: {right['content'][:overlap]!r}")
    print(f"configured overlap exact match: {exact_match}")
    print(f"longest exact overlap: {longest_overlap}")
    return {
        "exact_match": exact_match,
        "longest_overlap": longest_overlap,
        "same_page": left["page_number"] == right["page_number"],
    }


def find_occurrences(text: str, term: str) -> list[int]:
    """text 안에서 term이 등장하는 모든 시작 index를 구한다."""
    indices = []
    start = 0
    while True:
        index = text.find(term, start)
        if index == -1:
            return indices
        indices.append(index)
        start = index + 1


def print_term_positions(label: str, text: str) -> dict[str, list[int]]:
    """질문 관련 문자열의 모든 위치를 출력한다."""
    positions = {term: find_occurrences(text, term) for term in QUERY_TERMS}
    print(f"\n{label} query-term 위치(단순 literal string 확인):")
    for term, indices in positions.items():
        print(f"'{term}': {indices}")
    return positions


def merge_windows(positions: dict[str, list[int]], text_length: int) -> list[tuple[int, int]]:
    """서로 겹치는 query-term 주변 구간을 합친다."""
    windows = []
    for term, indices in positions.items():
        for index in indices:
            windows.append(
                (
                    max(0, index - QUERY_WINDOW_RADIUS),
                    min(text_length, index + len(term) + QUERY_WINDOW_RADIUS),
                )
            )
    merged = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def print_source4_windows(text: str, positions: dict[str, list[int]]) -> None:
    """Source 4의 query-term 주변 문맥을 출력한다."""
    windows = merge_windows(positions, len(text))
    if not windows:
        print("\nSource 4에서 query-term window를 만들 수 없습니다.")
        return
    for start, end in windows:
        print("\n===== SOURCE 4 QUERY-TERM WINDOW =====")
        print(f"range: {start}:{end}")
        print(text[start:end])
        print("===== END =====")


def print_boundaries(source_1: dict, source_4: dict, next_chunk: dict | None) -> None:
    """사람이 문장 경계를 관찰할 수 있도록 네 구간을 출력한다."""
    print("\n===== SOURCE 1 LAST 200 START =====")
    print(source_1["content"][-BOUNDARY_LENGTH:])
    print("===== SOURCE 1 LAST 200 END =====")
    print("\n===== SOURCE 4 FIRST 200 START =====")
    print(source_4["content"][:BOUNDARY_LENGTH])
    print("===== SOURCE 4 FIRST 200 END =====")
    print("\n===== SOURCE 4 LAST 200 START =====")
    print(source_4["content"][-BOUNDARY_LENGTH:])
    print("===== SOURCE 4 LAST 200 END =====")
    print("\n===== NEXT CHUNK FIRST 200 START =====")
    if next_chunk is None:
        print("Source 4의 다음 Chunk가 없습니다.")
    else:
        print(next_chunk["content"][:BOUNDARY_LENGTH])
    print("===== NEXT CHUNK FIRST 200 END =====")


def main() -> None:
    """HTTP Source와 DB Chunk의 content 및 경계 구조를 관찰한다."""
    case = get_target_case()
    try:
        status, body = call_query_api(case)
    except APIConnectionError as error:
        print(f"오류: {error}")
        print("FastAPI 서버가 127.0.0.1:8000에서 실행 중인지 확인하세요.")
        raise SystemExit(1) from None

    print(f"HTTP status: {status}")
    if status != 200:
        print(f"Response detail: {body.get('detail', '확인할 수 없음')}")
        raise SystemExit(1)

    print(f"Query: {body['query']}")
    print(f"document_name: {body['document_name']}")
    print("실제 Retrieval Top-5:")
    for result in body["results"]:
        is_gold = result["chunk_id"] in case["expected_chunk_ids"]
        print(
            f"rank={result['rank']}, chunk_id={result['chunk_id']}, "
            f"page_number={result['page_number']}, gold={is_gold}"
        )
    print("실제 HTTP answer:")
    print(body["answer"])

    try:
        sources = select_sources(body["results"])
        for source_number, source in sources.items():
            print_source(source_number, source, case["expected_chunk_ids"])

        database_config = get_database_config()
        with psycopg.connect(**database_config) as conn:
            configuration = find_configuration(
                conn, body["document_name"], sources[1]
            )
            source_4_chunk_id = sources[4]["chunk_id"]
            adjacent_ids = list(
                range(source_4_chunk_id - 2, source_4_chunk_id + 2)
            )
            requested_ids = sorted(
                set(adjacent_ids)
                | {source["chunk_id"] for source in sources.values()}
            )
            db_chunks = fetch_chunks(
                conn,
                body["document_name"],
                configuration,
                requested_ids,
            )
    except (OSError, ValueError, RuntimeError, psycopg.Error) as error:
        print(f"진단 실패: {error}")
        raise SystemExit(1) from None

    print("\n선택된 DB configuration:")
    print(f"embedding_model: {configuration['embedding_model']}")
    print(f"chunk_size: {configuration['chunk_size']}")
    print(f"chunk_overlap: {configuration['chunk_overlap']}")

    print("\nHTTP / DB text exact match:")
    for source_number, source in sources.items():
        db_chunk = db_chunks.get(source["chunk_id"])
        if db_chunk is None:
            print(f"Source {source_number}: 대응하는 DB Chunk가 없습니다.")
            continue
        matches = source["text"] == db_chunk["content"]
        print(f"Source {source_number} HTTP text == DB content: {matches}")
        if not matches:
            print_text_difference(source["text"], db_chunk["content"])

    print("\n인접 Chunk 전체 정보:")
    for chunk_id in adjacent_ids:
        chunk = db_chunks.get(chunk_id)
        if chunk is None:
            print(f"\nchunk_id={chunk_id}: 존재하지 않음")
        else:
            print_db_chunk(chunk)

    overlap = configuration["chunk_overlap"]
    pair_results = {}
    print("\n인접 Chunk overlap 검사:")
    for left_id, right_id in zip(adjacent_ids, adjacent_ids[1:]):
        left = db_chunks.get(left_id)
        right = db_chunks.get(right_id)
        if left is None or right is None:
            print(f"\n{left_id} -> {right_id}: Chunk가 없어 검사하지 못함")
            continue
        pair_results[(left_id, right_id)] = print_overlap(left, right, overlap)

    source_1_db = db_chunks.get(sources[1]["chunk_id"])
    source_4_db = db_chunks.get(sources[4]["chunk_id"])
    if source_1_db is None or source_4_db is None:
        print("Source 1 또는 Source 4 DB Chunk를 찾지 못했습니다.")
        raise SystemExit(1)

    source_pair = (source_1_db["chunk_id"], source_4_db["chunk_id"])
    source_pair_result = pair_results.get(source_pair)
    print("\nSource 1 -> Source 4 핵심 overlap:")
    print(f"Source 1 tail: {source_1_db['content'][-overlap:]!r}")
    print(f"Source 4 prefix: {source_4_db['content'][:overlap]!r}")
    if source_pair_result is None:
        print("configured overlap exact match: 확인할 수 없음")
        print("longest exact overlap: 확인할 수 없음")
        source_4_new_text = None
    else:
        print(
            "configured overlap exact match: "
            f"{source_pair_result['exact_match']}"
        )
        print(f"longest exact overlap: {source_pair_result['longest_overlap']}")
        if source_pair_result["exact_match"]:
            print("===== SOURCE 1 / SOURCE 4 OVERLAP START =====")
            print(source_4_db["content"][:overlap])
            print("===== SOURCE 1 / SOURCE 4 OVERLAP END =====")
            source_4_new_text = source_4_db["content"][overlap:]
            print("\n===== SOURCE 4 NEW PORTION START =====")
            print(source_4_new_text)
            print("===== SOURCE 4 NEW PORTION END =====")
            print(f"new portion length: {len(source_4_new_text)}")
        else:
            source_4_new_text = None
            print("Source 4 new portion을 configured overlap 기준으로 확정할 수 없음")

    source_positions = {}
    for source_number, source in sources.items():
        source_positions[source_number] = print_term_positions(
            f"Source {source_number}", source["text"]
        )
    print_source4_windows(sources[4]["text"], source_positions[4])

    if source_4_new_text is not None:
        print_term_positions("Source 4 new portion", source_4_new_text)
    else:
        print("\nSource 4 new portion의 query-term 위치를 확인할 수 없습니다.")

    next_chunk_id = source_4_db["chunk_id"] + 1
    next_chunk = db_chunks.get(next_chunk_id)
    print_boundaries(source_1_db, source_4_db, next_chunk)

    next_pair_result = pair_results.get((source_4_db["chunk_id"], next_chunk_id))
    if (
        next_chunk is not None
        and source_pair_result is not None
        and next_pair_result is not None
        and source_pair_result["exact_match"]
        and next_pair_result["exact_match"]
        and source_pair_result["same_page"]
        and next_pair_result["same_page"]
    ):
        reconstructed = (
            source_1_db["content"]
            + source_4_db["content"][overlap:]
            + next_chunk["content"][overlap:]
        )
        print(
            f"\n===== RECONSTRUCTED {source_1_db['chunk_id']}-"
            f"{next_chunk_id} TEXT START ====="
        )
        print(reconstructed)
        print(
            f"===== RECONSTRUCTED {source_1_db['chunk_id']}-"
            f"{next_chunk_id} TEXT END ====="
        )
    else:
        print("\nRECONSTRUCTED text 생략: 같은 페이지의 exact overlap 조건 불충족")

    print("\n관찰 Summary:")
    for source_number, source in sources.items():
        print(f"Source {source_number} page_number: {source['page_number']}")
    for source_number, source in sources.items():
        db_chunk = db_chunks.get(source["chunk_id"])
        matches = db_chunk is not None and source["text"] == db_chunk["content"]
        print(f"Source {source_number} HTTP/DB exact match: {matches}")
    if source_pair_result is not None:
        print(
            f"{source_1_db['chunk_id']} -> {source_4_db['chunk_id']} "
            f"overlap exact match: {source_pair_result['exact_match']}"
        )
    print(
        "Source 4 new portion length: "
        f"{len(source_4_new_text) if source_4_new_text is not None else '확인 불가'}"
    )
    print(f"Source 4 query-term 위치: {source_positions[4]}")
    if source_4_new_text is not None:
        new_positions = {
            term: find_occurrences(source_4_new_text, term)
            for term in QUERY_TERMS
        }
        print(f"Source 4 new portion query-term 위치: {new_positions}")
    if next_pair_result is not None:
        print(
            f"{source_4_db['chunk_id']} -> {next_chunk_id} "
            f"overlap exact match: {next_pair_result['exact_match']}"
        )


if __name__ == "__main__":
    main()
