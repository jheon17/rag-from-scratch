"""기존 Chunk와 Embedding을 PostgreSQL pgvector Table에 적재한다."""

import os
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_PATH,
    create_chunks,
    load_pages,
)
from rag_basic.embedding import MODEL_NAME, embed_texts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "sql" / "001_create_rag_chunks.sql"
EMBEDDING_DIMENSION = 384
PREVIEW_LENGTH = 120
REQUIRED_ENV_VARS = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")

UPSERT_SQL = """
INSERT INTO rag_chunks (
    document_name,
    chunk_id,
    page_number,
    content,
    embedding,
    embedding_model,
    chunk_size,
    chunk_overlap
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (
    document_name,
    chunk_id,
    embedding_model,
    chunk_size,
    chunk_overlap
)
DO UPDATE SET
    page_number = EXCLUDED.page_number,
    content = EXCLUDED.content,
    embedding = EXCLUDED.embedding;
"""


def get_database_config() -> dict:
    """환경변수에서 DB 연결 설정을 읽고 필수 항목을 검증한다."""
    missing_vars = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing_vars:
        raise ValueError(
            "필수 DB 환경변수가 설정되지 않았습니다: "
            + ", ".join(missing_vars)
        )

    return {
        "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
    }


def check_vector_extension(conn: psycopg.Connection) -> str:
    """vector extension 버전을 확인하고 없으면 적재를 중단한다."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT extversion
            FROM pg_extension
            WHERE extname = 'vector';
            """
        )
        row = cursor.fetchone()

    if row is None:
        raise RuntimeError(
            "PostgreSQL에 vector extension이 활성화되어 있지 않습니다."
        )

    return str(row[0])


def apply_schema(conn: psycopg.Connection) -> None:
    """버전 관리되는 SQL 파일을 실행하고 Table 생성을 확인한다."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cursor:
        cursor.execute(schema_sql)
        cursor.execute("SELECT to_regclass('public.rag_chunks');")
        table_name = cursor.fetchone()[0]

    if table_name != "rag_chunks":
        raise RuntimeError("rag_chunks Table 생성을 확인하지 못했습니다.")


def validate_embeddings(chunks: list[dict], embeddings: np.ndarray) -> None:
    """Chunk 수와 Embedding shape가 DB schema와 일치하는지 확인한다."""
    if embeddings.ndim != 2:
        raise ValueError(f"Embedding 배열이 2차원이 아닙니다: {embeddings.shape}")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "Chunk 수와 Embedding 행 수가 일치하지 않습니다: "
            f"{len(chunks)} != {embeddings.shape[0]}"
        )
    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            "Embedding 차원이 rag_chunks.embedding VECTOR(384)와 "
            f"일치하지 않습니다: {embeddings.shape[1]}"
        )


def build_rows(
    document_name: str,
    chunks: list[dict],
    embeddings: np.ndarray,
) -> list[tuple]:
    """Chunk metadata와 대응하는 Vector를 DB INSERT 행으로 구성한다."""
    return [
        (
            document_name,
            int(chunk["chunk_id"]),
            int(chunk["page_number"]),
            chunk["text"],
            embedding.astype(np.float32),
            MODEL_NAME,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]


def ingest_rows(conn: psycopg.Connection, rows: list[tuple]) -> None:
    """하나의 Connection과 transaction에서 전체 행을 batch Upsert한다."""
    with conn.cursor() as cursor:
        cursor.executemany(UPSERT_SQL, rows)


def dataset_parameters(document_name: str) -> tuple:
    """문서와 Embedding 설정으로 dataset을 식별하는 파라미터를 반환한다."""
    return (document_name, MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP)


def count_document_rows(
    conn: psycopg.Connection, document_name: str
) -> int:
    """동일 문서명과 현재 Embedding 설정으로 저장된 행 수를 반환한다."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM rag_chunks
            WHERE document_name = %s
              AND embedding_model = %s
              AND chunk_size = %s
              AND chunk_overlap = %s;
            """,
            dataset_parameters(document_name),
        )
        return int(cursor.fetchone()[0])


def validate_stored_data(
    conn: psycopg.Connection,
    document_name: str,
    chunks: list[dict],
) -> dict:
    """지정한 dataset의 행 수, NULL, 차원과 Chunk ID 범위를 검증한다."""
    filters = """
        document_name = %s
        AND embedding_model = %s
        AND chunk_size = %s
        AND chunk_overlap = %s
    """
    parameters = dataset_parameters(document_name)

    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (
                    WHERE chunk_id IS NULL
                       OR page_number IS NULL
                       OR content IS NULL
                       OR embedding IS NULL
                )
            FROM rag_chunks
            WHERE {filters};
            """,
            parameters,
        )
        row_count, null_count = cursor.fetchone()

        cursor.execute(
            f"""
            SELECT MIN(vector_dims(embedding)), MAX(vector_dims(embedding))
            FROM rag_chunks
            WHERE {filters};
            """,
            parameters,
        )
        min_dimension, max_dimension = cursor.fetchone()

        cursor.execute(
            f"""
            SELECT MIN(chunk_id), MAX(chunk_id)
            FROM rag_chunks
            WHERE {filters};
            """,
            parameters,
        )
        min_chunk_id, max_chunk_id = cursor.fetchone()

    expected_count = len(chunks)
    expected_min_chunk_id = min(chunk["chunk_id"] for chunk in chunks)
    expected_max_chunk_id = max(chunk["chunk_id"] for chunk in chunks)

    validation = {
        "row_count": int(row_count),
        "row_count_matches": row_count == expected_count,
        "null_count": int(null_count),
        "has_no_null": null_count == 0,
        "min_dimension": min_dimension,
        "max_dimension": max_dimension,
        "dimension_matches": (
            min_dimension == EMBEDDING_DIMENSION
            and max_dimension == EMBEDDING_DIMENSION
        ),
        "min_chunk_id": min_chunk_id,
        "max_chunk_id": max_chunk_id,
        "chunk_id_range_matches": (
            min_chunk_id == expected_min_chunk_id
            and max_chunk_id == expected_max_chunk_id
        ),
    }

    if not all(
        (
            validation["row_count_matches"],
            validation["has_no_null"],
            validation["dimension_matches"],
            validation["chunk_id_range_matches"],
        )
    ):
        raise RuntimeError(f"DB 적재 검증에 실패했습니다: {validation}")

    return validation


def fetch_sample_rows(
    conn: psycopg.Connection, document_name: str
) -> list[tuple]:
    """지정한 dataset의 첫 3개 행에서 Vector를 제외한 정보를 조회한다."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT chunk_id, page_number, content, vector_dims(embedding)
            FROM rag_chunks
            WHERE document_name = %s
              AND embedding_model = %s
              AND chunk_size = %s
              AND chunk_overlap = %s
            ORDER BY chunk_id
            LIMIT 3;
            """,
            dataset_parameters(document_name),
        )
        return cursor.fetchall()


def ingest_document(
    conn: psycopg.Connection,
    model: SentenceTransformer,
    pdf_path: Path,
    document_name: str,
) -> dict:
    """PDF를 Chunk와 Embedding으로 변환해 지정한 문서명으로 적재한다."""
    pages, _ = load_pages(pdf_path)
    chunks = create_chunks(pages)
    if not chunks:
        raise ValueError("PDF에서 적재할 텍스트 Chunk를 만들지 못했습니다.")

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(model, chunk_texts, "passage")
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    validate_embeddings(chunks, embeddings)

    rows = build_rows(document_name, chunks, embeddings)
    ingest_rows(conn, rows)
    validation = validate_stored_data(conn, document_name, chunks)

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "upsert_count": len(rows),
        "validation": validation,
    }


def print_validation(validation: dict, sample_rows: list[tuple]) -> None:
    """적재 검증과 Vector를 제외한 sample 행을 출력한다."""
    print("DB 적재 검증:")
    print(f"  Row count: {validation['row_count']}")
    print(f"  생성된 Chunk 수와 일치: {validation['row_count_matches']}")
    print(f"  NULL 행 수: {validation['null_count']}")
    print(f"  필수 데이터에 NULL이 없는가: {validation['has_no_null']}")
    print(
        "  Embedding dimension 최소/최대: "
        f"{validation['min_dimension']}/{validation['max_dimension']}"
    )
    print(f"  모든 dimension이 384인가: {validation['dimension_matches']}")
    print(
        "  Chunk ID 최소/최대: "
        f"{validation['min_chunk_id']}/{validation['max_chunk_id']}"
    )
    print(f"  Chunk ID 범위가 일치하는가: {validation['chunk_id_range_matches']}")

    print("\nSample rows:")
    for chunk_id, page_number, content, vector_dimension in sample_rows:
        preview = content[:PREVIEW_LENGTH]
        print(
            f"  chunk_id={chunk_id}, page_number={page_number}, "
            f"vector_dimension={vector_dimension}"
        )
        print(f"    content={preview}...")


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF 파일을 찾지 못했습니다: {PDF_PATH}")
        return
    if not SCHEMA_PATH.exists():
        print(f"SQL schema 파일을 찾지 못했습니다: {SCHEMA_PATH}")
        return

    try:
        database_config = get_database_config()

        model = SentenceTransformer(MODEL_NAME)

        with psycopg.connect(**database_config) as conn:
            pgvector_version = check_vector_extension(conn)
            print(f"pgvector version: {pgvector_version}")
            register_vector(conn)

            apply_schema(conn)
            print("rag_chunks Table 생성 확인: True")

            ingest_result = ingest_document(
                conn,
                model,
                PDF_PATH,
                PDF_PATH.name,
            )
            chunks = ingest_result["chunks"]
            embeddings = ingest_result["embeddings"]
            validation = ingest_result["validation"]
            print(f"생성된 Chunk 수: {len(chunks)}")
            print(f"Embedding shape: {embeddings.shape}")
            print(f"Upsert 처리 행 수: {ingest_result['upsert_count']}")

            sample_rows = fetch_sample_rows(conn, PDF_PATH.name)
            print_validation(validation, sample_rows)
    except (OSError, ValueError, RuntimeError, psycopg.Error) as error:
        print(f"적재 실패: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
