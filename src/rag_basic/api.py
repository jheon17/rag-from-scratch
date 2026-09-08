"""RAG 서비스화를 위한 최소 FastAPI 서버를 제공한다."""

from functools import lru_cache
from pathlib import Path
import tempfile

import psycopg
from fastapi import FastAPI, File, HTTPException, UploadFile
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer

from rag_basic.chunking import CHUNK_OVERLAP, CHUNK_SIZE
from rag_basic.embedding import MODEL_NAME
from rag_basic.pgvector_ingest import (
    count_document_rows,
    ingest_document,
)
from rag_basic.pgvector_retrieval import (
    create_query_embedding,
    get_database_config,
    search_pgvector,
)
from rag_basic.retrieval import build_context


MAX_UPLOAD_SIZE = 20 * 1024 * 1024
PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


class QueryRequest(BaseModel):
    """Retrieval API가 받을 문서명, 질문과 검색 결과 개수를 정의한다."""

    document_name: str = Field(min_length=1, max_length=255)
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResult(BaseModel):
    """pgvector에서 검색한 Chunk 한 개의 응답 구조다."""

    rank: int
    score: float
    cosine_distance: float
    chunk_id: int
    page_number: int
    text: str


class QueryResponse(BaseModel):
    """검색 문서, 질문, 검색 결과와 조합된 Context의 응답 구조다."""

    document_name: str
    query: str
    top_k: int
    results: list[RetrievalResult]
    context: str


class IngestResponse(BaseModel):
    """업로드한 PDF의 DB 적재 결과 구조다."""

    document_name: str
    chunk_count: int
    embedding_dimension: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


app = FastAPI(
    title="RAG from Scratch API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """FastAPI 프로세스가 요청에 응답할 수 있는지 확인한다."""
    return {"status": "ok"}


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """첫 Query 요청에서 모델을 로드하고 같은 프로세스에서 재사용한다."""
    return SentenceTransformer(MODEL_NAME)


def validate_document_name(document_name: str) -> str:
    """경로가 없는 PDF 파일명인지 확인한다."""
    if (
        Path(document_name).name != document_name
        or "\\" in document_name
        or Path(document_name).suffix.lower() != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="document_name은 경로가 없는 PDF 파일명이어야 합니다.",
        )
    return document_name


@app.post("/query", response_model=QueryResponse)
def query_retrieval(request: QueryRequest) -> QueryResponse:
    """질문을 Embedding하고 pgvector Top-K 검색 결과를 반환한다."""
    document_name = validate_document_name(request.document_name)

    try:
        model = get_embedding_model()
        query_embedding = create_query_embedding(model, request.query)
        database_config = get_database_config()

        with psycopg.connect(**database_config) as conn:
            register_vector(conn)
            if count_document_rows(conn, document_name) == 0:
                raise HTTPException(
                    status_code=404,
                    detail="요청한 문서를 찾을 수 없습니다.",
                )
            results = search_pgvector(
                conn,
                query_embedding,
                top_k=request.top_k,
                document_name=document_name,
            )

        return QueryResponse(
            document_name=document_name,
            query=request.query,
            top_k=request.top_k,
            results=results,
            context=build_context(results),
        )
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError, psycopg.Error):
        raise HTTPException(
            status_code=503,
            detail="Retrieval 서비스를 사용할 수 없습니다.",
        ) from None


@app.post("/ingest", response_model=IngestResponse)
def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """업로드한 PDF를 임시 파일에서 처리해 pgvector에 적재한다."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 필요합니다.")

    document_name = Path(file.filename).name
    if not document_name or Path(document_name).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=415,
            detail="PDF 파일만 업로드할 수 있습니다.",
        )
    if file.content_type not in PDF_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="PDF content type만 지원합니다.",
        )

    temporary_path: Path | None = None
    try:
        contents = file.file.read(MAX_UPLOAD_SIZE + 1)
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="파일 크기는 20 MiB 이하여야 합니다.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)

        database_config = get_database_config()
        with psycopg.connect(**database_config) as conn:
            register_vector(conn)
            if count_document_rows(conn, document_name) > 0:
                raise HTTPException(
                    status_code=409,
                    detail="동일한 이름의 문서가 이미 적재되어 있습니다.",
                )

            model = get_embedding_model()
            ingest_result = ingest_document(
                conn,
                model,
                temporary_path,
                document_name,
            )

        embeddings = ingest_result["embeddings"]
        return IngestResponse(
            document_name=document_name,
            chunk_count=len(ingest_result["chunks"]),
            embedding_dimension=int(embeddings.shape[1]),
            embedding_model=MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    except HTTPException:
        raise
    except PdfReadError:
        raise HTTPException(
            status_code=400,
            detail="PDF 파일을 읽을 수 없습니다.",
        ) from None
    except (OSError, ValueError, RuntimeError, psycopg.Error):
        raise HTTPException(
            status_code=503,
            detail="문서 적재 서비스를 사용할 수 없습니다.",
        ) from None
    finally:
        file.file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
