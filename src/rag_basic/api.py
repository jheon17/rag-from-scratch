"""RAG 서비스화를 위한 최소 FastAPI 서버를 제공한다."""

from functools import lru_cache

import psycopg
from fastapi import FastAPI, HTTPException
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from rag_basic.embedding import MODEL_NAME
from rag_basic.pgvector_retrieval import (
    count_baseline_rows,
    create_query_embedding,
    get_database_config,
    search_pgvector,
)
from rag_basic.retrieval import build_context


class QueryRequest(BaseModel):
    """Retrieval API가 받을 질문과 검색 결과 개수를 정의한다."""

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
    """질문, 검색 결과와 조합된 Context의 응답 구조다."""

    query: str
    top_k: int
    results: list[RetrievalResult]
    context: str


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


@app.post("/query", response_model=QueryResponse)
def query_retrieval(request: QueryRequest) -> QueryResponse:
    """질문을 Embedding하고 pgvector Top-K 검색 결과를 반환한다."""
    try:
        model = get_embedding_model()
        query_embedding = create_query_embedding(model, request.query)
        database_config = get_database_config()

        with psycopg.connect(**database_config) as conn:
            register_vector(conn)
            count_baseline_rows(conn)
            results = search_pgvector(
                conn,
                query_embedding,
                top_k=request.top_k,
            )

        return QueryResponse(
            query=request.query,
            top_k=request.top_k,
            results=results,
            context=build_context(results),
        )
    except (OSError, ValueError, RuntimeError, psycopg.Error):
        raise HTTPException(
            status_code=503,
            detail="Retrieval 서비스를 사용할 수 없습니다.",
        ) from None
