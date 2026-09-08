"""RAG 서비스화를 위한 최소 FastAPI 서버를 제공한다."""

from fastapi import FastAPI


app = FastAPI(
    title="RAG from Scratch API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """FastAPI 프로세스가 요청에 응답할 수 있는지 확인한다."""
    return {"status": "ok"}
