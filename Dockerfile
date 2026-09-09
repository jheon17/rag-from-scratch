FROM python:3.12-slim AS builder

ARG UV_VERSION=0.12.9

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync \
    --frozen \
    --no-dev \
    --no-editable \
    --no-cache

RUN set -eu; \
    /app/.venv/bin/python -c \
        'import torch; print("Builder original Torch:", torch.__version__); print("Builder original CUDA:", torch.version.cuda)'; \
    TORCH_BASE_VERSION="$(/app/.venv/bin/python -c 'import torch; print(torch.__version__.split("+", 1)[0])')"; \
    echo "Torch base version: ${TORCH_BASE_VERSION}"; \
    uv pip install \
        --python /app/.venv/bin/python \
        --reinstall \
        --no-deps \
        --no-cache \
        "torch==${TORCH_BASE_VERSION}+cpu" \
        --index-url https://download.pytorch.org/whl/cpu; \
    GPU_PACKAGES="$(/app/.venv/bin/python -c \
        'from importlib.metadata import distributions; print(" ".join(sorted(dist.metadata.get("Name", "") for dist in distributions() if dist.metadata.get("Name", "").lower().startswith("nvidia-") or dist.metadata.get("Name", "").lower() == "triton")))')"; \
    echo "GPU-only packages to remove: ${GPU_PACKAGES}"; \
    if [ -n "${GPU_PACKAGES}" ]; then \
        uv pip uninstall --python /app/.venv/bin/python ${GPU_PACKAGES}; \
    fi; \
    uv pip check --python /app/.venv/bin/python; \
    /app/.venv/bin/python -c \
        'import torch; from importlib.metadata import distributions; names=[dist.metadata.get("Name", "") for dist in distributions()]; nvidia=sorted(name for name in names if name.lower().startswith("nvidia-")); print("Builder CPU Torch:", torch.__version__); print("Builder torch.version.cuda:", torch.version.cuda); print("Builder torch.cuda.is_available:", torch.cuda.is_available()); print("Builder nvidia packages:", nvidia); print("Builder triton installed:", any(name.lower() == "triton" for name in names)); assert torch.version.cuda is None; assert not torch.cuda.is_available(); assert not nvidia; assert not any(name.lower() == "triton" for name in names)'

RUN HF_HUB_OFFLINE=0 /app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; from rag_basic.embedding import MODEL_NAME; SentenceTransformer(MODEL_NAME)"

RUN HF_HUB_OFFLINE=1 /app/.venv/bin/python -c \
    "import fastapi, pgvector, psycopg, sentence_transformers, torch, transformers; from rag_basic.api import app; from rag_basic.embedding import MODEL_NAME; from sentence_transformers import SentenceTransformer; model=SentenceTransformer(MODEL_NAME, local_files_only=True); embedding=model.encode(['query: 테스트 질문']); print('Builder core imports success: True'); print('Builder FastAPI import success:', app is not None); print('Builder embedding device:', model.device); print('Builder embedding shape:', embedding.shape); print('Builder offline embedding encode success: True'); assert embedding.shape == (1, 384)"

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/huggingface /opt/huggingface

CMD ["/app/.venv/bin/uvicorn", "rag_basic.api:app", "--host", "0.0.0.0", "--port", "8000"]
