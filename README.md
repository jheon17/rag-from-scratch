# RAG from Scratch

> LangChain 없이 구현한 pgvector 기반 Local RAG API

고수준 RAG 프레임워크에 의존하지 않고 PDF 로딩, Chunking, Embedding, Vector Retrieval,
Context 구성, Local LLM Generation을 직접 구현했습니다. FAISS exact-search baseline에서
PostgreSQL + pgvector 기반 Multi-document 검색으로 확장하고 FastAPI와 Docker로 API를 구성했습니다.

## Key Results

| 항목 | 검증 결과 |
| --- | --- |
| Retrieval | 수동 구성 in-domain 6 Case에서 `Hit@5 = 6/6`, `MRR = 0.8750` |
| Generation 구조 | in-domain Source citation 검증 `6/6`, out-of-domain exact refusal `3/3` |
| FAISS → pgvector | 6개 평가 Case의 Top-5 결과 순서 동일 |
| Context 개선 | Raw Retrieval 지표를 유지하며 near-duplicate Generation Context를 제거했고, 현재 regression set에서 assignment refusal 개선 |
| Multi-document | 162개 + 275개 Chunk 공존, 두 PDF 검증에서 cross-document contamination 미관찰 |
| Docker image | 약 `10.19 GB → 2.58 GB`, `74.71%` 감소 |

위 수치는 현재 문서와 소규모 수동 평가셋의 결과이며 일반적인 RAG 정확도나 완전한 faithfulness를 의미하지 않습니다.

## Overview

목적은 완성된 프레임워크를 빠르게 조립하는 것이 아니라 검색과 답변 생성 사이의 데이터 흐름을 코드 수준에서 이해하는 것입니다. 이를 위해 LangChain이나 LlamaIndex 없이 작은 함수와 모듈을 연결했습니다.

평가도 Retrieval과 Generation으로 분리했습니다. 정답 근거 검색과 LLM의 근거 활용은 서로 다른 문제이기 때문입니다. 실제로 gold Chunk가 검색 1위에 있어도 Local LLM이 부적절하게 거절하는 Case를 발견해 Source 단위 Context ablation으로 추적했습니다.

최종 서비스는 PDF를 pgvector에 적재하고 문서별로 검색한 Context를 Host Ollama의 `qwen3:8b`에 전달합니다. FastAPI와 PostgreSQL은 Docker Compose, LLM inference는 Ubuntu Host의 NVIDIA GPU에서 실행합니다.

## Architecture

### Logical RAG Pipeline

다음 그림은 PDF가 답변으로 변환되는 데이터 흐름입니다.

```text
PDF Upload
    ↓
Page Text Extraction
    ↓
Character-based Chunking
    ↓
multilingual-e5-small Embedding
    ↓
PostgreSQL + pgvector
    ↓
Document-aware Top-K Retrieval
    ↓
Near-Duplicate Context Selection
    ↓
Grounded Prompt
    ↓
Ollama qwen3:8b
    ↓
Answer + Source Citation
```

### Deployment Architecture

다음 그림은 각 구성 요소가 실제로 실행되는 위치입니다.

```text
Client
  │ HTTP
  ▼
FastAPI Container
  ├──────── PostgreSQL Container
  │            └─ pgvector
  │
  └──────── Ubuntu Host Ollama
               └─ qwen3:8b
                  └─ NVIDIA GPU
```

FastAPI container는 CPU로 Embedding을 생성하고 GPU Generation은 Host Ollama로 분리합니다.

## Tech Stack

| 영역 | Current Path | Baseline / Experiment |
| --- | --- | --- |
| Language | Python 3.12 | — |
| PDF | pypdf | — |
| Embedding | sentence-transformers, `intfloat/multilingual-e5-small` | — |
| Vector Search | PostgreSQL 16 + pgvector 0.8.6 | FAISS `IndexFlatIP` |
| LLM | Ollama + `qwen3:8b` | OpenAI Responses API 초기 비교 실험 |
| API | FastAPI, Uvicorn, Pydantic | — |
| Evaluation | 직접 구현한 Hit@5, MRR, citation/refusal 검사 | CrossEncoder, ablation 실험 |
| Infrastructure | Docker Compose, CPU API image, Host GPU Ollama | — |
| Package Management | uv | — |

OpenAI는 초기 Generation 비교 실험에만 사용했으며 현재 FastAPI는 Ollama의 `qwen3:8b`를 호출합니다.

## Core Implementation

### 1. PDF → Chunk

`pypdf`로 페이지별 텍스트를 추출하고 Python 문자열로 Chunk를 생성합니다. `chunk_size=500`,
`chunk_overlap=100`은 비교 baseline이며 NIA 평가 문서에서는 162개 Chunk가 생성됐습니다.

### 2. E5 Embedding

`intfloat/multilingual-e5-small`로 텍스트를 384차원 Vector로 변환합니다. 질문에는 `query:`,
문서에는 `passage:` prefix를 적용하고 cosine similarity 검색을 위해 정규화합니다.

### 3. pgvector Retrieval

FAISS `IndexFlatIP`로 exact-search를 검증한 뒤 PostgreSQL + pgvector로 확장했습니다. 현재 검색은 `document_name`, Embedding 모델, Chunk 설정 metadata로 범위를 제한하고 Top-K Chunk를 반환합니다.

6개 평가 Case에서 FAISS와 pgvector의 Top-5 순서가 같았으며 현재 API는 pgvector를 사용합니다.

### 4. Context Selection

검색된 Top-K는 그대로 보존하고 LLM Context에서만 상위 결과와 exact overlap coverage가 `0.90` 이상인 후순위 near-duplicate를 제외합니다.

따라서 API의 `results`와 Generation `context`의 Source 수는 다를 수 있습니다. `0.90`은 현재 regression set의 기준이며 최적값이 아닙니다.

### 5. Local LLM Generation

질문과 Context를 grounding Prompt로 구성해 Ollama `/api/chat`에 전달합니다. Context 밖의 내용을 추측하지 않고, 근거가 부족하면 거절하며, 근거를 `[Source N]`으로 표시하도록 요구합니다.

`qwen3:8b`는 `temperature=0`, `seed=42`, `think=false`로 호출하지만 모든 환경에서 완전히 결정적인 출력을 보장하지는 않습니다.

### 6. Document-aware FastAPI

`/ingest`는 PDF를 Chunking·Embedding해 pgvector에 저장하고 `/query`는 요청한 `document_name` 범위에서 Retrieval과 Local LLM Generation을 수행합니다.

Embedding 모델은 프로세스에서 재사용하며 응답에는 원본 Retrieval 결과, 선택된 Context, 답변과 LLM 모델명이 포함됩니다.

## Evaluation

평가 데이터는 PDF Chunk 본문을 사람이 읽고 질문과 gold evidence를 수동 구성했습니다.
전체 9개 Case 중 in-domain은 6개, out-of-domain은 3개입니다.

### Retrieval

| 지표 | 결과 |
| --- | ---: |
| in-domain Case | 6 |
| Hit@5 | `6/6` |
| MRR | `0.8750` |

Hit@5는 gold Chunk 중 하나가 Top-5에 존재하는지 확인합니다. MRR은 첫 gold Chunk 순위 역수의 평균입니다. 모든 gold evidence가 Top-5에는 포함됐지만 항상 1위는 아니었습니다.

### Generation

| 검사 | 결과 |
| --- | ---: |
| in-domain 답변의 Source citation 검증 | `6/6` |
| out-of-domain exact refusal | `3/3` |

Citation 검사는 `[Source N]`이 실제 Generation Context 범위 안의 번호인지 확인하며 semantic correctness나 faithfulness를 자동 증명하지는 않습니다.

Out-of-domain Case에서는 일반 지식으로 답하지 않고 다음 문장과 정확히 일치하는지
검사했습니다.

```text
제공된 문서에서 확인할 수 없습니다.
```

### Experimental Improvements

| 실험 | 관찰 결과 | 현재 서비스 적용 |
| --- | --- | --- |
| Top-K | K=5부터 Hit `6/6`; K=10은 같은 MRR에서 평균 Context 길이 증가 | K=5 사용 |
| Similarity threshold | 실험값 0.88에서 gold `6/6`, OOD rejected `3/3` | 미적용 |
| CrossEncoder reranking | MRR `0.8750 → 1.0000` | 미적용 |
| Chunk configuration | 비교한 세 설정 중 500/100의 Page MRR이 `0.8750`으로 가장 높았음 | 500/100 사용 |

CrossEncoder와 threshold 결과는 현재 소규모 평가셋의 실험 결과이며 최종 서비스 검색 경로에는 적용하지 않았습니다.

## Failure Diagnosis & Context Deduplication

### 문제

과제 제출 질문에서 gold Chunk가 검색 1위였지만 Local LLM은 부적절한 refusal을 생성했습니다. Retrieval 성공이 Generation 성공을 보장하지 않은 사례입니다.

### 진단과 관찰

Source 하나만 사용하거나 gold Source 조합을 바꾸는 ablation에서, Chunk overlap으로 생성된 near-duplicate의 포함 여부와 refusal behavior 변화 사이의 연관을 관찰했습니다.

이는 causal proof가 아니라 현재 모델, Prompt, 문서와 질문에서 관찰한 동작 차이입니다.

### 개선

Retrieval 결과나 순위를 바꾸지 않고 Generation Context에만 선택 함수를 적용했습니다.

```text
Raw Retrieval
chunk_ids: [69, 68, 76, 70, 75]

Generation Context
Source: [1, 2, 3, 5]
```

Hit@5와 MRR은 원본 Top-K로 계속 평가합니다. 현재 regression set에서는 assignment의 undesired refusal이 관찰되지 않았고 citation도 Context Source 범위 안에 있었습니다.

## Multi-Document Verification

| 문서 | 저장된 Chunk |
| --- | ---: |
| `ai_ethics_guide.pdf` | 162 |
| `nist.ai.100-1.pdf` | 275 |

두 PDF를 같은 PostgreSQL에 저장하고 `document_name` 기준 Retrieval isolation을
검증했습니다.

```text
All retrieval results scoped to requested document: True
Cross-document contamination detected: False
```

NIST 문서 질문에서는 AI RMF Core의 네 기능인 GOVERN, MAP, MEASURE, MANAGE를
`[Source 1]`과 함께 답변했습니다. 이 결과는 두 실제 PDF 범위의 integration
verification이며 모든 PDF 형식을 지원한다는 의미는 아닙니다.

## Docker Deployment & Image Optimization

FastAPI와 PostgreSQL은 container로 실행하고, qwen3:8b Generation은 Ubuntu Host의
Ollama와 NVIDIA GPU가 담당합니다.

```text
FastAPI Container    → CPU Embedding
PostgreSQL Container → pgvector
Host Ollama          → qwen3:8b → NVIDIA GPU
```

초기 API image에는 CUDA PyTorch, `nvidia-*`, Triton이 포함됐지만 container에 GPU를
전달하지 않아 Embedding은 CPU에서 실행되고 있었습니다. Multi-stage build에서
CPU-only Torch로 교체하고 GPU 전용 package를 제거했습니다.

| 항목 | 크기 |
| --- | ---: |
| 초기 image | `10,187,622,776 bytes` (약 10.19 GB) |
| 최적화 image | `2,576,568,589 bytes` (약 2.58 GB) |
| 감소율 | `74.71%` |

최종 image는 오프라인 상태에서 `multilingual-e5-small`을 CPU로 불러와 `(1, 384)`
Embedding을 생성했습니다. Host `.venv`의 `torch 2.14.0+cu130`, CUDA 13.0 환경은
유지했습니다.

Container networking은 다음 주소를 사용합니다.

```text
PostgreSQL: postgres:5432
Host Ollama: host.docker.internal:11434
```

Linux Docker에서는 `host.docker.internal:host-gateway` 매핑을 사용합니다. Ollama를
`0.0.0.0:11434`에 bind한 것은 로컬 통합 검증을 위한 구성입니다. 외부 공개 환경에서는
방화벽과 접근 제어가 필요합니다.

## API

| Method | Endpoint | 역할 |
| --- | --- | --- |
| GET | `/health` | API liveness 확인 |
| POST | `/ingest` | PDF Chunking, Embedding 및 DB 적재 |
| POST | `/query` | Document-aware Retrieval과 Local LLM Generation |

확인한 HTTP 동작은 다음과 같습니다.

```text
health             → 200 OK
정상 query         → 200 OK
duplicate ingest   → 409 Conflict
invalid top_k      → 422 Unprocessable Entity
unknown document   → 404 Not Found
```

`/query` 응답의 주요 field는 `document_name`, `query`, `top_k`, `results`, `context`,
`answer`, `llm_model`입니다. `results`는 원본 Top-K이고 `context`는 near-duplicate
선택 이후 실제 LLM에 전달된 문자열입니다.

## How to Run

**Prerequisites:** Docker Engine과 Compose plugin, Ollama, `qwen3:8b`, Docker container에서 Host Ollama에 접근할 수 있는 로컬 환경

**1. Repository와 환경변수 준비**

```bash
git clone https://github.com/jheon17/rag-from-scratch.git
cd rag-from-scratch
cp .env.example .env
```

`.env`에는 다음 PostgreSQL 항목만 설정하며 실제 비밀번호를 README나 Git에 기록하지 않습니다.

```text
POSTGRES_DB=rag_db
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=<직접_설정할_비밀번호>
```

**2. Ollama 준비**

```bash
ollama pull qwen3:8b
ollama list
```

Dockerized FastAPI가 Host Ollama에 접근하려면 `OLLAMA_HOST=0.0.0.0:11434`를 설정하고 service를 재시작해야 합니다. 로컬 개발 구성으로만 사용하고 네트워크 노출 범위를 확인합니다. 상세 과정은 [Development Log](docs/development-log.md)를 참고합니다.

**3. 서비스 실행**

```bash
sudo docker compose up -d --build
sudo docker compose ps
curl http://127.0.0.1:8000/health
```

첫 build에는 CPU PyTorch wheel과 Embedding model 다운로드가 필요하며 Runtime에서는 preload된 모델을 오프라인으로 사용합니다.

**4. PDF 적재**

```bash
curl -X POST \
  -F "file=@/absolute/path/document.pdf;type=application/pdf" \
  http://127.0.0.1:8000/ingest
```

응답에는 문서명, Chunk 수, Embedding 정보와 Chunk 설정이 포함되며 같은 파일명이 이미 있으면 `409`를 반환합니다.

**5. 문서 기반 질문**

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "document_name": "document.pdf",
    "query": "이 문서의 핵심 내용을 설명해 주세요.",
    "top_k": 5
  }'
```

API 문서는 서비스 실행 중 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.

**6. 서비스 종료**

```bash
sudo docker compose stop api
sudo docker compose stop postgres
```

Named volume의 PostgreSQL 데이터는 container를 정지해도 유지됩니다.

## Project Structure

핵심 설정과 API 구현: [Dockerfile](Dockerfile), [compose.yaml](compose.yaml), [.env.example](.env.example), [api.py](src/rag_basic/api.py)

```text
.
├── Dockerfile                         # CPU API runtime image
├── compose.yaml                       # FastAPI + PostgreSQL 구성
├── docker/postgres/init.sql           # pgvector extension 초기화
├── sql/001_create_rag_chunks.sql      # Chunk table schema
├── pyproject.toml                     # Python dependency
├── docs/development-log.md            # 전체 구현·실험 기록
└── src/rag_basic/
    ├── api.py                         # /health, /ingest, /query
    ├── chunking.py                    # PDF 로딩과 Chunking
    ├── embedding.py                   # E5 Embedding
    ├── pgvector_ingest.py             # Chunk와 Vector 적재
    ├── pgvector_retrieval.py          # Document-aware 검색
    ├── context_selection.py           # Generation Context 선택
    ├── local_llm.py                   # Ollama HTTP client
    ├── local_rag.py                   # Grounding Prompt 구성
    └── api_evaluation.py              # RAG API baseline 평가
```

실험용 script의 세부 역할과 실행 결과는 Development Log에서 확인할 수 있습니다.

## Design Decisions

1. **LangChain 미사용:** PDF부터 Prompt까지 데이터 변화를 직접 확인했습니다.
2. **FAISS → pgvector:** exact-search 검산 후 영속 저장과 문서별 검색으로 확장했습니다.
3. **Retrieval/Context 분리:** 원본 지표를 유지하며 LLM 입력에서만 near-duplicate를 제외합니다.
4. **API CPU/LLM GPU 분리:** FastAPI는 CPU Embedding, Host Ollama는 GPU Generation을 담당합니다.
5. **Embedding 모델 유지:** Docker 최적화 후에도 같은 E5 모델과 384차원 Vector를 사용합니다.

## Limitations

- Retrieval 평가는 사람이 구성한 in-domain 6개 Case에 기반합니다.
- Multi-document integration은 두 실제 PDF 범위에서만 검증했습니다.
- Near-duplicate coverage threshold `0.90`은 현재 regression set의 기준이며 최적값이
  아닙니다.
- Citation 번호의 존재와 유효성은 answer correctness나 faithfulness의 완전한 자동
  평가가 아닙니다.
- qwen3:8b는 동일한 재현성 설정에서도 실행환경에 따라 Generation이 달라질 수
  있습니다.
- 현재 배포 검증은 Dockerized API/PostgreSQL과 Host Ollama를 결합한 로컬 구성입니다.
  공개 Cloud 배포와 대규모 트래픽 검증은 수행하지 않았습니다.

## Detailed Development Log

[전체 구현·실험 기록](docs/development-log.md)에는 PDF loader, Chunking, Embedding,
FAISS, Top-K와 threshold, reranking, Generation ablation, Multi-document 및 Docker
최적화까지의 상세 과정과 terminal 출력을 원문 그대로 보존했습니다.

AI 도구는 코드 초안 작성과 반복 작업 보조에 활용했습니다. 설계, 검증, 실험 구성과
결과 해석은 각 단계에서 직접 확인하며 진행했습니다.
