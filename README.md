# Basic RAG from Scratch

LangChain 같은 고수준 프레임워크에 의존하기 전에,
PDF 로딩 → 청킹 → 임베딩 → 벡터 검색 → LLM 연결까지
RAG의 핵심 구조를 단계별로 직접 구현하고 검증하는 프로젝트입니다.

## 목표

다음 과정을 순서대로 구현하고 결과를 직접 검증합니다.

PDF 문서  
→ 텍스트 추출  
→ Chunking  
→ Embedding  
→ Vector Search  
→ Retrieval  
→ LLM  
→ RAG 평가

## 개발 환경

- Ubuntu
- Python 3.12
- uv
- VS Code
- PyTorch
- NVIDIA RTX 5070 Ti

## 1. PDF 로딩 및 텍스트 추출

`pypdf`를 사용해 PDF 문서를 페이지 단위로 읽고 텍스트를 추출했습니다.

실습 문서:

- NIA 생성형 AI 윤리 가이드북
- 총 74페이지

검증 결과:

- PDF 파일 정상 탐색
- 전체 74페이지 인식
- 한국어 본문 텍스트 추출 확인
- 텍스트가 없는 페이지 별도 확인
- 제어문자만 존재하는 페이지를 빈 페이지로 처리

PDF 로딩 단계에서는 아직 Chunking, Embedding, Vector DB, LLM을 구현하지 않았습니다.

## 실행 방법

```bash
uv run python -m rag_basic.pdf_loader
```

## 2. Chunking

추출한 PDF 텍스트를 검색 가능한 작은 단위로 나누기 위해
글자 수 기준 Chunking을 직접 구현했습니다.

LangChain 등의 Text Splitter는 사용하지 않고,
Python 문자열 슬라이싱을 이용해 Chunking의 기본 원리를 확인했습니다.

초기 기준값:

- `chunk_size = 500`
- `chunk_overlap = 100`

`500 / 100`은 최적값으로 가정한 것이 아니라,
추후 청킹 설정을 비교하기 위한 첫 번째 기준값으로 사용했습니다.

각 Chunk에는 다음 정보를 유지합니다.

- `chunk_id`
- `page_number`
- `text`

### 검증 결과

- 전체 Chunk: 162개
- 최소 Chunk 길이: 2자
- 최대 Chunk 길이: 500자
- 평균 Chunk 길이: 413.4자
- 같은 페이지 내 인접 Chunk 97쌍의 100자 overlap 확인
- 텍스트가 없는 9개 페이지에서는 Chunk가 생성되지 않음을 확인

### 확인한 한계

현재 방식은 페이지별로 독립적으로 Chunking하기 때문에
원본 페이지의 텍스트가 짧으면 매우 짧은 Chunk가 생성될 수 있습니다.

또한 일부 페이지에서는 PDF 내부의 글꼴 인코딩이나 구조로 인해
`pypdf`가 읽기 어려운 문자를 반환하는 경우가 있었습니다.

첫 구현에서는 이를 임의로 제거하지 않고 기본 방식의 한계로 기록했으며,
향후 검색 결과를 확인하면서 Chunking 및 전처리 방법을 개선할 예정입니다.

## 3. Embedding

Chunking한 문서를 의미 기반으로 비교할 수 있도록
`intfloat/multilingual-e5-small` 모델을 사용해 Vector로 변환했습니다.

Embedding에는 `sentence-transformers`를 사용했으며,
E5 모델의 검색 방식에 맞춰 다음 prefix를 적용했습니다.

- 검색 질문: `query: `
- 문서 Chunk: `passage: `

Embedding은 cosine similarity 비교를 위해 정규화했습니다.

### 의미 유사도 검증

간단한 문장 비교를 통해 의미가 가까운 문장이
관련 없는 문장보다 높은 유사도를 갖는지 확인했습니다.

- 개인정보 관련 문장 A-B: `0.9194`
- 개인정보 문장 A와 축구 문장 C: `0.7908`
- A-B의 유사도가 더 높음: 확인

유사도 점수 자체를 절대 기준으로 해석하기보다,
검색 후보 간 상대적인 순위를 중심으로 활용합니다.

### PDF Chunk Embedding 결과

- 전체 Chunk: 162개
- 생성된 Embedding: 162개
- Embedding shape: `(162, 384)`
- Vector 차원: 384
- NaN: 없음
- 무한대: 없음
- 영벡터: 없음
- Vector 정규화: 확인
- 실행 device: CUDA

### Token 길이 검증

글자 수 기준 Chunking과 Embedding 모델의 token 입력 길이는
동일하지 않으므로 별도로 token 수를 확인했습니다.

`passage: ` prefix와 tokenizer의 특수 토큰을 포함한 결과:

- 평균 token 수: 204.2
- 최대 token 수: 292
- 512 token 초과 Chunk: 0개
- 최대 token Chunk: `chunk_id=148`, `page_number=67`

현재 문서와 `chunk_size=500`, `chunk_overlap=100` 기준에서는
Embedding 과정에서 입력 길이 제한으로 잘리는 Chunk가 없음을 확인했습니다.

## 4. FAISS 기반 Vector Search

162개의 Chunk Vector에서 질문과 의미가 가까운 Chunk를 찾기 위해
CPU 기반 FAISS의 `IndexFlatIP`를 사용했습니다.

현재 검색 대상이 162개의 Chunk Vector로 작기 때문에,
모든 Vector를 직접 비교하는 exact search 방식으로도 충분합니다.
Embedding이 정규화되어 있어 Inner Product를 cosine similarity로 사용할 수 있으며,
FAISS에는 `float32` 타입의 Vector를 등록했습니다.

FAISS는 Chunk metadata를 직접 반환하지 않고 등록된 Vector의 index를 반환합니다.
이 index로 기존 `chunks` 목록에 접근하여 `chunk_id`, `page_number`, `text`를 연결합니다.

검증 질문과 검색 기준:

- 질문: `생성형 AI가 만든 이미지의 저작권은 누구에게 있나요?`
- `top_k = 5`
- FAISS: `IndexFlatIP`
- Chunk Vector: 162개
- Embedding shape: `(162, 384)`
- dtype: `float32`

1위 검색 결과:

- similarity: `0.9257`
- FAISS index: `27`
- `chunk_id`: `28`
- `page_number`: `18`

### NumPy 직접 계산 검증

동일한 Query Vector와 Chunk Vector의 Inner Product를 NumPy로 직접 계산하고,
FAISS의 Top-5 결과와 비교했습니다.

- NumPy Top-5: `[27, 38, 2, 92, 34]`
- FAISS Top-5: `[27, 38, 2, 92, 34]`
- index 순서 동일: `True`
- similarity 점수 동일: `True`

동일한 similarity를 가진 Vector가 있으면 FAISS와 NumPy가 동점 Vector의
순서를 다르게 반환할 수 있습니다. 따라서 index 순서의 완전 일치와
similarity 점수의 일치를 별도로 검증합니다.

### 실행 방법

```bash
uv run python -m rag_basic.vector_search
```

현재 FAISS index는 파일로 저장하지 않고 실행할 때마다 메모리에서 생성합니다.
Retrieval과 LLM 연결은 아직 구현하지 않았습니다.

## 진행 상황

- [x] PDF 로딩 및 텍스트 추출
- [x] Chunking
- [x] Embedding
- [x] FAISS 기반 Vector Search
- [ ] Retrieval
- [ ] LLM 연결
- [ ] RAG 품질 평가

## AI 도구 활용

Codex를 코드 초안 작성과 실행 보조에 활용했습니다.

RAG의 각 단계를 직접 이해하고,
구현 결과를 실행·검증하며 필요한 수정과 해석은 직접 수행합니다.
