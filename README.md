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

## 5. Retrieval

Vector Search는 질문과 가까운 Vector의 score와 index를 찾는 단계이고,
Retrieval은 이 결과를 원래 Chunk의 text와 metadata에 다시 연결하는 단계입니다.

`src/rag_basic/retrieval.py`에서는 기존 `embedding.py`의 `embed_texts()`와
`vector_search.py`의 `build_index()`를 재사용하며, `top_k = 5`로 검색합니다.
FAISS가 반환한 index로 기존 `chunks` 목록을 조회하여 검색 결과를
`list[dict]` 형태로 구성했습니다.

각 결과에는 다음 정보가 포함됩니다.

- `rank`
- `score`
- `faiss_index`
- `chunk_id`
- `page_number`
- `text`

FAISS와 NumPy가 반환한 score와 index는 이후 코드에서 쉽게 사용할 수 있도록
일반 Python의 `float`와 `int`로 변환했습니다.

### Context 구성

`build_context()`는 검색된 각 Chunk의 전체 text를 하나의 문자열로 합칩니다.
각 출처는 다음 header로 구분하여 원본 페이지와 Chunk를 추적할 수 있습니다.

```text
[Source N | page=... | chunk_id=...]
```

입력 list의 순서가 뒤섞여도 `rank`를 기준으로 정렬하여 Context의 Source 순서를
보장합니다. 이렇게 만든 Context는 LLM 연결 단계에서 참고 자료로 전달합니다.

### 검증 결과

- Retrieval 결과 개수: 5
- 검색된 FAISS index: `[27, 38, 2, 92, 34]`
- 이전 Vector Search 결과와 동일: `True`
- 생성된 Context 전체 글자 수: 2305
- Retrieval 결과가 정확히 5개인가: `True`
- rank가 1~5 순서인가: `True`
- 모든 결과에 필요한 key가 존재하는가: `True`
- Context에 모든 Chunk text가 포함되는가: `True`
- Context의 Source 순서가 rank와 동일한가: `True`
- 역순 입력에서도 Source 1~5 순서로 재정렬되는가: `True`

### 실행 방법

```bash
uv run python -m rag_basic.retrieval
```

## 6. LLM 연결

일반 LLM 호출은 모델이 학습한 일반 지식을 바탕으로 답하지만,
RAG의 LLM 호출은 Retrieval에서 찾은 Context와 질문을 함께 전달하여
검색된 문서를 근거로 답하도록 합니다.

`src/rag_basic/llm.py`에서는 OpenAI Python SDK의 Responses API와
`gpt-5.6-luna` 모델을 사용했습니다. 기존 단계의 다음 함수를 재사용합니다.

- `embedding.py`의 `embed_texts()`
- `vector_search.py`의 `build_index()`
- `retrieval.py`의 `retrieve()`
- `retrieval.py`의 `build_context()`

`generate_answer()`는 Retrieval이나 FAISS 검색을 직접 수행하지 않고,
질문과 Retrieval에서 생성된 Context를 받아 답변을 생성하는 Generation 역할만 담당합니다.

### 답변 규칙

LLM에는 다음 규칙을 전달합니다.

- 제공된 Context만 근거로 답변
- Context에 없는 내용을 일반 지식으로 보충하거나 추측하지 않음
- 근거가 부족하면 `제공된 문서에서 확인할 수 없습니다.`라고 답변
- 사용한 근거를 `[Source N]` 형태로 표시
- 간결한 한국어로 답변

Source 번호를 답변에 포함하여 어떤 검색 결과를 근거로 사용했는지 추적할 수 있습니다.

### 정상 질문 검증

질문: `생성형 AI가 만든 이미지의 저작권은 누구에게 있나요?`

- Retrieval FAISS index: `[27, 38, 2, 92, 34]`
- Context 글자 수: 2305
- 문서의 저작권 관련 내용을 근거로 답변 생성
- 답변에 `[Source 1] [Source 2] [Source 5]` 포함

### 문서 밖 질문 검증

질문: `프랑스의 수도는 어디인가요?`

- Retrieval FAISS index: `[135, 101, 64, 107, 95]`
- Context 글자 수: 2501
- 최종 답변: `제공된 문서에서 확인할 수 없습니다.`

문서 밖 질문에도 모델의 일반 지식으로 답하지 않는지 확인하여 grounding 동작을
검증했습니다.

검증 결과:

- 정상 질문의 답변이 비어 있지 않은가: `True`
- 정상 질문의 답변에 `[Source`가 포함되는가: `True`
- 문서 밖 질문의 답변이 지정된 문장과 정확히 일치하는가: `True`

### API Key 설정 및 실행 방법

API Key는 코드에 저장하지 않고 `OPENAI_API_KEY` 환경변수로 전달합니다.
실제 Key를 Git이나 README에 기록하지 않습니다.

```bash
export OPENAI_API_KEY="발급받은_API_KEY"
uv run python -m rag_basic.llm
```

현재 구현은 RAG의 전체 흐름을 연결한 첫 baseline입니다.

## 7. RAG 품질 평가

완성된 RAG가 단순히 답변을 생성하는지를 넘어 다음 항목을 확인하기 위해
소규모 baseline 평가를 직접 구현했습니다.

1. Retrieval이 사람이 지정한 정답 근거(gold evidence)를 찾는지
2. 검색 결과에서 정답 근거가 얼마나 높은 순위에 있는지
3. 생성 답변이 실제 Source 번호를 사용하는지
4. 문서 밖 질문에 일반 지식으로 답하지 않고 거절하는지

RAGAS, LangChain 평가 기능, LLM-as-a-Judge 및 별도 평가용 LLM은 사용하지 않고
기본 지표를 Python으로 직접 구현했습니다.

### 평가 데이터 구성

PDF의 실제 Chunk 본문을 사람이 직접 확인하여 질문과 gold evidence를
수동으로 구성했습니다.

- 전체 Case: 9개
- in-domain Case: 6개
- out-of-domain Case: 3개

in-domain Case는 다음과 같이 서로 다른 주제를 포함합니다.

- 생성형 AI 이미지 저작권
- 이용자의 창작적 기여와 저작권
- 생성형 AI 결과물의 과제 제출
- 미드저니 미술대회 사례
- 생성형 AI 가짜 뉴스 피해 신고
- 생성형 AI의 업무 활용 장점

질문에 따라 하나 이상의 Chunk가 모두 유효한 근거가 될 수 있으므로
복수의 `expected_chunk_ids`를 허용했습니다. 예를 들어 `[68, 69, 70]`이면
세 Chunk 중 하나라도 Top-5에 검색될 때 Retrieval Hit로 판단합니다.

### 평가 지표

#### Hit@5

사람이 지정한 gold evidence 중 하나라도 검색 결과 Top-5 안에 있는지 확인합니다.

#### Reciprocal Rank

검색 결과에서 가장 먼저 등장한 gold evidence 순위의 역수입니다.

- 1위 → `1.0`
- 2위 → `0.5`
- 4위 → `0.25`

#### MRR

각 in-domain 질문의 Reciprocal Rank 평균입니다. 이번 평가에서는
6개 in-domain 질문의 Reciprocal Rank 평균을 사용했습니다.

#### Source citation 검증

생성 답변에 `[Source N]` 형태의 Source 번호가 존재하고,
그 번호가 실제 Retrieval 결과 범위 안에 있는지 확인합니다.
Retrieval 결과가 Source 1~5까지라면 `[Source 1]`과 `[Source 5]`는 유효하지만,
`[Source 8]`은 유효하지 않습니다.

이 검사는 Source 번호의 형식과 존재 여부를 확인합니다. 답변의 모든 문장이
해당 Source에 의해 완전히 뒷받침되는지를 자동으로 판단하는 faithfulness 평가는 아닙니다.

#### Out-of-domain refusal

문서에 답이 없는 질문에 LLM의 일반 지식으로 답하지 않고 다음 문장으로
정확히 응답하는지 확인합니다.

```text
제공된 문서에서 확인할 수 없습니다.
```

### 실제 실행 결과

- 전체 평가 Case: 9
- in-domain Case: 6
- out-of-domain Case: 3
- in-domain Hit@5: `6/6`
- in-domain MRR: `0.8750`
- 정상 Source citation 검증: `6/6`
- out-of-domain refusal: `3/3`

MRR이 `1.0`이 아닌 이유는 `creative_contribution_copyright` 질문의
gold evidence인 `chunk_id=33`이 검색 결과 4위에 있었기 때문입니다.
이 Case의 Reciprocal Rank는 `1 / 4 = 0.25`입니다.

즉 모든 in-domain 질문에서 gold evidence가 Top-5 안에는 포함됐지만,
모든 질문의 정답 근거가 항상 검색 결과 1위였던 것은 아닙니다.

### 수동 답변 검토

6개 in-domain 질문의 생성 답변을 사람이 직접 읽고,
검색된 Context의 내용과 크게 어긋나지 않는지 확인했습니다.
이는 자동 faithfulness 평가가 아니라 수동 검토 결과입니다.

### 한계

- 평가 Case가 9개뿐인 소규모 baseline
- 사람이 직접 질문과 gold evidence를 구성한 수동 평가셋
- 동일 문서를 바탕으로 개발 과정에서 구성했으므로 대규모 독립 benchmark가 아님
- 자동 faithfulness 평가를 아직 구현하지 않음
- 자동 answer correctness 평가를 아직 구현하지 않음
- 현재 결과가 전체 RAG 품질을 대표하지 않음

### 평가 코드

주요 평가 파일:

- `src/rag_basic/evaluation.py`: 최종 9개 Case의 Retrieval과 Generation baseline 평가

평가셋 구성 과정에서 사용한 보조 스크립트:

- `src/rag_basic/evaluation_discovery.py`: 문서 전체의 실제 Chunk를 사람이 살펴보고 평가 질문 후보를 선정하기 위한 탐색용 스크립트
- `src/rag_basic/evaluation_retrieval_check.py`: 잠정 gold evidence가 실제 Retrieval에서 어떤 순위로 검색되는지 본문과 함께 사전 확인하는 스크립트

### 실행 방법

```bash
uv run python -m rag_basic.evaluation
```

현재까지 기본적인 from-scratch RAG baseline을 구현하고 소규모 품질 평가까지
수행했습니다. 이후에는 Local LLM, Retrieval 검색 품질 개선, Chunking 전략 비교,
더 큰 독립 평가셋, faithfulness 및 answer correctness 평가로 확장할 수 있습니다.

## 8. Local LLM 연결

기존 OpenAI API 기반 Generation을 바로 교체하기 전에,
내 PC에서 실행되는 Local LLM을 Python에서 직접 호출할 수 있는지 단계적으로
확인했습니다.

### Ollama 설치 및 모델 실행

- Ollama: 0.33.3
- Local model: `qwen3:8b`
- 모델 크기: 약 5.2GB
- NVIDIA RTX 5070 Ti에서 실행 확인

Ollama는 Local LLM 모델 자체가 아닙니다. 로컬에 설치된 모델을 실행하고,
터미널이나 HTTP API를 통해 Python 같은 프로그램이 모델에 요청할 수 있도록
연결해 주는 도구입니다.

다음 명령으로 모델을 단독 실행했습니다.

```bash
ollama run qwen3:8b
```

`RAG가 무엇인지 두 문장으로 설명해줘.`라는 질문에 한국어 답변이
정상적으로 생성되는 것을 확인했습니다.

### Python에서 Ollama API 호출

`src/rag_basic/local_llm.py`를 구현하여 외부 Python 패키지 없이
다음 표준 라이브러리만 사용했습니다.

- `json`
- `urllib.request`
- `urllib.error`

Python에서 다음 Ollama Local API로 HTTP 요청을 보냅니다.

```text
http://localhost:11434/api/chat
```

요청에는 다음 설정을 사용합니다.

- model: `qwen3:8b`
- `stream: false`
- `think: false`

`stream: false`는 응답을 토큰 단위로 나누어 받지 않고 완성된 JSON 응답을
한 번에 받기 위한 설정입니다. `think: false`는 Qwen3의 별도 thinking 출력을
끄고 최종 답변만 받기 위한 설정입니다.

Ollama가 반환한 JSON에서 `message.content`를 추출하여 최종 답변 문자열로
사용합니다.

### Python 실행 검증

```bash
uv run python -m rag_basic.local_llm
```

실제 테스트 질문은 다음과 같습니다.

```text
RAG에서 Retrieval이 필요한 이유를 두 문장으로 설명해줘.
```

실행 결과 `qwen3:8b`가 정상적인 한국어 답변을 반환했습니다.
추가로 `nvidia-smi`에서 `ollama/llama-server`가 약 5.6GB의 GPU Memory를
사용하는 것이 확인되어, Python API 호출 시에도 Local LLM이 GPU에서 실행되고
있음을 확인했습니다.

### 현재 단계의 범위

이번 단계에서는 다음 작업을 하지 않았습니다.

- PDF 및 Chunking 변경
- Embedding 변경
- FAISS 변경
- Retrieval 변경
- RAG Context와 Local LLM 연결
- OpenAI LLM과 Local LLM 품질 비교

현재는 다음 흐름까지만 확인한 상태입니다.

```text
Python
→ Ollama Local API
→ qwen3:8b
→ 답변
```

다음 단계에서는 기존 Retrieval과 Context를 그대로 재사용하고,
OpenAI API 대신 `qwen3:8b`가 문서 기반 답변을 생성하도록 Local RAG를
연결할 예정입니다.

## 진행 상황

- [x] PDF 로딩 및 텍스트 추출
- [x] Chunking
- [x] Embedding
- [x] FAISS 기반 Vector Search
- [x] Retrieval
- [x] LLM 연결
- [x] RAG 품질 평가
- [x] Ollama Local LLM 단독 연결
- [ ] Local RAG 연결
- [ ] OpenAI / Local LLM 비교

## AI 도구 활용

Codex를 코드 초안 작성과 실행 보조에 활용했습니다.

RAG의 각 단계를 직접 이해하고,
구현 결과를 실행·검증하며 필요한 수정과 해석은 직접 수행합니다.
