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

### Local API 단독 호출 단계의 범위

이 단독 호출 단계에서는 다음 작업을 하지 않았습니다.

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

이 확인을 바탕으로 기존 Retrieval 및 Context와 Local LLM을 연결했습니다.

### Local RAG 연결

`src/rag_basic/local_rag.py`를 구현하여 기존 RAG의 검색 부분을 새로 만들지 않고
다음 기능을 그대로 재사용했습니다.

- PDF Loading
- Chunking
- Embedding
- FAISS Vector Search
- Retrieval
- Context 생성

기존 OpenAI 기반 구조는 다음과 같습니다.

```text
질문
→ Retrieval
→ Context
→ OpenAI LLM
→ 답변
```

이 구조에서 Generation 부분만 교체하여 다음 Local RAG를 구성했습니다.

```text
질문
→ Retrieval
→ Context
→ Ollama
→ qwen3:8b
→ 답변
```

### Local RAG Prompt

Local LLM에도 기존 OpenAI RAG와 의미가 같은 grounding 규칙을 전달했습니다.

- 제공된 Context만 근거로 답변
- Context에 없는 내용을 일반 지식으로 보충하거나 추측하지 않음
- 근거가 없으면 `제공된 문서에서 확인할 수 없습니다.`라고 답변
- 사용한 근거를 `[Source N]` 형태로 표시
- 간결한 한국어로 답변

### 정상 질문 검증

질문: `생성형 AI가 만든 이미지의 저작권은 누구에게 있나요?`

- FAISS index: `[27, 38, 2, 92, 34]`
- `chunk_id`: `[28, 39, 3, 93, 35]`
- Context 글자 수: 2305

기존 OpenAI RAG에서 사용한 동일 질문과 같은 Retrieval 결과가 나왔습니다.
Retrieval 부분은 변경하지 않고 Generation 모델만 Local LLM으로 교체했기 때문입니다.

`qwen3:8b`는 검색된 Context를 바탕으로 저작권 관련 답변을 생성했고,
답변에 `[Source 1]`을 포함했습니다.

### 문서 밖 질문 검증

질문: `프랑스의 수도는 어디인가요?`

- FAISS index: `[135, 101, 64, 107, 95]`
- `chunk_id`: `[136, 102, 65, 108, 96]`
- Context 글자 수: 2501

`qwen3:8b`는 자신의 일반 지식으로 `파리`라고 답하지 않고 다음 문장으로
정확히 응답했습니다.

```text
제공된 문서에서 확인할 수 없습니다.
```

첫 Local RAG baseline의 두 테스트 질문에서 Context 밖 일반 지식을 사용하지
않도록 한 grounding 규칙이 의도대로 동작하는 것을 확인했습니다. 다만 두 Case만으로
Local RAG 전체 품질이 검증됐다고 볼 수는 없습니다.

### 검증 결과

- 정상 질문의 답변이 비어 있지 않은가: `True`
- 정상 질문의 답변에 `[Source`가 포함되는가: `True`
- 문서 밖 질문의 지정된 refusal 문장 일치: `True`

`nvidia-smi`에서 `ollama/llama-server`가 약 5.6GB의 GPU Memory를 사용하는 것이
확인되어, Local RAG의 `qwen3:8b` Generation이 NVIDIA RTX 5070 Ti에서 실행되고
있음을 확인했습니다.

### 현재 단계의 의미

RAG의 Retrieval 부분과 Generation 모델은 서로 분리할 수 있습니다.
현재 프로젝트에서는 다음 부분을 동일하게 유지했습니다.

```text
PDF
→ Chunking
→ Embedding
→ FAISS
→ Retrieval
→ Context
```

Generation만 `OpenAI LLM`에서 `Ollama + qwen3:8b`로 교체했습니다.
따라서 같은 Retrieval pipeline을 사용하면서 서로 다른 LLM을 비교할 수 있는
구조가 만들어졌습니다.

### 한계

- 정상 질문 1개와 문서 밖 질문 1개만 사용한 Local RAG 검증
- 두 Case만으로 Local RAG 전체 성능을 대표할 수 없음
- Source 표기의 존재만 확인했으며 자동 faithfulness 평가는 아님
- OpenAI와 Local LLM의 본격적인 품질 비교는 아직 수행하지 않음

### 실행 방법

```bash
uv run python -m rag_basic.local_rag
```

### OpenAI / Local LLM 비교

`src/rag_basic/model_comparison.py`를 구현하여 동일한 Retrieval 결과와 Context를
OpenAI GPT와 Local LLM에 각각 전달하고 Generation 결과를 비교했습니다.

#### 비교 방법

기존 `evaluation.py`의 9개 평가 Case를 그대로 재사용했습니다.

- 전체 Case: 9개
- in-domain Case: 6개
- out-of-domain Case: 3개

각 Case의 Retrieval은 한 번만 수행합니다. 검색 결과로 만든 동일한 Context를
OpenAI `gpt-5.6-luna`와 Local `qwen3:8b`에 각각 전달하므로, 두 답변의 차이는
Retrieval 결과가 아니라 Generation 모델의 차이에서 발생합니다.

#### 공통 Retrieval 결과

- in-domain Case: 6개
- Hit@5: `6/6`
- MRR: `0.8750`

이 값은 두 모델 각각의 점수가 아니라 두 모델이 공통으로 사용하는 Retrieval
pipeline의 결과입니다.

#### OpenAI Generation 결과

- in-domain `answer_non_empty`: `6/6`
- in-domain `unexpected_refusal`이 `False`: `6/6`
- 정상 Source citation 검증: `6/6`
- out-of-domain refusal: `3/3`

#### qwen3:8b Generation 결과

- in-domain `answer_non_empty`: `6/6`
- in-domain `unexpected_refusal`이 `False`: `6/6`
- 정상 Source citation 검증: `6/6`
- out-of-domain refusal: `3/3`

자동 지표만 보면 두 모델은 이번 소규모 baseline에서 동일한 결과를 기록했습니다.
이는 두 모델의 실제 답변 품질이 완전히 동일하다는 의미는 아닙니다.

#### 수동 답변 비교

자동 평가 후 6개 in-domain 답변을 사람이 직접 읽고 비교했습니다.

OpenAI `gpt-5.6-luna`에서 관찰한 내용:

- 여러 Source의 내용을 함께 사용하는 경우가 상대적으로 많았음
- 배경 설명이나 추가 조건을 포함하여 상대적으로 상세하게 설명하는 경향
- 가짜 뉴스 신고 질문처럼 여러 대응 경로가 Context에 있을 때 여러 항목을 종합하여 답변

Local `qwen3:8b`에서 관찰한 내용:

- 핵심 Source 한 개를 중심으로 답하는 경우가 많았음
- 상대적으로 짧고 직접적인 답변을 생성하는 경향
- 핵심 근거가 검색 순위 4위였던 `creative_contribution_copyright` 질문에서도
  `[Source 4]`를 사용하여 직접적인 정답 근거를 활용
- 세 개의 out-of-domain 질문 모두 일반 지식을 사용하지 않고 지정된 refusal 문장으로 응답

이번 소규모 평가에서는 두 모델 모두 기본적인 grounding과 문서 밖 질문 거절에
성공했습니다. OpenAI는 여러 근거를 종합해 상대적으로 풍부한 답변을 만드는 경향이
있었고, `qwen3:8b`는 핵심 근거 중심의 간결한 답변을 만드는 경향이 관찰되었습니다.
이 관찰은 현재 9개 Case에 한정되며 두 모델의 일반적인 성능 우열을 의미하지 않습니다.

#### 중요한 해석

이번 비교에서는 두 모델이 동일한 Retrieval 결과를 사용합니다.

```text
PDF
→ Chunking
→ Embedding
→ FAISS
→ Retrieval
→ Context
        ├→ OpenAI GPT
        └→ Ollama qwen3:8b
```

이를 통해 RAG의 Retrieval 부분과 Generation 모델을 분리하고, 서로 다른 LLM을
동일한 검색 조건에서 비교할 수 있음을 확인했습니다.

#### 자동 평가의 한계

- 9개 Case만 사용한 소규모 baseline
- Source 번호의 존재와 유효성만 자동 확인
- Source citation 검증은 자동 faithfulness 평가가 아님
- answer correctness 자동 평가는 아직 없음
- LLM-as-a-Judge를 사용하지 않음
- 답변 품질 비교는 사람이 직접 읽어 확인
- 실행시간과 비용 비교는 아직 수행하지 않음
- 현재 결과로 두 모델의 전체 성능 우열을 판단할 수 없음

#### 실행 방법

```bash
uv run python -m rag_basic.model_comparison
```

현재는 from-scratch RAG baseline과 Local LLM 연결, 소규모 모델 비교까지 완료한
상태입니다. 향후 Retrieval 품질 개선, Chunking 전략 비교, similarity threshold,
더 큰 독립 평가셋, faithfulness 및 answer correctness 평가, Vector DB 확장 등을
검토할 수 있습니다.

## 9. Retrieval 개선 실험

첫 Retrieval 개선 실험으로 Top-K 값에 따른 검색 품질과 Context 길이를
비교했습니다.

### Top-K란?

Top-K는 Vector Search에서 질문과 가장 유사한 Chunk를 몇 개까지 가져올지
정하는 값입니다.

- K=1: 가장 유사한 Chunk 1개
- K=3: 상위 3개
- K=5: 상위 5개
- K=10: 상위 10개

Top-K가 너무 작으면 답변에 필요한 정답 근거를 놓칠 수 있습니다. 반대로 너무
크면 관련성이 낮은 Chunk까지 Context에 포함되어 LLM에 전달할 입력이 길어질 수
있습니다.

### 실험 방법

`src/rag_basic/top_k_experiment.py`에서 기존 `evaluation.py`의 in-domain 6개
Case와 사람이 지정한 gold evidence를 그대로 재사용했습니다. 비교한 값은
K=1, K=3, K=5, K=10입니다.

Embedding model, PDF, Chunk, Chunk Embedding과 FAISS index는 한 번만 생성한 뒤
모든 Top-K 실험에서 재사용했습니다. 이번 실험은 Retrieval만 비교했으므로 다음
항목은 수행하지 않았습니다.

- OpenAI 호출
- Ollama 호출
- Generation
- Source citation 평가
- refusal 평가
- faithfulness 평가

### 평가 지표

- **Hit@K**: 사람이 지정한 gold evidence 중 하나라도 검색 결과 Top-K 안에
  존재하는지 확인합니다.
- **MRR**: 각 질문에서 가장 먼저 등장한 gold evidence 순위의 역수를 구한 뒤
  평균을 계산합니다.
- **평균 Context 글자 수**: Top-K 검색 결과를 `build_context()`로 합쳤을 때
  LLM에 전달될 입력 크기를 비교하기 위한 참고 지표입니다.

### 실제 실행 결과

|  K | Hit 통과 |    MRR | 평균 Context 글자 수 |
| -: | --------: | -----: | -------------------: |
|  1 |       5/6 | 0.8333 |                535.3 |
|  3 |       5/6 | 0.8333 |               1522.2 |
|  5 |       6/6 | 0.8750 |               2437.3 |
| 10 |       6/6 | 0.8750 |               4944.8 |

### 결과 해석

`creative_contribution_copyright` 질문의 gold evidence인 `chunk_id=33`은 검색
순위 4위에 있었습니다. 따라서 K=1과 K=3에서는 검색되지 않았고 K=5부터
검색됐습니다. 이 때문에 K=1과 K=3의 Hit는 `5/6`, K=5부터는 `6/6`이었습니다.

K=5와 K=10은 Hit가 `6/6`, MRR이 `0.8750`으로 같았습니다. 그러나 평균
Context 길이는 K=5의 2437.3자에서 K=10의 4944.8자로 약 두 배 증가했습니다.
현재 6개 평가 Case에서는 K=5가 K=10과 같은 Retrieval 지표를 유지하면서 더
짧은 Context를 사용했습니다.

이는 `K=5가 최적값이다`라는 결론이 아닙니다. 6개의 수동 평가 Case만 사용한
소규모 실험이므로 전체 문서나 다른 질문에서도 K=5가 최적이라고 일반화할 수
없습니다.

### Top-K의 trade-off

Top-K가 너무 작으면 정답 근거를 놓칠 수 있고, 너무 크면 검색 지표의 개선 없이
Context만 길어질 수 있습니다. 따라서 Retrieval에서는 정답 근거를 충분히
포함하면서 불필요하게 긴 Context를 만들지 않는 균형이 중요합니다.

### 실행 방법

```bash
uv run python -m rag_basic.top_k_experiment
```

첫 실행에서는 Hugging Face의 네트워크 확인이 제한되어 로컬 캐시에 있던 동일한
모델을 `HF_HUB_OFFLINE=1` 환경에서 불러와 정상 실행했습니다. 이는 실행 환경에서
적용한 설정이며, 소스 코드에 offline 설정을 추가한 것은 아닙니다.

### 한계

- in-domain 6개 Case만 사용
- 하나의 PDF 문서만 사용
- 기존 500자, overlap 100의 Chunking 방식을 그대로 사용
- 동일한 Embedding 모델 사용
- LLM Generation 품질은 이번 실험 범위에 포함하지 않음
- 현재 결과만으로 Top-K의 일반적인 최적값을 결정할 수 없음

### Similarity Threshold 실험

FAISS의 Top-K 검색은 질문이 문서와 관련이 없어도 가장 유사한 K개 Chunk를
항상 반환합니다. Top-K가 **최대 몇 개까지 가져올 것인지**를 정한다면,
Similarity Threshold는 **최소 어느 정도 유사해야 검색 결과로 인정할 것인지**를
정합니다.

```text
Top-K = 5

score >= threshold
→ 검색 결과 유지

score < threshold
→ 검색 결과 제거
```

Similarity score의 절대값은 Embedding 모델과 데이터에 따라 달라집니다. 따라서
한 시스템에서 확인한 threshold를 다른 RAG 시스템에 그대로 적용할 수는 없습니다.

#### 실험 방법

`src/rag_basic/similarity_threshold_experiment.py`에서 기존 `evaluation.py`의
9개 Case를 그대로 재사용했습니다.

- in-domain Case: 6개
- out-of-domain Case: 3개
- 고정 Top-K: `5`
- 비교 Threshold: `0.85`, `0.88`, `0.90`, `0.92`, `0.94`, `0.96`

각 Case의 FAISS Top-5 Retrieval은 한 번만 수행하고, 동일한 검색 결과에 여러
Threshold를 적용했습니다. Embedding model, PDF, Chunk, Chunk Embedding과 FAISS
index도 한 번 생성한 뒤 모든 Case에서 재사용했습니다.

이번 실험에서는 다음을 수행하지 않았습니다.

- OpenAI API 호출
- Ollama 호출
- Generation
- LLM refusal 평가
- faithfulness 평가
- answer correctness 평가

#### 평가 방법

in-domain Case에서는 Threshold 적용 후에도 gold evidence가 남아 있는지, MRR,
평균 surviving Chunk 수와 평균 Context 글자 수를 확인했습니다.

out-of-domain Case에서는 Threshold 적용 후 검색 결과가 하나도 남지 않으면
`rejected = True`로 평가했습니다. 이는 LLM이 답변을 거절하는 능력을 평가한 것이
아니라, Retrieval 단계에서 OOD 질문의 검색 결과를 제거할 수 있는지 확인한
실험입니다.

#### 실제 실행 결과

| Threshold | Gold retained |    MRR | OOD rejected | 평균 Chunk | 평균 Context |
| --------: | ------------: | -----: | -----------: | ---------: | -----------: |
|      0.85 |           6/6 | 0.8750 |          3/3 |       4.50 |       2201.8 |
|      0.88 |           6/6 | 0.8750 |          3/3 |       4.33 |       2112.3 |
|      0.90 |           5/6 | 0.7083 |          3/3 |       3.67 |       1793.7 |
|      0.92 |           5/6 | 0.7083 |          3/3 |       1.67 |        866.3 |
|      0.94 |           0/6 | 0.0000 |          3/3 |       0.00 |          0.0 |
|      0.96 |           0/6 | 0.0000 |          3/3 |       0.00 |          0.0 |

#### OOD 결과

세 out-of-domain 질문의 Top-5 최고 similarity score는 모두 `0.80` 미만이었습니다.
따라서 테스트한 `0.85` 이상의 모든 Threshold에서 검색 결과가 전부 제거되어
OOD rejected가 `3/3`이었습니다.

이번 세 질문에서 score 차이가 분명했다는 사실만으로 모든 OOD 질문을 하나의
threshold로 안정적으로 구분할 수 있다고 일반화할 수는 없습니다.

#### 정상 질문 결과

Threshold `0.85`와 `0.88`에서는 gold evidence `6/6`, MRR `0.8750`, OOD rejected
`3/3`을 유지했습니다. 특히 `0.88`에서는 평균 surviving Chunk가 4.33개,
평균 Context 길이가 2112.3자로 감소했습니다.

Threshold가 없던 기존 Top-5 실험의 평균 Context 길이는 2437.3자였습니다. 현재
평가 Case에서는 Threshold `0.88`이 일부 검색 결과를 제거하면서도 gold evidence와
MRR을 유지하는 결과를 보였습니다.

#### Threshold가 너무 높은 경우

Threshold가 `0.90`이 되면서 `midjourney_contest_controversy` Case의 gold
evidence가 제거됐습니다. 이 질문에서 가장 높은 gold evidence similarity score는
`0.8902`였기 때문에 `0.90` 기준을 통과하지 못했습니다.

그 결과 gold retained는 `5/6`, MRR은 `0.7083`으로 감소했습니다. Threshold
`0.94` 이상에서는 모든 in-domain 검색 결과까지 제거되어 gold retained `0/6`,
MRR `0.0000`, 평균 Context 0자가 됐습니다.

#### 결과 해석

Similarity Threshold는 FAISS가 항상 Top-K를 반환하는 문제를 완화하여 유사도가
낮은 검색 결과를 제거할 수 있었습니다. 현재 소규모 평가에서는 `0.85`와 `0.88`
모두 gold evidence와 MRR을 유지하면서 세 OOD 질문을 모두 제거했습니다.

테스트한 값 중 `0.88`은 `0.85`와 같은 gold retained, MRR, OOD rejected 결과를
유지하면서 평균 surviving Chunk와 Context 길이가 조금 더 작았습니다. 이는 현재
9개 평가 Case와 현재 Embedding 모델에 한정된 관찰이며, `0.88`이 최적 Threshold이거나
새 문서와 질문에도 같은 값이 적합하다는 의미는 아닙니다.

#### Threshold의 trade-off

Threshold가 너무 낮으면 관련성이 낮은 Chunk가 많이 남을 수 있고, 너무 높으면
답변에 필요한 정답 근거까지 제거될 수 있습니다. Threshold는 단순히 높이는 것이
목적이 아니라 **정답 근거 보존**과 **불필요한 검색 결과 제거** 사이의 균형을
검증하며 정해야 합니다.

#### 실행 방법

```bash
uv run python -m rag_basic.similarity_threshold_experiment
```

#### 실험의 한계

- 전체 평가 Case가 9개뿐임
- in-domain 6개, OOD 3개만 사용
- 하나의 PDF만 사용
- 하나의 Embedding 모델만 사용
- Threshold 후보도 제한적
- OOD 질문 3개에서 나타난 score 분리를 일반화할 수 없음
- LLM Generation 품질은 이번 실험에서 평가하지 않음
- production threshold를 결정한 실험이 아님

### Reranking 실험

현재 FAISS Vector Search는 질문과 Chunk를 각각 Embedding Vector로 만든 뒤
Vector 유사도를 비교하여 후보를 빠르게 찾습니다. 이것이 1차 검색입니다.

Reranking은 FAISS가 먼저 찾은 소수의 후보를 질문과 Chunk를 함께 비교하는
별도의 모델로 다시 평가하여 순서를 재정렬하는 2차 단계입니다.

```text
Query
↓
Embedding
↓
FAISS Top-5
↓
동일한 5개 후보
↓
CrossEncoder Reranker
↓
후보 순서 재정렬
```

Reranker가 새로운 Chunk를 검색한 것은 아니며, 기존 FAISS Top-5 후보의 순서만
다시 정렬했습니다.

#### Bi-Encoder와 Cross-Encoder

현재 Embedding 검색은 다음처럼 질문과 Chunk를 각각 Vector로 만든 뒤 두 Vector의
유사도를 비교합니다.

```text
Query → Vector
Chunk → Vector
→ 두 Vector 유사도 비교
```

이 방식은 많은 문서를 빠르게 검색하기에 적합합니다. 반면 Reranker는 다음처럼
질문과 Chunk를 하나의 pair로 함께 입력하여 관련성 점수를 계산합니다.

```text
(Query, Chunk)
→ 둘을 함께 입력
→ 관련성 점수 계산
```

후보를 더 자세히 비교할 수 있지만 계산량이 더 크기 때문에, 전체 약 162개
Chunk가 아니라 FAISS가 먼저 좁힌 Top-5 후보에만 적용했습니다.

#### 사용 모델

Reranker는 multilingual query-document 관련성 비교에 사용할 수 있는
`BAAI/bge-reranker-v2-m3`를 선택하고 `sentence_transformers.CrossEncoder`로
로드했습니다. 모델은 한 번만 로드한 뒤 모든 Case에서 재사용했습니다.

#### 실험 방법

`src/rag_basic/reranking_experiment.py`에서 기존 `evaluation.py`의 in-domain
6개 Case와 `expected_chunk_ids`를 그대로 재사용했습니다. Top-K는 `5`로
고정했고 Similarity Threshold는 적용하지 않았습니다.

각 질문에서 다음 순서로 비교했습니다.

1. 기존 FAISS Top-5 검색
2. 동일한 5개의 `(query, chunk text)` pair 생성
3. CrossEncoder로 reranker score 계산
4. reranker score가 높은 순서로 재정렬
5. 기존 FAISS 순위와 Reranking 이후 순위 비교

이번 실험에서는 다음을 수행하지 않았습니다.

- OpenAI 호출
- Ollama 호출
- Generation 및 Prompt 변경
- Chunking 및 Embedding 모델 변경
- Similarity Threshold 적용
- Hybrid Search 및 BM25

#### 평가 지표

- **Hit@5**: gold evidence가 Top-5 후보 안에 존재하는지 확인합니다.
- **MRR**: 각 질문에서 가장 먼저 등장한 gold evidence 순위의 역수를 구한 뒤
  평균을 계산합니다.
- **Top-1 gold**: 검색 결과 1위가 gold evidence인 Case 수를 확인합니다.

Reranker는 기존 Top-5 후보의 순서만 바꾸므로 후보 집합이 동일한 이번 실험에서
Hit@5는 원칙적으로 변하지 않습니다. 핵심 비교 대상은 gold evidence 순위, MRR,
Top-1 gold의 변화입니다.

#### 실제 전체 결과

| 지표       | FAISS Baseline | Reranked |
| ---------- | -------------: | -------: |
| Hit@5      |            6/6 |      6/6 |
| MRR        |         0.8750 |   1.0000 |
| Top-1 gold |            5/6 |      6/6 |

Rank 변화는 개선 1개, 동일 5개, 악화 0개였습니다.

#### Case별 결과

| Case                              | FAISS gold rank | Reranked gold rank |
| --------------------------------- | --------------: | -----------------: |
| copyright_in_domain               |               1 |                  1 |
| creative_contribution_copyright   |               4 |                  1 |
| ai_assignment_submission          |               1 |                  1 |
| midjourney_contest_controversy    |               1 |                  1 |
| fake_news_damage_report           |               1 |                  1 |
| generative_ai_work_benefits       |               1 |                  1 |

#### 가장 중요한 Case

`creative_contribution_copyright`의 gold evidence인 `chunk_id=33`은 기존 FAISS
검색에서 4위였습니다. Reranking 후에는 동일한 Top-5 후보 안에서 1위로
이동했습니다.

이 변화로 전체 MRR은 `0.8750 → 1.0000`, Top-1 gold는 `5/6 → 6/6`으로
증가했습니다.

#### 결과 해석

현재 6개 in-domain Case에서는 Reranker가 FAISS의 후보 집합을 바꾸지 않으면서
gold evidence의 순서를 개선했습니다. 특히 기존 4위였던 한 정답 근거를 1위로
재정렬하면서 MRR과 Top-1 gold 지표가 개선됐습니다.

하지만 Reranking이 항상 성능을 높이거나 `bge-reranker-v2-m3`가 최적이라는
의미는 아닙니다. 이번 실험에서는 LLM Generation을 수행하지 않았으므로 최종 RAG
답변 품질까지 실제로 개선됐다고 결론 내릴 수도 없습니다.

#### 현재 RAG에서의 의미

현재 RAG는 Top-5 Chunk를 모두 LLM에 전달합니다. 따라서 정답 근거가 FAISS
4위여도 LLM이 이를 읽고 사용할 수 있으며, 이전 Local LLM 비교에서 실제로
`qwen3:8b`가 `[Source 4]`를 사용한 사례가 있었습니다. 현재 작은 baseline에서
Reranking이 필수적인 기능이라고 볼 수는 없습니다.

Reranking은 다음과 같은 상황에서 의미가 더 커질 수 있습니다.

- LLM에 더 적은 수의 Chunk만 전달하려는 경우
- 1차 검색 후보 수가 많아지는 경우
- 관련 근거를 검색 결과 상위에 배치하려는 경우

#### 계산 비용의 trade-off

FAISS Embedding 검색은 많은 Chunk를 빠르게 검색하는 1차 단계입니다. CrossEncoder
Reranker는 각 `(query, chunk)` pair를 함께 처리하므로 상대적으로 계산량이 더
큽니다. 이 때문에 이번 실험에서도 전체 Chunk가 아닌 Top-5 후보에만 Reranker를
적용했습니다.

#### 실행 방법

```bash
uv run python -m rag_basic.reranking_experiment
```

#### 실험의 한계

- in-domain 6개 Case만 사용
- 하나의 PDF만 사용
- Top-K=5만 사용
- 하나의 Reranker 모델만 사용
- 한 Case에서만 순위 개선이 발생
- LLM Generation 품질은 평가하지 않음
- Reranking의 latency나 GPU 비용은 측정하지 않음
- 현재 결과를 일반적인 Reranking 성능으로 일반화할 수 없음

### Chunking configuration 비교

현재 기본 Chunking은 `chunk_size=500`, `chunk_overlap=100`이며, 기존
`create_chunks()`는 페이지 경계를 유지하면서 글자 수를 기준으로 Chunk를
생성합니다.

이번 실험에서는 단순히 Chunk Size만 바꾼 것이 아니라 약 20% overlap 비율을
유지한 세 Chunking configuration을 비교했습니다. 설정이 달라질 때 전체 Chunk
개수, 평균 Chunk 길이, Retrieval 순위와 Top-5 Context 길이가 어떻게 변하는지
확인했습니다.

#### Gold 평가 방식

기존 `evaluation.py`의 `expected_chunk_ids`는 baseline인 500/100 설정에서
만들어진 Chunk ID입니다. Chunking configuration이 달라지면 Chunk ID와 Chunk
text 자체도 달라지므로 기존 ID를 새로운 Chunk에 직접 적용할 수 없습니다.

따라서 다음 과정으로 공통 평가 기준을 만들었습니다.

1. baseline 500/100 Chunk 생성
2. 기존 `expected_chunk_ids`를 baseline Chunk와 연결
3. 각 gold Chunk가 위치한 `page_number` 추출
4. 새로운 Chunking 결과에서 해당 gold page가 검색되는지 평가

#### Page-level 평가의 의미

**Page Hit@5**는 gold evidence가 존재하는 페이지 중 하나라도 Top-5 검색 결과에
포함되면 성공으로 판단합니다. **Page-level MRR**은 가장 먼저 등장한 gold page의
검색 순위로 Reciprocal Rank를 계산한 뒤 전체 질문의 평균을 구한 값입니다.

이는 기존 Chunk ID 기반 MRR과 동일한 평가가 아닙니다. Chunking이 바뀌어도
공통으로 사용할 수 있도록 만든 더 느슨한 page-level 평가입니다.

#### 실험 설정

| Chunk Size | Overlap |
| ---------: | ------: |
|        300 |      60 |
|        500 |     100 |
|        800 |     160 |

세 설정 모두 약 20% overlap 비율을 유지했습니다. Top-K는 `5`로 고정하고 동일한
Embedding 모델을 사용했으며 Similarity Threshold와 Reranking은 적용하지
않았습니다. OpenAI, Ollama 및 Generation도 사용하지 않았습니다.

PDF와 Embedding 모델은 한 번만 로드하고, 내용이 달라지는 다음 자원은 각
configuration마다 새로 생성했습니다.

- Chunks
- Chunk Embeddings
- FAISS index

#### 실제 실행 결과

| Chunk Size | Overlap | Chunk Count | Avg Length | Page Hit@5 | Page MRR | Avg Context |
| ---------: | ------: | ----------: | ---------: | ---------: | -------: | ----------: |
|        300 |      60 |         260 |      265.3 |        6/6 |   0.7917 |      1534.0 |
|        500 |     100 |         162 |      413.4 |        6/6 |   0.8750 |      2437.3 |
|        800 |     160 |         111 |      582.2 |        6/6 |   0.7833 |      3579.2 |

#### Chunk 개수 변화

Chunk Size가 작으면 문서를 더 잘게 나누므로 전체 Chunk 수가 증가하고, Chunk가
커지면 전체 Chunk 수가 감소했습니다.

- 300/60: 260개
- 500/100: 162개
- 800/160: 111개

#### Context 길이 변화

Top-K를 5로 동일하게 유지했기 때문에 큰 Chunk를 사용할수록 LLM에 전달할 수
있는 Context 길이도 증가했습니다.

- 300/60: 평균 1534.0자
- 500/100: 평균 2437.3자
- 800/160: 평균 3579.2자

큰 Chunk는 더 많은 문맥을 포함하는 대신 Context 입력량도 늘어나므로 항상 더
효율적이라고 볼 수는 없습니다.

#### Retrieval 결과

세 configuration 모두 현재 6개 평가 Case에서 Page Hit@5 `6/6`을 기록했습니다.
즉 모든 설정이 Top-5 안에서 gold evidence가 위치한 페이지를 검색했습니다.

Page-level MRR은 300/60에서 `0.7917`, 500/100에서 `0.8750`, 800/160에서
`0.7833`이었습니다. 현재 6개 Case에서는 500/100 configuration이 세 설정 중
가장 높은 Page-level MRR을 보였지만, 이것이 일반적인 최적 Chunking이라는
의미는 아닙니다.

#### 결과 해석

300/60 설정:

- Chunk 수가 가장 많음
- 평균 Context가 가장 짧음
- Page Hit@5 `6/6`
- Page-level MRR `0.7917`

500/100 설정:

- 현재 baseline
- Page Hit@5 `6/6`
- 세 configuration 중 가장 높은 Page-level MRR `0.8750`
- Context 길이는 중간 수준

800/160 설정:

- Chunk 수가 가장 적음
- 평균 Context가 가장 김
- Page Hit@5 `6/6`
- Page-level MRR `0.7833`

현재 소규모 평가에서는 큰 Chunk가 더 많은 Context를 제공했지만 Retrieval 순위
지표가 반드시 좋아지지는 않았습니다. 작은 Chunk는 Context 길이를 줄였지만
Page-level MRR이 baseline보다 낮았습니다.

#### Chunking의 trade-off

작은 Chunk의 특징:

- 더 세밀한 검색 단위
- 전체 Chunk 수 증가
- Context가 짧아질 수 있음
- 문맥이 여러 Chunk로 나뉠 수 있음

큰 Chunk의 특징:

- 한 Chunk에 더 많은 문맥 포함
- 전체 Chunk 수 감소
- Context가 길어질 수 있음
- 질문과 직접 관련 없는 내용도 함께 포함될 수 있음

Chunk Size를 무조건 작게 또는 크게 만드는 것이 목적은 아닙니다. 검색 정확도와
문맥 보존 사이의 균형을 실제 데이터로 확인해야 합니다.

#### 평가 한계

- in-domain 6개 Case만 사용
- 하나의 PDF만 사용
- 세 configuration만 비교
- 약 20% overlap 비율을 유지했지만 Chunk Size와 Overlap이 함께 변경됨
- 순수하게 Chunk Size 하나의 효과만 분리한 실험이 아님
- Page-level gold는 Chunk-level gold evidence보다 느슨한 평가 기준
- 같은 gold page의 관련 없는 Chunk가 검색돼도 Hit로 계산될 수 있음
- LLM Generation 품질은 평가하지 않음
- 현재 결과로 500/100을 일반적인 최적값이라고 결론 내릴 수 없음

#### 실행 방법

```bash
uv run python -m rag_basic.chunk_size_experiment
```

## 10. Vector DB

### Docker + PostgreSQL + pgvector 환경 구성

현재 RAG는 FAISS를 사용해 다음 흐름으로 Vector Search를 수행합니다.

```text
PDF
↓
Chunk
↓
Embedding
↓
FAISS Index
↓
Vector Search
```

FAISS는 현재 실험에서 정상적으로 동작하지만, 프로그램을 실행할 때마다 Chunk
Embedding과 Index를 메모리에 다시 구성하는 구조입니다. 이번 확장에서는 Chunk,
metadata와 Embedding을 PostgreSQL에 영속적으로 저장하고 SQL 기반 Vector
Search를 구현하기 위한 기반 환경을 만들었습니다.

이는 FAISS가 잘못된 방식이어서 교체하는 것이 아니라, 영속 저장과 데이터 관리가
가능한 Vector DB 구조를 추가로 학습하기 위한 확장입니다.

#### PostgreSQL과 pgvector

PostgreSQL은 일반적인 SQL 데이터를 저장하고 조회할 수 있는 관계형 DBMS입니다.
pgvector extension을 추가하면 PostgreSQL 안에서 Vector 타입과 Vector 거리
연산도 사용할 수 있습니다.

앞으로 하나의 DB Table에서 다음 정보를 관리할 예정입니다.

```text
chunk_id
page_number
text
embedding
```

이번 단계에서는 아직 Table이나 Embedding column을 만들지 않았습니다.

pgvector는 별도의 DB가 아니라 PostgreSQL에 Vector 기능을 추가하는 extension입니다.
다음 SQL로 extension을 활성화했습니다.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

실제 환경에서 PostgreSQL `16.15`와 pgvector `0.8.6`이 동작하는 것을
확인했습니다.

#### Docker를 사용한 이유

PostgreSQL과 pgvector를 Ubuntu host에 직접 설치하지 않고 Docker Container에서
실행했습니다.

- DB 실행 환경을 host와 분리
- 동일한 설정을 다시 구성하기 쉬움
- Container를 다시 생성해도 named volume으로 데이터 보존 가능
- 프로젝트 실행 설정을 `compose.yaml`로 관리 가능

#### 생성된 파일

```text
rag-basic/
├── compose.yaml
├── .env.example
└── docker/
    └── postgres/
        └── init.sql
```

- `compose.yaml`: PostgreSQL + pgvector Container 실행 설정
- `.env.example`: 실행에 필요한 DB 환경변수 예시
- `docker/postgres/init.sql`: 최초 DB 초기화 시 vector extension 활성화

실제 `.env`는 로컬 실행용 설정 파일이며 `.gitignore`로 제외합니다. 실제 DB
password는 Git과 README에 저장하지 않습니다.

#### Docker 구성

- Image: `pgvector/pgvector:pg16`
- Container: `rag-postgres`
- Host port: `5432`
- Container port: `5432`
- Named volume: `rag-basic_postgres_data`
- PostgreSQL 준비 상태를 확인하는 healthcheck 사용
- `unless-stopped` restart policy 사용

#### 검증 결과

```text
Container: rag-postgres
Status: healthy
PostgreSQL: 16.15
pgvector: 0.8.6
Volume: rag-basic_postgres_data
```

Vector 타입이 활성화됐는지 다음 SQL로 확인했습니다.

```sql
SELECT '[1,2,3]'::vector;
```

정상적으로 다음 Vector가 반환됐습니다.

```text
[1,2,3]
```

pgvector의 cosine distance 연산도 확인했습니다.

```sql
SELECT
    '[1,0,0]'::vector <=> '[1,0,0]'::vector,
    '[1,0,0]'::vector <=> '[0,1,0]'::vector;
```

- 같은 방향 Vector의 distance: `0`
- 서로 직교하는 Vector의 distance: `1`

#### Cosine distance

`<=>`는 pgvector의 cosine distance 연산자이며, 값이 작을수록 두 Vector의 방향이
더 비슷하다는 뜻입니다.

```text
같은 Vector
→ distance 0

서로 직교하는 Vector
→ distance 1
```

Cosine similarity와 cosine distance는 해석 방향이 반대입니다.

```text
Cosine similarity
→ 높을수록 유사

Cosine distance
→ 낮을수록 유사
```

#### Volume의 의미

PostgreSQL 데이터는 Container 자체가 아니라 Docker named volume인
`rag-basic_postgres_data`에 저장됩니다. Container를 `docker compose restart`로
재시작한 뒤에도 vector extension `0.8.6`이 유지되는 것을 확인했습니다.

이는 Container 재시작 수준의 확인이며, Volume 삭제와 복구까지 검증한 실험은
아닙니다.

#### 환경 구성 단계에서 수행하지 않은 작업

- chunks Table 생성
- Embedding column 생성
- Python에서 PostgreSQL 연결
- PDF 및 Chunk 적재
- Embedding 적재
- SQL Vector Search
- FAISS와 pgvector 비교
- FastAPI
- LangChain

#### 환경 구성 다음 단계

다음 단계는 **Chunk + Embedding PostgreSQL 적재**입니다.

```text
PDF
↓
Chunk
↓
Embedding
↓
PostgreSQL + pgvector
```

이 환경 구성을 마친 뒤 아래 단계에서 실제 데이터 적재를 구현했습니다.

### Chunk + Embedding PostgreSQL 적재

기존 FAISS baseline에서는 프로그램을 실행할 때 Chunk Embedding과 FAISS Index를
메모리에 구성했습니다. 이번 단계에서는 같은 PDF, Chunking과 Embedding 모델을
재사용하면서 Chunk와 Embedding을 PostgreSQL에 영속적으로 저장했습니다.

```text
PDF
↓
Chunk
↓
Embedding
↓
SQL INSERT / UPSERT
↓
PostgreSQL + pgvector
```

#### 기존 RAG 설정 재사용

- PDF: `data/ai_ethics_guide.pdf`
- `chunk_size`: `500`
- `chunk_overlap`: `100`
- 생성 Chunk: `162`
- Embedding model: `intfloat/multilingual-e5-small`
- Embedding dimension: `384`

새 Chunking이나 Embedding 모델을 만든 것이 아니라 기존 baseline pipeline을
그대로 재사용했습니다.

#### Python에서 PostgreSQL 연결

PostgreSQL 연결과 Vector 타입 처리를 위해 다음 dependency를 추가했습니다.

```text
pgvector>=0.5.0
psycopg[binary]>=3.3.5
```

`psycopg`는 Python과 PostgreSQL의 연결을 담당합니다. `pgvector` Python package는
다음 import를 통해 PostgreSQL Connection에 Vector 타입 adapter를 등록하고,
NumPy Vector를 pgvector 타입으로 전달할 수 있게 합니다.

```python
from pgvector.psycopg import register_vector
```

PostgreSQL password는 코드나 README에 저장하지 않고 Git에서 제외된 로컬 `.env`
환경변수에서 읽습니다.

#### SQL Schema

버전 관리 가능한 [SQL schema](sql/001_create_rag_chunks.sql)에 `rag_chunks`
Table을 정의했습니다.

```text
rag_chunks
├── id: BIGSERIAL PRIMARY KEY
├── document_name: TEXT
├── chunk_id: INTEGER
├── page_number: INTEGER
├── content: TEXT
├── embedding: VECTOR(384)
├── embedding_model: TEXT
├── chunk_size: INTEGER
└── chunk_overlap: INTEGER
```

Embedding은 `VECTOR(384) NOT NULL`로 저장하여 기존 모델의 실제 384차원 출력과
schema가 일치하도록 했습니다.

#### metadata를 함께 저장한 이유

Embedding만 별도로 저장하지 않고 원문과 출처 정보를 같은 행에서 관리합니다.

```text
Chunk Vector
↕
chunk_id
page_number
content
document_name
chunking 설정
embedding model
```

향후 Vector Search로 가까운 Embedding을 찾았을 때 원문 text와 page metadata를
함께 조회하여 RAG Context와 Source로 사용할 수 있기 때문입니다.

#### Upsert와 idempotency

Upsert는 데이터가 없으면 INSERT하고, 이미 있으면 UPDATE하는 방식입니다.

```text
없으면
→ INSERT

이미 있으면
→ UPDATE
```

다음 조합을 UNIQUE constraint로 만들어 동일 dataset 행을 식별했습니다.

```text
document_name
chunk_id
embedding_model
chunk_size
chunk_overlap
```

INSERT에는 `ON CONFLICT ... DO UPDATE`를 사용하여 충돌 시 `page_number`, `content`,
`embedding`을 갱신합니다.

Idempotency는 같은 작업을 여러 번 실행해도 최종 DB 상태가 불필요하게 달라지지
않는 성질입니다. 실제 실행 결과는 다음과 같았습니다.

```text
첫 번째 실행
→ 162 rows

두 번째 실행
→ 162 rows
```

단순 INSERT에서 생길 수 있는 `162 → 324` 중복 적재가 발생하지 않았습니다.
이는 모든 데이터 파이프라인의 완전한 idempotency를 증명한 것이 아니라, 현재
baseline dataset과 UNIQUE key를 기준으로 재실행 안전성을 확인한 결과입니다.

#### 실제 적재 검증 결과

```text
생성된 Chunk 수: 162
Embedding shape: (162, 384)
pgvector version: 0.8.6
Upsert 처리 행 수: 162

DB Row count: 162
NULL 행 수: 0
Embedding dimension: 384 / 384
Chunk ID 범위: 1 / 162
```

- **Row count**: Python에서 생성한 Chunk 수와 DB 저장 행 수가 일치하는지 확인
- **NULL**: 필수 metadata, content 및 embedding의 누락 여부 확인
- **vector_dims**: 모든 Embedding이 schema의 384차원과 일치하는지 확인
- **Chunk ID**: baseline Chunk가 실제 생성 범위인 1~162로 저장됐는지 확인

Embedding 전체 숫자는 출력하지 않고 첫 3개 행의 `chunk_id`, `page_number`,
content preview와 Vector dimension만 확인했습니다.

#### Persistence 검증

PostgreSQL Container를 재시작한 뒤에도 다음 상태가 유지됐습니다.

```text
Container: healthy
rag_chunks baseline rows: 162
```

메모리에서 다시 구성하는 기존 FAISS Index와 달리, Chunk와 Embedding이 PostgreSQL
named volume에 영속 저장됐음을 Container 재시작 수준에서 확인했습니다. Volume
삭제와 복구까지 시험한 것은 아닙니다.

#### FAISS와 현재 저장 구조의 차이

기존 FAISS baseline:

```text
PDF
↓
Chunk
↓
Embedding
↓
FAISS Index
↓
Memory
```

현재 pgvector 저장 단계:

```text
PDF
↓
Chunk
↓
Embedding
↓
PostgreSQL
↓
Persistent Storage
```

FAISS는 Vector Search에 특화된 라이브러리입니다. PostgreSQL + pgvector는 일반
데이터와 Vector를 DB에서 함께 관리하고 영속 저장할 수 있다는 차이가 있으며,
FAISS가 잘못된 기술이라는 의미는 아닙니다.

#### 적재 단계에서 수행하지 않은 작업

- Query Embedding을 이용한 SQL Vector Search
- `<=>` 기반 Top-K Retrieval
- pgvector Retrieval 함수
- FAISS와 pgvector 검색 결과 비교
- Vector index인 HNSW 및 IVFFlat
- OpenAI 및 Ollama Generation 연결
- FastAPI
- LangChain

#### 실행 방법

실제 비밀번호를 명령에 직접 작성하지 않고 `.env`를 현재 shell의 환경변수로
불러온 뒤 실행합니다. `.env`는 Git에 포함하지 않습니다.

```bash
set -a
source .env
set +a

uv run python -m rag_basic.pgvector_ingest
```

#### 적재 다음 단계

다음 단계는 **SQL Vector Search 구현**입니다.

```text
User Query
↓
Query Embedding
↓
PostgreSQL
↓
ORDER BY embedding <=> query_embedding
↓
Top-5 Chunk
```

이 적재를 마친 뒤 아래 단계에서 SQL Vector Search를 구현했습니다.

### SQL Vector Search

PostgreSQL에 이미 저장된 162개 Chunk Embedding은 다시 생성하지 않고, 사용자
Query 하나만 Embedding한 뒤 pgvector의 cosine distance 연산으로 Top-5 Chunk를
검색하도록 구현했습니다.

```text
User Query
↓
Query Embedding
↓
PostgreSQL + pgvector
↓
Cosine distance
↓
Top-5 Chunk
↓
build_context()
```

#### Query Embedding

기존 Embedding 모델인 `intfloat/multilingual-e5-small`을 그대로 사용했습니다.
문서 Chunk Vector는 DB에 저장되어 있으므로 검색할 때 다시 만들지 않고 질문
하나만 Embedding합니다.

E5 모델의 검색 규칙에 따라 기존 `embed_texts(..., "query")`를 재사용하여
`query: ` prefix를 적용했습니다.

```text
Query Embedding shape: (1, 384)
```

#### pgvector SQL 검색

핵심 검색 연산은 pgvector의 cosine distance 연산자 `<=>`입니다.

```sql
ORDER BY embedding <=> query_embedding
LIMIT 5
```

Cosine distance는 작을수록 두 Vector의 방향이 비슷하므로 오름차순으로 정렬합니다.
실제 Query Vector를 SQL 문자열에 직접 삽입하지 않고 parameterized query로
전달했습니다.

#### Distance와 Similarity

두 값의 관계는 다음과 같습니다.

```text
cosine similarity = 1 - cosine distance
```

```text
Cosine similarity
→ 높을수록 유사

Cosine distance
→ 낮을수록 유사
```

pgvector 내부 검색은 distance를 기준으로 합니다. Python 결과의 `score`는 기존
FAISS Retrieval 구조와 맞추기 위해 cosine similarity로 반환합니다.

#### metadata filter

DB에는 앞으로 다른 문서, Embedding 모델과 Chunking configuration도 저장할 수
있습니다. 서로 다른 dataset이 검색에 섞이지 않도록 현재 baseline metadata를
WHERE 조건으로 사용했습니다.

- `document_name`: `ai_ethics_guide.pdf`
- `embedding_model`: `intfloat/multilingual-e5-small`
- `chunk_size`: `500`
- `chunk_overlap`: `100`

#### 검색 결과 구조와 Context 재사용

pgvector 검색 결과는 기존 Retrieval 결과와 최대한 비슷한 `list[dict]` 구조로
구성했습니다.

```text
rank
score
cosine_distance
chunk_id
page_number
text
```

이 구조를 통해 기존 `build_context()`를 변경 없이 재사용했습니다.

```text
FAISS retrieve()
        ↓
    result dict
        ↓
build_context()

pgvector search()
        ↓
    result dict
        ↓
build_context()
```

Retrieval backend가 달라져도 같은 Context 생성 로직을 사용할 수 있음을
확인했습니다.

#### 실제 검색 결과

```text
copyright_in_domain
Top-5: [28, 39, 3, 93, 35]
Gold Hit@5: True
First gold rank: 1
Context: 2305자

creative_contribution_copyright
Top-5: [28, 39, 30, 33, 31]
Gold Hit@5: True
First gold rank: 4
Context: 2519자

france_out_of_domain
Top-5: [136, 102, 65, 108, 96]
Context: 2501자
```

두 in-domain Case에서는 기존 evaluation gold Chunk가 Top-5에 존재하는지와 첫
순위를 확인했습니다. 전체 Evaluation Case의 MRR을 다시 계산한 것은 아닙니다.

#### OOD 검색의 의미

France 질문도 Top-5 결과를 반환했습니다. 이는 오류가 아니라 Vector Search가
현재 저장된 Vector 중 질문에 가장 가까운 후보를 반환하기 때문입니다.

이번 단계에서는 Similarity Threshold, OOD classifier 및 LLM refusal을 적용하지
않았습니다.

```text
Top-K 검색
≠
질문이 문서와 관련 있다는 판정
```

#### Exact Search

현재 `rag_chunks.embedding`에는 HNSW나 IVFFlat Vector index가 없습니다. dataset이
162개로 작기 때문에 이번 단계에서는 성능 최적화보다 pgvector SQL 검색 동작과
결과 확인에 초점을 맞춰 exact nearest-neighbor search를 사용했습니다.

HNSW 또는 IVFFlat을 구현하거나 성능을 검증한 단계는 아닙니다.

#### 검증 결과

```text
DB Connection: 성공
baseline rows: 162
Query Embedding shape: (1, 384)

각 Case:
- Top-5 반환
- rank 1~5
- cosine distance 오름차순
- cosine similarity 내림차순
- score ≈ 1 - cosine_distance
- build_context() 생성 성공
```

Embedding Vector 전체 숫자는 출력하거나 README에 기록하지 않았습니다.

#### 실행 방법

실제 password를 명령에 작성하지 않고 Git에서 제외된 `.env`를 환경변수로
불러옵니다.

```bash
set -a
source .env
set +a

uv run python -m rag_basic.pgvector_retrieval
```

#### 아직 수행하지 않은 작업

- FAISS와 pgvector 정식 비교
- 전체 Evaluation Case 비교
- HNSW
- IVFFlat
- pgvector Retrieval과 LLM Generation 연결
- FastAPI
- LangChain

일부 결과가 기존 FAISS 실행과 같아 보이더라도 두 backend가 완전히 동일하다고
검증한 것은 아닙니다. 정식 비교는 다음 단계에서 수행합니다.

#### 다음 단계

다음 단계는 **FAISS vs pgvector 검색 결과 비교**입니다.

```text
                Query
                  ↓
          Query Embedding
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
      FAISS             pgvector
        ↓                   ↓
      Top-5               Top-5
        └─────────┬─────────┘
                  ↓
       rank / chunk / score 비교
```

### FAISS vs pgvector 비교

이번 실험은 FAISS와 pgvector 중 어느 기술이 더 좋은지 판단하기 위한 것이
아닙니다. 기존 FAISS Retrieval을 PostgreSQL + pgvector backend로 변경해도,
동일한 Vector와 cosine 기준의 exact search 조건에서 Retrieval 결과가 유지되는지
검증하는 것이 목적입니다.

비교를 재현하는 코드는
`src/rag_basic/faiss_pgvector_comparison.py`에 구현했습니다.

#### 공정한 비교 조건

PostgreSQL에 저장된 Document Embedding을 읽어 FAISS에서도 그대로 사용했습니다.

```text
PostgreSQL rag_chunks
        ↓
동일 Document Embeddings
        ↓
   ┌────┴────┐
   ↓         ↓
 FAISS    pgvector
```

FAISS용 Document Embedding을 별도로 계산하거나 다시 정규화하지 않아, Embedding
재계산에 따른 차이를 비교 변수에서 제외했습니다.

```text
shape: (162, 384)
dtype: float32

norm min: 0.99999994
norm max: 1.00000012
norm average: 1.00000000
```

기존에 정규화한 E5 Embedding이 DB에 저장된 뒤에도 L2 norm 약 1을 유지하는 것을
확인했습니다.

각 Evaluation Case에서는 Query Embedding도 정확히 한 번만 생성하고, 같은 Query
Vector를 두 backend에 전달했습니다. 모든 Query Vector의 norm도 약 1.0이었습니다.

```text
Query
↓
Query Embedding 1회
↓
├─ FAISS
└─ pgvector
```

#### 검색 방식

FAISS는 정규화된 Vector를 `IndexFlatIP`로 검색했습니다. 두 Vector의 norm이 1이면
Inner Product를 cosine similarity로 해석할 수 있으며, score가 높을수록
유사합니다.

pgvector는 `<=>` 연산자로 cosine distance를 계산합니다. Distance는 낮을수록
유사하며, 비교용 score는 다음과 같이 cosine similarity로 변환했습니다.

```text
cosine similarity = 1 - cosine distance
```

이번 실험의 FAISS `IndexFlatIP`와 index가 없는 pgvector 검색은 모두 approximate
search가 아니라 전체 후보를 비교하는 exact search입니다.

#### Top-5 비교 결과

기존 평가셋의 in-domain Case 6개에서 Top-5 `chunk_id`의 순서와 집합을 각각
비교했습니다.

```text
Exact Top-5 order match: 6/6
Top-5 set match: 6/6
```

| Case | FAISS Top-5 | pgvector Top-5 | First gold rank | RR |
| --- | --- | --- | ---: | ---: |
| `copyright_in_domain` | `[28, 39, 3, 93, 35]` | `[28, 39, 3, 93, 35]` | 1 | 1.0000 |
| `creative_contribution_copyright` | `[28, 39, 30, 33, 31]` | `[28, 39, 30, 33, 31]` | 4 | 0.2500 |
| `ai_assignment_submission` | `[69, 68, 76, 70, 75]` | `[69, 68, 76, 70, 75]` | 1 | 1.0000 |
| `midjourney_contest_controversy` | `[65, 74, 76, 53, 93]` | `[65, 74, 76, 53, 93]` | 1 | 1.0000 |
| `fake_news_damage_report` | `[104, 99, 105, 7, 8]` | `[104, 99, 105, 7, 8]` | 1 | 1.0000 |
| `generative_ai_work_benefits` | `[137, 13, 78, 138, 16]` | `[137, 13, 78, 138, 16]` | 1 | 1.0000 |

기존 gold evidence로 backend별 Retrieval 지표도 다시 계산했습니다.

```text
FAISS
Hit@5: 6/6
MRR: 0.8750

pgvector
Hit@5: 6/6
MRR: 0.8750
```

두 결과 모두 기존 FAISS baseline과 일치했습니다.

#### Score 차이

FAISS의 Inner Product score와 pgvector의 distance를 similarity로 변환한 score를
같은 순위의 동일 Chunk끼리 비교했습니다.

```text
maximum absolute difference: 0.0000001747
average absolute difference: 0.0000000474
```

모든 score 비교는 `atol=1e-5`, `rtol=1e-5` 범위에서 일치했습니다. 이는 score가
수학적으로 완전히 동일하다는 뜻이 아니라, 두 구현의 부동소수점 계산 차이
범위에서 일치했다는 의미입니다.

#### 실험 해석

```text
기존 FAISS Retrieval
↓
PostgreSQL + pgvector Retrieval로 확장
↓
현재 baseline에서 Top-5 / Hit@5 / MRR 유지 확인
```

기존 FAISS Retrieval의 검색 품질을 유지하면서 PostgreSQL 기반의 Embedding 영속
저장, metadata filtering, SQL Vector Search 구조로 확장했습니다.

다만 이는 다음 조건에서 얻은 결과에 한정됩니다.

```text
162개 baseline Chunk
multilingual-e5-small
384 dimensions
normalized Vector
Top-K=5
exact search
6개 in-domain Evaluation Case
```

이번 실험은 pgvector가 FAISS보다 정확하거나 빠르다는 것을 확인한 것이 아닙니다.
또한 모든 dataset에서 같은 결과를 반환하는지, production 환경에서 어느 쪽이 더
우수한지, 대규모 dataset에서 검색 성능이 어떤지도 검증하지 않았습니다.

#### Vector DB 단계의 현재 구조

```text
PDF
↓
Chunk
↓
Embedding
↓
PostgreSQL + pgvector
↓
Persistent Vector Storage

User Query
↓
Query Embedding
↓
SQL Vector Search
↓
Top-K Chunk
↓
build_context()
```

검색 backend 비교 결과는 다음과 같습니다.

```text
FAISS exact search
        ↕
pgvector exact search

Top-5 order: 6/6 일치
Hit@5: 동일
MRR: 동일
```

#### 아직 수행하지 않은 작업

- pgvector Retrieval + LLM Generation 최종 연결
- FastAPI
- `/ingest`
- `/query`
- HNSW 및 IVFFlat
- 대규모 성능 benchmark
- LangChain

HNSW와 IVFFlat은 현재 162개 dataset의 exact search에 필수적이지 않으므로 Vector
DB 단계의 완료 조건으로 두지 않았습니다.

#### 실행 방법

실제 password는 README에 기록하지 않고, Git에서 제외된 `.env`를 현재 shell의
환경변수로 불러옵니다.

```bash
set -a
source .env
set +a

uv run python -m rag_basic.faiss_pgvector_comparison
```

#### 다음 단계: 11. 서비스화

다음 단계에서는 아직 구현하지 않은 FastAPI endpoint를 구성할 수 있습니다.

```text
FastAPI

POST /ingest
POST /query
```

향후 목표 구조는 다음과 같습니다.

```text
Query
↓
FastAPI
↓
pgvector Retrieval
↓
Context
↓
OpenAI / qwen3:8b
↓
Answer + Source
```

## 11. 서비스화

기존 프로젝트는 필요한 Python module을 터미널에서 직접 실행하는 구조였습니다.

```text
Terminal
↓
python module
↓
Retrieval / RAG 실행
```

FastAPI를 추가하는 목적은 기존 Python 기능을 HTTP API를 통해 다른 프로그램에서도
호출할 수 있는 구조로 확장하는 것입니다. 현재 단계에서는 FastAPI 서버의 기본
동작만 확인했으며, 기존 RAG 기능은 아직 API에 연결하지 않았습니다.

### FastAPI 최소 서버

FastAPI는 Python으로 HTTP API를 만들기 위한 웹 프레임워크입니다. 이번 단계에서는
다음 Endpoint 하나만 구현했습니다.

```text
GET /health
```

응답은 다음과 같습니다.

```json
{
  "status": "ok"
}
```

현재 `/health`는 PostgreSQL, pgvector, Ollama, OpenAI 또는 GPU 상태를 검사하지
않습니다. FastAPI 프로세스가 HTTP 요청을 받아 정상적으로 응답할 수 있는지만
확인합니다.

#### FastAPI와 Uvicorn의 역할

```text
FastAPI
→ API endpoint 및 요청 처리 로직 정의

Uvicorn
→ FastAPI application을 실제 HTTP server로 실행
```

서버는 다음 명령으로 실행했습니다.

```bash
uv run uvicorn rag_basic.api:app \
  --host 127.0.0.1 \
  --port 8000
```

`127.0.0.1`은 localhost에만 bind합니다. 따라서 현재 서버는 인터넷에 공개된 API가
아니라 이 컴퓨터 안에서만 접근하는 로컬 개발 및 검증용 API입니다.

#### Dependency

이번 단계에서 다음 dependency를 추가했습니다.

```text
fastapi>=0.141.1
uvicorn[standard]>=0.52.4
```

실제 검증 당시 설치된 버전은 다음과 같습니다.

```text
fastapi==0.141.1
uvicorn==0.52.4
```

#### 최소 API 코드

최소 FastAPI application은 `src/rag_basic/api.py`에 구현했습니다.

```python
app = FastAPI(...)


@app.get("/health")
def health_check():
    return {"status": "ok"}
```

이번 단계에서는 `/health` 이외의 Endpoint를 추가하지 않았습니다.

#### 실제 HTTP 검증

실행 중인 서버에 다음 요청을 실제로 보냈습니다.

```bash
curl -i http://127.0.0.1:8000/health
```

응답:

```text
HTTP/1.1 200 OK
content-type: application/json

{"status":"ok"}
```

HTTP status `200`은 서버가 요청을 정상적으로 처리했다는 의미입니다. 검증이 끝난
뒤 테스트용 Uvicorn 프로세스를 정상 종료하고 port 8000이 해제된 것도
확인했습니다.

#### OpenAPI와 Swagger UI

FastAPI는 코드에 정의된 API 정보를 바탕으로 OpenAPI schema를 자동 생성합니다.
실제 schema에서 다음 내용을 확인했습니다.

```text
title: RAG from Scratch API
version: 0.1.0
/health path 존재: True
GET method 존재: True
```

Swagger UI는 브라우저에서 Endpoint를 확인하고 직접 요청해 볼 수 있는 자동 생성
문서 화면입니다.

```text
http://127.0.0.1:8000/docs
```

#### 현재 구조

현재 구현된 서비스 흐름은 다음과 같습니다.

```text
Client
↓
HTTP
↓
Uvicorn
↓
FastAPI
↓
GET /health
```

다음 Retrieval API 구조는 아직 구현 전입니다.

```text
POST /query
↓
FastAPI
↓
Query Embedding
↓
pgvector
↓
Top-K
```

#### 이번 단계에서 하지 않은 작업

- `POST /query`
- `POST /ingest`
- PostgreSQL 연결
- Query Embedding
- pgvector Retrieval
- `build_context()`
- OpenAI
- Ollama
- LLM Generation
- FastAPI Docker화
- 외부 인터넷 배포
- 인증
- CORS
- LangChain

#### 실행 방법

서버 실행:

```bash
uv run uvicorn rag_basic.api:app \
  --host 127.0.0.1 \
  --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok"}
```

#### 다음 단계: POST /query Retrieval API

다음 단계에서는 다음 흐름의 Retrieval API를 구현할 예정입니다.

```text
POST /query
↓
FastAPI
↓
Query Embedding
↓
PostgreSQL + pgvector
↓
Top-5 Chunk
↓
JSON Response
```

이 단계에서도 LLM 답변 생성은 연결하지 않고 Retrieval 결과를 JSON으로 반환하는
것까지만 구현할 예정입니다.

### POST /query Retrieval API

기존에는 pgvector Retrieval을 Python module에서 직접 실행해야 했습니다. 이번
단계에서는 Embedding과 SQL 검색을 새로 구현하지 않고 기존 Retrieval 함수를
FastAPI Endpoint 뒤에 연결했습니다.

```text
HTTP Request
↓
FastAPI
↓
기존 Retrieval 함수
↓
PostgreSQL + pgvector
↓
JSON Response
```

현재 `/query`는 검색 결과와 Context를 반환하는 Retrieval API입니다. OpenAI,
Ollama 또는 자연어 Answer를 생성하는 LLM은 아직 연결하지 않았습니다.

#### Endpoint와 Request Body

새 Endpoint는 다음과 같습니다.

```text
POST /query
```

기본 Request Body 구조:

```json
{
  "query": "질문",
  "top_k": 5
}
```

`query`는 검색할 질문 문자열이고, `top_k`는 반환할 검색 결과의 최대 개수입니다.

#### Pydantic 입력 및 출력 검증

Pydantic은 Request가 실제 Retrieval 로직에 들어가기 전에 입력 조건을 확인하고,
응답 구조를 OpenAPI에 명확하게 표현합니다.

`QueryRequest`의 조건:

```text
query
- 최소 1자
- 최대 1000자

top_k
- 기본값 5
- 최소 1
- 최대 20
```

`QueryResponse`는 다음 항목을 포함합니다.

```text
query
top_k
results
context
```

각 Retrieval result에는 다음 metadata와 원문이 포함됩니다.

```text
rank
score
cosine_distance
chunk_id
page_number
text
```

`api.py`에서 Pydantic을 직접 import하므로 FastAPI의 간접 dependency에만 의존하지
않고 프로젝트의 direct dependency로 명시했습니다.

```text
pydantic>=2.13.5
```

#### 기존 Retrieval 코드 재사용

API용 Embedding이나 SQL Vector Search를 다시 작성하지 않았습니다.
`rag_basic.pgvector_retrieval`에서 다음 함수를 재사용했습니다.

```text
get_database_config()
count_baseline_rows()
create_query_embedding()
search_pgvector()
```

검색 결과를 LLM에 전달할 수 있는 문자열로 합칠 때는 `rag_basic.retrieval`의
`build_context()`를 그대로 사용했습니다.

```text
POST /query
↓
create_query_embedding()
↓
search_pgvector()
↓
build_context()
↓
JSON Response
```

#### Embedding Model 재사용

Query Embedding에는 기존 `intfloat/multilingual-e5-small` 모델을 사용합니다.
SentenceTransformer 모델을 요청마다 다시 로드하지 않도록
`lru_cache(maxsize=1)`를 적용했습니다.

```text
FastAPI import
→ Model 미로드

첫 /query
→ Model 로드

이후 /query
→ 같은 Process에서 Model 재사용
```

실제 import 검증에서도 다음 결과를 확인했습니다.

```text
모델 cache size: 0
```

따라서 module import만으로 모델을 즉시 로드하지 않고, 첫 `/query` 요청에서
필요할 때 로드합니다.

#### PostgreSQL Connection

현재는 `/query` 요청마다 PostgreSQL Connection을 하나 열고 검색이 끝나면
context manager를 통해 닫습니다.

```text
/query
↓
DB Connection 생성
↓
pgvector Search
↓
Connection 종료
```

현재 학습용 규모에서는 이해하기 쉬운 단순한 구조를 우선했으며 Connection Pool은
아직 구현하지 않았습니다.

#### 실제 HTTP 검증

`/query`를 추가한 뒤 기존 Health Endpoint도 회귀 검증했습니다.

```text
GET /health

HTTP 200
{"status":"ok"}
```

기존 `copyright_in_domain` 질문을 실제 `POST /query` 요청으로 전달한 결과는 다음과
같습니다.

```text
HTTP status: 200
top_k: 5
result 수: 5
ranks: [1, 2, 3, 4, 5]
Top-5 chunk_id: [28, 39, 3, 93, 35]
context length: 2305
```

Top-5는 기존에 Python module로 직접 실행한 pgvector Retrieval 결과인
`[28, 39, 3, 93, 35]`와 동일했습니다. 현재 조건에서는 HTTP API 계층을 추가한
뒤에도 기존 Retrieval 결과가 유지되는 것을 확인했습니다.

문서 밖 질문인 `france_out_of_domain`도 실제로 요청했습니다.

```text
HTTP status: 200
result 수: 5
Top-5 chunk_id: [136, 102, 65, 108, 96]
context length: 2501
```

현재 API에는 Similarity Threshold, OOD Classifier 또는 LLM refusal이 없습니다.
따라서 문서 밖 질문에도 pgvector가 저장된 Vector 중 가장 가까운 Top-5 Chunk를
반환하는 것이 정상이며, 이 결과가 France 질문의 정답을 제공한다는 의미는
아닙니다.

#### Validation 검증

잘못된 Request도 실제 HTTP 요청으로 확인했습니다.

```text
top_k=0
→ HTTP 422

query 누락
→ HTTP 422
```

HTTP `422`는 Request 데이터가 `QueryRequest`의 조건을 만족하지 않아
FastAPI/Pydantic 검증 단계에서 거절됐다는 의미입니다.

#### 오류 처리

DB 연결이나 Retrieval 의존성에 문제가 생기면 password와 Connection String 같은
내부 정보를 HTTP Response에 직접 노출하지 않고 다음과 같은 일반적인 오류로
처리합니다.

```text
HTTP 503
Retrieval 서비스를 사용할 수 없습니다.
```

이번 단계에서는 복잡한 Exception hierarchy는 구현하지 않았습니다.

#### OpenAPI와 Swagger UI

실제 OpenAPI schema에서 다음 항목을 확인했습니다.

```text
/health GET: True
/query POST: True
QueryRequest schema: True
QueryResponse schema: True
```

Swagger UI에서도 Request와 Response 구조를 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

#### 현재 API 구조

```text
Client
↓
HTTP POST /query
↓
Uvicorn
↓
FastAPI
↓
Pydantic Validation
↓
Query Embedding
↓
PostgreSQL + pgvector
↓
Top-K Chunk
↓
build_context()
↓
JSON Response
```

이 구조에는 아직 LLM Answer Generation이 포함되어 있지 않습니다.

#### 실행 예시

실제 password는 README에 기록하지 않고 Git에서 제외된 `.env`를 shell
환경변수로 불러옵니다. 검증에서는 로컬에 캐시된 동일 모델을 사용하도록
`HF_HUB_OFFLINE=1`을 설정했습니다.

```bash
set -a
source .env
set +a

HF_HUB_OFFLINE=1 \
uv run uvicorn rag_basic.api:app \
  --host 127.0.0.1 \
  --port 8000
```

일반적인 Query 요청 예시:

```bash
curl -X POST \
  http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "질문",
    "top_k": 5
  }'
```

#### 아직 수행하지 않은 작업

- `POST /ingest`
- PDF Upload API
- OpenAI Generation
- Ollama Generation
- Answer + Source 최종 응답
- Similarity Threshold API 적용
- Reranking API 적용
- DB Connection Pool
- HNSW 및 IVFFlat
- 인증
- CORS
- FastAPI Docker화
- 외부 인터넷 배포
- LangChain

#### 다음 단계: POST /ingest

다음 단계에서는 아직 구현하지 않은 문서 적재 API를 구성할 예정입니다.

```text
Document
↓
POST /ingest
↓
FastAPI
↓
Chunk
↓
Embedding
↓
PostgreSQL + pgvector
```

`/ingest` 이후 최종 RAG 단계에서는 다음 구조로 확장할 수 있습니다.

```text
POST /query
↓
Retrieval
↓
Context
↓
OpenAI / qwen3:8b
↓
Answer + Source
```

현재 `/query`는 Retrieval 결과와 Context를 반환하는 API이며 최종 자연어 Answer
API는 아닙니다.

### POST /ingest PDF Upload API

기존에는 서버 컴퓨터에 있는 특정 PDF를 Python CLI로 적재했습니다. 이번 단계에서는
사용자가 자신의 PDF를 HTTP `multipart/form-data`로 FastAPI에 업로드하고,
PostgreSQL + pgvector에 적재할 수 있도록 확장했습니다.

```text
User PDF
↓
multipart/form-data
↓
POST /ingest
↓
Temporary PDF
↓
PDF Text Extraction
↓
Chunking
↓
Embedding
↓
PostgreSQL + pgvector
```

PDF 같은 파일은 JSON 문자열 안에 넣지 않고 `multipart/form-data`로 전송합니다.
다음 명령의 `-F` 옵션은 사용자의 로컬 PDF를 FastAPI에 전달합니다.

```bash
curl -X POST \
  http://127.0.0.1:8000/ingest \
  -F 'file=@sample.pdf;type=application/pdf'
```

업로드된 원본 PDF는 임시 파일에서 처리하며 서버에 영구 보관하지 않습니다.

#### Upload 기본 검증

- filename 필수
- `.pdf` 확장자 확인
- PDF content type 확인
- 최대 크기 20 MiB
- 경로를 제외한 basename만 `document_name`으로 사용
- 임시 파일에서 PDF 처리
- 처리 완료 후 임시 파일 삭제

20 MiB는 현재 학습용 API에서 정한 제한이며 일반적인 production 표준을 의미하지
않습니다.

#### Ingest Response

`POST /ingest` 응답에는 다음 정보가 포함됩니다.

```text
document_name
chunk_count
embedding_dimension
embedding_model
chunk_size
chunk_overlap
```

응답 크기와 정보 노출 범위를 줄이기 위해 Embedding Vector 전체와 PDF 전체 내용은
반환하지 않습니다.

#### 기존 ingestion 코드 재사용

API를 위해 Chunking, Embedding, DB INSERT 로직을 복사하지 않았습니다. 기존
`pgvector_ingest.py`를 특정 `PDF_PATH`뿐 아니라 임의의 `document_name`도 처리할
수 있도록 최소한으로 일반화하고, CLI와 FastAPI가 다음 pipeline을 함께
재사용하도록 구성했습니다.

```text
load_pages()
↓
create_chunks()
↓
embed_texts(..., "passage")
↓
validate_embeddings()
↓
ingest_rows()
↓
validate_stored_data()
```

#### Duplicate 정책

다음 metadata 조합이 이미 DB에 있으면 `HTTP 409 Conflict`를 반환합니다.

```text
document_name
embedding_model
chunk_size
chunk_overlap
```

이는 기존 데이터를 의도치 않게 섞거나 남기는 일을 피하기 위한 단순한 초기
정책입니다. overwrite, replace 및 delete 기능은 아직 구현하지 않았습니다.

#### 실제 /ingest 검증

baseline PDF의 bytes를 사용하되 multipart filename을
`upload_test_ai_ethics.pdf`로 바꾸어 실제 업로드를 검증했습니다.

```text
HTTP success
chunk_count: 162
embedding_dimension: 384
```

DB에서도 row count, NULL 존재 여부, Embedding 차원과 Chunk ID 범위를 확인했습니다.
같은 파일명으로 다시 업로드했을 때는 HTTP 409가 반환되고 row 수가 증가하지
않았습니다. 검증 후 테스트 row만 삭제하고 baseline 문서는 유지했습니다.

### Document-aware POST /query

`POST /ingest`로 새 PDF를 DB에 넣을 수 있게 된 뒤에도 기존 `/query`는 baseline
PDF만 검색하도록 고정되어 있었습니다. 따라서 업로드한 문서를 선택해서 검색할 수
없었습니다. 이를 해결하기 위해 Query Request에 필수 `document_name`을
추가했습니다.

#### Query Request

```json
{
  "document_name": "report.pdf",
  "query": "질문",
  "top_k": 5
}
```

- `document_name`: 검색할 PDF
- `query`: 검색 질문
- `top_k`: 반환할 Chunk 수

#### Metadata Filtering

Vector Search가 질문과 의미적으로 가까운 Chunk를 찾는다면, `document_name`
filter는 어느 PDF 안에서 검색할지 선택합니다.

```text
document_name으로 문서 범위 제한
              +
cosine Vector Search
              ↓
선택한 PDF 안에서 의미적으로 가까운 Chunk
```

SQL에는 문자열을 직접 조합하지 않고 parameter binding을 사용합니다.

```text
WHERE rag_chunks.document_name = %s
```

기존 `search_pgvector()`는 `PDF_PATH.name`을 고정해서 사용했지만, 이제
`document_name`을 인자로 받을 수 있습니다.

```python
search_pgvector(
    ...,
    document_name=document_name,
)
```

기본값은 기존 `PDF_PATH.name`을 유지합니다. 따라서 기존 CLI와 평가 코드가
인자를 추가하지 않아도 baseline 문서를 검색하는 동작은 유지됩니다.

#### Query Response

응답에도 `document_name`을 포함하여 어느 PDF를 검색했는지 확인할 수 있습니다.

```text
document_name
query
top_k
results
context
```

#### Baseline 회귀 검증

기존 `ai_ethics_guide.pdf`를 지정해 `/query`를 호출한 실제 결과입니다.

```text
HTTP 200
Top-5 chunk_id: [28, 39, 3, 93, 35]
context length: 2305
```

기존 pgvector CLI도 다시 실행해 다음 baseline 결과가 유지되는 것을 확인했습니다.

```text
copyright: [28, 39, 3, 93, 35]
creative contribution: [28, 39, 30, 33, 31]
France OOD: [136, 102, 65, 108, 96]
```

FAISS와 pgvector 비교 실험의 회귀 결과도 유지됐습니다.

```text
Exact Top-5 order match: 6/6

FAISS
Hit@5: 6/6
MRR: 0.8750

pgvector
Hit@5: 6/6
MRR: 0.8750
```

이는 현재 baseline 조건에서 함수 일반화로 인한 기존 검색 동작의 회귀가 없음을
확인한 결과입니다.

#### Upload → Query 연결 검증

`upload_query_test_ai_ethics.pdf`를 `/ingest`로 적재한 뒤 해당
`document_name`을 명시해 `/query`를 호출했습니다.

```text
HTTP 200
document_name: upload_query_test_ai_ethics.pdf
result count: 5
Top-5 chunk_id: [28, 39, 3, 93, 35]
context length: 2305
```

baseline과 같은 PDF bytes를 사용했기 때문에 실제 Top-5 결과도 같았습니다. 검증이
끝난 뒤 업로드 테스트 row만 삭제했으며 baseline 162개 row는 유지했습니다.

#### API 오류 검증

```text
document_name 누락
→ HTTP 422

존재하지 않는 document_name
→ HTTP 404

../../report.pdf
→ HTTP 400
```

존재하지 않는 문서의 404가 내부 서비스 오류인 503으로 바뀌지 않고 의도한 HTTP
의미를 유지하는 것도 확인했습니다.

#### OpenAPI와 Swagger UI

현재 OpenAPI에는 다음 Endpoint가 포함됩니다.

```text
GET  /health
POST /query
POST /ingest
```

`QueryRequest`와 `QueryResponse` 구조는 다음과 같습니다.

```text
QueryRequest
- document_name (required)
- query (required)
- top_k

QueryResponse
- document_name
- query
- top_k
- results
- context
```

Swagger UI에서도 같은 구조를 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

#### 서비스화 최종 구조

```text
                ┌─────────────┐
User PDF ──────→│ POST /ingest│
                └──────┬──────┘
                       ↓
                 PDF Processing
                       ↓
                    Chunk
                       ↓
                   Embedding
                       ↓
             PostgreSQL + pgvector
                       ↑
                       │
                 document_name
                       │
                ┌──────┴──────┐
User Query ────→│ POST /query │
                └──────┬──────┘
                       ↓
                Query Embedding
                       ↓
                Metadata Filter
                       ↓
                 Vector Search
                       ↓
               Top-K + Context
```

현재 API에는 자연어 Answer Generation이 포함되지 않았습니다.

#### 실행 예시

FastAPI 실행:

```bash
set -a
source .env
set +a

HF_HUB_OFFLINE=1 \
uv run uvicorn rag_basic.api:app \
  --host 127.0.0.1 \
  --port 8000
```

PDF Upload:

```bash
curl -X POST \
  http://127.0.0.1:8000/ingest \
  -F 'file=@sample.pdf;type=application/pdf'
```

문서별 Query:

```bash
curl -X POST \
  http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "document_name": "sample.pdf",
    "query": "질문",
    "top_k": 5
  }'
```

#### 현재 제한

- LLM Answer Generation
- OpenAI 및 Ollama API 연결
- Answer + Source API
- 여러 문서 동시 검색
- `document_id` 또는 UUID
- 문서 목록 및 삭제 API
- overwrite 및 replace
- Similarity Threshold API 적용
- Reranking API 적용
- HNSW 및 IVFFlat
- Connection Pool
- 인증
- CORS
- FastAPI Docker화
- 외부 인터넷 배포
- LangChain

현재 학습용 구현에서는 `document_name`을 문서 식별자로 사용합니다. 같은 파일명
관리나 문서 rename이 필요한 서비스에서는 별도 `document_id` 사용을 고려할 수
있지만 이번 단계에서는 구현하지 않았습니다.

#### 다음 단계: pgvector Retrieval + LLM Generation

다음 12단계에서는 다음 구조로 확장할 예정입니다.

```text
POST /query
↓
document_name
↓
Query Embedding
↓
pgvector Retrieval
↓
Context
↓
OpenAI / qwen3:8b
↓
Answer + Source
```

이는 다음 단계의 예정 구조이며 아직 구현된 기능은 아닙니다.

## 12. pgvector Retrieval + Local LLM Generation

기존 `/query`는 pgvector에서 관련 Chunk를 검색하고 Context를 JSON으로 반환하는
단계까지 수행했습니다. 이번 단계에서는 기존 Local LLM 코드를 재사용하여 다음
흐름을 실제 FastAPI HTTP 경로로 연결했습니다.

```text
POST /query
↓
document_name
↓
Query Embedding
↓
PostgreSQL + pgvector
↓
Top-K Retrieval
↓
Context
↓
Grounding Prompt
↓
Ollama
↓
qwen3:8b
↓
Answer + Source
```

### 최종 Generation 모델 선택

프로젝트 과정에서 `src/rag_basic/llm.py`를 통해 OpenAI 기반 RAG를 구현했고,
`local_llm.py`와 `local_rag.py`를 통해 `qwen3:8b` 기반 Local RAG도 구현했습니다.
두 Generation 방식을 기존 소규모 평가셋에서 비교해 본 뒤, 최종 FastAPI 서비스
경로에는 외부 API Key 없이 로컬에서 재현할 수 있는 `Ollama + qwen3:8b`를
선택했습니다.

이는 현재 소규모 평가와 개발 환경을 기준으로 선택한 구조입니다. `qwen3:8b`가
OpenAI와 일반적으로 동일하거나 더 우수한 성능을 갖는다는 의미는 아닙니다. 기존
OpenAI 실험 코드는 비교와 학습 기록으로 계속 유지합니다.

### 기존 Local RAG 코드 재사용

FastAPI 안에 Ollama HTTP 호출이나 Grounding Prompt를 복사하지 않았습니다.
`local_llm.py`의 다음 요소를 재사용합니다.

- `generate_local_answer()`
- `LOCAL_MODEL_NAME`
- `LocalLLMError`

Prompt는 `local_rag.py`의 `build_local_rag_prompt()`를 재사용합니다.

```text
pgvector results
↓
build_context()
↓
build_local_rag_prompt()
↓
generate_local_answer()
```

### Retrieval과 Generation의 역할

Retrieval은 질문에 답하는 데 사용할 근거 Chunk를 찾는 단계이고, Generation은
검색된 Context를 바탕으로 자연어 답변을 만드는 단계입니다. 현재 API에서도 두
역할을 분리했습니다.

동일한 `/query` 요청 안에서 Retrieval은 한 번만 수행합니다. 검색 결과로 Context를
한 번 생성하고, 그 문자열을 응답과 `qwen3:8b` Generation에서 함께 사용합니다.

### Grounding 규칙

기존 Local RAG Prompt의 다음 규칙을 그대로 사용했습니다.

- 제공된 Context만 근거로 답변
- 일반 지식으로 내용을 보충하지 않음
- 근거가 부족하면 정확히 `제공된 문서에서 확인할 수 없습니다.`라고 답변
- `[Source N]` 형태로 근거 표시
- 간결한 한국어로 답변

이번 연결 단계에서는 Prompt 자체를 새로 튜닝하지 않았습니다.

### Query Response

현재 `/query` 응답은 다음 정보를 포함합니다.

```text
document_name
query
top_k
results
context
answer
llm_model
```

`answer`는 `qwen3:8b`가 생성한 최종 자연어 답변이며, `llm_model`은 실제 사용한
모델인 `qwen3:8b`를 나타냅니다.

### Source 연결

검색 결과와 Source 번호는 다음처럼 연결됩니다.

```text
results[0] → Source 1
results[1] → Source 2
...
```

`build_context()`는 각 검색 결과를 다음 header와 함께 Context에 넣습니다.

```text
[Source 1 | page=... | chunk_id=...]
```

LLM 답변의 `[Source 1]`은 이에 대응하는 첫 번째 Retrieval result를 가리킵니다.
이번 단계에서는 별도 Citation parser나 Source 객체를 API 응답에 추가하지
않았습니다.

### Ollama 환경

실제 검증 환경은 다음과 같습니다.

```text
Ollama: 0.33.3
Model: qwen3:8b
Model size: 약 5.2 GB
Ollama API: http://127.0.0.1:11434
```

Ollama API에 실제 연결되고 모델 목록에 `qwen3:8b`가 존재하는 것을 확인했습니다.
현재 최종 FastAPI API에는 OpenAI API Key가 필요하지 않습니다.

### 기존 Local LLM 회귀 검증

다음 명령을 다시 실행하여 Retrieval이 없는 `qwen3:8b` 단독 호출이 정상
동작하는지 확인했습니다.

```bash
uv run python -m rag_basic.local_llm
```

`RAG에서 Retrieval이 필요한 이유`를 묻는 테스트에서 비어 있지 않은 한국어
답변이 생성됐습니다.

기존 FAISS 기반 Local RAG도 다시 검증했습니다.

```bash
HF_HUB_OFFLINE=1 \
uv run python -m rag_basic.local_rag
```

검증 결과:

```text
정상 질문 chunk_id: [28, 39, 3, 93, 35]
Context length: 2305
[Source] Citation 존재: True

France OOD 답변:
제공된 문서에서 확인할 수 없습니다.
정확한 refusal 문장 일치: True
```

이는 기존 FAISS 기반 Local RAG의 회귀 검증입니다. 최종 FastAPI 경로는 FAISS가
아닌 PostgreSQL + pgvector Retrieval을 사용합니다.

### FastAPI 정상 질문 검증

baseline 문서 `ai_ethics_guide.pdf`와 기존 저작권 평가 질문으로 실제 HTTP
`POST /query`를 호출했습니다.

```text
HTTP status: 200
document_name: ai_ethics_guide.pdf
top_k: 5
result count: 5
Top-5 chunk_id: [28, 39, 3, 93, 35]
context length: 2305
llm_model: qwen3:8b
answer non-empty: True
```

실제 생성된 답변은 다음과 같습니다.

```text
생성형 AI가 만든 이미지의 저작권은 일반적으로 인간의 창작물로 인정되지 않기 때문에 특정한 개인이나 기관에 소유권이 귀속되지 않습니다. [Source 1]
```

이는 현재 검색 문서와 Prompt를 기반으로 `qwen3:8b`가 생성한 검증 결과이며,
README에서 별도의 법률적 일반 사실로 해석하지 않습니다.

답변의 Citation도 다음과 같이 확인했습니다.

```text
Citation Source 번호: [1]
Source 번호 범위 유효: True
```

이 검증은 Citation 형식이 맞는지와 Source 번호가 현재 Retrieval result 범위 안에
있는지만 확인합니다. 자동 faithfulness 또는 답변 정확성 평가가 아닙니다. Source
내용이 답변을 충분히 뒷받침하는지는 후속 최종 평가에서 별도로 확인할 예정입니다.

### FastAPI OOD 검증

France OOD 질문도 같은 API 경로로 확인했습니다.

```text
HTTP status: 200
Top-5 chunk_id: [136, 102, 65, 108, 96]
context length: 2501
llm_model: qwen3:8b
answer: 제공된 문서에서 확인할 수 없습니다.
exact refusal: True
```

```text
Retrieval
→ OOD 질문에도 가장 가까운 Top-5 반환

Generation
→ Context에 답이 없으므로 refusal
```

Retrieval 결과가 존재한다는 사실만으로 그 Chunk가 질문에 대한 실제 근거라는 뜻은
아닙니다.

### HTTP 오류 회귀

Generation 연결 후에도 기존 오류 의미가 유지됐습니다.

```text
document_name 누락
→ HTTP 422

존재하지 않는 document
→ HTTP 404

../../report.pdf
→ HTTP 400
```

세 요청은 모두 Local LLM Generation 단계에 도달하기 전에 종료됩니다.

Ollama 호출 실패는 기존 `LocalLLMError`를 받아 HTTP 503의 일반적인 서비스 오류로
변환하도록 구성했습니다. 내부 오류 상세와 stack trace는 HTTP 응답에 직접
전달하지 않습니다. 이번 작업에서는 실제 Ollama 장애 상황의 HTTP 503을 별도로
검증하지 않았습니다.

### OpenAPI

현재 Endpoint는 다음과 같습니다.

```text
GET  /health
POST /ingest
POST /query
```

OpenAPI에서도 다음 `QueryResponse` 구조를 확인했습니다.

```text
document_name
query
top_k
results
context
answer
llm_model
```

### 현재 Architecture

```text
User
 │
 │ PDF
 ▼
POST /ingest
 │
 ▼
PDF Parsing
 │
 ▼
Chunking
 │
 ▼
Embedding
 │
 ▼
PostgreSQL + pgvector
 ▲
 │
 │ document_name
 │
POST /query
 │
 ▼
Query Embedding
 │
 ▼
Metadata Filtering
 │
 ▼
Vector Search
 │
 ▼
Top-K Retrieval
 │
 ▼
Context
 │
 ▼
Grounding Prompt
 │
 ▼
Ollama
 │
 ▼
qwen3:8b
 │
 ▼
Answer + [Source N]
```

현재 검증 범위에서는 PDF Upload부터 Chunking, Embedding, PostgreSQL/pgvector
Retrieval, Context, Local LLM Generation과 Source가 포함된 답변까지 실제 FastAPI
HTTP 경로로 연결했습니다. 이는 production-ready, 대규모 트래픽 지원, 완전한
보안 또는 RAG 답변 품질 보장을 의미하지 않습니다.

### 실행 조건과 방법

FastAPI를 실행하기 전에 Ollama 서비스와 `qwen3:8b`가 준비되어 있어야 합니다.

```bash
ollama list
```

DB 환경변수를 불러온 뒤 FastAPI를 실행합니다. 실제 password는 README에 기록하지
않습니다.

```bash
set -a
source .env
set +a

HF_HUB_OFFLINE=1 \
uv run uvicorn rag_basic.api:app \
  --host 127.0.0.1 \
  --port 8000
```

### 최종 RAG API Evaluation Baseline

`src/rag_basic/api_evaluation.py`에서 기존 Python 함수를 직접 평가하는 대신 다음
실제 서비스 경로 전체를 HTTP Response 기준으로 평가했습니다.

```text
Evaluation Script
↓
HTTP POST /query
↓
FastAPI
↓
PostgreSQL + pgvector
↓
Ollama qwen3:8b
↓
HTTP Response
```

기존 `EVAL_CASES`의 9개 질문과 gold evidence를 그대로 재사용했습니다.

- In-domain: 6개
- Out-of-domain: 3개

평가 결과에 맞추기 위해 질문이나 `expected_chunk_ids`를 새로 만들거나 변경하지
않았습니다.

#### Retrieval 결과

```text
Hit@5: 6/6
MRR: 0.8750
```

이는 기존 Python-level baseline의 Hit@5 `6/6`, MRR `0.8750`과 같았습니다.
최종 HTTP → FastAPI → pgvector 경로에서도 기존 baseline Retrieval 동작이
유지됐다는 의미로만 해석합니다.

Case별 실제 Top-5 `chunk_id`는 다음과 같습니다.

```text
copyright_in_domain
[28, 39, 3, 93, 35]

creative_contribution_copyright
[28, 39, 30, 33, 31]

ai_assignment_submission
[69, 68, 76, 70, 75]

midjourney_contest_controversy
[65, 74, 76, 53, 93]

fake_news_damage_report
[104, 99, 105, 7, 8]

generative_ai_work_benefits
[137, 13, 78, 138, 16]

france_out_of_domain
[136, 102, 65, 108, 96]

solar_system_out_of_domain
[146, 9, 36, 145, 155]

triangle_out_of_domain
[139, 74, 114, 75, 113]
```

#### Generation 자동 검증

```text
In-domain answer non-empty: 6/6
In-domain citation present: 5/6
In-domain citation number valid: 5/6
OOD exact refusal: 3/3
```

기존 Generation baseline의 citation `6/6`과 달리 이번 HTTP API 평가에서는
`5/6`이었습니다. Retrieval 결과가 유지됐지만 Generation 결과까지 자동으로
동일하게 유지된 것은 아닙니다.

#### Retrieval 성공, Generation 실패 Case

`ai_assignment_submission`의 Retrieval 결과는 다음과 같습니다.

```text
expected gold: [68, 69, 70]
Top-5: [69, 68, 76, 70, 75]
first gold rank: 1
```

정답 근거 검색에는 성공했고, 답변이 인용한 Source 1의 `chunk_id=69`에도 생성형
AI 결과물을 그대로 과제로 제출해서는 안 된다는 내용이 있었습니다. 그러나 생성
답변에는 다음 refusal 문장이 포함됐습니다.

```text
제공된 문서에서 확인할 수 없습니다.
```

Citation 번호 자체는 유효했지만 검색된 근거를 적절한 답변으로 활용하지 못했으므로
이 Case는 Retrieval 성공, Generation 실패 사례로 기록합니다.

`fake_news_damage_report`의 Retrieval 결과는 다음과 같습니다.

```text
expected gold: [104, 105]
Top-5: [104, 99, 105, 7, 8]
```

gold evidence가 rank 1과 rank 3에 있었지만 Generation은 다음과 같이 거절했고
Citation도 생성하지 않았습니다.

```text
제공된 문서에서 확인할 수 없습니다.
```

이 Case도 Retrieval 성공, Generation 실패 사례입니다.

#### Citation 평가의 한계

`ai_assignment_submission`은 Citation 번호가 실제 Retrieval result 범위에 있었지만
답변 내용은 적절하지 않았습니다. 따라서 이번 결과에서 다음 세 항목은 서로 같은
의미가 아님을 확인했습니다.

```text
Citation 존재 및 번호 유효
≠ Citation faithfulness
≠ Answer correctness
```

현재 자동 검증은 Citation 존재와 Source 번호 범위만 확인합니다. 자동
faithfulness 또는 answer correctness 평가가 아닙니다.

#### OOD 결과

다음 세 문서 밖 질문은 모두 지정된 문장으로 정확히 거절됐습니다.

- 프랑스의 수도
- 태양계에서 가장 큰 행성
- 삼각형 내각의 합

```text
제공된 문서에서 확인할 수 없습니다.
```

따라서 OOD exact refusal은 `3/3`이었습니다. 현재 Vector Search에는 similarity
threshold가 적용되지 않아 OOD 질문에도 nearest Top-5 Retrieval 결과가 존재할 수
있습니다. 이 평가에서는 해당 Context를 보고 Generation이 답변을 거절하는지를
확인했습니다.

#### 이번 평가에서 확인한 점

Retrieval 지표가 유지돼도 최종 Generation 품질이 자동으로 보장되지는 않습니다.
또한 단순히 답변이 비어 있지 않거나 Citation이 존재한다는 사실만으로 RAG 답변
품질을 충분히 평가하기 어렵다는 점을 실제 Case에서 확인했습니다.

### Local LLM Generation 변동성 Baseline

명시적인 Generation 설정을 추가하기 전에 동일한 서비스 입력을 반복했을 때
답변이 실제로 달라지는지 확인했습니다. 현재 `local_llm.py`의 Ollama 요청에는
`temperature`와 `seed`가 명시되어 있지 않습니다.

이번 결과만으로 temperature, seed 또는 `qwen3:8b` 자체가 변동의 원인이라고
단정하지 않습니다. 이 단계의 목적은 현재 설정에서 출력 변동이 관찰되는지를 먼저
측정하는 것입니다.

#### 평가 구조

Generation 함수를 직접 호출하지 않고 다음 최종 서비스 경로 전체를 반복
호출했습니다.

```text
generation_stability.py
↓
POST /query
↓
FastAPI
↓
PostgreSQL + pgvector
↓
Context
↓
Ollama
↓
qwen3:8b
↓
HTTP Response
```

기존 Evaluation Case에서 다음 세 질문을 선택하고 각각 5회씩, 총 15회 HTTP
요청을 실행했습니다.

- `copyright_in_domain`: 기존 정상 동작 control
- `ai_assignment_submission`: 이전 Generation 실패 Case
- `fake_news_damage_report`: 이전 Generation 실패 Case

새 질문이나 gold evidence를 만들지 않았습니다.

#### Retrieval 결과

각 Case 안에서 Top-5 Retrieval 결과는 5회 모두 같았습니다.

```text
copyright_in_domain
[28, 39, 3, 93, 35]

ai_assignment_submission
[69, 68, 76, 70, 75]

fake_news_damage_report
[104, 99, 105, 7, 8]
```

세 Case 모두 Retrieval variants는 `1`이었습니다. 따라서 이번 반복에서는
Retrieval 결과의 변동이 관찰되지 않았습니다.

#### copyright_in_domain 결과

```text
HTTP success: 5/5
Retrieval variants: 1
Unique answer count: 5
Exact refusal: 0/5
Citation present: 5/5
```

5회 모두 질문에 답하고 Citation을 생성했지만, exact string 기준으로 답변 문자열은
모두 달랐습니다. 일부 Run은 Source 1만 사용했고 다른 Run은 여러 Source를
사용했습니다. 이 차이를 자동으로 semantic quality 차이라고 판정하지는 않습니다.

#### ai_assignment_submission 결과

```text
HTTP success: 5/5
Retrieval variants: 1
Unique answer count: 3
Exact refusal: 0/5
Citation present: 5/5
```

Retrieval Top-5와 Context는 5회 동안 같았지만 Generation 결과는 달랐습니다.
Run 1~3은 질문을 반복한 뒤 다음 refusal 문구를 포함했습니다.

```text
제공된 문서에서 확인할 수 없습니다.
```

Run 4에서는 검색된 근거를 이용하여 그대로 과제로 제출해서는 안 되며 보조적으로
활용하고 최종 과제는 학습자가 완성해야 한다는 취지로 직접 답했습니다. Run 5도
refusal 문구를 포함했지만 질문과 Citation 등의 문자열이 함께 있었습니다.

현재 exact refusal은 답변 전체가 다음 문장과 정확히 같을 때만 `True`입니다.

```text
제공된 문서에서 확인할 수 없습니다.
```

예를 들어 `제공된 문서에서 확인할 수 없습니다. [Source 1]`처럼 Citation이 붙거나
질문 문장과 refusal이 함께 출력되면 exact refusal에 포함되지 않습니다. 따라서
이 Case의 `Exact refusal: 0/5`는 거절 취지의 답변이 없었다는 뜻이 아닙니다. 이번
baseline에서는 기존 metric을 변경하지 않았습니다.

#### fake_news_damage_report 결과

```text
HTTP success: 5/5
Retrieval variants: 1
Unique answer count: 4
Exact refusal: 0/5
Citation present: 5/5
```

이전 12-2A의 단일 실행에서는 지정된 문장으로 거절했지만, 이번 5회 반복에서는
모두 검색된 근거를 이용해 신고·상담 관련 답변을 생성했습니다. 따라서 한 번의
Generation 결과만으로 이 Case의 답변 행동을 대표하기 어렵다는 점이
관찰됐습니다.

#### 전체 결과

| Case | Retrieval variants | Unique answers | Exact refusal | Citation |
| --- | ---: | ---: | ---: | ---: |
| copyright_in_domain | 1 | 5 | 0/5 | 5/5 |
| ai_assignment_submission | 1 | 3 | 0/5 | 5/5 |
| fake_news_damage_report | 1 | 4 | 0/5 | 5/5 |

이번 실험에서는 동일 질문, 동일 Retrieval Top-5와 동일 Context 조건에서도
Generation answer 문자열이 여러 형태로 달라지는 현상을 관찰했습니다.

```text
Retrieval 재현성
≠ Generation 재현성
```

따라서 한 번의 Generation 결과만으로 Local RAG의 답변 행동을 평가하면 실험
결과가 흔들릴 수 있습니다.

#### 해석의 한계

이번 비교는 공백 정규화나 의미 기반 비교가 아닌 exact string comparison만
사용했습니다. 문장 표현만 조금 다르거나 의미가 사실상 같은 답변도 서로 다른
variant로 계산됩니다. 따라서 Unique answer count를 의미적으로 서로 다른 답변의
개수라고 해석할 수 없습니다.

또한 이 실험만으로 변동의 원인을 temperature, seed 또는 모델 자체라고 확정하지
않습니다.

### Generation 재현성 설정 비교

`local_llm.py`에 다음 Generation 설정을 명시했습니다.

```python
LOCAL_TEMPERATURE = 0
LOCAL_SEED = 42
```

Ollama 요청에는 다음과 같이 숫자 값으로 전달합니다.

```python
"options": {
    "temperature": LOCAL_TEMPERATURE,
    "seed": LOCAL_SEED,
}
```

`temperature = 0`은 sampling의 무작위성을 낮추기 위한 설정이고, fixed seed `42`는
동일한 조건에서 반복 비교할 때 재현성을 높이기 위한 설정입니다. 기존
`qwen3:8b`, `stream = False`, `think = False`, timeout 및 `LocalLLMError` 처리는
유지했습니다.

이 설정이 모든 시스템, GPU 및 Ollama 버전에서 100% deterministic한 출력을
보장한다는 의미는 아닙니다. 현재 로컬 실행 환경에서 실제 반복 결과를
비교했습니다.

#### Local LLM 단독 반복

Retrieval이 없는 동일 TEST_PROMPT를 세 번 실행했습니다.

```text
3회 answer exact 동일: True
```

#### 동일 HTTP 반복 실험

기존 `generation_stability.py`를 수정하지 않고 그대로 재사용해 동일한 3 Case ×
5회, 총 15회 HTTP 요청을 다시 실행했습니다.

```text
generation_stability.py
↓
POST /query
↓
FastAPI
↓
PostgreSQL + pgvector
↓
Context
↓
Ollama qwen3:8b
↓
HTTP Response
```

#### Before / After 비교

| Case | Baseline unique answers | Fixed settings unique answers | Retrieval variants | Citation |
| --- | ---: | ---: | ---: | ---: |
| copyright_in_domain | 5 | 1 | 1 | 5/5 |
| ai_assignment_submission | 3 | 1 | 1 | 5/5 |
| fake_news_damage_report | 4 | 1 | 1 | 5/5 |

세 Case 모두 Retrieval variants가 `1`, fixed settings의 Unique answers가 `1`이
됐습니다. 현재 환경에서는 명시적인 Generation 설정을 적용한 뒤 exact string
기준 반복 결과의 재현성이 개선됐습니다.

#### copyright_in_domain 결과

5회 모두 다음 Top-5를 사용했고 Generation answer도 같았습니다.

```text
Top-5: [28, 39, 3, 93, 35]
Citation: [Source 1] [Source 2] [Source 5]
Unique answers: 1
```

이는 반복 출력이 같았다는 결과입니다. 답변의 법률적 정확성이나 Citation
faithfulness를 자동으로 보장한 결과는 아닙니다.

#### ai_assignment_submission 결과

5회 모두 다음 Top-5와 같은 답변을 반환했습니다.

```text
Top-5: [69, 68, 76, 70, 75]

생성형 AI가 만든 결과물을 그대로 과제로 제출해도 되나요?
제공된 문서에서 확인할 수 없습니다. [Source 1]
```

기존 gold evidence는 Retrieval rank 1에 존재하지만 모델은 이를 적절한 답변으로
활용하지 못했습니다.

```text
재현성은 개선
Generation 품질 문제는 유지
```

즉 답변이 매번 동일하다는 사실이 답변의 올바름을 의미하지는 않습니다.

```text
답변이 매번 동일함
≠ 답변이 올바름
```

#### fake_news_damage_report 결과

5회 모두 다음 Top-5를 사용하고 같은 신고·상담센터 관련 답변을 생성했습니다.

```text
Top-5: [104, 99, 105, 7, 8]
Unique answers: 1
Citation present: 5/5
```

12-2A의 단일 실행에서는 refusal이 발생했지만, 설정을 고정한 이번 반복에서는
하나의 답변으로 고정됐습니다. 이 결과만으로 Prompt quality 문제가 완전히
해결됐다고 판단하지 않습니다.

#### 세 가지 개념의 구분

```text
Retrieval 재현성
Generation 재현성
Generation 품질
```

이번 실험에서 Retrieval 결과는 반복해도 고정됐고 Generation 재현성은 설정 전보다
개선됐습니다. 그러나 assignment Case의 Generation 품질 문제는 계속
관찰됐습니다.

이번 비교는 exact string 기준이며, 현재 환경에서 수행한 제한된 반복 실험입니다.
`temperature=0`과 `seed=42`가 모든 환경에서 완전한 determinism을 보장한다고
일반화할 수 없습니다.

### 다음 단계: Generation 실패 원인 분석

다음 12-2C 단계에서는 우선 `ai_assignment_submission` Case의 다음 내용을
확인할 예정입니다.

- 실제 Top-5 Context 전체
- gold Chunk 내용
- 실제 `build_local_rag_prompt()` 결과
- 질문, Context 및 Grounding rule 사이의 충돌 여부

아직 Prompt는 수정하지 않습니다. 원인을 먼저 확인한 뒤 최소 Prompt 변경이
필요한지 결정할 예정입니다.

## 진행 상황

- [x] PDF 로딩 및 텍스트 추출
- [x] Chunking
- [x] Embedding
- [x] FAISS 기반 Vector Search
- [x] Retrieval
- [x] LLM 연결
- [x] RAG 품질 평가
- [x] Ollama Local LLM 단독 연결
- [x] Local RAG 연결
- [x] OpenAI / Local LLM 비교
- [x] Top-K Retrieval 비교
- [x] similarity threshold 실험
- [x] Reranking 비교
- [x] Chunking configuration 비교
- [ ] 추가 Retrieval 개선
- [x] Docker 환경 구성
- [x] PostgreSQL + pgvector 실행
- [x] Chunk + Embedding DB 적재
- [x] SQL Vector Search
- [x] FAISS vs pgvector 비교
- [x] FastAPI 기본 서버
- [x] POST `/query` Retrieval API
- [x] POST `/ingest` PDF Upload API
- [x] Document-aware `/query`
- [x] pgvector Retrieval + qwen3:8b Generation
- [x] 최종 RAG API Evaluation Baseline
- [x] Local LLM Generation 변동성 Baseline
- [x] Generation 재현성 설정 비교
- [ ] Generation 실패 원인 분석

## AI 도구 활용

Codex를 코드 초안 작성과 실행 보조에 활용했습니다.

RAG의 각 단계를 직접 이해하고,
구현 결과를 실행·검증하며 필요한 수정과 해석은 직접 수행합니다.
