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

## 진행 상황

- [x] PDF 로딩 및 텍스트 추출
- [x] Chunking
- [ ] Embedding
- [ ] FAISS 기반 Vector Search
- [ ] Retrieval
- [ ] LLM 연결
- [ ] RAG 품질 평가

## AI 도구 활용

Codex를 코드 초안 작성과 실행 보조에 활용했습니다.

RAG의 각 단계를 직접 이해하고,
구현 결과를 실행·검증하며 필요한 수정과 해석은 직접 수행합니다.