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

현재 단계에서는 아직 Chunking, Embedding, Vector DB, LLM을 구현하지 않았습니다.

## 실행 방법

```bash
uv run python -m rag_basic.pdf_loader
```

## 진행 상황

- [x] PDF 로딩 및 텍스트 추출
- [ ] Chunking
- [ ] Embedding
- [ ] FAISS 기반 Vector Search
- [ ] Retrieval
- [ ] LLM 연결
- [ ] RAG 품질 평가

## AI 도구 활용

Codex를 코드 초안 작성과 실행 보조에 활용했습니다.

RAG의 각 단계를 직접 이해하고,
구현 결과를 실행·검증하며 필요한 수정과 해석은 직접 수행합니다.