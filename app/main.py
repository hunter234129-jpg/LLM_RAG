import os
import requests
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

app = FastAPI()

# 1. 경로 및 템플릿 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))

# 2. 고성능 AI 모델 및 DB 인프라 로드
print("인프라 및 AI 모델 로드 중...")

# 1) Qdrant 연결
qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "baseball_rules"

# 2) 1단계 적재때와 동일한 768차원 임베딩 모델
bi_encoder = SentenceTransformer('intfloat/multilingual-e5-base')

# 3) [Reranker 추가] 1차 추출된 문서 조각들의 순위를 재정렬하는 크로스 인코더 모델
# 최초 실행 시 다운로드로 인해 시간이 다소 걸릴 수 있습니다.
reranker = CrossEncoder('BAAI/bge-reranker-large')

print("모든 AI 모델 및 DB 연결 완료! 서버를 가동합니다.")

@app.get("/")
async def index_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"question": "", "pure_llm": "", "rag_llm": "", "context_used": []}
    )

@app.post("/search")
async def handle_rag_query(request: Request, question: str = Form(...)):
    print(f"\n 유저 질문 입력됨: {question}")
    
    # ----------------------------------------------------
    # STEP 1: 순수 LLM 답변 (야구 규칙 데이터가 없을 때)
    # ----------------------------------------------------
    pure_llm_answer = "Ollama 통신 실패"
    try:
        ollama_url = "http://localhost:11434/api/generate"
        pure_payload = {
            "model": "llama3",
            "prompt": question,
            "stream": False
        }
        res = requests.post(ollama_url, json=pure_payload)
        if res.status_code == 200:
            pure_llm_answer = res.json().get("response", "")
    except Exception as e:
        print(f" 순수 LLM 요청 실패: {e}")

    # ----------------------------------------------------
    # STEP 2: Qdrant 고밀도 벡터 검색 (1차 Dense Retrieval)
    # ----------------------------------------------------
    # Multilingual-E5 규칙에 따라 검색 질문 앞에는 'query: ' 프리픽스를 붙여줍니다.
    query_vector = bi_encoder.encode(f"query: {question}").tolist()
    
    # 여유 있게 관련 문서 조각 10개를 1차로 긁어옵니다.
    search_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=10
    )
    search_results = search_response.points
    print("=" * 50)
    print("search_results type:", type(search_results))
    print("search_results:", search_results)

    for i, item in enumerate(search_results):
        print(f"{i}: {type(item)}")
        print(item)
    print("=" * 50)

    # ----------------------------------------------------
    # STEP 3: 리랭커(Reranker) 작동 (2차 Re-ranking 및 엄선)
    # ----------------------------------------------------
    # 크로스 인코더에 입력할 형태로 데이터 페어(Pair) 조립: [[질문, 문서1], [질문, 문서2], ...]
    rerank_pairs = [[question, hit.payload["text"]] for hit in search_results]
    
    # 각 페어별 매칭 점수 계산
    rerank_scores = reranker.predict(rerank_pairs)
    
    # 검색 조각 결과에 리랭커 점수를 매핑
    for idx, score in enumerate(rerank_scores):
        search_results[idx].score = float(score)
        
    # 리랭킹 점수가 높은 순으로 역정렬(내림차순)한 뒤, 진짜 알짜배기 상위 3개만 엄선합니다.
    search_results.sort(key=lambda x: x.score, reverse=True)
    top_k_chunks = search_results[:3]
    
    # LLM 프롬프트에 주입할 컨텍스트 문자열 생성
    context_str = "\n\n".join([f"[참고 규칙 {i+1}]: {hit.payload['text']}" for i, hit in enumerate(top_k_chunks)])
    
    # 웹 화면 하단에 디버깅용으로 띄워줄 컨텍스트 리스트 데이터 정돈
    context_used = [
        {"text": hit.payload["text"], "page": hit.payload.get("page", "알수없음"), "score": round(hit.score, 4)} 
        for hit in top_k_chunks
    ]

    # ----------------------------------------------------
    # STEP 4: 지식 주입형 RAG LLM 답변 생성
    # ----------------------------------------------------
    rag_llm_answer = "Ollama 통신 실패"
    
    rag_prompt = f"""당신은 공식 야구 규칙에 기반하여 답변하는 전문 봇입니다. 
아래 제공된 [야구 규칙 데이터]를 기반으로 사용자의 질문에 정확하게 답변하세요. 
주어진 데이터에 없는 내용은 유추해서 지어내지 말고, 모른다면 솔직하게 모른다고 답하세요.
반드시! 무조건! 한국어(Korean)로만 자연스럽게 답변하세요. 영어로 답변하지 마세요.

[야구 규칙 데이터]
{context_str}

[사용자 질문]
{question}

답변:"""

    try:
        rag_payload = {
            "model": "llama3",
            "prompt": rag_prompt,
            "stream": False
        }
        res = requests.post(ollama_url, json=rag_payload)
        if res.status_code == 200:
            rag_llm_answer = res.json().get("response", "")
    except Exception as e:
        print(f" RAG LLM 요청 실패: {e}")

    # ----------------------------------------------------
    # STEP 5: 연동 결과를 대시보드 템플릿에 바인딩하여 렌더링
    # ----------------------------------------------------

    print("=" * 50)
    print("context_used")
    print(context_used)
    print("=" * 50)
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "question": question,
            "pure_llm": pure_llm_answer,
            "rag_llm": rag_llm_answer,
            "context_used": context_used
        }
    )