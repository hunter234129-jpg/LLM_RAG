import os
import requests
import json
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer, CrossEncoder
import httpx

app = FastAPI()

# 1. 경로 및 템플릿 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app", "static")))

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

async_http_client = httpx.AsyncClient(base_url="http://localhost:11434", timeout=60.0)

print("모든 AI 모델 및 DB 연결 완료! 서버를 가동합니다.")

@app.get("/")
async def index_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"question": "", "pure_llm": "", "rag_llm": "", "context_used": []}
    )

async def generate_ollama_stream(prompt: str, model: str = "llama3"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.3
        }
    }
    try:
        async with async_http_client.stream("POST", "/api/generate", json=payload) as response:
            async for chunk in response.aiter_lines():
                if chunk:
                    data = json.loads(chunk)
                    text_chunk = data.get("response", "")
                    yield f"data: {json.dumps({'text': text_chunk})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'text': f' [에러 발생]: {str(e)}'})}\n\n"

@app.get("/api/stream/pure")
async def stream_pure_llm(question: str):
    korean_prompt = f"다음 질문에 대해 반드시 한국어(Korean)로만 답변해 주세요.\n질문: {question}"
    return StreamingResponse(generate_ollama_stream(korean_prompt), media_type="text/event-stream")

@app.get("/api/stream/rag")
async def stream_rag_llm(question: str):
    query_vector = bi_encoder.encode(f"query: {question}").tolist()
    search_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=10
    )
    search_results = search_response.points
    rerank_pairs = [[question, hit.payload["text"]] for hit in search_results]
    rerank_scores = reranker.predict(rerank_pairs)
    for idx, score in enumerate(rerank_scores):
        search_results[idx].score = float(score)
    
    search_results.sort(key=lambda x: x.score, reverse=True)
    top_k_chunks = search_results[:3]

    context_str = "\n\n".join([f"[참고 규칙 {i+1}]: {hit.payload['text']}" for i, hit in enumerate(top_k_chunks)])
    context_used = [
        {"text": hit.payload["text"], "page": hit.payload.get("page", "알수없음"), "score": round(hit.score, 4)} 
        for hit in top_k_chunks
    ]

    rag_prompt = f"""당신은 공식 야구 규칙에 기반하여 답변하는 전문 봇입니다. 
아래 제공된 [야구 규칙 데이터]를 기반으로 사용자의 질문에 정확하게 답변하세요. 
주어진 데이터에 없는 내용은 유추해서 지어내지 말고, 모른다면 솔직하게 모른다고 답하세요.
반드시! 무조건! 한국어(Korean)로만 자연스럽게 답변하세요. 영어로 답변하지 마세요.

[야구 규칙 데이터]
{context_str}

[사용자 질문]
{question}

답변:"""
    
    async def rag_stream_wrapper():
        yield f"data: {json.dumps({'context_used': context_used})}\n\n"
        async for text_chunk in generate_ollama_stream(rag_prompt):
            yield text_chunk
    
    return StreamingResponse(rag_stream_wrapper(), media_type="text/event-stream")