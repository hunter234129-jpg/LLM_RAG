import os
import requests
import uuid
from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from qdrant_client import AsyncQdrantClient
from fastapi.responses import StreamingResponse, RedirectResponse
from sentence_transformers import SentenceTransformer, CrossEncoder
import httpx
import bcrypt
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv() 

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True, pool_recycle=3600)
redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
qdrant_client = AsyncQdrantClient(host="localhost", port=6333)
async_http_client = httpx.AsyncClient(base_url="http://localhost:11434", timeout=60.0)

# FastAPI Lifespan 자원 관리 (서버 온/오프시 안정적으로 커넥션 닫기)
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await async_http_client.get('/')
    except Exception:
        print("Ollama 서버가 실행 중인지 확인하세요. (http://localhost:11434)")
    app.state.bi_encoder = SentenceTransformer('intfloat/multilingual-e5-base')
    app.state.reranker = CrossEncoder('BAAI/bge-reranker-large')
    print("모든 AI 모델 및 DB 연결 완료! 서버를 가동합니다.")
    yield
    print("[인프라 셧다운] Redis, Qdrant, Ollama 서버 연결 종료 중... 자원 회수")
    await engine.dispose()
    await redis_client.close()
    await qdrant_client.close()
    await async_http_client.aclose()
    print("[인프라 셧다운] 모든 연결 종료 완료. 서버 종료.")

app = FastAPI(lifespan=lifespan) # 서버클라이언트들 연결된채 종료 방지

# 경로 및 템플릿 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app", "static")))

COLLECTION_NAME = "baseball_rules"

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

async def get_current_user(request: Request):
    session_token = request.cookies.get("session_token")
    if not session_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다.")    
    user_id = await redis_client.get(session_token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="세션이 만료되었습니다. 다시 로그인하세요.")
    
    return int(user_id)

@app.post("/auth/signup")
async def signup(username: str = Form(...), password: str = Form(...)):
    hashed = hash_password(password)
    async with AsyncSession(engine) as session:
        async with session.begin():
            result = await session.execute(text("SELECT id FROM users WHERE username = :username"), {"username": username})
            if result.scalar():
                raise HTTPException(status_code=400, detail="이미 존재하는 사용자 이름입니다.")
            await session.execute(
                text("INSERT INTO users (username, hashed_password) VALUES (:username, :hashed)"),
                {"username": username, "hashed": hashed}
            )
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    async with AsyncSession(engine) as session:
        result = await session.execute(
            text("SELECT id, hashed_password FROM users WHERE username = :username"),
            {"username": username}
        )
        user = result.fetchone()

        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=400, detail="사용자 이름 또는 비밀번호가 잘못되었습니다.")
        
        user_id = user.id
    
    session_token = str(uuid.uuid4())
    await redis_client.set(f"session:{session_token}", user_id, ex=3600)

    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="session_token", value=session_token, httponly=True)
    return response

@app.post("/auth/logout")
async def logout(request: Request):
    session_token = request.cookies.get("session_token")
    if session_token:
        await redis_client.delete(f"session:{session_token}")
    response = RedirectResponse(url="/")
    response.delete_cookie("session_token")
    return response

@app.get("/")
async def index_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"session_id": None}
    )

@app.get("/chat/{session_id}")
async def chat_room_page(session_id: int, request: Request, user_id: int = Depends(get_current_user)):
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"session_id": session_id}
    )

@app.get("api/sessions")
async def get_user_sessions(user_id: int = Depends(get_current_user)):
    async with AsyncSession(engine) as session:
        result = await session.execute(
            text("SELECT id, session_title, created_at FROM chat_sessions WHERE user_id = :user_id ORDER BY created_at DESC"),
            {"user_id": user_id}
        )
        sessions_list = result.fetchall()

        return [
            {"id": s.id, "session_title": s.session_title, "created_at": s.created_at.strftime("%Y-%m-%d %H:%M")}
            for s in sessions_list
        ]
    
@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: int, user_id: int = Depends(get_current_user)):
    async with AsyncSession(engine) as session:
        session_check = await session.execute(
            text("SELECT id FROM chat_sessions WHERE id = :session_id AND user_id = :user_id"),
            {"session_id": session_id, "user_id": user_id}
        )
        if not session_check.scalar():
            raise HTTPException(status_code=403, detail="이 대화방에 접근할 권한이 없습니다.")

        result = await session.execute(
            text("SELECT question, answer, created_at FROM chat_messages WHERE session_id = :session_id ORDER BY created_at ASC"),
            {"session_id": session_id}
        )
        messages_list = result.fetchall()
        
        return [
            {"question": m.question, "answer": m.answer, "created_at": m.created_at.strftime("%Y-%m-%d %H:%M")}
            for m in messages_list
        ]

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

@app.get("/api/stream/rag")
async def stream_rag_llm(question: str, request: Request, session_id: int = None, user_id: int = Depends(get_current_user)):
    bi_encoder = request.app.state.bi_encoder
    reranker = request.app.state.reranker

    # 1. Qdrant 벡터 검색
    query_vector = bi_encoder.encode(f"query: {question}").tolist()
    search_response = await qdrant_client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=10)
    search_results = search_response.points
    
    # 2. CrossEncoder 리랭킹
    rerank_pairs = [[question, hit.payload["text"]] for hit in search_results]
    rerank_scores = reranker.predict(rerank_pairs)
    for idx, score in enumerate(rerank_scores):
        search_results[idx].score = float(score)
    
    search_results.sort(key=lambda x: x.score, reverse=True)
    top_k_chunks = search_results[:3]

    # 3. LLM에 주입할 컨텍스트 문자열만 결합 (프론트 전달용 리스트 생성을 통째로 삭제! 🌟)
    context_str = "\n\n".join([f"[참고 규칙 {i+1}]: {hit.payload['text']}" for i, hit in enumerate(top_k_chunks)])

    rag_prompt = f"""당신은 공식 야구 규칙에 기반하여 답변하는 전문 봇입니다. 
아래 제공된 [야구 규칙 데이터]를 기반으로 사용자의 질문에 정확하게 답변하세요. 
주어진 데이터에 없는 내용은 유추해서 지어내지 말고, 모른다면 솔직하게 모른다고 답하세요.
반드시 한국어(Korean)로만 답변하세요.

[야구 규칙 데이터]
{context_str}

[사용자 질문]
{question}

답변:"""
    
    async def rag_stream_wrapper():
        # ❌ yield f"data: {json.dumps({'context_used': ...})}\n\n" <- 첫 줄 출처 전송 코드 삭제!
        
        full_answer = ""
        current_sess_id = session_id
        
        # 4. Ollama 답변 스트리밍 실행
        async for text_chunk in generate_ollama_stream(rag_prompt):
            full_answer += text_chunk
            # 만약 헬퍼 함수가 이미 'data: ...' 포맷팅을 해서 yield 한다면 그대로 뱉어줍니다.
            yield text_chunk 
            
        # 5. 스트리밍 완료 후 MariaDB 대화 히스토리 보존 처리
        async with AsyncSession(engine) as db_session:
            async with db_session.begin():
                if not current_sess_id:
                    title = question[:20] + "..." if len(question) > 20 else question
                    res = await db_session.execute(
                        text("INSERT INTO chat_sessions (user_id, session_title) VALUES (:user_id, :title)"),
                        {"user_id": user_id, "title": title}
                    )
                    current_sess_id = res.lastrowid
                    
                    # 신규 방인 경우 프론트엔드가 주소창을 이동할 수 있게 새 방 고유 ID 전송
                    yield f"data: {json.dumps({'new_session_id': current_sess_id})}\n\n"
                
                await db_session.execute(
                    text("INSERT INTO chat_messages (session_id, question, answer) VALUES (:session_id, :q, :a)"),
                    {"session_id": current_sess_id, "q": question, "a": full_answer}
                )
                print(f"[연동 성공] 유저 {user_id}의 대화 조각이 세션 {current_sess_id}번에 무사히 누적되었습니다.")
                
    return StreamingResponse(rag_stream_wrapper(), media_type="text/event-stream")