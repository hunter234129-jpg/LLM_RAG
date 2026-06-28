import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 1단계에서 만든 청킹 함수 가져오기
from extract_and_chunk import extract_and_chunk_pdf

def upload_data_to_qdrant():
    # --- [STEP 1] 1단계 청킹 데이터 불러오기 ---
    chunks = extract_and_chunk_pdf()
    if not chunks:
        print("에러: 청킹된 데이터가 없어 Qdrant 적재를 중단.")
        return

    # --- [STEP 2] 로컬 인프라 및 모델 준비 ---
    print("\n🔄 2. 한국어 임베딩 모델 로드 중 (최초 실행 시 다운로드로 인해 시간이 조금 걸림)...")
    # 한국어를 지원하는 강력한 오픈소스 고성능 임베딩 모델, 문장의 의미적 표현을 벡터 공간에 매핑하는 데 효과적
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')
    
    # 이 모델이 생성하는 벡터의 차원 수는 768차원입니다.
    VECTOR_SIZE = 768 
    COLLECTION_NAME = "baseball_rules"

    print("🔄 3. 로컬 Qdrant DB 연결 중 (localhost:6333)...")
    # 로컬에 실행 중인 Qdrant 서버와 통신할 클라이언트 객체 생성, 데이터베이스명시X 따라서 default데이터베이스 연결
    qdrant_client = QdrantClient(host="localhost", port=6333)

    # --- [STEP 3] Qdrant 컬렉션(테이블) 생성 ---
    # 기존에 똑같은 컬렉션이 있다면 초기화하고 새로 만듭니다.
    try:
        if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"기존에 존재하는 '{COLLECTION_NAME}' 컬렉션을 삭제합니다.")
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        
        print(f"🆕 '{COLLECTION_NAME}' 컬렉션을 새로 생성합니다 (차원: {VECTOR_SIZE}).")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    except Exception as e:
        print(f"Qdrant 연결 또는 컬렉션 생성 실패! Qdrant 프로세스가 켜져 있는지 확인하세요. 에러: {e}")
        return

    # --- [STEP 4] 텍스트 임베딩 및 DB 적재 (Upsert) ---
    print(f"{len(chunks)}개의 텍스트 조각 임베딩 및 DB 업로드 시작...")
    
    points = []
    for idx, item in enumerate(chunks):
        # Multilingual-E5 모델의 특성상, 검색 대상이 되는 문서 본문 앞에는 
        # 'passage: ' 라는 프리픽스를 붙여주면 검색 정확도가 극대화됩니다.
        formatted_text = f"passage: {item['text']}"
        
        # 텍스트를 768차원의 숫자 배열(벡터)로 변환 numpy -> list
        vector = embedding_model.encode(formatted_text).tolist()
        
        # Qdrant에 저장할 데이터 규격(Point) 조립
        point = PointStruct(
            id=idx,                                 # 고유 ID (숫자)
            vector=vector,                          # 임베딩 벡터
            payload={                               # 함께 저장할 원본 데이터 및 메타데이터
                "text": item['text'],
                "page": item['page']
            }
        )
        points.append(point)

    # Qdrant DB에 한방에 밀어 넣기 (Batch Upload)
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points
    )
    
    print(f"총 {len(points)}개의 야구 규칙 벡터가 Qdrant 로컬 DB(`qdrant_data/`)에 안전하게 저장되었습니다.")

if __name__ == "__main__":
    upload_data_to_qdrant()