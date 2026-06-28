import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_and_chunk_pdf():
    # 1. 경로 설정 (scripts 폴더 기준 부모 폴더의 data/baseball_rules.pdf)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(BASE_DIR, "data", "2026_야구규칙.pdf")
    
    print(f"🔄 1. PDF 파일 탐색 중: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ 에러: '{pdf_path}' 위치에 파일이 없습니다!")
        print("💡 data/ 폴더 안에 PDF 파일을 'baseball_rules.pdf' 이름으로 넣어주세요.")
        return None

    # 2. PDF에서 텍스트 추출 (페이지 번호 매핑)
    reader = PdfReader(pdf_path)
    raw_documents = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():  # 비어있지 않은 텍스트만 추출
            raw_documents.append({
                "text": text,
                "metadata": {"page": page_num}
            })
            
    print(f"✅ PDF 텍스트 추출 완료! (총 {len(reader.pages)} 페이지 분석됨)")

    # 3. LangChain 텍스트 스플리터로 영리하게 쪼개기 (청킹)
    # chunk_size: 토막당 글자 수 (약 500자)
    # chunk_overlap: 문맥 보존을 위해 앞 토막과 겹치게 할 글자 수 (100자)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    
    final_chunks = []
    for doc in raw_documents:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            final_chunks.append({
                "text": chunk,
                "page": doc["metadata"]["page"]
            })
            
    print(f"✂️ 2. 텍스트 청킹 완료! (총 {len(final_chunks)} 개의 문서 조각 생성됨)")
    
    # 4. [요청 사항] 청킹 예시 시각화 출력 (1번 조각과 2번 조각의 오버랩 확인용)
    if len(final_chunks) >= 2:
        print("\n" + "="*60)
        print("🔍 [RAG 전처리 디버깅] 텍스트 청킹(Chunking) 연속 예시")
        print("="*60)
        
        print(f"📄 [문서 조각 1] (출처: 야구규칙백과 {final_chunks[0]['page']}p / 글자수: {len(final_chunks[0]['text'])}자)")
        print("-"*60)
        print(final_chunks[0]['text'])
        print("-"*60)
        
        print(f"\n📄 [문서 조각 2] (출처: 야구규칙백과 {final_chunks[1]['page']}p / 글자수: {len(final_chunks[1]['text'])}자)")
        print("-"*60)
        print(final_chunks[1]['text'])
        print("-"*60)
        
        print("\n💡 팁: 조각 1의 뒷부분과 조각 2의 앞부분 글자가 약 100자 정도")
        print("   중복되게 겹쳐져 있다면 문맥 보존 청킹이 완벽하게 성공한 것입니다!")
        print("="*60 + "\n")
    elif len(final_chunks) == 1:
        print("\n⚠️ 데이터 조각이 1개밖에 생성되지 않아 연속 청킹 예시를 출력할 수 없습니다.")
        print(f"📝 조각 1 내용:\n{final_chunks[0]['text']}\n")
        
    return final_chunks

if __name__ == "__main__":
    extract_and_chunk_pdf()