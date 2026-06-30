function startStreamingSearch(event) {
    // 1. 폼 제출 시 브라우저가 화면을 새로고침하는 기본 본능 차단
    event.preventDefault();

    const question = document.getElementById("questionInput").value;
    
    // UI 전환: 플레이스홀더 숨기고 실제 대시보드 그리드 컴포넌트 활성화
    document.getElementById("welcomePlaceholder").style.display = "none";
    document.getElementById("mainDashboard").style.display = "block";

    // 결과 출력 박스 포인터 스코프 확보
    const pureContentBox = document.getElementById("pureResultContent");
    const ragContentBox = document.getElementById("ragResultContent");
    const contextListBox = document.getElementById("contextResultList");

    // 매 검색 시 이전 기록 데이터 초기화 세팅
    pureContentBox.innerText = "답변을 실시간으로 생성 중...";
    ragContentBox.innerText = "참고 규칙 검색 및 답변 생성 중...";
    contextListBox.innerHTML = "";

    // ==========================================
    // 🟦 1. 순수 LLM 스트리밍 채널 파이프라인 수신
    // ==========================================
    const pureSource = new EventSource(`/api/stream/pure?question=${encodeURIComponent(question)}`);

    pureSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (pureContentBox.innerText === "답변을 실시간으로 생성 중...") {
            pureContentBox.innerText = "";
        }
        // 화면 새로고침 없이 문자열 실시간 누적 연산
        pureContentBox.innerText += data.text;
    };

    pureSource.onerror = function() {
        pureSource.close(); // 연결 수명 만료 시 브라우저 커넥션 반환
    };

    // ==========================================
    // 🟩 2. RAG 기반 LLM 스트리밍 채널 파이프라인 수신
    // ==========================================
    const ragSource = new EventSource(`/api/stream/rag?question=${encodeURIComponent(question)}`);

    ragSource.onmessage = function(event) {
        const data = JSON.parse(event.data);

        // 첫 패킷으로 엄선된 Context 출처 정보가 들어왔을 때의 분기 처리
        if (data.context_used) {
            contextListBox.innerHTML = ""; // 기존 대기 메시지 클리어
            
            // 상위 3개 문서 데이터를 순회하며 동적으로 HTML 엘리먼트 카드 노출
            data.context_used.forEach((item, index) => {
                const itemHtml = `
                    <div class="context-item">
                        <div class="context-meta">
                            <span>📄 출처: 야구규칙백과 PDF - ${item.page} 페이지</span>
                            <span class="score-badge">Rerank 정밀 매칭 점수: ${item.score}</span>
                        </div>
                        <div style="font-size: 14px; line-height: 1.5; color: #334155;">${item.text}</div>
                    </div>
                `;
                contextListBox.innerHTML += itemHtml;
            });
            return; // 텍스트 렌더링으로 넘어가지 않고 종료
        }

        // 이후 연속 수신되는 대사 텍스트 조각 렌더링 분기
        if (data.text) {
            if (ragContentBox.innerText === "참고 규칙 검색 및 답변 생성 중...") {
                ragContentBox.innerText = "";
            }
            ragContentBox.innerText += data.text;
        }
    };

    ragSource.onerror = function() {
        ragSource.close(); // 채널 안전 종료
    };
}