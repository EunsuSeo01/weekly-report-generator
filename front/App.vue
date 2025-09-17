<script setup>
import { ref, computed } from 'vue';
import axios from 'axios';
import { marked } from 'marked';

// --- 상태 관리 ---
const isLoading = ref(false); // 로딩 중인지 여부
const reportContent = ref(''); // 생성된 보고서 내용 (마크다운)
const error = ref(null); // 에러 메시지

// --- Computed 속성 ---
// 마크다운 텍스트를 HTML로 변환
const formattedReport = computed(() => {
  if (reportContent.value) {
    return marked(reportContent.value);
  }
  return '';
});

// --- 메서드 ---
const generateReport = async () => {
  // 초기화
  isLoading.value = true;
  error.value = null;
  reportContent.value = '';

  try {
    // 백엔드 API 호출 (실제 주소로 변경 필요)
    const response = await axios.post('http://localhost:3306/generate-report', {
      user_request: '이번 주(9/15~9/19) 업무보고서 초안 만들어줘', 
    });
    
    // API 응답 결과를 reportContent에 저장
    reportContent.value = response.data.report;

  } catch (err) {
    console.error("보고서 생성 중 오류 발생:", err);
    error.value = '보고서 생성에 실패했습니다. 잠시 후 다시 시도해주세요.';
  } finally {
    isLoading.value = false;
  }
};
</script>

<template>
  <div id="app-container">
    <header>
      <h1>🤖 AI 주간업무 보고서 생성기</h1>
      <p>버튼 하나로 이번 주 업무 기록을 멋진 보고서로 만들어보세요.</p>
    </header>

    <main>
      <div class="control-panel">
        <button @click="generateReport" :disabled="isLoading">
          {{ isLoading ? '생성 중...' : '이번 주 보고서 생성하기' }}
        </button>
        <p class="guide-text">
          지난 월요일부터 오늘까지의 슬랙, 노션, 원드라이브, 아웃룩 기록을 바탕으로 생성됩니다.
        </p>
      </div>

      <div class="result-panel">
        <div v-if="isLoading" class="loading-state">
          <div class="spinner"></div>
          <p>AI가 보고서를 작성하고 있습니다.<br>잠시만 기다려주세요...</p>
        </div>

        <div v-else-if="error" class="error-state">
          <p>⚠️ {{ error }}</p>
        </div>

        <div v-else-if="reportContent" class="report-view" v-html="formattedReport"></div>

        <div v-else class="initial-state">
          <p>☝️ 버튼을 눌러 보고서 생성을 시작하세요.</p>
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
#app-container {
  max-width: 800px;
  margin: 40px auto;
  padding: 20px;
  font-family: 'Pretendard', sans-serif;
  text-align: center;
  color: #333;
}

header {
  margin-bottom: 40px;
}

header h1 {
  font-size: 2.5em;
  color: #2c3e50;
}

header p {
  color: #666;
  font-size: 1.1em;
}

.control-panel button {
  background-color: #42b983;
  color: white;
  border: none;
  padding: 15px 30px;
  font-size: 1.2em;
  font-weight: bold;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.control-panel button:hover:not(:disabled) {
  background-color: #36a471;
}

.control-panel button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.guide-text {
  font-size: 0.9em;
  color: #888;
  margin-top: 15px;
}

.result-panel {
  margin-top: 40px;
  min-height: 300px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  text-align: left;
  background-color: #f9f9f9;
}

.initial-state, .error-state {
  text-align: center;
  color: #888;
  font-size: 1.2em;
}

.error-state {
  color: #e53935;
}

.loading-state {
  text-align: center;
}

.spinner {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #42b983;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.report-view {
  width: 100%;
  white-space: pre-wrap; /* 줄바꿈 및 공백 유지 */
}

/* 마크다운 렌더링 스타일 */
.report-view :deep(h1),
.report-view :deep(h2),
.report-view :deep(h3) {
  border-bottom: 1px solid #ddd;
  padding-bottom: 10px;
  margin-top: 20px;
}
.report-view :deep(ul) {
  padding-left: 20px;
}
.report-view :deep(li) {
  margin-bottom: 8px;
}
</style>