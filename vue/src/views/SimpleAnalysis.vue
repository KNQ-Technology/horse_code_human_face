<script setup lang="ts">
import { ref, computed } from 'vue';
import { useRoute } from 'vue-router';
import { Upload, FileVideo, CheckCircle, Loader2, PlayCircle, History, Plus, Info, Camera, CameraOff } from 'lucide-vue-next';
import axios from 'axios';

const route = useRoute();

const API_BASE = import.meta.env.VITE_API_BASE || window.location.origin;

const fileInput = ref<HTMLInputElement | null>(null);
const videoFile = ref<File | null>(null);
const videoUrl = ref<string | null>(null);
const processedVideoUrl = ref<string | null>(null);
const videoMode = ref<'original' | 'processed'>('original');
const uploadProgress = ref(0);
const processingProgress = ref(0);
const processingStatus = ref<'idle' | 'uploading' | 'processing' | 'completed' | 'error'>('idle');
const processingResult = ref<any>(null);
const taskId = ref<string | null>(null);
const errorMessage = ref<string>('');

const simpleDetectionsFiltered = computed(() => {
  const list = processingResult.value?.detections;
  if (!Array.isArray(list)) return [];
  return list.filter((item: { horse_id?: string }) => !!item.horse_id);
});

const handleFileUpload = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files[0]) {
    videoFile.value = target.files[0];
    videoUrl.value = URL.createObjectURL(videoFile.value);
    processedVideoUrl.value = null;
    videoMode.value = 'original';
    uploadProgress.value = 0;
    processingProgress.value = 0;
    processingStatus.value = 'idle';
    processingResult.value = null;
    errorMessage.value = '';
  }
};

let statusRetryCount = 0;
const MAX_STATUS_RETRIES = 5;

const checkStatus = async () => {
  if (!taskId.value) return;

  try {
    const response = await axios.get(`${API_BASE}/api/status?task_id=${taskId.value}`);
    const { code, data } = response.data;
    statusRetryCount = 0;

    if (code === 200) {
      processingProgress.value = data.progress;
      processingStatus.value = data.status;

      if (data.status === 'completed') {
        const videoPath = data.processed_video_url as string;
        processedVideoUrl.value = videoPath.startsWith('http') ? videoPath : `${API_BASE}${videoPath}`;
        videoMode.value = 'processed';
        processingResult.value = data.result;
      } else if (data.status === 'error') {
        errorMessage.value = data.message || '后端处理时出错';
      } else if (data.status === 'processing' || data.status === 'uploading') {
        setTimeout(checkStatus, 1000);
      }
    } else if (code === 404) {
      errorMessage.value = '任务不存在，可能后端已重启';
      processingStatus.value = 'error';
    }
  } catch (error) {
    console.error('Status check failed', error);
    statusRetryCount++;
    if (statusRetryCount < MAX_STATUS_RETRIES) {
      setTimeout(checkStatus, 2000);
    } else {
      errorMessage.value = '无法连接后端服务，请检查后端是否正在运行';
      processingStatus.value = 'error';
    }
  }
};

const resetUpload = () => {
  videoFile.value = null;
  videoUrl.value = null;
  processedVideoUrl.value = null;
  videoMode.value = 'original';
  uploadProgress.value = 0;
  processingProgress.value = 0;
  processingStatus.value = 'idle';
  processingResult.value = null;
  taskId.value = null;
  errorMessage.value = '';
};

const startUpload = async () => {
  if (!videoFile.value) return;

  processingStatus.value = 'uploading';
  uploadProgress.value = 0;

  const formData = new FormData();
  formData.append('video', videoFile.value);
  formData.append('mode', 'simple');

  try {
    const response = await axios.post(`${API_BASE}/api/upload`, formData, {
      onUploadProgress: (progressEvent) => {
        uploadProgress.value = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
      }
    });

    if (response.data.code === 200) {
      taskId.value = response.data.data.task_id;
      checkStatus();
    } else {
      processingStatus.value = 'error';
    }
  } catch (error) {
    console.error('Upload failed', error);
    processingStatus.value = 'error';
  }
};
</script>

<template>
  <div class="page-content">
    <section class="guide-module">
      <div class="guide-header">
        <h2 class="guide-title">机位二（通道内部）· 号码识别</h2>
        <nav class="mode-tabs">
          <router-link to="/" class="mode-tab" :class="{ active: route.path === '/' }">
            骑手识别
          </router-link>
          <router-link to="/simple" class="mode-tab" :class="{ active: route.path === '/simple' }">
            号码识别
          </router-link>
        </nav>
      </div>

      <p class="guide-desc">
        <Info :size="16" class="guide-desc-icon" />
        上传通道侧向拍摄视频，AI 将自动识别画面中马匹鞍垫上的号码。
      </p>

      <div class="guide-steps">
        <div class="step">
          <span class="step-num">1</span>
          <span>在下方选择或拖拽视频上传</span>
        </div>
        <div class="step-arrow">→</div>
        <div class="step">
          <span class="step-num">2</span>
          <span>点击「开始上传处理」</span>
        </div>
        <div class="step-arrow">→</div>
        <div class="step">
          <span class="step-num">3</span>
          <span>右侧查看识别到的鞍垫号码</span>
        </div>
      </div>

      <div class="guide-examples">
        <div class="example good">
          <img src="/examples/num-good-1.png" alt="清晰侧面" />
          <div class="example-label">
            <Camera :size="14" />
            <span>清晰侧面 · 号码可见</span>
          </div>
        </div>
        <div class="example bad">
          <img src="/examples/num-bad-1.png" alt="遮挡严重" />
          <div class="example-label">
            <CameraOff :size="14" />
            <span>马匹遮挡 · 号码不可见</span>
          </div>
        </div>
        <div class="example bad">
          <img src="/examples/num-bad-2.png" alt="光线问题" />
          <div class="example-label">
            <CameraOff :size="14" />
            <span>光线不足 · 画面偏暗</span>
          </div>
        </div>
        <div class="example bad">
          <img src="/examples/num-bad-3.png" alt="角度不对" />
          <div class="example-label">
            <CameraOff :size="14" />
            <span>拍摄角度偏 · 号码变形</span>
          </div>
        </div>
      </div>

      <div class="guide-notice">
        ⚠ 要求画面清晰无遮挡 · 建议视频时长 &lt; 20s，文件 &lt; 100MB · 分辨率不低于 1080p
      </div>
    </section>

    <div class="content-grid">
      <section class="panel preview-panel">
        <div class="panel-header">
          <h2 class="panel-title">视频上传与预览</h2>
        </div>

        <div class="panel-body">
          <div
            class="upload-area"
            :class="{ 'has-file': videoFile }"
            @click="fileInput?.click()"
          >
            <input
              type="file"
              ref="fileInput"
              accept="video/*"
              @change="handleFileUpload"
              style="display: none"
            />

            <div v-if="!videoFile" class="upload-placeholder">
              <div class="upload-icon-wrapper">
                <Upload :size="40" class="icon" />
              </div>
              <p class="upload-text">点击或拖拽视频到此处上传</p>
              <p class="upload-hint">支持 MP4, AVI, MOV 等格式</p>
            </div>

            <div v-else class="video-preview-container">
              <div class="video-controls-overlay">
                <div class="video-mode-toggle" @click.stop>
                  <button
                    :class="{ active: videoMode === 'original' }"
                    @click="videoMode = 'original'"
                  >
                    <History :size="14" />
                    原视频
                  </button>
                  <button
                    :class="{ active: videoMode === 'processed', disabled: !processedVideoUrl }"
                    :disabled="!processedVideoUrl"
                    @click="videoMode = 'processed'"
                  >
                    <PlayCircle :size="14" />
                    处理后视频 {{ !processedVideoUrl ? '(等待中)' : '' }}
                  </button>
                </div>
              </div>

              <video
                v-if="videoMode === 'original' && videoUrl"
                :src="videoUrl"
                key="original"
                controls
                class="video-preview"
              ></video>
              <video
                v-if="videoMode === 'processed' && processedVideoUrl"
                :src="processedVideoUrl"
                key="processed"
                controls
                autoplay
                class="video-preview"
              ></video>

              <div class="file-tag">
                <FileVideo :size="14" />
                <span>{{ videoMode === 'original' ? videoFile.name : 'Processed_' + videoFile.name }}</span>
              </div>
            </div>
          </div>

          <button
            class="primary-button"
            :class="{ 'secondary': processingStatus === 'completed' }"
            :disabled="!videoFile || (processingStatus !== 'idle' && processingStatus !== 'completed')"
            @click="processingStatus === 'completed' ? resetUpload() : startUpload()"
          >
            <template v-if="processingStatus === 'idle'">
              <Upload :size="20" />
              <span>开始上传处理</span>
            </template>
            <template v-else-if="processingStatus === 'completed'">
              <Plus :size="20" />
              <span>上传新视频</span>
            </template>
            <template v-else>
              <Loader2 class="animate-spin" :size="20" />
              <span>正在执行 AI 分析...</span>
            </template>
          </button>
        </div>
      </section>

      <section class="panel status-panel">
        <div class="panel-header">
          <h2 class="panel-title">鞍垫号码识别结果</h2>
        </div>

        <div class="panel-body">
          <div class="status-section">
            <div class="progress-card">
              <div class="progress-group">
                <div class="progress-label">
                  <span>上传进度</span>
                  <span class="percent">{{ uploadProgress }}%</span>
                </div>
                <div class="progress-track">
                  <div class="progress-fill upload" :style="{ width: uploadProgress + '%' }"></div>
                </div>
              </div>

              <div class="progress-group">
                <div class="progress-label">
                  <span>AI 处理进度</span>
                  <span class="percent">{{ processingProgress }}%</span>
                </div>
                <div class="progress-track">
                  <div class="progress-fill processing" :style="{ width: processingProgress + '%' }"></div>
                </div>
              </div>
            </div>
          </div>

          <div class="result-section">
            <div v-if="processingStatus === 'completed'" class="result-content-wrapper">
              <div class="success-banner">
                <CheckCircle class="success-icon" :size="20" />
                <span>鞍垫号码识别完成</span>
              </div>

              <div class="metadata-grid">
                <div class="meta-item">
                  <span class="label">原始文件</span>
                  <span class="value">{{ processingResult.filename }}</span>
                </div>
                <div class="meta-item">
                  <span class="label">完成时间</span>
                  <span class="value">{{ processingResult.processed_at }}</span>
                </div>
              </div>

              <div class="horse-id-section">
                <h3 class="section-title">检测到的马匹号码</h3>
                <div v-if="simpleDetectionsFiltered.length > 0" class="horse-id-grid">
                  <div
                    v-for="(item, index) in simpleDetectionsFiltered"
                    :key="index"
                    class="horse-id-card"
                  >
                    <span class="horse-id-number">{{ item.horse_id }}</span>
                    <span class="horse-id-conf">鞍垫号码</span>
                  </div>
                </div>
                <div v-else class="no-detection">
                  <p>未检测到有效马匹号码</p>
                </div>
              </div>

              <details class="json-details">
                <summary>查看原始分析数据 (JSON)</summary>
                <pre class="json-block">{{ JSON.stringify(processingResult, null, 2) }}</pre>
              </details>
            </div>

            <div v-else-if="processingStatus === 'idle'" class="empty-state">
              <div class="empty-icon-wrapper">
                <FileVideo :size="48" />
              </div>
              <h3>等待任务提交</h3>
              <p>请在左侧上传视频并点击开始处理按钮</p>
            </div>

            <div v-else-if="processingStatus === 'error'" class="error-state">
              <div class="error-icon-wrapper">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              </div>
              <h3>处理失败</h3>
              <p>{{ errorMessage || '未知错误，请重试' }}</p>
              <button class="retry-button" @click="resetUpload()">重新上传</button>
            </div>

            <div v-else class="loading-state">
              <div class="loader-visual">
                <div class="pulse-ring"></div>
                <Loader2 class="animate-spin" :size="40" />
              </div>
              <h3>{{ processingStatus === 'uploading' ? '正在同步文件...' : '后端正在分析...' }}</h3>
              <p>这可能需要几十秒钟，请稍候</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.page-content {
  flex: 1;
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

/* --- guide module --- */
.guide-module {
  background-color: #0f111a;
  border: 1px solid #1e293b;
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}

.guide-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}

.guide-title {
  font-size: 1.2rem;
  font-weight: 700;
  color: #f1f5f9;
  margin: 0;
}

.mode-tabs {
  display: flex;
  gap: 0.25rem;
  background-color: #1e293b;
  padding: 4px;
  border-radius: 10px;
  border: 1px solid #334155;
}

.mode-tab {
  padding: 6px 18px;
  border: none;
  background: transparent;
  color: #94a3b8;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  text-decoration: none;
  transition: all 0.2s;
}

.mode-tab.active {
  background-color: #6366f1;
  color: #fff;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.mode-tab:not(.active):hover {
  color: #f1f5f9;
  background-color: rgba(51, 65, 85, 0.4);
}

.guide-desc {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
  color: #94a3b8;
  margin: 0 0 1rem 0;
  line-height: 1.5;
}

.guide-desc-icon {
  color: #6366f1;
  flex-shrink: 0;
}

.guide-steps {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
  flex-wrap: wrap;
}

.step {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background-color: rgba(30, 41, 59, 0.5);
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 0.85rem;
  color: #e2e8f0;
}

.step-num {
  width: 22px;
  height: 22px;
  background: linear-gradient(135deg, #6366f1, #4f46e5);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 700;
  color: #fff;
  flex-shrink: 0;
}

.step-arrow {
  color: #475569;
  font-size: 1rem;
}

.guide-examples {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.example {
  border-radius: 10px;
  overflow: hidden;
  border: 2px solid transparent;
  transition: all 0.2s;
}

.example.good {
  border-color: rgba(16, 185, 129, 0.3);
}

.example.bad {
  border-color: rgba(239, 68, 68, 0.3);
}

.example img {
  width: 100%;
  aspect-ratio: 16 / 9;
  object-fit: cover;
  display: block;
  background-color: #1e293b;
}

.example-label {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.6rem;
  font-size: 0.75rem;
  font-weight: 600;
}

.example.good .example-label {
  background-color: rgba(16, 185, 129, 0.1);
  color: #10b981;
}

.example.bad .example-label {
  background-color: rgba(239, 68, 68, 0.1);
  color: #ef4444;
}

.guide-notice {
  font-size: 0.8rem;
  color: #f59e0b;
  background-color: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.15);
  border-radius: 8px;
  padding: 0.6rem 1rem;
  line-height: 1.5;
}

.content-grid {
  display: grid;
  grid-template-columns: 1.6fr 1fr;
  gap: 2rem;
  height: calc(100vh - 128px);
}

.panel {
  background-color: #0f111a;
  border: 1px solid #1e293b;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}

.panel-header {
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid #1e293b;
  background-color: rgba(30, 41, 59, 0.2);
}

.panel-title {
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
  color: #f1f5f9;
}

.panel-body {
  flex: 1;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.upload-area {
  width: 100%;
  aspect-ratio: 16 / 9;
  border: 2px dashed #334155;
  border-radius: 12px;
  background-color: #0a0c12;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
  margin-bottom: 1.5rem;
}

.upload-area:hover {
  border-color: #6366f1;
  background-color: #111420;
}

.upload-area.has-file {
  border-style: solid;
  border-color: #1e293b;
}

.upload-placeholder {
  text-align: center;
}

.upload-icon-wrapper {
  width: 80px;
  height: 80px;
  background-color: #1e293b;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1.5rem;
  color: #94a3b8;
  transition: all 0.3s;
}

.upload-area:hover .upload-icon-wrapper {
  color: #6366f1;
  transform: translateY(-4px);
  background-color: #262c45;
}

.upload-text {
  font-size: 1.1rem;
  font-weight: 600;
  color: #e2e8f0;
  margin-bottom: 0.5rem;
}

.upload-hint {
  font-size: 0.85rem;
  color: #64748b;
}

.video-preview-container {
  width: 100%;
  height: 100%;
  position: relative;
}

.video-preview {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background-color: #000;
}

.video-controls-overlay {
  position: absolute;
  top: 1rem;
  left: 1rem;
  z-index: 10;
}

.video-mode-toggle {
  display: flex;
  gap: 0.25rem;
  background-color: rgba(15, 23, 42, 0.85);
  backdrop-filter: blur(8px);
  padding: 4px;
  border-radius: 10px;
  border: 1px solid rgba(51, 65, 85, 0.5);
}

.video-mode-toggle button {
  padding: 6px 14px;
  border: none;
  background: transparent;
  color: #94a3b8;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s;
}

.video-mode-toggle button.active {
  background-color: #6366f1;
  color: #fff;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.video-mode-toggle button:not(.active):hover {
  color: #f1f5f9;
  background-color: rgba(51, 65, 85, 0.4);
}

.file-tag {
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.8rem;
  background-color: rgba(0, 0, 0, 0.6);
  border-radius: 8px;
  font-size: 0.75rem;
  color: #cbd5e0;
}

.primary-button {
  width: 100%;
  padding: 1rem;
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  color: white;
  border: none;
  border-radius: 12px;
  font-weight: 600;
  font-size: 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  transition: all 0.3s;
  box-shadow: 0 4px 16px rgba(79, 70, 229, 0.4);
}

.primary-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
}

.primary-button:active:not(:disabled) {
  transform: translateY(0);
}

.primary-button.secondary {
  background: #1e293b;
  border: 1px solid #334155;
  box-shadow: none;
}

.primary-button.secondary:hover {
  background-color: #26334a;
  border-color: #475569;
  transform: none;
}

.primary-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  box-shadow: none;
}

.progress-card {
  background-color: #0a0c12;
  border-radius: 14px;
  padding: 1.5rem;
  border: 1px solid #1e293b;
  margin-bottom: 2rem;
}

.progress-group {
  margin-bottom: 1.5rem;
}

.progress-group:last-child {
  margin-bottom: 0;
}

.progress-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
  color: #94a3b8;
}

.percent {
  color: #f1f5f9;
  font-weight: 700;
}

.progress-track {
  height: 10px;
  background-color: #1e293b;
  border-radius: 5px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.progress-fill.upload { background: linear-gradient(to right, #3b82f6, #60a5fa); }
.progress-fill.processing { background: linear-gradient(to right, #8b5cf6, #a78bfa); }

.success-banner {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background-color: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.2);
  border-radius: 10px;
  color: #10b981;
  font-weight: 600;
  margin-bottom: 1.5rem;
}

.metadata-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.meta-item {
  background-color: #0a0c12;
  padding: 0.75rem 1rem;
  border-radius: 10px;
  border: 1px solid #1e293b;
}

.meta-item .label {
  display: block;
  font-size: 0.75rem;
  color: #64748b;
  margin-bottom: 0.25rem;
}

.meta-item .value {
  font-size: 0.85rem;
  color: #e2e8f0;
  font-weight: 500;
  word-break: break-all;
}

/* --- horse id cards (replaces table) --- */
.horse-id-section {
  margin-bottom: 1.5rem;
}

.section-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: #94a3b8;
  margin: 0 0 0.35rem 0;
}

.section-hint {
  font-size: 0.75rem;
  color: #64748b;
  margin: 0 0 1rem 0;
  line-height: 1.4;
}

.horse-id-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.horse-id-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.4rem;
  padding: 1rem 1.5rem;
  background-color: #0a0c12;
  border: 1px solid #1e293b;
  border-radius: 12px;
  min-width: 100px;
  transition: all 0.2s;
}

.horse-id-card:hover {
  border-color: #6366f1;
  background-color: #111420;
}

.horse-id-number {
  font-size: 1.4rem;
  font-weight: 700;
  font-family: monospace;
  color: #6366f1;
  letter-spacing: 0.05em;
}

.horse-id-conf {
  font-size: 0.7rem;
  color: #10b981;
  font-family: monospace;
}

.no-detection {
  text-align: center;
  padding: 2rem 1rem;
  color: #64748b;
  font-size: 0.9rem;
}

.json-details summary {
  padding: 1rem 0;
  font-size: 0.85rem;
  color: #64748b;
  cursor: pointer;
  transition: color 0.2s;
}

.json-details summary:hover { color: #94a3b8; }

.json-block {
  background-color: #050505;
  padding: 1.5rem;
  border-radius: 12px;
  font-size: 0.8rem;
  color: #a0aec0;
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid #1e293b;
}

.error-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 3rem 1rem;
}

.error-icon-wrapper {
  margin-bottom: 1.5rem;
  color: #ef4444;
}

.error-state h3 {
  font-size: 1.1rem;
  margin-bottom: 0.75rem;
  color: #f1f5f9;
}

.error-state p {
  font-size: 0.9rem;
  color: #94a3b8;
  max-width: 300px;
  margin-bottom: 1.5rem;
}

.retry-button {
  padding: 0.6rem 1.5rem;
  background-color: #1e293b;
  color: #e2e8f0;
  border: 1px solid #334155;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  transition: all 0.2s;
}

.retry-button:hover {
  background-color: #334155;
  border-color: #475569;
}

.empty-state, .loading-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 3rem 1rem;
}

.empty-icon-wrapper, .loader-visual {
  margin-bottom: 1.5rem;
  position: relative;
}

.empty-icon-wrapper {
  color: #1e293b;
}

.loader-visual {
  color: #6366f1;
}

.pulse-ring {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 60px;
  height: 60px;
  border: 2px solid #6366f1;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.empty-state h3, .loading-state h3 {
  font-size: 1.1rem;
  margin-bottom: 0.75rem;
  color: #f1f5f9;
}

.empty-state p, .loading-state p {
  font-size: 0.9rem;
  color: #64748b;
  max-width: 250px;
}

.animate-spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes pulse {
  0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.8; }
  100% { transform: translate(-50%, -50%) scale(1.5); opacity: 0; }
}

::-webkit-scrollbar {
  width: 6px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: #1e293b;
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
  background: #334155;
}
</style>
