<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue';
import { useRoute } from 'vue-router';
import { Upload, FileVideo, CheckCircle, Loader2, PlayCircle, History, Plus, Info, Camera, CameraOff, Clock, X, RefreshCw, Download } from 'lucide-vue-next';
import axios from 'axios';
import html2canvas from 'html2canvas';

type HistoryItem = {
  task_id: string;
  original_filename: string;
  processed_at: string | null;
  mtime: number;
  duration: string | null;
  resolution: string | null;
  mode: string | null;
  process_duration_seconds: number | null;
  detection_count: number;
  processed_video_url: string | null;
  original_video_url: string | null;
  summary_url: string;
};
type HistoryDetail = HistoryItem & {
  detections: Array<{ timestamp: string; horse_id?: string; person_name?: string; confidence?: string }>;
  profiling: Record<string, number>;
  stage_devices: Record<string, string>;
};

const route = useRoute();

const API_BASE = import.meta.env.VITE_API_BASE || window.location.origin;

const fileInput = ref<HTMLInputElement | null>(null);
const videoFile = ref<File | null>(null);
const videoUrl = ref<string | null>(null);
const processedVideoUrl = ref<string | null>(null);
const videoMode = ref<'original' | 'processed'>('original');
const uploadProgress = ref(0);
const processingProgress = ref(0);
const processingStatus = ref<'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'error'>('idle');
const processingResult = ref<any>(null);
const taskId = ref<string | null>(null);
const errorMessage = ref<string>('');
const queuePosition = ref(0);

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
        fetchHistory();
      } else if (data.status === 'error') {
        errorMessage.value = data.message || '后端处理时出错';
      } else if (data.status === 'queued') {
        queuePosition.value = data.queue_position ?? 0;
        processingStatus.value = 'queued';
        setTimeout(checkStatus, 2000);
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
  queuePosition.value = 0;
};

const historyItems = ref<HistoryItem[]>([]);
const historyLoading = ref(false);
const historyError = ref('');
const selectedHistory = ref<HistoryDetail | null>(null);
const detailVideoMode = ref<'original' | 'processed'>('processed');
const historyFilter = ref<'all' | 'full' | 'simple' | 'face'>('all');
let historyTimer: ReturnType<typeof setInterval> | undefined;

const filteredHistoryItems = computed(() => {
  if (historyFilter.value === 'all') return historyItems.value;
  return historyItems.value.filter((it) => it.mode === historyFilter.value);
});
const historyModeCount = computed(() => {
  const c = { all: historyItems.value.length, full: 0, simple: 0, face: 0 };
  for (const it of historyItems.value) {
    if (it.mode === 'full') c.full++;
    else if (it.mode === 'simple') c.simple++;
    else if (it.mode === 'face') c.face++;
  }
  return c;
});

const toAbsUrl = (u: string | null) => (!u ? null : (u.startsWith('http') ? u : `${API_BASE}${u}`));

const fetchHistory = async () => {
  historyLoading.value = true;
  try {
    const r = await axios.get(`${API_BASE}/api/history?limit=200`);
    if (r.data?.code === 200) {
      historyItems.value = (r.data.data.items as HistoryItem[]) || [];
      historyError.value = '';
    }
  } catch (e) {
    historyError.value = '加载历史失败';
  } finally {
    historyLoading.value = false;
  }
};

const openHistoryDetail = async (item: HistoryItem) => {
  try {
    const r = await axios.get(`${API_BASE}${item.summary_url}`);
    if (r.data?.code === 200) {
      selectedHistory.value = r.data.data as HistoryDetail;
      detailVideoMode.value = selectedHistory.value.processed_video_url ? 'processed' : 'original';
    }
  } catch (e) {
    selectedHistory.value = { ...item, detections: [], profiling: {}, stage_devices: {} };
    detailVideoMode.value = item.processed_video_url ? 'processed' : 'original';
  }
};

const closeHistoryDetail = () => { selectedHistory.value = null; };

const exportingHistory = ref(false);
const exportProgress = ref({ current: 0, total: 0, phase: '' });
const exportCardItem = ref<HistoryDetail | null>(null);
const exportCardEl = ref<HTMLElement | null>(null);

const captureItemPng = async (item: HistoryItem): Promise<Blob | null> => {
  try {
    const r = await axios.get(`${API_BASE}${item.summary_url}`);
    if (r.data?.code !== 200) return null;
    exportCardItem.value = r.data.data as HistoryDetail;
    await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
    const el = exportCardEl.value;
    if (!el) return null;
    const canvas = await html2canvas(el, { backgroundColor: '#0f111a', scale: 1.5, useCORS: true, logging: false });
    return await new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), 'image/png'));
  } catch {
    return null;
  }
};

const exportHistory = async () => {
  if (exportingHistory.value) return;
  const items = filteredHistoryItems.value;
  if (!items.length) return;
  exportingHistory.value = true;
  exportProgress.value = { current: 0, total: items.length, phase: '渲染截图' };

  const form = new FormData();
  form.append('mode', historyFilter.value);
  try {
    for (let i = 0; i < items.length; i++) {
      exportProgress.value = { current: i + 1, total: items.length, phase: '渲染截图' };
      const blob = await captureItemPng(items[i]);
      if (blob) form.append('screenshots', blob, `${items[i].task_id}.png`);
    }
    exportCardItem.value = null;
    exportProgress.value = { current: items.length, total: items.length, phase: '后端打包（视频较大需 1-2 分钟）' };
    const resp = await fetch(`${API_BASE}/api/export`, { method: 'POST', body: form });
    if (!resp.ok) throw new Error(`export failed: ${resp.status}`);
    const zipBlob = await resp.blob();
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const url = URL.createObjectURL(zipBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `export_${ts}_${historyFilter.value}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error(e);
    alert('导出失败，看控制台');
  } finally {
    exportingHistory.value = false;
    exportProgress.value = { current: 0, total: 0, phase: '' };
    exportCardItem.value = null;
  }
};

const formatDuration = (s: number | null | undefined) => {
  if (s == null || !Number.isFinite(s)) return '—';
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = (s - m * 60).toFixed(1);
  return `${m}m${r}s`;
};

const parseVideoDurationSeconds = (raw: string | null | undefined): number | null => {
  if (!raw) return null;
  const parts = raw.split(':').map((p) => Number(p));
  if (parts.some((n) => !Number.isFinite(n))) return null;
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 1) return parts[0];
  return null;
};

const speedRatio = (proc: number | null | undefined, videoRaw: string | null | undefined): string | null => {
  const video = parseVideoDurationSeconds(videoRaw);
  if (proc == null || !Number.isFinite(proc) || !video || video <= 0) return null;
  return `${(proc / video).toFixed(2)}×`;
};

const formatRelTime = (mtime: number) => {
  const now = Date.now() / 1000;
  const diff = now - mtime;
  if (diff < 60) return '刚刚';
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
  return `${Math.floor(diff / 86400)} 天前`;
};

const modeLabel = (m: string | null | undefined) => {
  if (m === 'full') return '骑手识别';
  if (m === 'simple') return '号码识别';
  if (m === 'face') return '仅人脸';
  return m || '—';
};

const PROFILING_LABELS: Record<string, string> = {
  read: '视频读取', yolo: 'YOLO+追踪', face_det: '人脸检测',
  enhance: '图像增强', ocr: 'OCR识别', vlm: 'VLM回退',
  rider_id: '骑手识别', viz: '结果渲染', write: '视频编码',
};
const MAIN_KEYS = ['read', 'yolo', 'face_det', 'enhance', 'ocr', 'vlm', 'rider_id'];
const PARALLEL_KEYS = ['viz', 'write'];

const buildRows = (keys: string[], p: Record<string, number>) =>
  keys
    .filter((k) => typeof p[k] === 'number' && p[k] > 0)
    .map((k) => ({ key: k, label: PROFILING_LABELS[k], value: Number(p[k]) }));

const detailMainProfiling = computed(() => {
  const p = selectedHistory.value?.profiling || {};
  return buildRows(MAIN_KEYS, p);
});
const detailParallelProfiling = computed(() => {
  const p = selectedHistory.value?.profiling || {};
  return buildRows(PARALLEL_KEYS, p);
});
const detailMainSum = computed(() => detailMainProfiling.value.reduce((s, r) => s + r.value, 0));
const detailParallelSum = computed(() => detailParallelProfiling.value.reduce((s, r) => s + r.value, 0));

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

onMounted(() => {
  fetchHistory();
  historyTimer = setInterval(fetchHistory, 15000);
});
onUnmounted(() => {
  if (historyTimer !== undefined) clearInterval(historyTimer);
});
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
<!--          <router-link to="/face" class="mode-tab" :class="{ active: route.path === '/face' }">
            仅人脸
          </router-link>-->
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
            <template v-else-if="processingStatus === 'queued'">
              <Loader2 class="animate-spin" :size="20" />
              <span>排队中{{ queuePosition > 0 ? `，前方还有 ${queuePosition} 个任务` : '，即将开始' }}...</span>
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
              <h3>{{ processingStatus === 'uploading' ? '正在同步文件...' : processingStatus === 'queued' ? '排队等待中...' : '后端正在分析...' }}</h3>
              <p v-if="processingStatus === 'queued'">{{ queuePosition > 0 ? `当前排在第 ${queuePosition + 1} 位，请耐心等待` : '即将开始处理' }}</p>
              <p v-else>这可能需要几十秒钟，请稍候</p>
            </div>
          </div>
        </div>
      </section>

      <section class="panel history-panel">
        <div class="panel-header history-panel-header">
          <h2 class="panel-title">
            <History :size="16" class="history-title-icon" />
            分析记录
            <span class="history-count" v-if="historyItems.length">{{ historyItems.length }}</span>
          </h2>
          <div class="history-actions">
            <button class="history-refresh" :disabled="exportingHistory || !filteredHistoryItems.length" @click="exportHistory" :title="`导出当前 (${filteredHistoryItems.length} 条)`">
              <Download :size="14" />
            </button>
            <button class="history-refresh" :disabled="historyLoading" @click="fetchHistory" title="刷新">
              <RefreshCw :size="14" :class="{ 'animate-spin': historyLoading }" />
            </button>
          </div>
        </div>
        <div class="history-filter-tabs">
          <button :class="{ active: historyFilter === 'all' }" @click="historyFilter = 'all'">
            全部 <span class="tab-count">{{ historyModeCount.all }}</span>
          </button>
          <button :class="{ active: historyFilter === 'full' }" @click="historyFilter = 'full'">
            骑手识别 <span class="tab-count">{{ historyModeCount.full }}</span>
          </button>
          <button :class="{ active: historyFilter === 'simple' }" @click="historyFilter = 'simple'">
            号码识别 <span class="tab-count">{{ historyModeCount.simple }}</span>
          </button>
        </div>
        <div class="panel-body history-panel-body">
          <div v-if="historyError" class="history-error">{{ historyError }}</div>
          <div v-else-if="!filteredHistoryItems.length && !historyLoading" class="history-empty">
            <FileVideo :size="32" />
            <p>{{ historyItems.length ? '该模式下暂无记录' : '暂无历史记录' }}</p>
          </div>
          <ul v-else class="history-list">
            <li
              v-for="item in filteredHistoryItems"
              :key="item.task_id"
              class="history-item"
              @click="openHistoryDetail(item)"
            >
              <div class="history-item-top">
                <span class="history-filename" :title="item.original_filename">{{ item.original_filename }}</span>
                <span class="history-mode" :class="'mode-' + item.mode">{{ modeLabel(item.mode) }}</span>
              </div>
              <div class="history-item-meta">
                <span class="history-time" :title="formatRelTime(item.mtime)"><Clock :size="11" /> {{ item.processed_at || formatRelTime(item.mtime) }}</span>
              </div>
              <div class="history-item-meta history-item-meta-sub" v-if="item.duration || item.resolution">
                <span v-if="item.duration">{{ item.duration }}</span>
                <span v-if="item.duration && item.resolution" class="history-dot">·</span>
                <span v-if="item.resolution">{{ item.resolution }}</span>
              </div>
              <div class="history-item-bottom">
                <span class="history-chip">检测 {{ item.detection_count }}</span>
                <span v-if="item.process_duration_seconds != null" class="history-chip chip-time">
                  耗时 {{ formatDuration(item.process_duration_seconds) }}
                </span>
                <span v-if="speedRatio(item.process_duration_seconds, item.duration)" class="history-chip chip-ratio" title="处理耗时 / 视频时长">
                  {{ speedRatio(item.process_duration_seconds, item.duration) }}
                </span>
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>

    <div v-if="exportingHistory" class="export-overlay">
      <div class="export-card">
        <Loader2 class="animate-spin" :size="28" />
        <div class="export-phase">{{ exportProgress.phase }}</div>
        <div v-if="exportProgress.total" class="export-progress">
          {{ exportProgress.current }} / {{ exportProgress.total }}
        </div>
        <div v-if="exportProgress.total" class="export-bar-track">
          <div class="export-bar-fill" :style="{ width: (exportProgress.current / exportProgress.total * 100) + '%' }"></div>
        </div>
      </div>
    </div>

    <div class="export-offscreen" aria-hidden="true">
      <div v-if="exportCardItem" ref="exportCardEl" class="export-render-card">
        <div class="export-render-head">
          <h3>{{ exportCardItem.original_filename }}</h3>
          <span class="export-render-mode" :class="'mode-' + exportCardItem.mode">{{ modeLabel(exportCardItem.mode) }}</span>
        </div>
        <div class="detail-meta-grid">
          <div class="meta-item"><span class="label">完成时间</span><span class="value">{{ exportCardItem.processed_at || '—' }}</span></div>
          <div class="meta-item"><span class="label">视频时长</span><span class="value">{{ exportCardItem.duration || '—' }}</span></div>
          <div class="meta-item"><span class="label">分辨率</span><span class="value">{{ exportCardItem.resolution || '—' }}</span></div>
          <div class="meta-item"><span class="label">处理耗时</span><span class="value">{{ formatDuration(exportCardItem.process_duration_seconds) }}<span v-if="speedRatio(exportCardItem.process_duration_seconds, exportCardItem.duration)" class="value-sub">（{{ speedRatio(exportCardItem.process_duration_seconds, exportCardItem.duration) }}）</span></span></div>
          <div class="meta-item"><span class="label">分析模式</span><span class="value">{{ modeLabel(exportCardItem.mode) }}</span></div>
          <div class="meta-item"><span class="label">检测条数</span><span class="value">{{ exportCardItem.detection_count }}</span></div>
        </div>
        <div v-if="exportCardItem.detections.length" class="detail-block">
          <h3 class="detail-block-title">检测结果</h3>
          <div class="table-container">
            <table class="data-table">
              <thead><tr><th>时间</th><th>目标名称</th><th>置信度</th></tr></thead>
              <tbody>
                <tr v-for="(d, i) in exportCardItem.detections" :key="i">
                  <td>{{ d.timestamp }}</td>
                  <td>{{ d.person_name || d.horse_id || '—' }}</td>
                  <td class="conf-cell">{{ d.confidence || '—' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div v-if="(exportCardItem.profiling && Object.keys(exportCardItem.profiling).length)" class="detail-block">
          <h3 class="detail-block-title">各阶段耗时占比 <span class="detail-block-hint">(相对墙钟时间)</span></h3>
          <div class="detail-profiling-group">
            <div class="detail-profiling-group-head"><span>主管线（串行）</span></div>
            <ul class="detail-profiling-list">
              <li v-for="k in ['read','yolo','face_det','enhance','ocr','vlm','rider_id']" :key="k" v-show="exportCardItem.profiling[k]" class="detail-profiling-row">
                <span class="detail-profiling-label">{{ PROFILING_LABELS[k] }}</span>
                <div class="detail-profiling-track"><div class="detail-profiling-fill" :style="{ width: (exportCardItem.profiling[k] || 0) + '%' }"></div></div>
                <span class="detail-profiling-value">{{ (exportCardItem.profiling[k] || 0).toFixed(1) }}%</span>
              </li>
            </ul>
          </div>
          <div class="detail-profiling-group">
            <div class="detail-profiling-group-head"><span>编码线程（与主管线并行）</span></div>
            <ul class="detail-profiling-list">
              <li v-for="k in ['viz','write']" :key="k" v-show="exportCardItem.profiling[k]" class="detail-profiling-row">
                <span class="detail-profiling-label">{{ PROFILING_LABELS[k] }}</span>
                <div class="detail-profiling-track"><div class="detail-profiling-fill detail-profiling-fill-parallel" :style="{ width: (exportCardItem.profiling[k] || 0) + '%' }"></div></div>
                <span class="detail-profiling-value">{{ (exportCardItem.profiling[k] || 0).toFixed(1) }}%</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div v-if="selectedHistory" class="detail-modal-backdrop" @click.self="closeHistoryDetail">
      <div class="detail-modal">
        <div class="detail-modal-header">
          <div class="detail-modal-title">
            <FileVideo :size="18" />
            <span>{{ selectedHistory.original_filename }}</span>
            <span class="detail-mode-tag" :class="'mode-' + selectedHistory.mode">{{ modeLabel(selectedHistory.mode) }}</span>
          </div>
          <button class="detail-close" @click="closeHistoryDetail" title="关闭"><X :size="18" /></button>
        </div>
        <div class="detail-modal-body">
          <div class="detail-video-section">
            <div class="detail-video-toggle">
              <button
                :class="{ active: detailVideoMode === 'original', disabled: !selectedHistory.original_video_url }"
                :disabled="!selectedHistory.original_video_url"
                @click="detailVideoMode = 'original'"
              ><History :size="14" /> 原视频</button>
              <button
                :class="{ active: detailVideoMode === 'processed', disabled: !selectedHistory.processed_video_url }"
                :disabled="!selectedHistory.processed_video_url"
                @click="detailVideoMode = 'processed'"
              ><PlayCircle :size="14" /> 分析视频</button>
            </div>
            <video
              v-if="detailVideoMode === 'original' && selectedHistory.original_video_url"
              :src="toAbsUrl(selectedHistory.original_video_url) || undefined"
              :key="'orig-' + selectedHistory.task_id"
              controls
              class="detail-video"
            ></video>
            <video
              v-else-if="detailVideoMode === 'processed' && selectedHistory.processed_video_url"
              :src="toAbsUrl(selectedHistory.processed_video_url) || undefined"
              :key="'proc-' + selectedHistory.task_id"
              controls
              class="detail-video"
            ></video>
            <div v-else class="detail-video-missing">视频文件不存在</div>
          </div>

          <div class="detail-info-section">
            <div class="detail-meta-grid">
              <div class="meta-item"><span class="label">完成时间</span><span class="value">{{ selectedHistory.processed_at || '—' }}</span></div>
              <div class="meta-item"><span class="label">视频时长</span><span class="value">{{ selectedHistory.duration || '—' }}</span></div>
              <div class="meta-item"><span class="label">分辨率</span><span class="value">{{ selectedHistory.resolution || '—' }}</span></div>
              <div class="meta-item"><span class="label">处理耗时</span><span class="value">{{ formatDuration(selectedHistory.process_duration_seconds) }}<span v-if="speedRatio(selectedHistory.process_duration_seconds, selectedHistory.duration)" class="value-sub">（{{ speedRatio(selectedHistory.process_duration_seconds, selectedHistory.duration) }}）</span></span></div>
              <div class="meta-item"><span class="label">分析模式</span><span class="value">{{ modeLabel(selectedHistory.mode) }}</span></div>
              <div class="meta-item"><span class="label">检测条数</span><span class="value">{{ selectedHistory.detection_count }}</span></div>
            </div>

            <div v-if="selectedHistory.detections.length" class="detail-block">
              <h3 class="detail-block-title">检测结果</h3>
              <div class="table-container">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>时间</th>
                      <th>目标名称</th>
                      <th>置信度</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(d, i) in selectedHistory.detections" :key="i">
                      <td>{{ d.timestamp }}</td>
                      <td>{{ d.person_name || d.horse_id || '—' }}</td>
                      <td class="conf-cell">{{ d.confidence || '—' }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div v-if="detailMainProfiling.length || detailParallelProfiling.length" class="detail-block">
              <h3 class="detail-block-title">各阶段耗时占比 <span class="detail-block-hint">(相对墙钟时间)</span></h3>
              <div v-if="detailMainProfiling.length" class="detail-profiling-group">
                <div class="detail-profiling-group-head">
                  <span>主管线（串行）</span>
                  <span class="detail-profiling-group-sum">{{ detailMainSum.toFixed(1) }}%</span>
                </div>
                <ul class="detail-profiling-list">
                  <li v-for="row in detailMainProfiling" :key="row.key" class="detail-profiling-row">
                    <span class="detail-profiling-label">{{ row.label }}</span>
                    <div class="detail-profiling-track"><div class="detail-profiling-fill" :style="{ width: row.value + '%' }"></div></div>
                    <span class="detail-profiling-value">{{ row.value.toFixed(1) }}%</span>
                  </li>
                </ul>
              </div>
              <div v-if="detailParallelProfiling.length" class="detail-profiling-group">
                <div class="detail-profiling-group-head">
                  <span>编码线程（与主管线并行）</span>
                  <span class="detail-profiling-group-sum">{{ detailParallelSum.toFixed(1) }}%</span>
                </div>
                <ul class="detail-profiling-list">
                  <li v-for="row in detailParallelProfiling" :key="row.key" class="detail-profiling-row">
                    <span class="detail-profiling-label">{{ row.label }}</span>
                    <div class="detail-profiling-track"><div class="detail-profiling-fill detail-profiling-fill-parallel" :style="{ width: row.value + '%' }"></div></div>
                    <span class="detail-profiling-value">{{ row.value.toFixed(1) }}%</span>
                  </li>
                </ul>
              </div>
            </div>

            <details class="json-details">
              <summary>查看原始 JSON</summary>
              <pre class="json-block">{{ JSON.stringify(selectedHistory, null, 2) }}</pre>
            </details>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page-content {
  flex: 1;
  padding: 1.5rem 2rem;
  max-width: min(1800px, 100%);
  margin: 0 auto;
  width: 100%;
  box-sizing: border-box;
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
  grid-template-columns: 0.9fr 1.6fr 1fr;
  gap: 1.25rem;
  height: calc(100vh - 128px);
}
.history-panel { order: -1; }

@media (max-width: 1280px) {
  .content-grid {
    grid-template-columns: 1.4fr 1fr;
  }
  .history-panel {
    order: 99;
    grid-column: 1 / -1;
    height: auto;
    max-height: 320px;
  }
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

/* --- history panel (right column) --- */
.history-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.history-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}
.history-title-icon { color: #a78bfa; vertical-align: -2px; margin-right: 0.35rem; }
.history-count {
  display: inline-block;
  margin-left: 0.4rem;
  padding: 1px 7px;
  font-size: 0.68rem;
  font-weight: 700;
  color: #a78bfa;
  background: rgba(167, 139, 250, 0.12);
  border: 1px solid rgba(167, 139, 250, 0.3);
  border-radius: 10px;
  vertical-align: 2px;
}
.history-actions { display: flex; gap: 0.35rem; }
.history-refresh {
  background: transparent;
  border: 1px solid #1e293b;
  color: #94a3b8;
  padding: 0.3rem;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}
.history-refresh:hover:not(:disabled) { color: #e2e8f0; border-color: #334155; }
.history-refresh:disabled { opacity: 0.5; cursor: not-allowed; }

.history-filter-tabs {
  display: flex;
  gap: 0.3rem;
  padding: 0.35rem 0.5rem 0.5rem;
  border-bottom: 1px solid #1e293b;
}
.history-filter-tabs button {
  flex: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.3rem;
  padding: 0.35rem 0.4rem;
  font-size: 0.72rem;
  font-weight: 600;
  background: transparent;
  color: #64748b;
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}
.history-filter-tabs button:hover { color: #cbd5e1; background: rgba(148, 163, 184, 0.06); }
.history-filter-tabs button.active {
  background: rgba(99, 102, 241, 0.15);
  color: #c7d2fe;
  border-color: rgba(99, 102, 241, 0.35);
}
.history-filter-tabs .tab-count {
  font-size: 0.62rem;
  padding: 0 5px;
  border-radius: 8px;
  background: rgba(148, 163, 184, 0.12);
  color: #94a3b8;
  font-weight: 700;
}
.history-filter-tabs button.active .tab-count {
  background: rgba(99, 102, 241, 0.25);
  color: #e0e7ff;
}

.history-item-meta-sub {
  margin-top: -0.1rem;
  opacity: 0.85;
}

.history-panel-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 0.25rem 0.5rem 0.75rem;
}
.history-error, .history-empty {
  color: #64748b;
  text-align: center;
  padding: 2rem 1rem;
  font-size: 0.85rem;
}
.history-empty { display: flex; flex-direction: column; gap: 0.75rem; align-items: center; }
.history-empty p { margin: 0; }

.history-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.history-item {
  background: #0b0d14;
  border: 1px solid #1e293b;
  border-radius: 10px;
  padding: 0.65rem 0.75rem;
  cursor: pointer;
  transition: all 0.18s;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.history-item:hover {
  border-color: #6366f1;
  background: #111425;
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(99, 102, 241, 0.18);
}
.history-item-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}
.history-filename {
  flex: 1;
  font-size: 0.8rem;
  color: #e2e8f0;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.history-mode {
  flex-shrink: 0;
  font-size: 0.62rem;
  font-weight: 700;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(99, 102, 241, 0.1);
  color: #a5b4fc;
  border: 1px solid rgba(99, 102, 241, 0.25);
  letter-spacing: 0.02em;
}
.history-mode.mode-simple { background: rgba(56, 189, 248, 0.1); color: #7dd3fc; border-color: rgba(56, 189, 248, 0.25); }
.history-mode.mode-face   { background: rgba(244, 114, 182, 0.1); color: #f9a8d4; border-color: rgba(244, 114, 182, 0.25); }
.history-item-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
  align-items: center;
  font-size: 0.68rem;
  color: #64748b;
  font-variant-numeric: tabular-nums;
}
.history-time { display: inline-flex; align-items: center; gap: 0.2rem; }
.history-dot { opacity: 0.5; }
.history-item-bottom {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}
.history-chip {
  font-size: 0.65rem;
  padding: 2px 7px;
  border-radius: 3px;
  background: rgba(148, 163, 184, 0.08);
  color: #94a3b8;
  border: 1px solid rgba(148, 163, 184, 0.15);
}
.history-chip.chip-time {
  background: rgba(52, 211, 153, 0.08);
  color: #6ee7b7;
  border-color: rgba(52, 211, 153, 0.25);
}
.history-chip.chip-ratio {
  background: rgba(251, 146, 60, 0.08);
  color: #fdba74;
  border-color: rgba(251, 146, 60, 0.25);
}
.detail-meta-grid .value-sub {
  font-size: 0.7rem;
  color: #94a3b8;
  font-weight: 500;
  margin-left: 0.1rem;
}

/* --- detail modal --- */
.detail-modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(5, 7, 14, 0.8);
  backdrop-filter: blur(6px);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  animation: fadeIn 0.18s ease;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.detail-modal {
  background: #0f111a;
  border: 1px solid #1e293b;
  border-radius: 14px;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.6);
  width: min(1200px, 96vw);
  max-height: 92vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.detail-modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.9rem 1.25rem;
  border-bottom: 1px solid #1e293b;
  gap: 1rem;
}
.detail-modal-title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  color: #f1f5f9;
  font-weight: 600;
  font-size: 0.95rem;
  overflow: hidden;
}
.detail-modal-title > span:first-of-type {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.detail-mode-tag {
  font-size: 0.65rem;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 4px;
  background: rgba(99, 102, 241, 0.1);
  color: #a5b4fc;
  border: 1px solid rgba(99, 102, 241, 0.25);
  flex-shrink: 0;
}
.detail-mode-tag.mode-simple { background: rgba(56, 189, 248, 0.1); color: #7dd3fc; border-color: rgba(56, 189, 248, 0.25); }
.detail-mode-tag.mode-face   { background: rgba(244, 114, 182, 0.1); color: #f9a8d4; border-color: rgba(244, 114, 182, 0.25); }
.detail-close {
  background: transparent;
  border: 1px solid #1e293b;
  color: #94a3b8;
  padding: 0.4rem;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.detail-close:hover { color: #f87171; border-color: #7f1d1d; }

.detail-modal-body {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 1.25rem;
  display: grid;
  grid-template-columns: 1.3fr 1fr;
  gap: 1.5rem;
}
@media (max-width: 960px) {
  .detail-modal-body { grid-template-columns: 1fr; }
}

.detail-video-section {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  min-width: 0;
}
.detail-video-toggle {
  display: flex;
  gap: 0.4rem;
}
.detail-video-toggle button {
  flex: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.35rem;
  padding: 0.5rem 0.6rem;
  font-size: 0.78rem;
  font-weight: 600;
  background: #0b0d14;
  color: #94a3b8;
  border: 1px solid #1e293b;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}
.detail-video-toggle button.active {
  background: rgba(99, 102, 241, 0.15);
  color: #c7d2fe;
  border-color: #6366f1;
}
.detail-video-toggle button.disabled,
.detail-video-toggle button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.detail-video {
  width: 100%;
  max-height: 60vh;
  background: #000;
  border-radius: 10px;
  border: 1px solid #1e293b;
}
.detail-video-missing {
  padding: 3rem;
  text-align: center;
  color: #64748b;
  background: #0b0d14;
  border: 1px dashed #1e293b;
  border-radius: 10px;
}

.detail-info-section {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 0;
}
.detail-meta-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
}
.detail-meta-grid .meta-item {
  background: #0b0d14;
  border: 1px solid #1e293b;
  border-radius: 8px;
  padding: 0.5rem 0.7rem;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}
.detail-meta-grid .label {
  font-size: 0.65rem;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.detail-meta-grid .value {
  font-size: 0.85rem;
  color: #f1f5f9;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.detail-block-title {
  font-size: 0.8rem;
  font-weight: 700;
  color: #cbd5e1;
  margin: 0 0 0.5rem 0;
  letter-spacing: 0.02em;
}

.detail-profiling-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.35rem; }
.detail-profiling-row {
  display: grid;
  grid-template-columns: 5.5rem 1fr 2.8rem;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
}
.detail-profiling-label { color: #94a3b8; }
.detail-profiling-track { height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
.detail-profiling-fill {
  height: 100%;
  background: linear-gradient(90deg, #6366f1, #a78bfa);
  border-radius: 3px;
  transition: width 0.3s;
}
.detail-profiling-fill-parallel {
  background: linear-gradient(90deg, #0ea5e9, #38bdf8);
}
.detail-profiling-value { text-align: right; color: #e2e8f0; font-variant-numeric: tabular-nums; }
.detail-block-hint {
  font-size: 0.68rem;
  font-weight: 500;
  color: #64748b;
  margin-left: 0.35rem;
  letter-spacing: 0;
}
.detail-profiling-group + .detail-profiling-group { margin-top: 0.75rem; }
.detail-profiling-group-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0 0.1rem 0.35rem;
  font-size: 0.7rem;
  color: #94a3b8;
  font-weight: 600;
  border-bottom: 1px dashed #1e293b;
  margin-bottom: 0.4rem;
}
.detail-profiling-group-sum {
  color: #e2e8f0;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}

/* --- export overlay + offscreen render card --- */
.export-overlay {
  position: fixed; inset: 0; background: rgba(5, 7, 14, 0.85); z-index: 2000;
  display: flex; align-items: center; justify-content: center; backdrop-filter: blur(6px);
}
.export-card {
  background: #0f111a; border: 1px solid #1e293b; border-radius: 12px;
  padding: 1.5rem 2rem; min-width: 260px; color: #e2e8f0;
  display: flex; flex-direction: column; align-items: center; gap: 0.6rem;
}
.export-phase { font-size: 0.85rem; font-weight: 600; color: #cbd5e1; }
.export-progress { font-size: 0.75rem; color: #94a3b8; font-variant-numeric: tabular-nums; }
.export-bar-track { width: 220px; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
.export-bar-fill { height: 100%; background: linear-gradient(90deg, #6366f1, #a78bfa); transition: width 0.25s; }

.export-offscreen { position: fixed; left: -99999px; top: 0; opacity: 1; pointer-events: none; }
.export-render-card {
  width: 720px; background: #0f111a; padding: 1.25rem 1.5rem; color: #e2e8f0;
  font-family: inherit; display: flex; flex-direction: column; gap: 1rem;
}
.export-render-head {
  display: flex; align-items: center; gap: 0.75rem;
  border-bottom: 1px solid #1e293b; padding-bottom: 0.6rem;
}
.export-render-head h3 {
  flex: 1; margin: 0; font-size: 1rem; color: #f1f5f9;
  overflow: hidden; text-overflow: ellipsis;
}
.export-render-mode {
  font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;
  background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3);
}
.export-render-mode.mode-simple { background: rgba(56, 189, 248, 0.15); color: #7dd3fc; border-color: rgba(56, 189, 248, 0.3); }
.export-render-mode.mode-face { background: rgba(244, 114, 182, 0.15); color: #f9a8d4; border-color: rgba(244, 114, 182, 0.3); }
</style>
