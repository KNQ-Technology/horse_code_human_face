<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, reactive } from 'vue';
import { FileVideo, Cpu, HardDrive, Zap, ListOrdered, Activity } from 'lucide-vue-next';

const API_BASE = import.meta.env.VITE_API_BASE || window.location.origin;

type GpuRow = {
  index: number;
  utilization_gpu: number;
  memory_used_mib?: number;
  memory_total_mib?: number;
};

const cpuPercent = ref<number | null>(null);
const memoryPercent = ref<number | null>(null);
const memoryUsedGb = ref<number | null>(null);
const memoryTotalGb = ref<number | null>(null);
const gpuPercent = ref<number | null>(null);
const gpuVramPercent = ref<number | null>(null);
const gpuMemoryUsedMib = ref<number | null>(null);
const gpuMemoryTotalMib = ref<number | null>(null);
const gpuAvailable = ref(false);
const gpuList = ref<GpuRow[]>([]);
const metricsError = ref(false);

const queueProcessingCount = ref(0);
const queueWaitingCount = ref(0);

type ProfilingData = {
  read: number; yolo: number; face_det: number; enhance: number;
  ocr: number; vlm: number; rider_id: number; viz: number; write: number;
  other: number;
};

const profilingLabels: Record<string, string> = {
  read: '视频读取', yolo: 'YOLO+追踪', face_det: '人脸检测',
  enhance: '图像增强', ocr: 'OCR识别', vlm: 'VLM回退',
  rider_id: '骑手识别', viz: '结果渲染', write: '视频编码',
};

const profilingColors: Record<string, string> = {
  read: '#38bdf8', yolo: '#f472b6', face_det: '#c084fc',
  enhance: '#34d399', ocr: '#fb923c', vlm: '#fbbf24',
  rider_id: '#a78bfa', viz: '#2dd4bf', write: '#94a3b8',
};

const profiling = reactive<ProfilingData>({
  read: 0, yolo: 0, face_det: 0, enhance: 0,
  ocr: 0, vlm: 0, rider_id: 0, viz: 0, write: 0, other: 0,
});
const profilingActive = ref(false);
const profilingProgress = ref(0);
const profilingStatus = ref<'processing' | 'completed' | 'error' | ''>('');
const stageDevices = ref<Record<string, string>>({});

let metricsTimer: ReturnType<typeof setInterval> | undefined;
let queueTimer: ReturnType<typeof setInterval> | undefined;
let profilingTimer: ReturnType<typeof setInterval> | undefined;

function clampPct(n: number | null | undefined): number {
  if (n == null || Number.isNaN(n)) return 0;
  return Math.min(100, Math.max(0, n));
}

const gpuSubtitle = computed(() => {
  if (!gpuAvailable.value) return '';
  const u = gpuMemoryUsedMib.value;
  const t = gpuMemoryTotalMib.value;
  const vp = gpuVramPercent.value;
  if (typeof u === 'number' && typeof t === 'number' && t > 0) {
    const ug = (u / 1024).toFixed(1);
    const tg = (t / 1024).toFixed(1);
    const tail = typeof vp === 'number' ? ` · ${vp}%` : '';
    return `显存 ${ug} / ${tg} GiB${tail}`;
  }
  if (typeof vp === 'number') return `显存占用 ${vp}%`;
  return '';
});

const gpuCardTitle = computed(() => {
  const base =
    '后端所在机器：GPU 算力利用率、显存占用；多卡时为平均值 / 显存合计。每秒更新。';
  if (gpuList.value.length === 0) return base;
  const per = gpuList.value
    .map((g) => {
      const m =
        typeof g.memory_used_mib === 'number' && typeof g.memory_total_mib === 'number'
          ? `，显存 ${g.memory_used_mib} / ${g.memory_total_mib} MiB`
          : '';
      return `GPU ${g.index}: ${g.utilization_gpu}%${m}`;
    })
    .join('\n');
  return `${base}\n${per}`;
});

async function fetchMetrics() {
  try {
    const r = await fetch(`${API_BASE}/api/system/metrics`);
    const j = await r.json();
    metricsError.value = false;
    if (j.code === 200 && j.data) {
      const d = j.data;
      cpuPercent.value = typeof d.cpu_percent === 'number' ? d.cpu_percent : null;
      memoryPercent.value = typeof d.memory_percent === 'number' ? d.memory_percent : null;
      memoryUsedGb.value = typeof d.memory_used_gb === 'number' ? d.memory_used_gb : null;
      memoryTotalGb.value = typeof d.memory_total_gb === 'number' ? d.memory_total_gb : null;
      gpuAvailable.value = !!d.gpu_available;
      gpuPercent.value = typeof d.gpu_percent === 'number' ? d.gpu_percent : null;
      gpuVramPercent.value = typeof d.gpu_vram_percent === 'number' ? d.gpu_vram_percent : null;
      gpuMemoryUsedMib.value =
        typeof d.gpu_memory_used_mib === 'number' ? d.gpu_memory_used_mib : null;
      gpuMemoryTotalMib.value =
        typeof d.gpu_memory_total_mib === 'number' ? d.gpu_memory_total_mib : null;
      const gl = d.gpus;
      gpuList.value = Array.isArray(gl)
        ? gl.filter((g: GpuRow) => g && typeof g.index === 'number')
        : [];
    }
  } catch {
    metricsError.value = true;
  }
}

async function fetchProfiling() {
  try {
    const r = await fetch(`${API_BASE}/api/profiling`);
    const j = await r.json();
    if (j.code === 200 && j.data && j.data.profiling) {
      const p = j.data.profiling;
      for (const k of Object.keys(profilingLabels)) {
        (profiling as any)[k] = typeof p[k] === 'number' ? p[k] : 0;
      }
      profiling.other = typeof p.other === 'number' ? p.other : 0;
      profilingActive.value = true;
      profilingProgress.value = j.data.progress ?? 0;
      profilingStatus.value = j.data.status ?? '';
      stageDevices.value = j.data.devices ?? {};
    } else {
      profilingActive.value = false;
    }
  } catch { profilingActive.value = false; }
}

async function fetchQueue() {
  try {
    const r = await fetch(`${API_BASE}/api/queue`);
    const j = await r.json();
    if (j.code === 200 && j.data) {
      queueProcessingCount.value = j.data.processing?.length ?? 0;
      queueWaitingCount.value = j.data.queue_length ?? 0;
    }
  } catch { /* 静默失败，不影响主体 */ }
}

onMounted(() => {
  fetchMetrics();
  fetchQueue();
  fetchProfiling();
  metricsTimer = setInterval(fetchMetrics, 1000);
  queueTimer = setInterval(fetchQueue, 2000);
  profilingTimer = setInterval(fetchProfiling, 1000);
});

onUnmounted(() => {
  if (metricsTimer !== undefined) clearInterval(metricsTimer);
  if (queueTimer !== undefined) clearInterval(queueTimer);
  if (profilingTimer !== undefined) clearInterval(profilingTimer);
});
</script>

<template>
  <div class="app-layout">
    <header class="app-header">
      <div class="header-content">
        <div class="logo">
          <FileVideo class="logo-icon" :size="24" />
          <h1>赛马识别系统</h1>
        </div>
        <div v-if="profilingActive" class="profiling-panel">
          <div class="profiling-left">
            <div class="profiling-task-row">
              <Activity class="profiling-icon" :size="14" />
              <span class="profiling-task-label">TASK</span>
              <span class="profiling-task-count">{{ queueProcessingCount }}</span>
            </div>
            <div class="profiling-bar-track">
              <div
                class="profiling-bar-fill"
                :class="{ 'profiling-bar-done': profilingStatus === 'completed' }"
                :style="{ width: profilingProgress + '%' }"
              ></div>
            </div>
          </div>
          <div class="profiling-grid">
            <span v-for="key in ['read','yolo','face_det','enhance','ocr','vlm','rider_id','viz','write']" :key="key" class="profiling-tag">
              <span class="profiling-tag-label">{{ profilingLabels[key] }}</span>
              <span class="profiling-tag-value" :style="{ color: profilingColors[key] }">{{ (profiling as any)[key] }}%</span>
              <span
                v-if="stageDevices[key]"
                class="profiling-device"
                :class="{
                  'profiling-device-gpu': stageDevices[key] === 'GPU',
                  'profiling-device-cpu': stageDevices[key] === 'CPU',
                  'profiling-device-api': stageDevices[key] === 'API',
                }"
              >{{ stageDevices[key] }}</span>
            </span>
          </div>
        </div>

        <div
          class="header-metrics"
          role="group"
          aria-label="本机资源监控"
          title="与视频处理后端同一台机器上的 CPU、内存、GPU 实时占用"
        >
          <div class="resource-meter" title="当前任务队列状态">
            <div class="meter-top">
              <ListOrdered class="meter-icon meter-icon-queue" :size="15" aria-hidden="true" />
              <span class="meter-label">处理中</span>
              <span class="meter-value">{{ queueProcessingCount }}</span>
            </div>
            <div class="meter-top">
              <span class="meter-icon" style="width:15px" aria-hidden="true"></span>
              <span class="meter-label">排队中</span>
              <span class="meter-value meter-value-waiting">{{ queueWaitingCount }}</span>
            </div>
          </div>
          <div
            class="resource-meter"
            title="系统 CPU 使用率（多核整体）"
          >
            <div class="meter-top">
              <Cpu class="meter-icon meter-icon-cpu" :size="15" aria-hidden="true" />
              <span class="meter-label">CPU</span>
              <span class="meter-value">{{
                cpuPercent !== null ? `${cpuPercent.toFixed(1)}%` : '—'
              }}</span>
            </div>
            <div
              class="meter-track"
              role="progressbar"
              :aria-valuenow="clampPct(cpuPercent)"
              aria-valuemin="0"
              aria-valuemax="100"
              aria-label="CPU 占用"
            >
              <div
                class="meter-fill meter-fill-cpu"
                :style="{ width: clampPct(cpuPercent) + '%' }"
              />
            </div>
          </div>

          <div
            class="resource-meter"
            :title="
              memoryUsedGb != null && memoryTotalGb != null
                ? `已用 ${memoryUsedGb} / ${memoryTotalGb} GiB`
                : '系统物理内存占用'
            "
          >
            <div class="meter-top">
              <HardDrive class="meter-icon meter-icon-mem" :size="15" aria-hidden="true" />
              <span class="meter-label">内存</span>
              <span class="meter-value">{{
                memoryPercent !== null ? `${memoryPercent.toFixed(1)}%` : '—'
              }}</span>
            </div>
            <div
              class="meter-track"
              role="progressbar"
              :aria-valuenow="clampPct(memoryPercent)"
              aria-valuemin="0"
              aria-valuemax="100"
              aria-label="内存占用"
            >
              <div
                class="meter-fill meter-fill-mem"
                :style="{ width: clampPct(memoryPercent) + '%' }"
              />
            </div>
            <div v-if="memoryUsedGb != null && memoryTotalGb != null" class="meter-sub">
              {{ memoryUsedGb }} / {{ memoryTotalGb }} GiB
            </div>
          </div>

          <!-- 多卡：每张 GPU 独立一个 meter -->
          <template v-if="gpuList.length > 1">
            <div
              v-for="g in gpuList"
              :key="g.index"
              class="resource-meter resource-meter-gpu"
              :title="`GPU ${g.index}: 算力 ${g.utilization_gpu}%` + (g.memory_used_mib != null && g.memory_total_mib != null ? `，显存 ${g.memory_used_mib} / ${g.memory_total_mib} MiB` : '')"
            >
              <div class="meter-top">
                <Zap class="meter-icon meter-icon-gpu" :size="15" aria-hidden="true" />
                <span class="meter-label">GPU {{ g.index }}</span>
                <span class="meter-value">{{ g.utilization_gpu.toFixed(1) }}%</span>
              </div>
              <div class="meter-track">
                <div class="meter-fill meter-fill-gpu" :style="{ width: clampPct(g.utilization_gpu) + '%' }" />
              </div>
              <template v-if="g.memory_used_mib != null && g.memory_total_mib != null">
                <div class="meter-sub">
                  显存 {{ (g.memory_used_mib / 1024).toFixed(1) }} / {{ (g.memory_total_mib / 1024).toFixed(1) }} GiB
                </div>
                <div class="meter-track meter-track-sub">
                  <div class="meter-fill meter-fill-vram" :style="{ width: clampPct(g.memory_used_mib / g.memory_total_mib * 100) + '%' }" />
                </div>
              </template>
            </div>
          </template>

          <!-- 单卡或无卡：保持原有汇总样式 -->
          <div v-else class="resource-meter resource-meter-gpu" :title="gpuCardTitle">
            <div class="meter-top">
              <Zap class="meter-icon meter-icon-gpu" :size="15" aria-hidden="true" />
              <span class="meter-label">GPU</span>
              <span class="meter-value">
                <template v-if="gpuPercent !== null">{{ gpuPercent.toFixed(1) }}%</template>
                <template v-else-if="gpuAvailable">—</template>
                <template v-else>N/A</template>
              </span>
            </div>
            <div class="meter-track">
              <div class="meter-fill meter-fill-gpu" :style="{ width: clampPct(gpuPercent) + '%' }" />
            </div>
            <div v-if="gpuSubtitle" class="meter-sub">{{ gpuSubtitle }}</div>
            <div v-if="gpuAvailable && gpuVramPercent !== null" class="meter-track meter-track-sub">
              <div class="meter-fill meter-fill-vram" :style="{ width: clampPct(gpuVramPercent) + '%' }" />
            </div>
          </div>

          <span v-if="metricsError" class="metric-hint" title="无法连接后端指标接口">!</span>
        </div>
      </div>
    </header>

    <main class="main-container">
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.app-layout {
  min-height: 100vh;
  background-color: #090a0f;
  color: #e2e8f0;
  display: flex;
  flex-direction: column;
}

.app-header {
  min-height: 64px;
  background-color: rgba(15, 17, 26, 0.8);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid #1e293b;
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-content {
  max-width: min(1800px, 100%);
  margin: 0 auto;
  min-height: 64px;
  padding: 0.5rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  box-sizing: border-box;
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-shrink: 0;
}

.logo-icon {
  color: #6366f1;
}

.logo h1 {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(to right, #fff, #94a3b8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.header-metrics {
  display: flex;
  align-items: stretch;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.65rem 0.85rem;
  max-width: min(800px, 70vw);
  font-size: 0.75rem;
  color: #94a3b8;
  font-variant-numeric: tabular-nums;
}

.resource-meter {
  min-width: 6.5rem;
  flex: 1 1 5.5rem;
  max-width: 10rem;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.resource-meter-gpu {
  max-width: 11rem;
}

.meter-top {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  min-height: 1.1rem;
}

.meter-icon {
  flex-shrink: 0;
  opacity: 0.9;
}

.meter-icon-cpu {
  color: #38bdf8;
}

.meter-icon-mem {
  color: #34d399;
}

.meter-icon-gpu {
  color: #fbbf24;
}

.meter-label {
  color: #64748b;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  flex: 1;
}

.meter-value {
  color: #f1f5f9;
  font-weight: 700;
  font-size: 0.78rem;
  font-family: ui-monospace, 'Cascadia Code', 'Segoe UI Mono', monospace;
  text-align: right;
  white-space: nowrap;
}

.meter-multi {
  font-weight: 600;
  font-size: 0.62rem;
  color: #94a3b8;
  margin-left: 0.15rem;
}

.meter-track {
  height: 4px;
  background: rgba(30, 41, 59, 0.9);
  border-radius: 2px;
  overflow: hidden;
  border: 1px solid rgba(51, 65, 85, 0.6);
}

.meter-track-sub {
  margin-top: 0.1rem;
  height: 3px;
  opacity: 0.95;
}

.meter-fill {
  height: 100%;
  border-radius: 1px;
  transition: width 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}

.meter-fill-cpu {
  background: linear-gradient(90deg, #0ea5e9, #38bdf8);
  box-shadow: 0 0 8px rgba(56, 189, 248, 0.35);
}

.meter-fill-mem {
  background: linear-gradient(90deg, #059669, #34d399);
  box-shadow: 0 0 8px rgba(52, 211, 153, 0.3);
}

.meter-fill-gpu {
  background: linear-gradient(90deg, #d97706, #fbbf24);
  box-shadow: 0 0 8px rgba(251, 191, 36, 0.35);
}

.meter-fill-vram {
  background: linear-gradient(90deg, #a78bfa, #c4b5fd);
  box-shadow: 0 0 6px rgba(167, 139, 250, 0.35);
}

.meter-sub {
  font-size: 0.62rem;
  color: #64748b;
  line-height: 1.2;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.metric-hint {
  color: #f59e0b;
  font-weight: 700;
  align-self: center;
  margin-left: 0.1rem;
}

/* --- profiling panel --- */
.profiling-panel {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  background-color: rgba(15, 23, 42, 0.7);
  border: 1px solid #1e293b;
  border-radius: 10px;
  padding: 0.4rem 0.75rem;
  flex-shrink: 1;
  min-width: 0;
}

.profiling-left {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  min-width: 58px;
}

.profiling-task-row {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.profiling-icon {
  color: #6366f1;
  flex-shrink: 0;
}

.profiling-task-label {
  font-size: 0.65rem;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
}

.profiling-task-count {
  font-size: 0.8rem;
  font-weight: 700;
  color: #f1f5f9;
  font-family: ui-monospace, 'Cascadia Code', 'Segoe UI Mono', monospace;
}

.profiling-bar-track {
  height: 4px;
  background: rgba(30, 41, 59, 0.9);
  border-radius: 2px;
  overflow: hidden;
  border: 1px solid rgba(51, 65, 85, 0.6);
}

.profiling-bar-fill {
  height: 100%;
  border-radius: 1px;
  background: linear-gradient(90deg, #ef4444, #f97316);
  box-shadow: 0 0 6px rgba(239, 68, 68, 0.4);
  transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.profiling-bar-done {
  background: linear-gradient(90deg, #059669, #10b981);
  box-shadow: 0 0 6px rgba(16, 185, 129, 0.4);
}

.profiling-grid {
  display: grid;
  grid-template-columns: repeat(3, auto);
  gap: 0.15rem 0.6rem;
  font-size: 0.7rem;
  white-space: nowrap;
}

.profiling-tag {
  display: flex;
  align-items: center;
  gap: 0.3rem;
}

.profiling-tag-label {
  color: #64748b;
}

.profiling-tag-value {
  font-weight: 700;
  font-family: ui-monospace, 'Cascadia Code', 'Segoe UI Mono', monospace;
  font-size: 0.72rem;
}

.profiling-device {
  font-size: 0.55rem;
  font-weight: 700;
  padding: 0px 3px;
  border-radius: 3px;
  letter-spacing: 0.03em;
  line-height: 1.4;
}

.profiling-device-gpu {
  background-color: rgba(251, 191, 36, 0.15);
  color: #fbbf24;
  border: 1px solid rgba(251, 191, 36, 0.3);
}

.profiling-device-cpu {
  background-color: rgba(56, 189, 248, 0.1);
  color: #38bdf8;
  border: 1px solid rgba(56, 189, 248, 0.2);
}

.profiling-device-api {
  background-color: rgba(167, 139, 250, 0.1);
  color: #a78bfa;
  border: 1px solid rgba(167, 139, 250, 0.2);
}

/* --- queue meter --- */
.meter-icon-queue {
  color: #6366f1;
}

.meter-value-waiting {
  color: #f59e0b !important;
}

.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
}
</style>
