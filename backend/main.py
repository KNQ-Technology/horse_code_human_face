from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import traceback
import time
import uuid
import os
import re
import shutil
import threading
from collections import deque
from typing import Dict

from processor import process_video
from system_metrics import get_system_metrics

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_VIDEOS_DIR = "/home/data/horse_video"
APP_PORT = int(os.getenv("PORT", "8002"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks: Dict[str, dict] = {}

CONFIG_PATH = os.getenv("PIPELINE_CONFIG", os.path.join(BASE_DIR, "config", "pipeline.yaml"))
_VALID_PIPELINE_MODES = frozenset({"full", "simple", "face"})

# --------------- 串行任务队列 ---------------
_task_queue: deque = deque()          # 元素: (task_id, filename, mode)
_queue_lock = threading.Lock()
_queue_event = threading.Event()      # 通知 worker 有新任务


def _queue_worker():
    """后台线程：从队列中逐个取出任务并执行，确保同一时间只处理一个视频。"""
    while True:
        _queue_event.wait()           # 等待有任务入队
        while True:
            with _queue_lock:
                if not _task_queue:
                    _queue_event.clear()
                    break
                job = _task_queue.popleft()
                # 更新剩余排队任务的 queue_position
                for idx, (qid, _, _) in enumerate(_task_queue):
                    if qid in tasks:
                        tasks[qid]["queue_position"] = idx + 1
            task_id, filename, mode = job
            process_video_task(task_id, filename, mode)


_worker_thread = threading.Thread(target=_queue_worker, daemon=True)
_worker_thread.start()


def _sanitize_error_message(exc: Exception) -> str:
    raw_message = str(exc).strip() or exc.__class__.__name__
    sanitized = re.sub(r"(?:[A-Za-z]:)?/[^\s]+", "<path>", raw_message)
    return sanitized


def _choose_videos_dir() -> str:
    env_dir = os.getenv("VIDEOS_DIR", "").strip()
    candidates = [env_dir] if env_dir else [LEGACY_VIDEOS_DIR, os.path.join(BASE_DIR, "outputs", "videos")]
    for candidate in candidates:
        try:
            upload_dir = os.path.join(candidate, "upload")
            os.makedirs(upload_dir, exist_ok=True)
            probe_path = os.path.join(upload_dir, ".write_probe")
            with open(probe_path, "w", encoding="utf-8") as probe:
                probe.write("ok")
            os.remove(probe_path)
            return candidate
        except OSError:
            continue
    raise RuntimeError("No writable video directory available. Set VIDEOS_DIR to a writable path.")


def _public_task_state(task: dict) -> dict:
    return {key: value for key, value in task.items() if key != "file_path"}


VIDEOS_DIR = _choose_videos_dir()
UPLOAD_DIR = os.path.join(VIDEOS_DIR, "upload")


def process_video_task(task_id: str, filename: str, mode: str = "full"):
    tasks[task_id]["status"] = "processing"

    uploaded_path = os.path.join(UPLOAD_DIR, f"{task_id}_{filename}")

    base_name, ext = os.path.splitext(filename)
    processed_filename = f"processed_{task_id}_{base_name}.mp4"
    processed_path = os.path.join(VIDEOS_DIR, processed_filename)

    _job_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    _job_t0 = time.perf_counter()
    print(
        f"[STEP] 开始: 后台任务 process_video | {_job_ts} | "
        f"task={task_id[:8]} file={filename} mode={mode}"
    )
    try:
        result = process_video(
            task_id=task_id,
            video_path=uploaded_path,
            output_video_path=processed_path,
            tasks=tasks,
            config_path=CONFIG_PATH,
            mode=mode,
        )
        _job_dt = time.perf_counter() - _job_t0
        print(
            f"[STEP] 结束: 后台任务 process_video | {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"耗时={_job_dt:.2f}s | task={task_id[:8]} status=completed"
        )
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["processed_video_url"] = f"/videos/{processed_filename}"
        tasks[task_id]["result"] = result
    except Exception as e:
        _job_dt = time.perf_counter() - _job_t0
        print(
            f"[STEP] 结束: 后台任务 process_video | {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"耗时={_job_dt:.2f}s | task={task_id[:8]} status=error"
        )
        traceback.print_exc()
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = f"处理失败: {_sanitize_error_message(e)}"

@app.post("/api/upload")
async def upload_video(
    video: UploadFile = File(...),
    mode: str = Form("full"),
):
    if mode not in _VALID_PIPELINE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode 必须是 full | simple | face，收到: {mode!r}",
        )
    task_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{video.filename}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    with _queue_lock:
        queue_position = len(_task_queue) + 1  # 排在已有队列之后
        # 如果队列为空且没有正在处理的任务，直接开始处理（position=0）
        has_processing = any(t.get("status") == "processing" for t in tasks.values())
        if not _task_queue and not has_processing:
            initial_status = "queued"
            queue_position = 0
        else:
            initial_status = "queued"

        tasks[task_id] = {
            "status": initial_status,
            "progress": 0,
            "message": f"排队中，前方还有 {queue_position} 个任务" if queue_position > 0 else "即将开始处理",
            "queue_position": queue_position,
            "file_path": file_path,
        }
        _task_queue.append((task_id, video.filename, mode))
        _queue_event.set()

    return {
        "code": 200,
        "message": "上传成功",
        "data": {
            "task_id": task_id
        }
    }

@app.get("/api/status")
async def get_status(task_id: str):
    if task_id not in tasks:
        return {"code": 404, "message": "任务不存在"}
    
    return {
        "code": 200,
        "data": _public_task_state(tasks[task_id])
    }


@app.get("/api/queue")
async def get_queue_status():
    """返回当前队列状态：正在处理的任务和排队中的任务列表。"""
    processing = []
    queued = []
    for tid, t in tasks.items():
        if t.get("status") == "processing":
            processing.append({"task_id": tid, "progress": t.get("progress", 0)})
        elif t.get("status") == "queued":
            queued.append({"task_id": tid, "queue_position": t.get("queue_position", 0)})
    queued.sort(key=lambda x: x["queue_position"])
    return {
        "code": 200,
        "data": {
            "processing": processing,
            "queued": queued,
            "queue_length": len(queued),
        }
    }


@app.get("/api/profiling")
async def get_profiling():
    """返回当前或最近一次任务的各阶段耗时占比。优先返回正在处理的，否则返回最近完成的。"""
    def _build_profiling_response(tid: str, t: dict) -> dict:
        return {
            "code": 200,
            "data": {
                "task_id": tid,
                "progress": t.get("progress", 0),
                "status": t["status"],
                "profiling": t["profiling"],
                "devices": t.get("stage_devices", {}),
            }
        }
    # 优先：正在处理的任务
    for tid, t in tasks.items():
        if t.get("status") == "processing" and "profiling" in t:
            return _build_profiling_response(tid, t)
    # 其次：最近完成/出错且有 profiling 的任务
    for tid, t in reversed(list(tasks.items())):
        if t.get("status") in ("completed", "error") and "profiling" in t:
            return _build_profiling_response(tid, t)
    return {"code": 200, "data": None}


@app.get("/api/system/metrics")
async def system_metrics():
    """CPU / GPU 利用率（与后端进程所在机器一致），供前端轮询展示。"""
    try:
        data = get_system_metrics()
        return {"code": 200, "data": data}
    except Exception as e:
        return {
            "code": 500,
            "message": f"读取系统指标失败: {_sanitize_error_message(e)}",
            "data": {
                "cpu_percent": None,
                "memory_percent": None,
                "memory_used_gb": None,
                "memory_total_gb": None,
                "gpu_percent": None,
                "gpu_available": False,
                "gpus": [],
                "gpu_vram_percent": None,
                "gpu_memory_used_mib": None,
                "gpu_memory_total_mib": None,
            },
        }


# 1. Map assets (js, css)
vue_dist = os.path.abspath(os.path.join(BASE_DIR, "..", "vue", "dist"))
if os.path.exists(vue_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(vue_dist, "assets")), name="assets")

# 3. Serve video files
if os.path.exists(VIDEOS_DIR):
    app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")

# 2. Map all other routes to index.html
@app.get("/{full_path:path}")
async def serve_vue(full_path: str):
    index_file = os.path.join(vue_dist, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Vue dist folder not found. Please run 'npm run build' first."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
