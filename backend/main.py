from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import traceback
import uuid
import os
import re
import shutil
from typing import Dict

from processor import process_video

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_VIDEOS_DIR = "/mnt/nas/【赛马会识别】/temp"
APP_PORT = int(os.getenv("PORT", "8001"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks: Dict[str, dict] = {}

CONFIG_PATH = os.getenv("PIPELINE_CONFIG", os.path.join(BASE_DIR, "config", "pipeline.yaml"))


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

    try:
        result = process_video(
            task_id=task_id,
            video_path=uploaded_path,
            output_video_path=processed_path,
            tasks=tasks,
            config_path=CONFIG_PATH,
            mode=mode,
        )
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["processed_video_url"] = f"/videos/{processed_filename}"
        tasks[task_id]["result"] = result
    except Exception as e:
        traceback.print_exc()
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = f"处理失败: {_sanitize_error_message(e)}"

@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    mode: str = Form("full"),
):
    task_id = str(uuid.uuid4())
    
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{video.filename}")
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    tasks[task_id] = {
        "status": "uploading",
        "progress": 0,
        "message": "文件已接收，等待处理",
        "file_path": file_path
    }
    
    background_tasks.add_task(process_video_task, task_id, video.filename, mode)
    
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
