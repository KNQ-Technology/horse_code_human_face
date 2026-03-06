from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uuid
import time
import os
import shutil
from typing import Dict

app = FastAPI()

# Video storage directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
UPLOAD_DIR = os.path.join(VIDEOS_DIR, "upload")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Enable CORS (Optional in production if same origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for task status
tasks: Dict[str, dict] = {}

def process_video_task(task_id: str, filename: str):
    """Simulate backend processing logic"""
    tasks[task_id]["status"] = "processing"
    
    # Path of uploaded video
    uploaded_path = os.path.join(UPLOAD_DIR, f"{task_id}_{filename}")
    # Path for processed video
    processed_filename = f"processed_{task_id}_{filename}"
    processed_path = os.path.join(VIDEOS_DIR, processed_filename)
    
    for i in range(0, 101, 10):
        tasks[task_id]["progress"] = i
        tasks[task_id]["message"] = f"正在进行 AI 识别... {i}%"
        time.sleep(1)  # Simulate work
        
    # Simulate processing result by copying the original file
    if os.path.exists(uploaded_path):
        shutil.copy(uploaded_path, processed_path)
        
    tasks[task_id]["status"] = "completed"
    tasks[task_id]["progress"] = 100
    # Point to the locally served video
    tasks[task_id]["processed_video_url"] = f"http://localhost:8000/videos/{processed_filename}"
    tasks[task_id]["result"] = {
        "filename": filename,
        "duration": "00:30",
        "resolution": "1920x1080",
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detections": [
            {
                "timestamp": "00:05",
                "horse_id": "H001",
                "person_name": "张三",
                "confidence": "0.98_0.95"
            },
            {
                "timestamp": "00:12",
                "horse_id": "H005",
                "person_name": "李四",
                "confidence": "0.92_0.88"
            },
            {
                "timestamp": "00:25",
                "horse_id": "H003",
                "person_name": "王五",
                "confidence": "0.95_0.91"
            }
        ]
    }

@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    
    # Path to save uploaded video
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{video.filename}")
    
    # Ensure directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    tasks[task_id] = {
        "status": "uploading",
        "progress": 0,
        "message": "文件已接收，等待处理",
        "file_path": file_path
    }
    
    # Start background processing
    background_tasks.add_task(process_video_task, task_id, video.filename)
    
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
        "data": tasks[task_id]
    }

# 1. Map assets (js, css)
vue_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vue", "dist"))
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
