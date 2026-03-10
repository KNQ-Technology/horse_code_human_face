# 人马识别系统 — 部署与运行文档

## 1. 项目概述

本系统通过 Web 界面上传赛马视频，后端调用 AI 算法（YOLO 马匹检测 + PaddleOCR 号码识别 + SCRFD/ArcFace 人脸识别）自动识别视频中的马匹编号与骑手身份，并输出带标注的可视化视频。

### 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Vue 3 + TypeScript + Vite |
| 后端 | Python 3.10+ / FastAPI / Uvicorn |
| 算法 | YOLOv8 (马匹检测) + ByteTrack (多目标追踪) + PaddleOCR (号码识别) + SCRFD + ArcFace (人脸识别) |
| 视频编码 | FFmpeg (H.264) |

### 项目结构

```
horse_code_human_face/
├── backend/
│   ├── main.py                  # FastAPI 入口（API 路由、静态文件托管）
│   ├── processor.py             # 算法集成层（核心处理逻辑）
│   ├── requirements.txt         # Python 依赖
│   ├── config/
│   │   └── pipeline.yaml        # 算法参数配置
│   ├── horse_id/                # 算法模块
│   │   ├── detector.py          #   YOLO 马匹检测 + ByteTrack
│   │   ├── ocr_engine.py        #   PaddleOCR 号码识别
│   │   ├── rider_identity.py    #   骑手身份识别
│   │   ├── visualizer.py        #   可视化绘制
│   │   ├── face/                #   人脸检测/识别子模块
│   │   └── ...                  #   其他模块
│   ├── save_rider_faces_to_milvus.py  # 人脸入库工具
│   └── venv/                    # Python 虚拟环境（不纳入版本控制）
├── vue/
│   ├── src/
│   │   ├── App.vue              # 主界面（上传、预览、结果展示）
│   │   └── main.ts              # Vue 入口
│   ├── package.json
│   └── vite.config.ts
├── videos/                      # 运行时视频存储（上传 + 处理结果）
├── API_INTERFACE.md             # API 接口文档
├── DEPLOY.md                    # 本文档
└── .gitignore
```

---

## 2. 环境要求

### 2.1 操作系统

Linux（推荐 Ubuntu 20.04+）。macOS 也可运行但未充分测试。

### 2.2 系统依赖

| 软件 | 最低版本 | 用途 |
|---|---|---|
| Python | 3.10+ | 后端运行时 |
| Node.js | 18+ | 前端构建 |
| npm | 8+ | 前端包管理 |
| FFmpeg | 4.0+ | 视频 H.264 编码（可选但强烈推荐） |
| Git | 2.0+ | 版本控制 |

**安装系统依赖（Ubuntu）：**

```bash
# Python 3.10+（Ubuntu 22.04 自带）
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

# Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# FFmpeg
sudo apt install -y ffmpeg

# 其他编译依赖（部分 Python 包可能需要）
sudo apt install -y build-essential libgl1-mesa-glx libglib2.0-0
```

---

## 3. 获取代码

```bash
git clone <仓库地址> horse_code_human_face
cd horse_code_human_face

# 切换到 test 分支（包含算法集成）
git checkout test
```

---

## 4. 后端配置与启动

### 4.1 创建 Python 虚拟环境

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

### 4.2 安装 Python 依赖

```bash
pip install -r requirements.txt
```

这会安装全部依赖，包括 FastAPI、Ultralytics (YOLO)、PaddleOCR、ONNX Runtime 等。首次安装可能需要 3-5 分钟。

### 4.3 模型文件

算法需要以下模型文件，它们会按需自动下载：

| 模型 | 大小 | 下载方式 |
|---|---|---|
| `yolov8n.pt` (马匹检测) | ~6 MB | 首次运行时自动从 GitHub 下载到 `backend/` |
| PaddleOCR 模型 | ~100 MB | 首次运行时自动下载到 `~/.paddlex/` |

人脸识别模型（可选）需手动放置：

| 模型 | 位置 |
|---|---|
| `det_10g.onnx` (SCRFD 人脸检测) | `backend/horse_id/models/` |
| `w600k_r50.onnx` (ArcFace 人脸识别) | `backend/horse_id/models/` |

> 如果不需要人脸识别功能，无需放置这些文件，系统会自动禁用该模块。

### 4.4 配置文件

算法参数位于 `backend/config/pipeline.yaml`，通常无需修改。关键参数说明：

```yaml
detector:
  model_path: "yolov8n.pt"    # YOLO 模型路径
  conf_threshold: 0.4          # 检测置信度阈值

ocr:
  conf_threshold: 0.5          # OCR 置信度阈值
  regex_pattern: "^[A-Z][0-9]{2,3}$"  # 号码格式正则

vlm_fallback:
  enabled: false               # VLM 兜底默认关闭（需要 API Key）
```

### 4.5 环境变量（可选）

以下环境变量可启用高级功能：

| 变量名 | 说明 | 示例 |
|---|---|---|
| `VLM_API_KEY` | 启用 Qwen VL 鞍垫号码识别兜底 | `sk-xxxx` |
| `FACE_DB_URI` | Milvus 人脸向量库路径，启用骑手人脸识别 | `./face_db.db` |
| `FACE_MODELS_DIR` | ONNX 人脸模型目录（默认 `horse_id/models/`） | `/path/to/models` |
| `HORSE_RIDER_MAP` | 马号到骑手的映射 JSON 文件路径 | `./horse_rider.json` |
| `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` | 跳过 PaddleOCR 网络检查（离线环境设为 `True`） | `True` |

### 4.6 启动后端

```bash
cd backend
source venv/bin/activate

# 基本启动
python main.py

# 带环境变量启动（启用人脸识别 + VLM）
FACE_DB_URI=./face_db.db \
VLM_API_KEY=sk-你的密钥 \
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
python main.py
```

启动成功后输出：

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**后端监听 `http://0.0.0.0:8000`。**

---

## 5. 前端配置与启动

### 5.1 开发模式（前后端分离）

开发模式下前端 Vite 开发服务器独立运行，通过 CORS 与后端通信。

```bash
cd vue
npm install
npm run dev
```

启动成功后输出：

```
VITE vX.X.X ready
➜  Local:   http://localhost:5173/
```

浏览器访问 `http://localhost:5173` 即可。前端会自动请求 `http://localhost:8000` 的后端 API。

> **注意**：开发模式需要同时运行后端（端口 8000）和前端开发服务器（端口 5173）。

### 5.2 生产模式（一体化部署）

生产模式下后端直接托管前端构建产物，只需一个端口。

**第一步：构建前端**

```bash
cd vue
npm install
npm run build
```

构建产物输出到 `vue/dist/` 目录。

**第二步：启动后端**

```bash
cd backend
source venv/bin/activate
python main.py
```

浏览器访问 `http://<服务器IP>:8000` 即可使用完整系统。后端会自动托管 `vue/dist/` 下的静态文件。

---

## 6. 使用流程

1. 打开浏览器访问系统页面
2. 点击上传区域，选择一个赛马视频文件（支持 MP4、AVI、MOV 等格式）
3. 点击 **"开始上传处理"** 按钮
4. 观察右侧面板：
   - **上传进度**：文件上传到服务器的进度
   - **AI 处理进度**：算法逐帧分析的进度（基于已处理帧数/总帧数）
5. 处理完成后：
   - 视频区域自动切换到 **处理后视频**（带检测框和标注）
   - 右侧表格展示检测到的马匹编号、骑手名称和置信度
   - 可展开查看完整的 JSON 原始数据
6. 点击 **"上传新视频"** 可重新开始

---

## 7. API 接口

### 7.1 上传视频

```
POST /api/upload
Content-Type: multipart/form-data

参数：video (File) — 视频文件
```

响应：

```json
{
  "code": 200,
  "message": "上传成功",
  "data": { "task_id": "uuid-xxx" }
}
```

### 7.2 查询状态

```
GET /api/status?task_id=<uuid>
```

处理中：

```json
{
  "code": 200,
  "data": {
    "status": "processing",
    "progress": 45,
    "message": "正在进行 AI 识别... 45%"
  }
}
```

处理完成：

```json
{
  "code": 200,
  "data": {
    "status": "completed",
    "progress": 100,
    "processed_video_url": "http://localhost:8000/videos/processed_xxx.mp4",
    "result": {
      "filename": "race.mp4",
      "duration": "01:30",
      "resolution": "1920x1080",
      "processed_at": "2026-03-10 12:00:00",
      "detections": [
        {
          "timestamp": "00:05",
          "horse_id": "A12",
          "person_name": "张三",
          "confidence": "0.95_0.88"
        }
      ]
    }
  }
}
```

---

## 8. 算法流水线说明

每一帧视频经过以下处理步骤：

```
输入帧
  │
  ├─ 1. HorseDetector (YOLOv8 + ByteTrack)
  │     检测马匹位置，分配 track_id 跨帧追踪
  │
  ├─ 2. ROIExtractor
  │     从马匹框中提取左肩区域 (ROI)
  │
  ├─ 3. ROIEnhancer
  │     CLAHE、Gamma、双边滤波、锐化、自适应二值化
  │
  ├─ 4. OCREngine (PaddleOCR)
  │     识别 ROI 中的马号文字
  │     └─ VLMFallback (Qwen VL，可选)
  │        如果 OCR 失败，调用视觉语言模型兜底
  │
  ├─ 5. OCRTrackFuser
  │     轨迹级 OCR 滑窗投票，输出稳定的 stable_id
  │
  ├─ 6. TrackStateMachine
  │     UNCONFIRMED → CONFIRMED → HOLD 状态流转
  │
  ├─ 7. RiderIdentityModule (可选)
  │     人脸检测(SCRFD) + 人脸识别(ArcFace) + 颜色特征 + 马号映射
  │     融合多特征识别骑手身份
  │
  └─ 8. ResultVisualizer
        绘制检测框、马号、骑手信息到帧上
        输出带标注的可视化视频
```

---

## 9. 常见问题

### Q: 首次启动很慢？

首次运行时算法模型会自动下载（YOLO ~6MB，PaddleOCR ~100MB），需要网络访问。后续启动会使用缓存，速度正常。

如果处于离线环境，设置 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` 跳过网络检查（前提是模型已缓存在 `~/.paddlex/`）。

### Q: 报错 `No module named 'lap'`？

`lap` 是 ByteTrack 追踪器的依赖。确保在虚拟环境内安装：

```bash
source venv/bin/activate
pip install lap>=0.5.12
```

### Q: 处理后视频无法播放？

确保系统安装了 FFmpeg。没有 FFmpeg 时系统会退化为 OpenCV 的 `mp4v` 编码，部分浏览器不支持。

```bash
sudo apt install -y ffmpeg
```

### Q: 如何启用人脸识别功能？

需要三步：

1. 将 ONNX 模型文件（`det_10g.onnx`、`w600k_r50.onnx`）放入 `backend/horse_id/models/`
2. 准备人脸向量库（使用 `save_rider_faces_to_milvus.py` 入库）
3. 启动时指定环境变量：`FACE_DB_URI=./face_db.db python main.py`

### Q: 如何在远程服务器部署后从本地访问？

后端默认监听 `0.0.0.0:8000`，可通过服务器 IP 直接访问。如果需要修改端口：

```python
# backend/main.py 最后一行
uvicorn.run(app, host="0.0.0.0", port=你的端口)
```

前端开发模式下 API 地址硬编码为 `http://localhost:8000`，远程部署建议使用**生产模式**（前端构建后由后端托管）。

### Q: GPU 加速？

- **YOLO**：安装 `torch` 的 CUDA 版本即可自动使用 GPU
- **PaddleOCR**：安装 `paddlepaddle-gpu` 替代 `paddlepaddle`
- **ONNX (人脸)**：安装 `onnxruntime-gpu` 替代 `onnxruntime`

---

## 10. 快速启动（TL;DR）

```bash
# 1. 切换分支
cd horse_code_human_face
git checkout test

# 2. 后端
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py                # 后端运行在 http://0.0.0.0:8000

# 3. 前端（新开终端）
cd vue
npm install
npm run build                 # 构建前端

# 4. 访问
# 浏览器打开 http://localhost:8000
```
