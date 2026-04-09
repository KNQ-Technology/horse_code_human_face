# 人马识别系统 — 部署与运行文档

## 1. 项目概述

本系统通过 Web 界面上传赛马视频，后端调用 AI 算法（YOLO 马匹检测 + PaddleOCR 号码识别 + SCRFD/ArcFace 人脸识别）自动识别视频中的马匹编号与骑手身份，并输出带标注的可视化视频。

### 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Vue 3 + TypeScript + Vite |
| 后端 | Python 3.11 / FastAPI / Uvicorn |
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
│   ├── migrate_milvus_to_sqlite.py    # Milvus → SQLite 迁移工具（Windows 专用）
│   └── venv/                    # Python 虚拟环境（不纳入版本控制）
├── vue/
│   ├── src/
│   │   ├── App.vue              # 主界面（上传、预览、结果展示）
│   │   └── main.ts              # Vue 入口
│   ├── package.json
│   └── vite.config.ts
├── API_INTERFACE.md             # API 接口文档
├── DEPLOY.md                    # 本文档
└── .gitignore
```

---

## 2. 环境准备

### 2.1 通用要求

| 软件 | 版本 | 用途 |
|---|---|---|
| Python | 3.11 | 后端运行时 |
| Node.js | 18+ | 前端构建 |
| npm | 8+ | 前端包管理 |
| FFmpeg | 4.0+ | 视频 H.264 编码 |
| Git | 2.0+ | 版本控制 |
| NVIDIA 驱动 + CUDA | 12.6（推荐） | GPU 加速（YOLO / PaddleOCR / ONNX 推理） |

### 2.2 Ubuntu 环境安装

```bash
# Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# FFmpeg
sudo apt install -y ffmpeg

# 编译依赖（部分 Python 包需要）
sudo apt install -y build-essential libgl1-mesa-glx libglib2.0-0

# CUDA 12.6 驱动（如尚未安装）
# 参考 https://developer.nvidia.com/cuda-12-6-0-download-archive 选择对应 Ubuntu 版本
```

### 2.3 Windows 环境安装

1. **Python 3.11**：从 https://www.python.org/downloads/ 下载安装，勾选 "Add Python to PATH"。
2. **Node.js 18+**：从 https://nodejs.org/ 下载 LTS 版本安装。
3. **FFmpeg**：从 https://github.com/BtbN/FFmpeg-Builds/releases 下载，解压后将 `bin` 目录加入系统 `PATH`。
4. **CUDA 12.6**：从 https://developer.nvidia.com/cuda-12-6-0-download-archive 下载安装。安装后在命令行运行 `nvcc --version` 确认版本。
5. **Git**：从 https://git-scm.com/download/win 下载安装。

---

## 3. 获取代码

```bash
git clone <仓库地址> horse_code_human_face
cd horse_code_human_face
git checkout test
```

---

## 4. 后端部署

### 4.1 创建虚拟环境

**Ubuntu：**

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate
```

**Windows（PowerShell）：**

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4.2 安装 Python 依赖

Ubuntu 和 Windows 的依赖安装存在显著差异，请对照下表操作。

#### Ubuntu

```bash
# 直接安装全部依赖即可
pip install -r requirements.txt
```

`requirements.txt` 中的 `paddlepaddle-gpu`、`pymilvus[milvus_lite]`、`onnxruntime-gpu` 在 Ubuntu 下均可正常安装使用。

#### Windows

Windows 需要逐步处理以下差异：

**1) paddlepaddle-gpu：需从百度镜像源安装**

```bash
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

直接 `pip install paddlepaddle-gpu` 可能找不到对应 Windows + CUDA 12.6 的预编译包，必须指定百度官方源。

**2) pymilvus / milvus-lite：不支持 Windows**

`milvus-lite`（嵌入式 Milvus）不支持 Windows 平台，安装会失败。解决方案是使用 SQLite 降级方案：

```bash
# 先安装 pymilvus（不带 milvus_lite extra）
pip install pymilvus>=2.3.0

# 如果你有 rider.db（Milvus Lite 格式的人脸数据库），在 Ubuntu 或 WSL 中执行迁移：
python migrate_milvus_to_sqlite.py --src rider.db
# 会生成 rider.sqlite.db，将其复制到 Windows 的 backend/ 目录下
```

> 注意：迁移脚本 `migrate_milvus_to_sqlite.py` 需要在支持 milvus-lite 的环境（Ubuntu / WSL）中运行。

**3) onnxruntime-gpu vs onnxruntime**

```bash
# 如果 CUDA 已正确安装且版本兼容，可以安装 GPU 版本
pip install onnxruntime-gpu>=1.23.0

# 如果遇到兼容性问题，退回 CPU 版本
pip install onnxruntime>=1.23.0
```

**4) 其余依赖正常安装**

```bash
# 排除上面三个特殊包后，安装其余依赖
pip install fastapi uvicorn[standard] python-multipart "numpy>=1.24.0,<2.0.0" "setuptools>=65.0.0,<82" opencv-python==4.6.0.66 "ultralytics>=8.2.0" "lap>=0.5.12" "paddleocr>=2.7.0" "PyYAML>=6.0.0" "scikit-image>=0.19.0" "openai>=1.0.0" onnx
```

#### 依赖差异汇总

| 依赖包 | Ubuntu | Windows |
|---|---|---|
| `paddlepaddle-gpu` | `pip install paddlepaddle-gpu>=3.3.0` | 需从百度源安装：`pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/` |
| `pymilvus[milvus_lite]` | 直接可用 | `milvus-lite` 不支持 Windows，只安装 `pymilvus`，人脸库需用 SQLite 方案 |
| `onnxruntime-gpu` | 直接可用，加速 SCRFD 人脸检测 | 需确认 CUDA 版本兼容，不兼容时退回 `onnxruntime`（CPU） |

### 4.3 模型文件

算法需要以下模型文件：

| 模型 | 大小 | 下载方式 |
|---|---|---|
| `yolov8n.pt` (马匹检测) | ~6 MB | 首次运行时自动从 GitHub 下载到 `backend/` |
| PaddleOCR 模型 | ~100 MB | 首次运行时自动下载到 `~/.paddlex/` |

**人脸识别模型（需手动放置）：**

| 模型文件 | 放置位置 |
|---|---|
| `det_10g.onnx`（SCRFD 人脸检测） | `backend/horse_id/models/` |
| `w600k_r50.onnx`（ArcFace 人脸识别） | `backend/horse_id/models/` |

> 如果不需要人脸识别功能，无需放置这些文件，系统会自动禁用 `rider_identity` 模块。

### 4.4 人脸数据库

| 平台 | 数据库文件 | 说明 |
|---|---|---|
| Ubuntu | `rider.db` | Milvus Lite 格式，直接可用 |
| Windows | `rider.sqlite.db` | 需在 Ubuntu/WSL 中用 `migrate_milvus_to_sqlite.py` 从 `rider.db` 迁移生成 |

**迁移步骤（在 Ubuntu 或 WSL 中执行）：**

```bash
cd backend
python migrate_milvus_to_sqlite.py --src rider.db
# 输出: rider.sqlite.db
# 将 rider.sqlite.db 复制到 Windows 机器的 backend/ 目录
```

### 4.5 配置文件

算法参数位于 `backend/config/pipeline.yaml`，关键配置项说明：

```yaml
detector:
  model_path: "yolov8n.pt"    # YOLO 模型路径
  conf_threshold: 0.4          # 检测置信度阈值
  device: "0"                  # GPU 设备编号，"cpu" 表示纯 CPU

ocr:
  conf_threshold: 0.5          # OCR 置信度阈值
  regex_pattern: "^[A-Z][0-9]{3}$"  # 号码格式正则

rider_identity:
  face_db_uri: "rider.db"      # 人脸数据库路径（见下方 Windows 注意事项）
  face_device: "cpu"           # 人脸推理设备，"cpu" 或 "0"(GPU)
  face_models_dir: "horse_id/models"  # ONNX 模型目录

vlm_fallback:
  enabled: true                # VLM 兜底是否开启（需 API Key）
```

#### Windows 配置注意事项

1. **`rider_identity.face_db_uri`**：Windows 下 milvus-lite 不可用，如果使用 SQLite 方案，需将此项改为 `"rider.sqlite.db"`，并确保系统代码已支持 SQLite 后端。
2. **`rider_identity.face_device`**：无独立显卡或 CUDA 不可用时，确保设为 `"cpu"`。
3. **`detector.device`**：若无 GPU，改为 `"cpu"`。

### 4.6 环境变量（可选）

| 变量名 | 说明 | 示例 |
|---|---|---|
| `VLM_API_KEY` | 启用 Qwen VL 鞍垫号码识别兜底 | `sk-xxxx` |
| `FACE_DB_URI` | 人脸向量库路径（覆盖 pipeline.yaml 中的配置） | `./rider.db`（Ubuntu）或 `./rider.sqlite.db`（Windows） |
| `FACE_MODELS_DIR` | ONNX 人脸模型目录（默认 `horse_id/models/`） | `/path/to/models` |
| `HORSE_RIDER_MAP` | 马号到骑手的映射 JSON 文件路径 | `./horse_rider.json` |
| `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` | 跳过 PaddleOCR 网络检查（离线环境设为 `True`） | `True` |

### 4.7 启动后端

**Ubuntu：**

```bash
cd backend
source venv/bin/activate

# 基本启动
python main.py

# 带环境变量启动（启用人脸识别 + VLM）
FACE_DB_URI=./rider.db \
VLM_API_KEY=sk-你的密钥 \
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
python main.py
```

**Windows（PowerShell）：**

```powershell
cd backend
.\venv\Scripts\Activate.ps1

# 基本启动
python main.py

# 带环境变量启动
$env:FACE_DB_URI="./rider.sqlite.db"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
python main.py
```

启动成功后输出：

```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

**后端监听 `http://0.0.0.0:8001`。**

---

## 5. 前端部署

前端为标准 Vue 3 + Vite 项目，**无平台差异**，Ubuntu 和 Windows 操作一致。

### 5.1 开发模式（前后端分离）

```bash
cd vue
npm install
npm run dev
```

启动后访问 `http://localhost:5173`。前端会自动请求 `http://<当前主机名>:8001` 的后端 API（可通过 `VITE_API_BASE` 环境变量覆盖）。

> 开发模式需同时运行后端（8001）和前端开发服务器（5173）。

### 5.2 生产模式（一体化部署）

```bash
# 构建前端
cd vue
npm install
npm run build    # 产物输出到 vue/dist/

# 启动后端（后端自动托管 vue/dist/ 静态文件）
cd backend
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
python main.py
```

浏览器访问 `http://<服务器IP>:8001` 即可使用完整系统。

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
    "processed_video_url": "http://localhost:8001/videos/processed_xxx.mp4",
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

## 9. 常见问题排查

### Q: PaddleOCR 报 "GPU not available, switching to CPU"

**原因**：安装的是 `paddlepaddle`（CPU 版）而非 `paddlepaddle-gpu`。

**解决**：

```bash
pip uninstall paddlepaddle -y

# Ubuntu
pip install paddlepaddle-gpu>=3.3.0

# Windows（必须指定百度源）
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

安装后验证：

```python
import paddle
print(paddle.device.is_compiled_with_cuda())  # 应输出 True
```

### Q: 启动日志显示 "rider_identity disabled"

**可能原因：**

1. **缺少模型文件**：`horse_id/models/` 下缺少 `det_10g.onnx` 或 `w600k_r50.onnx`，需手动放入。
2. **缺少人脸数据库**：`rider.db` 文件不存在或路径不正确。
3. **Windows 上 milvus-lite 不可用**：Windows 不支持 milvus-lite，需要执行迁移生成 `rider.sqlite.db`，并修改 `pipeline.yaml` 中 `face_db_uri` 为 `"rider.sqlite.db"`（或设置环境变量 `FACE_DB_URI=./rider.sqlite.db`）。

### Q: 首次启动很慢

首次运行时算法模型会自动下载（YOLO ~6MB，PaddleOCR ~100MB），需要网络访问。后续启动使用缓存。

离线环境下设置 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` 跳过网络检查（前提是模型已缓存在 `~/.paddlex/`）。

### Q: 报错 `No module named 'lap'`

`lap` 是 ByteTrack 追踪器的依赖：

```bash
pip install lap>=0.5.12
```

### Q: 处理后视频无法播放

确保系统安装了 FFmpeg。没有 FFmpeg 时系统退化为 OpenCV 的 `mp4v` 编码，部分浏览器不支持。

```bash
# Ubuntu
sudo apt install -y ffmpeg

# Windows：下载 FFmpeg 并加入 PATH
```

### Q: 端口冲突（8001 已被占用）

**Ubuntu：**

```bash
# 查看占用 8001 端口的进程
lsof -i :8001
# 杀掉进程后重启，或修改 main.py 中的端口号
```

**Windows：**

```powershell
# 查看占用 8001 端口的进程
netstat -ano | findstr :8001
# 根据 PID 终止进程
taskkill /PID <PID> /F
```

也可修改 `backend/main.py` 最后一行的端口号：

```python
uvicorn.run(app, host="0.0.0.0", port=你的端口)
```

### Q: GPU 加速配置

| 组件 | 加速方式 |
|---|---|
| YOLO (马匹检测) | 安装 `torch` CUDA 版本，自动使用 GPU |
| PaddleOCR | 安装 `paddlepaddle-gpu` 替代 `paddlepaddle` |
| SCRFD/ArcFace (人脸) | 安装 `onnxruntime-gpu` 替代 `onnxruntime`；`pipeline.yaml` 中 `face_device` 设为 `"0"` |

---

## 10. 快速启动（TL;DR）

### Ubuntu

```bash
cd horse_code_human_face
git checkout test

# 后端
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py                # http://0.0.0.0:8001

# 前端（新开终端）
cd vue
npm install
npm run build

# 浏览器打开 http://localhost:8001
```

### Windows

```powershell
cd horse_code_human_face
git checkout test

# 后端
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1

# 分步安装依赖（注意差异项）
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install pymilvus>=2.3.0
pip install onnxruntime-gpu>=1.23.0          # 或 onnxruntime（无 GPU 时）
pip install fastapi uvicorn[standard] python-multipart "numpy>=1.24.0,<2.0.0" "setuptools>=65.0.0,<82" opencv-python==4.6.0.66 "ultralytics>=8.2.0" "lap>=0.5.12" "paddleocr>=2.7.0" "PyYAML>=6.0.0" "scikit-image>=0.19.0" "openai>=1.0.0" onnx

# 如需人脸识别：将 rider.sqlite.db 放入 backend/，设置环境变量
$env:FACE_DB_URI="./rider.sqlite.db"

python main.py                # http://0.0.0.0:8001

# 前端（新开终端）
cd vue
npm install
npm run build

# 浏览器打开 http://localhost:8001
```
