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
├── /mnt/nas/【赛马会识别】/temp   # 运行时视频存储（上传 + 处理结果，NAS 挂载路径，可在 main.py 中修改）
├── API_INTERFACE.md             # API 接口文档
├── DEPLOY.md                    # 本文档
└── .gitignore
```

---

## 2. 环境要求

### 2.1 操作系统

- **Linux**（推荐 Ubuntu 20.04+）— 功能最完整
- **Windows 10/11** — 完整支持（见下方 Windows 特别说明）
- macOS 可运行但未充分测试

### 2.2 系统依赖

| 软件 | 最低版本 | 用途 |
|---|---|---|
| Python | 3.10+（推荐 3.11） | 后端运行时 |
| Node.js | 18+ | 前端构建 |
| npm | 8+ | 前端包管理 |
| FFmpeg | 4.0+ | 视频 H.264 编码（可选但强烈推荐） |
| Git | 2.0+ | 版本控制 |
| NVIDIA 驱动 | ≥ 525（对应 CUDA 12.x） | GPU 加速必需 |

### 2.3 Ubuntu 安装系统依赖

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

### 2.4 Windows 安装系统依赖

推荐工具链：

| 软件 | 下载方式 |
|---|---|
| Python 3.11 | https://www.python.org/downloads/ （勾选 "Add Python to PATH"） |
| Node.js 18 LTS | https://nodejs.org/ |
| FFmpeg | https://www.gyan.dev/ffmpeg/builds/ （将 `ffmpeg.exe` 所在 `bin/` 加入 PATH） |
| Git for Windows | https://git-scm.com/download/win |
| NVIDIA 驱动 | GeForce Experience 或 https://www.nvidia.com/Download/index.aspx |

**验证安装：** 打开 PowerShell 执行

```powershell
python --version           # Python 3.11.x
node --version             # v18.x+
ffmpeg -version            # ffmpeg version 4.x+
nvidia-smi                 # 显示 GPU 信息与驱动版本
```

> 所有命令在 **PowerShell** 中执行（非 cmd.exe）。项目已在 Windows 11 + RTX 4070 + Python 3.11 + CUDA 13.1 驱动环境下验证通过。

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

**Ubuntu / macOS：**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

**Windows（PowerShell）：**
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1

# 如遇到 "禁止运行脚本" 错误，先执行（仅一次）：
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 4.2 安装 Python 依赖

```bash
pip install -r requirements.txt
```

这会安装全部基础依赖，包括 FastAPI、Ultralytics (YOLO)、PaddleOCR、ONNX Runtime 等。首次安装可能需要 3-5 分钟。

#### 4.2.1 GPU 加速包（强烈推荐）

本项目针对 GPU 做了深度优化，OCR 和人脸检测在 GPU 上比 CPU 快 10-60 倍。**不装 GPU 包会导致处理速度慢一个数量级**。

**Ubuntu：**
```bash
# PaddlePaddle GPU 版（替换 requirements.txt 中的 paddlepaddle）
pip uninstall -y paddlepaddle
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# onnxruntime GPU 版
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.23.0
```

**Windows（PowerShell）：**
```powershell
# PaddlePaddle GPU 版
pip uninstall -y paddlepaddle
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# onnxruntime GPU 版
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.23.0

# 重要：Windows 下 paddlepaddle-gpu 会拉入 nvidia-cudnn-cu12 等独立 CUDA 包，
# 它们与 PyTorch 自带的 CUDA 运行时冲突。需要卸载：
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 `
                  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 `
                  nvidia-cusparse-cu12 nvidia-nvjitlink-cu12
```

> **Windows 特别说明**：已在 `backend/_nvidia_dll_fix.py` 中处理了 cuDNN DLL 加载路径问题（复用 `torch/lib` 目录下的 CUDA 库），`main.py` 会在启动时自动加载，无需手动配置 PATH。

### 4.3 模型文件

算法需要以下模型文件，它们会按需自动下载：

| 模型 | 大小 | 下载方式 |
|---|---|---|
| `yolov8n.pt` (马匹检测) | ~6 MB | 首次运行时自动从 GitHub 下载到 `backend/` |
| PaddleOCR 模型 | ~100 MB | 首次运行时自动下载到 `~/.paddlex/` |

人脸识别模型（可选）需手动放置：

| 模型 | 位置 | 大小 |
|---|---|---|
| `det_10g.onnx` (SCRFD 人脸检测) | `backend/horse_id/models/` | ~17 MB |
| `w600k_r50.onnx` (ArcFace 人脸识别) | `backend/horse_id/models/` | ~175 MB |

> 如果不需要人脸识别功能，无需放置这些文件，系统会自动禁用该模块。

#### 4.3.1 人脸特征库（rider.db）

骑手识别需要预先录入的人脸特征库。两种格式：

| 格式 | 后缀 | 生成方式 | 平台 |
|---|---|---|---|
| Milvus Lite | `rider.db` | `python save_rider_faces_to_milvus.py --uri rider.db --rider-dir <照片目录>` | 仅 Ubuntu |
| 标准 SQLite | `rider.sqlite.db` | 同上（Windows 自动使用） 或从 Milvus Lite 迁移 | Windows + Ubuntu |

**Windows 必须使用 SQLite 格式**（milvus-lite 不支持 Windows）。如果已有 Ubuntu 生成的 `rider.db`，拷贝到 Windows 后执行迁移：

```powershell
cd backend
python migrate_milvus_to_sqlite.py --src rider.db
# 输出: rider.sqlite.db（项目会自动加载）
```

系统会在 `rider_identity` 模块初始化时自动检测：Ubuntu 优先用 `rider.db`，Windows 自动降级到 `rider.sqlite.db`。

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

**Ubuntu / macOS：**
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
$env:FACE_DB_URI = "./rider.sqlite.db"
$env:VLM_API_KEY = "sk-你的密钥"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
python main.py
```

启动成功后输出：

```
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
```

**后端监听 `http://0.0.0.0:8002`。** 可通过 `PORT` 环境变量修改端口。

---

## 5. 前端配置与启动

### 5.1 开发模式（前后端分离）

开发模式下前端 Vite 开发服务器独立运行，并通过 Vite 的 `/api` 代理转发到后端。

**Ubuntu / macOS / Windows（命令相同）：**
```bash
cd vue
npm install
npm run dev
```

启动成功后输出：

```
VITE v7.3.1  ready in 400 ms

➜  Local:   http://localhost:8001/
➜  Network: http://<局域网 IP>:8001/
```

浏览器访问 `http://localhost:8001` 即可。Vite 会自动将 `/api/*` 和 `/videos/*` 请求转发到后端 `http://localhost:8002`。

> **端口约定**（由 `vue/vite.config.ts` 定义）：
> - 前端 Vite：`8001`
> - 后端 FastAPI：`8002`（代理目标）
>
> **注意**：开发模式需要同时运行后端（端口 8002）和前端开发服务器（端口 8001），顺序随意。

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

Ubuntu：
```bash
cd backend
source venv/bin/activate
python main.py
```

Windows：
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python main.py
```

浏览器访问 `http://<服务器IP>:8002` 即可使用完整系统。后端会自动托管 `vue/dist/` 下的静态文件。

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
    "processed_video_url": "http://localhost:8002/videos/processed_xxx.mp4",
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
2. 准备人脸向量库：
   - Ubuntu：`python save_rider_faces_to_milvus.py --uri rider.db --rider-dir <照片目录>`
   - Windows：同上命令自动生成 `rider.sqlite.db`；或从已有 `rider.db` 迁移：`python migrate_milvus_to_sqlite.py --src rider.db`
3. `pipeline.yaml` 已默认启用（`face_db_uri: "rider.db"`），直接启动即可

### Q: 如何在远程服务器部署后从本地访问？

后端默认监听 `0.0.0.0:8002`，可通过服务器 IP 直接访问。修改端口：

```bash
# Ubuntu
PORT=9000 python main.py
```
```powershell
# Windows
$env:PORT = "9000"; python main.py
```

远程部署建议使用**生产模式**（前端构建后由后端托管，只开一个端口）。

### Q: Windows 下启动报错 `[WinError 127] 找不到指定的程序. Error loading cudnn_cnn64_9.dll`？

这是 PaddleOCR GPU 版在 Windows 下的 cuDNN 加载问题。项目已通过 `backend/_nvidia_dll_fix.py` 自动修复 — 它复用了 PyTorch 自带的 CUDA 库。如果仍报错：

1. 确认同时装了 `torch`（PyTorch 会带完整 CUDA 运行时）
2. 确认已卸载独立的 nvidia-* 包（见 4.2.1 节命令）
3. 确认 `backend/main.py` 和 `backend/processor.py` 顶部都 `import _nvidia_dll_fix`

### Q: Windows 下启动报错 `No module named 'milvus_lite'`？

`milvus-lite` 不支持 Windows。系统会自动降级到 SQLite 后端，但需要 `rider.sqlite.db`（而非 `rider.db`）。执行迁移：

```powershell
cd backend
python migrate_milvus_to_sqlite.py --src rider.db
```

### Q: Windows 下 PowerShell 报错 `无法加载文件 Activate.ps1，因为在此系统上禁止运行脚本`？

一次性放开当前用户的脚本执行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# 然后再运行:
.\venv\Scripts\Activate.ps1
```

### Q: 端口 8001 或 8002 被占用？

查找并终止占用进程：

```powershell
# Windows
netstat -ano | findstr :8002
taskkill /F /PID <PID>
```
```bash
# Ubuntu
lsof -i :8002
kill -9 <PID>
```

或者通过 `PORT` 环境变量改用其他端口（后端），前端端口在 `vue/vite.config.ts` 的 `server.port` 修改。

### Q: GPU 加速？

- **YOLO**：安装 `torch` 的 CUDA 版本即可自动使用 GPU
- **PaddleOCR**：安装 `paddlepaddle-gpu` 替代 `paddlepaddle`（见 4.2.1）
- **ONNX (人脸)**：安装 `onnxruntime-gpu` 替代 `onnxruntime`（见 4.2.1）

装完后在 `pipeline.yaml` 中：
```yaml
detector:
  device: "0"        # YOLO 用 GPU
rider_identity:
  face_device: "cuda"  # SCRFD/ArcFace 用 GPU
```
OCR 在代码中硬编码用 `gpu:0`，只要装了 `paddlepaddle-gpu` 就自动生效。

---

## 10. 快速启动（TL;DR）

### Ubuntu

```bash
# 1. 后端（新开一个终端）
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip uninstall -y paddlepaddle onnxruntime
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install onnxruntime-gpu==1.23.0
python main.py                # 后端: http://0.0.0.0:8002

# 2. 前端（新开一个终端）
cd vue
npm install
npm run dev                   # 前端: http://localhost:8001

# 3. 浏览器打开 http://localhost:8001
```

### Windows（PowerShell）

```powershell
# 1. 后端（新开一个 PowerShell 窗口）
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# GPU 加速包（强烈推荐）
pip uninstall -y paddlepaddle onnxruntime
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install onnxruntime-gpu==1.23.0

# 卸载与 torch 冲突的独立 CUDA 包
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 `
                  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 `
                  nvidia-cusparse-cu12 nvidia-nvjitlink-cu12

# 如果已有 Ubuntu 的 rider.db，迁移到 SQLite 格式
python migrate_milvus_to_sqlite.py --src rider.db

python main.py                # 后端: http://0.0.0.0:8002

# 2. 前端（新开一个 PowerShell 窗口）
cd vue
npm install
npm run dev                   # 前端: http://localhost:8001

# 3. 浏览器打开 http://localhost:8001
```

### 验证启动成功

- 前端页面能打开 → Vite 正常
- 页面底部监控条显示 CPU/GPU 数据 → 后端 API 正常
- 上传视频后日志出现 `[rider_identity] enabled`（若放了人脸库） → 全链路正常
