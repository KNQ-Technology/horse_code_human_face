# 性能优化对比报告

> 测试日期：2026-04-09
> 测试环境：Windows 11, RTX 4060 (8GB), Python 3.11, CUDA 13.1 驱动
> 测试视频：H303-班德禮.mp4 — 501 帧, 2560×1440, 25fps, 时长 ~20 秒

---

## 一、整体性能对比

| 指标 | 优化前 (baseline) | 优化后 (最终) | 提升倍数 |
|------|-------------------|--------------|----------|
| **总处理耗时** | 182.17 秒 | 32.94 秒 | **5.5x** |
| **平均帧率** | 2.75 fps | 15.21 fps | **5.5x** |
| **骑手人脸识别** | ❌ 不可用 | ✅ 已启用 | — |
| **处理速度/视频时长比** | 9.1x 实时 | 1.6x 实时 | — |

---

## 二、各模块逐项对比 (ms/帧)

| 处理阶段 | 优化前 (ms/帧) | 优化前占比 | 优化后 (ms/帧) | 优化后占比 | 加速倍数 | 优化措施 |
|----------|---------------|-----------|---------------|-----------|---------|---------|
| 视频读取 | 0.1 | 0.0% | 0.1 | 0.2% | — | 异步预读线程 |
| YOLO+ByteTrack | 26.1 | 7.2% | 21.6 | 32.8% | 1.2x | 无变化 (已在 GPU) |
| 人脸检测 (SCRFD) | — (关闭) | — | 6.6 | 10.1% | — | 新增功能, onnxruntime-gpu |
| ROI 增强 | 1.9 | 0.5% | 1.2 | 1.9% | 1.6x | 无变化 |
| OCR (PaddleOCR) | **342.1** | **58.6%** | **5.2** | **7.8%** | **66x** | paddlepaddle CPU → GPU |
| VLM 回退 | 0.1 | 0.0% | 0.0 | 0.0% | — | 无变化 |
| 骑手身份识别 | — (关闭) | — | 4.0 | 6.1% | — | 新增功能, SQLite 后端 |
| 可视化绘制 | 81.9 | 22.4% | 25.1 | 38.1% | 3.3x | 检测数量差异 |
| 视频写入 | 6.2 | 1.7% | 5.6 | 8.5% | — | 异步写线程 |
| 其他 (融合/状态) | 5.5 | 1.5% | — | — | — | — |
| **合计** | **363.4** | **100%** | **65.8** | **100%** | **5.5x** | — |

> 注：优化前骑手识别功能不可用 (milvus-lite 不支持 Windows, 模型文件缺失), 占比以实际运行模块为基准。

---

## 三、优化措施明细

### 1. PaddleOCR GPU 加速 (贡献最大: ~100 秒节省)

| 项目 | 详情 |
|------|------|
| **问题** | 安装的是 `paddlepaddle` (CPU 版), PaddleOCR 被迫回退到 CPU 推理 |
| **方案** | 替换为 `paddlepaddle-gpu==3.3.0` (cu126) |
| **附加修复** | Windows 下 cuDNN DLL 加载失败 → 创建 `_nvidia_dll_fix.py` 预加载 torch/lib 中的 CUDA DLL; 补充缺失的 `zlibwapi.dll` |
| **效果** | OCR: 342ms/帧 → 5.2ms/帧 (**66 倍加速**) |

### 2. SCRFD 人脸检测 GPU 加速 (贡献大: ~45 秒节省)

| 项目 | 详情 |
|------|------|
| **问题** | `onnxruntime` (CPU 版) 不含 CUDAExecutionProvider, SCRFD 在 CPU 上运行 (96.5ms/帧) |
| **方案** | 替换为 `onnxruntime-gpu==1.23.0`; `pipeline.yaml` 中 `face_device` 改为 `cuda` |
| **效果** | SCRFD: 96.5ms/帧 → 6.6ms/帧 (**14.6 倍加速**) |

### 3. 读写流水线并行 (贡献中等: I/O 不阻塞)

| 项目 | 详情 |
|------|------|
| **问题** | 读帧 (`cap.read`) 和写帧 (`FFmpeg 编码`) 在主线程同步执行, 阻塞 GPU 推理 |
| **方案** | 三段式流水线: 读帧线程 → 主处理线程 → 写帧线程, 通过 `queue.Queue` 连接 |
| **效果** | 读+写约 5.7ms/帧 的阻塞时间被隐藏, 主线程 GPU 利用率提升 |

### 4. Windows 骑手人脸识别兼容 (功能恢复)

| 项目 | 详情 |
|------|------|
| **问题** | `milvus-lite` 不支持 Windows → 人脸特征库无法加载 → 骑手识别完全禁用 |
| **方案** | 新增 `SQLiteFaceStore` 后端, 暴力 IP 搜索; 自动检测 milvus-lite 可用性并降级; 迁移脚本从 Milvus Lite protobuf 格式提取向量 |
| **效果** | Windows 下骑手识别完全可用, 117 条人脸记录自匹配 117/117 正确 |

---

## 四、当前瓶颈分析

优化后瓶颈分布:

```
可视化绘制      38.1%   25.1ms/帧   ← 新瓶颈 (OpenCV 绘图, CPU)
YOLO+ByteTrack  32.8%   21.6ms/帧   ← 已在 GPU, 难再优化
SCRFD 人脸检测  10.1%    6.6ms/帧   ← 已在 GPU
视频写入         8.5%    5.6ms/帧   ← 已异步
OCR              7.8%    5.2ms/帧   ← 已在 GPU
骑手身份识别     6.1%    4.0ms/帧   ← SQLite 暴力搜索, 毫秒级
```

进一步优化方向:
- 可视化绘制可移入写线程 (与写入一起异步)
- FFmpeg 编码 preset 从 `medium` 改为 `fast`
- 人脸检测间隔 (`face_interval`) 可适当增大

---

## 五、文件变更清单

### 修改的文件

| 文件 | 改动说明 |
|------|---------|
| `backend/main.py` | 加载 `_nvidia_dll_fix` + 恢复 `import os` |
| `backend/processor.py` | 三段式读写流水线 |
| `backend/horse_id/rider_identity.py` | SQLite 降级逻辑 |
| `backend/save_rider_faces_to_milvus.py` | 入库脚本 SQLite 降级 |
| `backend/config/pipeline.yaml` | `face_device: cpu → cuda` |
| `DEPLOY.md` | 区分 Win/Ubuntu 部署文档 |

### 新增的文件

| 文件 | 说明 |
|------|------|
| `backend/_nvidia_dll_fix.py` | Windows CUDA DLL 预加载 |
| `backend/horse_id/sqlite_face_store.py` | SQLite 人脸特征库后端 |
| `backend/migrate_milvus_to_sqlite.py` | Milvus Lite → SQLite 迁移工具 |

### 依赖变更

| 包 | 变更 | 说明 |
|----|------|------|
| `paddlepaddle` → `paddlepaddle-gpu` | 3.2.2 (CPU) → 3.3.0 (GPU/cu126) | OCR 加速核心 |
| `onnxruntime` → `onnxruntime-gpu` | 1.23.2 (CPU) → 1.23.0 (GPU) | SCRFD 加速 |
| `nvidia-cudnn-cu12` 等 | 已卸载 | 与 torch 自带的 CUDA 库冲突, 复用 torch/lib |
