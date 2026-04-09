# 性能优化对比报告

> 测试日期：2026-04-09
> 测试环境：Windows 11, RTX 4060 (8GB), Python 3.11, CUDA 13.1 驱动
> 测试视频：H303-班德禮.mp4 — 501 帧, 2560×1440, 25fps, 时长 ~20 秒

---

## 一、整体性能对比

| 指标 | 优化前 (baseline) | 优化后 (最终) | 提升倍数 |
|------|-------------------|--------------|----------|
| **总处理耗时** | 182.17 秒 | 21.50 秒 | **8.5x** |
| **平均帧率** | 2.75 fps | 23.30 fps | **8.5x** |
| **骑手人脸识别** | ❌ 不可用 | ✅ 已启用 | — |
| **处理速度/视频时长比** | 9.1x 实时 | 1.1x 实时 | — |

---

## 二、优化历程

| 阶段 | 总耗时 | fps | 关键改动 |
|------|--------|-----|---------|
| baseline (CPU OCR, 无人脸) | 182s | 2.75 | — |
| + PaddleOCR GPU | 87s | 5.76 | paddlepaddle-gpu, DLL 修复 |
| + SCRFD GPU | 33s | 15.21 | onnxruntime-gpu, face_device: cuda |
| + viz 移入写线程 | 25s | 19.74 | 主线程不阻塞渲染 |
| + PIL 局部渲染 | **21.5s** | **23.30** | 避免全帧 BGR↔RGB 转换 |

---

## 三、各模块逐项对比 (ms/帧)

| 处理阶段 | 优化前 (ms/帧) | 优化前占比 | 优化后 (ms/帧) | 优化后占比 | 加速倍数 | 优化措施 |
|----------|---------------|-----------|---------------|-----------|---------|---------|
| 视频读取 | 0.1 | 0.0% | 0.1 | 0.2% | — | 异步预读线程 |
| YOLO+ByteTrack | 26.1 | 7.2% | 23.5 | 54.8% | 1.1x | 无变化 (已在 GPU) |
| 人脸检测 (SCRFD) | — (关闭) | — | 6.5 | 15.1% | — | 新增功能, onnxruntime-gpu |
| ROI 增强 | 1.9 | 0.5% | 1.4 | 3.3% | 1.4x | 无变化 |
| OCR (PaddleOCR) | **342.1** | **58.6%** | **5.6** | **13.0%** | **61x** | paddlepaddle CPU → GPU |
| VLM 回退 | 0.1 | 0.0% | 0.0 | 0.0% | — | 无变化 |
| 骑手身份识别 | — (关闭) | — | 4.4 | 10.3% | — | 新增功能, SQLite 后端 |
| 可视化绘制 | 81.9 | 22.4% | 11.3 | 26.4% | **7.2x** | PIL 局部渲染 + 移入写线程 |
| 视频写入 | 6.2 | 1.7% | 4.9 | 11.5% | 1.3x | 异步写线程 |
| 其他 (融合/状态) | 5.5 | 1.5% | — | — | — | — |
| **合计** | **363.4** | **100%** | **42.9** | **100%** | **8.5x** | — |

> 注：优化前骑手识别功能不可用 (milvus-lite 不支持 Windows, 模型文件缺失), 占比以实际运行模块为基准。
> 优化后 viz + write 在异步线程执行，不阻塞主线程，主线程有效耗时约 41.5ms/帧。

---

## 四、优化措施明细

### 1. PaddleOCR GPU 加速 (贡献最大: ~100 秒节省)

| 项目 | 详情 |
|------|------|
| **问题** | 安装的是 `paddlepaddle` (CPU 版), PaddleOCR 被迫回退到 CPU 推理 |
| **方案** | 替换为 `paddlepaddle-gpu==3.3.0` (cu126) |
| **附加修复** | Windows 下 cuDNN DLL 加载失败 → 创建 `_nvidia_dll_fix.py` 预加载 torch/lib 中的 CUDA DLL; 补充缺失的 `zlibwapi.dll` |
| **效果** | OCR: 342ms/帧 → 5.6ms/帧 (**61 倍加速**) |

### 2. SCRFD 人脸检测 GPU 加速 (贡献大: ~54 秒节省)

| 项目 | 详情 |
|------|------|
| **问题** | `onnxruntime` (CPU 版) 不含 CUDAExecutionProvider, SCRFD 在 CPU 上运行 (96.5ms/帧) |
| **方案** | 替换为 `onnxruntime-gpu==1.23.0`; `pipeline.yaml` 中 `face_device` 改为 `cuda` |
| **效果** | SCRFD: 96.5ms/帧 → 6.5ms/帧 (**14.8 倍加速**) |

### 3. viz + write 异步流水线 (贡献大: ~12 秒节省)

| 项目 | 详情 |
|------|------|
| **问题** | 读帧、可视化绘制、视频编码在主线程同步执行, 阻塞 GPU 推理 |
| **方案** | 三段式流水线: 读帧线程 → 主处理线程 → viz+写帧线程, 通过 `queue.Queue` 连接 |
| **效果** | viz(11.3ms) + write(4.9ms) ≈ 16ms/帧 的开销完全隐藏, 不阻塞主线程 |

### 4. PIL 文字渲染局部化 (贡献中: ~7 秒节省)

| 项目 | 详情 |
|------|------|
| **问题** | `_put_text_pil` 和 `PilBatchRenderer` 每次将整个 2560×1440 帧做 BGR→RGB→PIL→RGB→BGR 转换, 仅为渲染几行文字 |
| **方案** | 改为只提取文字区域的局部 patch 做转换, 渲染后贴回 |
| **效果** | viz: 31.2ms/帧 → 11.3ms/帧 (**2.8 倍加速**) |

### 5. Windows 骑手人脸识别兼容 (功能恢复)

| 项目 | 详情 |
|------|------|
| **问题** | `milvus-lite` 不支持 Windows → 人脸特征库无法加载 → 骑手识别完全禁用 |
| **方案** | 新增 `SQLiteFaceStore` 后端, 暴力 IP 搜索; 自动检测 milvus-lite 可用性并降级; 迁移脚本从 Milvus Lite protobuf 格式提取向量 |
| **效果** | Windows 下骑手识别完全可用, 117 条人脸记录自匹配 117/117 正确 |

---

## 五、当前瓶颈分析

优化后主线程瓶颈分布 (viz/write 已异步, 不阻塞):

```
YOLO+ByteTrack  54.8%   23.5ms/帧   ← 主瓶颈 (已在 GPU, 难再优化)
SCRFD 人脸检测  15.1%    6.5ms/帧   ← 已在 GPU
OCR              13.0%    5.6ms/帧   ← 已在 GPU
骑手身份识别     10.3%    4.4ms/帧   ← SQLite 暴力搜索, 毫秒级
ROI 增强         3.3%    1.4ms/帧   ← CPU, 已很快
```

进一步优化空间有限, 主要方向:
- 人脸检测间隔 (`face_interval`) 可适当增大以减少 SCRFD 调用
- YOLO 模型量化 (INT8/FP16) 可能小幅提速
- FFmpeg 编码 preset 从 `medium` 改为 `fast`

---

## 六、文件变更清单

### 修改的文件

| 文件 | 改动说明 |
|------|---------|
| `backend/main.py` | 加载 `_nvidia_dll_fix` + 恢复 `import os` |
| `backend/processor.py` | 三段式流水线 (读帧→处理→viz+写帧), viz 移入写线程 |
| `backend/horse_id/rider_identity.py` | SQLite 降级逻辑 |
| `backend/horse_id/visualizer.py` | PIL 文字渲染局部化 |
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
