# hydtest 分支变更日志

> 对比基准: `origin/master` (commit `407a8c7`)
> 当前分支: `hydtest` (commit `bef427e`)
> 统计: **72 个文件变更, +15,331 行, -782 行**

---

## 一、变更概览

hydtest 分支在 master 基础上完成了以下工作:

1. **后端算法集成** — 将 Hor-Peo_detect 马匹/骑手识别算法集成到 Web 后端
2. **三种分析模式** — 全流程 (full)、简化 VLM (simple)、仅人脸 (face)
3. **前端重构** — 路由拆分 + 三种分析视图 + 实时监控面板
4. **评测系统** — 端到端精度评测框架
5. **Windows 兼容** — SQLite 降级方案解决 milvus-lite 不可用问题
6. **性能优化** — OCR/SCRFD GPU 加速 + 读写流水线, 整体 5.5x 提速

---

## 二、后端改动 (49 文件)

### 2.1 核心算法模块 (`backend/horse_id/`)

| 文件 | 说明 |
|------|------|
| `config.py` | 管线配置加载: 检测器、ROI、增强、OCR、融合、运行时等参数 |
| `detector.py` | YOLO 马匹检测器封装, 支持 ByteTrack 追踪 |
| `direction_filter.py` | 运动方向过滤, 只处理指定方向 (如右→左) 的马匹 |
| `enhancer.py` | ROI 图像增强: CLAHE + 锐化 + 二值化 |
| `ocr_engine.py` | PaddleOCR 鞍垫号码识别, 正则校验 `^[A-Z]\d{3}$` |
| `roi_extractor.py` | 从马匹检测框提取鞍垫 ROI 区域 |
| `track_fusion.py` | OCR 结果时序融合: 滑动窗口投票 + 锁定机制 |
| `track_state.py` | 轨迹状态机: UNCONFIRMED → CONFIRMED → LOST |
| `visualizer.py` | 结果可视化: 检测框、号码、骑手名、人脸框绘制; PIL 文字渲染局部化优化 |
| `vlm_fallback.py` | VLM 回退: OCR 失败时调用 Qwen API 识别鞍垫号码 |
| `vlm_video.py` | 简化模式: 对裁剪图调用多模态 VLM 识别号码 |
| `types.py` | 数据类型定义: HorseDetection, ROIBox, OCRResult |
| `light_detector.py` | 轻量检测器 |
| `pose_detector.py` | 姿态检测器 |
| `redis_utils.py` | Redis 工具类 |

### 2.2 人脸识别模块 (`backend/horse_id/face/`)

| 文件 | 说明 |
|------|------|
| `scrfd.py` | SCRFD 人脸检测模型 (ONNX Runtime 推理) |
| `arcface_onnx.py` | ArcFace 人脸特征提取 (ONNX Runtime) |
| `face_align.py` | 人脸对齐 (5 点关键点仿射变换) |
| `face_detector.py` | 人脸检测器封装: 检测 + 关键点 + 特征提取 |
| `face_feature_db.py` | 人脸特征数据库 |
| `face_recognizer.py` | 人脸识别器 |

### 2.3 骑手身份识别 (`backend/horse_id/`)

| 文件 | 说明 |
|------|------|
| `rider_identity.py` | 骑手身份模块: 人脸匹配 + 服装颜色 + 马号映射 + 时序投票 + 锁定机制 |
| `rider_feature_store.py` | 骑手特征存储: 跨轨迹身份关联 |
| `sqlite_face_store.py` | **[新增]** SQLite 人脸特征库 — milvus-lite 的 Windows 降级方案 |

### 2.4 视频处理管线 (`backend/`)

| 文件 | 说明 |
|------|------|
| `processor.py` | 视频处理主流程: 逐帧检测→追踪→OCR→融合→人脸→可视化→编码; 三段式读写流水线 |
| `main.py` | FastAPI 服务: 上传接口、任务队列、状态查询、系统监控; 新增 NVIDIA DLL 预加载 |
| `system_metrics.py` | CPU/GPU/内存利用率采集 |
| `save_rider_faces_to_milvus.py` | 骑手人脸入库脚本: Milvus + SQLite 双后端支持 |
| `migrate_milvus_to_sqlite.py` | **[新增]** Milvus Lite → SQLite 迁移工具 |
| `_nvidia_dll_fix.py` | **[新增]** Windows CUDA DLL 预加载 (解决 cudnn 加载失败) |
| `config/pipeline.yaml` | 管线配置: 检测阈值、OCR 参数、人脸设备、VLM 接口等 |
| `requirements.txt` | Python 依赖清单 |

### 2.5 评测系统 (`backend/` + `eval/`)

| 文件 | 说明 |
|------|------|
| `eval_runner.py` | 评测主入口: 加载 ground truth, 计算 P/R/F1, 错误归因, 生成报告 |
| `tests/test_eval_runner.py` | 评测单元测试 |
| `tools/run_eval.sh` | 评测运行脚本 |
| `eval/ground_truth.json` | 标注数据 |
| `eval/ground_truth_template.json` | 标注模板 |

### 2.6 测试

| 文件 | 说明 |
|------|------|
| `test_aggregate.py` | 检测结果汇总逻辑测试 |
| `test_api.py` | API 接口测试 |
| `test_display_name_fix.py` | 显示名修复测试 |

---

## 三、前端改动 (19 文件)

### 3.1 架构重构

| 改动 | 说明 |
|------|------|
| **路由系统** | 新增 `router.ts`, 从单页应用改为多视图路由 |
| **视图拆分** | `App.vue` 拆分为导航壳 + 3 个独立视图 |
| **依赖** | 新增 `vue-router`, `lucide-vue-next` (图标库) |
| **代理配置** | `vite.config.ts` 新增 `/api` + `/videos` 代理到后端 8001 |

### 3.2 三种分析视图

| 文件 | 说明 |
|------|------|
| `views/FullAnalysis.vue` | 全流程分析: 上传视频 → 检测+OCR+人脸 → 展示结果 |
| `views/SimpleAnalysis.vue` | 简化分析: 上传视频 → VLM 鞍垫号码识别 |
| `views/FaceAnalysis.vue` | 人脸分析: 上传视频 → 仅检测+追踪+人脸识别 |

### 3.3 实时监控

- Header 集成系统指标面板: CPU/GPU/内存利用率
- 任务队列状态实时展示
- 各阶段处理耗时占比 (profiling)

### 3.4 示例图片

新增 9 张示例图片 (`vue/public/examples/`), 展示识别效果好/差的对比案例。

---

## 四、文档 (3 文件)

| 文件 | 说明 |
|------|------|
| `DEPLOY.md` | **[重写]** 部署文档, 区分 Windows / Ubuntu, 覆盖依赖安装、配置、常见问题 |
| `docs/performance_report.md` | **[新增]** 性能优化对比报告: 182s → 33s (5.5x), 各模块逐项分析 |
| `docs/hydtest_branch_changelog.md` | **[新增]** 本文档 |
| `docs/plans/` | 评测系统设计文档 (2 篇) |

---

## 五、关键性能指标

测试视频: 501 帧, 2560×1440, 25fps

| 指标 | master (无算法) | hydtest 初始 | hydtest 最终 |
|------|----------------|-------------|-------------|
| 处理耗时 | — | 182 秒 | **21.5 秒** |
| 帧率 | — | 2.75 fps | **23.3 fps** |
| OCR | — | 342 ms/帧 (CPU) | **5.6 ms/帧 (GPU)** |
| SCRFD | — | 不可用 | **6.5 ms/帧 (GPU)** |
| 可视化 | — | 81.9 ms/帧 | **11.3 ms/帧 (局部PIL + 异步)** |
| 骑手识别 | — | 不可用 | **已启用** |
| 总提升 | — | — | **8.5x** |

---

## 六、依赖变更

| 依赖 | master | hydtest | 说明 |
|------|--------|---------|------|
| paddlepaddle | 无 | **paddlepaddle-gpu 3.3.0** | OCR GPU 推理 |
| paddleocr | 无 | ≥2.7.0 | PaddleOCR 引擎 |
| onnxruntime | 无 | **onnxruntime-gpu 1.23.0** | SCRFD/ArcFace GPU 推理 |
| ultralytics | 无 | ≥8.2.0 | YOLOv8 检测+追踪 |
| pymilvus | 无 | ≥2.3.0 | 人脸特征库 (Ubuntu) |
| opencv-python | 无 | 4.6.0.66 | 图像处理 |
| openai | 无 | ≥1.0.0 | VLM API 调用 |
| vue-router | 无 | ^4.6.4 | 前端路由 |
| lucide-vue-next | 无 | ^0.577.0 | 前端图标 |
