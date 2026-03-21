#!/usr/bin/env bash
# 赛马识别系统 — 批量评测脚本
# 用法: bash backend/tools/run_eval.sh
#
# 前提: 将待评测的 .mp4 视频放入 eval/videos/ 目录，
#       并确保 eval/ground_truth.json 已正确填写。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_DIR="$PROJECT_ROOT/eval"
VIDEOS_DIR="$EVAL_DIR/videos"
RESULTS_DIR="$EVAL_DIR/results"
BACKEND_DIR="$PROJECT_ROOT/backend"
CONFIG_PATH="$BACKEND_DIR/config/pipeline.yaml"

echo "=========================================="
echo "  赛马识别系统 · 批量评测"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "评测目录:   $EVAL_DIR"
echo ""

# -------- 前置检查 --------
if [ ! -f "$EVAL_DIR/ground_truth.json" ]; then
    echo "[ERROR] 未找到 ground_truth.json，请先生成标注文件。"
    exit 1
fi

VIDEO_COUNT=$(find "$VIDEOS_DIR" -maxdepth 1 -name "*.mp4" -type f | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "[ERROR] eval/videos/ 下未找到 .mp4 文件，请先放入待评测视频。"
    exit 1
fi

echo "[INFO] 发现 $VIDEO_COUNT 个视频待处理"
echo ""

# -------- 清空旧结果 --------
rm -f "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.mp4 2>/dev/null || true
echo "[INFO] 已清空 eval/results/ 旧文件"

# -------- 第一阶段: 批量处理视频 --------
echo ""
echo "========== 第一阶段: 批量处理视频 =========="
echo ""

cd "$BACKEND_DIR"

python3 -c "
import sys, os, time
sys.path.insert(0, '.')
from processor import process_video
from pathlib import Path

videos_dir = Path('$VIDEOS_DIR')
results_dir = Path('$RESULTS_DIR')
config_path = '$CONFIG_PATH'

mp4_files = sorted(videos_dir.glob('*.mp4'))
total = len(mp4_files)
tasks = {}

for i, mp4 in enumerate(mp4_files, 1):
    print(f'\\n[{i}/{total}] 处理: {mp4.name}')
    print('-' * 50)
    out_video = results_dir / f'processed_{mp4.name}'
    t0 = time.time()
    try:
        process_video(
            task_id=mp4.stem,
            video_path=str(mp4),
            output_video_path=str(out_video),
            tasks=tasks,
            config_path=config_path,
        )
        elapsed = time.time() - t0
        print(f'  完成 ({elapsed:.1f}s)')
    except Exception as e:
        elapsed = time.time() - t0
        print(f'  失败 ({elapsed:.1f}s): {e}')

print(f'\\n视频处理全部完成。')
"

# -------- 第二阶段: 运行评测 --------
echo ""
echo "========== 第二阶段: 运行评测 =========="
echo ""

python3 eval_runner.py --eval-dir "$EVAL_DIR"

# -------- 输出报告路径 --------
echo ""
echo "=========================================="
echo "  评测完成!"
echo "=========================================="
REPORT_DATE=$(date +%Y-%m-%d)
echo "Markdown 报告: eval/reports/eval_report_${REPORT_DATE}.md"
echo "JSON 报告:     eval/reports/eval_report_${REPORT_DATE}.json"
echo ""
echo "查看报告: cat eval/reports/eval_report_${REPORT_DATE}.md"
