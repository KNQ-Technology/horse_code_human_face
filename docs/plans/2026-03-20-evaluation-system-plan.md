# 赛马识别系统评测工具 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个端到端评测工具，对比系统输出的 `(马号, 骑手)` 集合与人工标注的 Ground Truth，输出 P/R/F1 指标和错误归因报告。

**Architecture:** 纯 Python 脚本，读取 `ground_truth.json` 和系统生成的 `_summary.json` 文件，计算集合匹配指标，生成 JSON + Markdown 格式的评测报告。不依赖额外服务，可在现有 venv 中运行。

**Tech Stack:** Python 3, pytest, json, pathlib（无额外第三方依赖）

---

## Task 1: 创建评测目录结构与 Ground Truth 模板

**Files:**
- Create: `eval/ground_truth_template.json`
- Create: `eval/videos/.gitkeep`
- Create: `eval/results/.gitkeep`
- Create: `eval/reports/.gitkeep`

**Step 1: 创建目录结构**

```bash
mkdir -p eval/videos eval/results eval/reports
touch eval/videos/.gitkeep eval/results/.gitkeep eval/reports/.gitkeep
```

**Step 2: 创建 Ground Truth 模板文件**

创建 `eval/ground_truth_template.json`：

```json
{
  "version": "1.0",
  "annotator": "",
  "created_at": "",
  "videos": [
    {
      "video_file": "example_race.mp4",
      "race_info": {
        "date": "2026-01-01",
        "race_number": 1,
        "venue": ""
      },
      "ground_truth": [
        {
          "horse_id": "B123",
          "rider_name": "骑手姓名（须与人脸库注册名一致）",
          "notes": ""
        }
      ]
    }
  ]
}
```

**Step 3: Commit**

```bash
git add eval/
git commit -m "chore: 创建评测目录结构与 ground_truth 模板"
```

---

## Task 2: 实现评测核心逻辑 — 集合匹配与 P/R/F1 计算

**Files:**
- Create: `backend/eval_runner.py`
- Test: `backend/tests/test_eval_runner.py`

**Step 1: 编写失败测试 — 单视频 P/R/F1 计算**

创建 `backend/tests/__init__.py`（如不存在）和 `backend/tests/test_eval_runner.py`：

```python
"""Tests for eval_runner module."""
from __future__ import annotations

import pytest
from eval_runner import compute_set_metrics


class TestComputeSetMetrics:
    """Test Precision / Recall / F1 for set matching."""

    def test_perfect_match(self):
        pred = {"B123", "A045"}
        gt = {"B123", "A045"}
        m = compute_set_metrics(pred, gt)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_partial_match(self):
        pred = {"B123", "A045", "C999"}
        gt = {"B123", "A045"}
        m = compute_set_metrics(pred, gt)
        assert m["tp"] == 2
        assert m["fp"] == 1
        assert m["fn"] == 0
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == 1.0

    def test_no_predictions(self):
        pred = set()
        gt = {"B123"}
        m = compute_set_metrics(pred, gt)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_no_ground_truth(self):
        pred = {"B123"}
        gt = set()
        m = compute_set_metrics(pred, gt)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_both_empty(self):
        m = compute_set_metrics(set(), set())
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_complete_mismatch(self):
        pred = {"X001", "X002"}
        gt = {"Y001", "Y002"}
        m = compute_set_metrics(pred, gt)
        assert m["tp"] == 0
        assert m["fp"] == 2
        assert m["fn"] == 2
        assert m["f1"] == 0.0
```

**Step 2: 运行测试确认失败**

```bash
cd backend && python -m pytest tests/test_eval_runner.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'eval_runner'`）

**Step 3: 实现 `compute_set_metrics`**

创建 `backend/eval_runner.py`：

```python
"""
评测工具：对比系统输出与 Ground Truth，计算 P/R/F1 并生成错误归因报告。
"""
from __future__ import annotations

from typing import Any


def compute_set_metrics(pred: set, gt: set) -> dict[str, Any]:
    """
    @param pred 系统预测集合
    @param gt Ground Truth 集合
    @return 包含 precision, recall, f1, tp, fp, fn 的字典
    """
    if not pred and not gt:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
```

**Step 4: 运行测试确认通过**

```bash
cd backend && python -m pytest tests/test_eval_runner.py -v
```

预期：全部 PASS

**Step 5: Commit**

```bash
git add backend/eval_runner.py backend/tests/
git commit -m "feat(eval): 实现集合匹配 P/R/F1 计算"
```

---

## Task 3: 实现三维度评测 — 马号 / 骑手 / 配对

**Files:**
- Modify: `backend/eval_runner.py`
- Test: `backend/tests/test_eval_runner.py`

**Step 1: 编写失败测试 — 单视频三维度评测**

在 `backend/tests/test_eval_runner.py` 中新增：

```python
from eval_runner import evaluate_single_video


class TestEvaluateSingleVideo:
    """Test three-dimensional evaluation for a single video."""

    def test_all_correct(self):
        gt_entries = [
            {"horse_id": "B123", "rider_name": "潘顿"},
            {"horse_id": "A045", "rider_name": "莫雷拉"},
        ]
        pred_entries = [
            {"horse_id": "B123", "person_name": "潘顿"},
            {"horse_id": "A045", "person_name": "莫雷拉"},
        ]
        result = evaluate_single_video(gt_entries, pred_entries)
        assert result["horse_id"]["f1"] == 1.0
        assert result["rider"]["f1"] == 1.0
        assert result["pair"]["f1"] == 1.0

    def test_horse_correct_rider_wrong(self):
        gt_entries = [{"horse_id": "B123", "rider_name": "潘顿"}]
        pred_entries = [{"horse_id": "B123", "person_name": "莫雷拉"}]
        result = evaluate_single_video(gt_entries, pred_entries)
        assert result["horse_id"]["f1"] == 1.0
        assert result["rider"]["f1"] == 0.0
        assert result["pair"]["f1"] == 0.0

    def test_missing_prediction(self):
        gt_entries = [
            {"horse_id": "B123", "rider_name": "潘顿"},
            {"horse_id": "A045", "rider_name": "莫雷拉"},
        ]
        pred_entries = [{"horse_id": "B123", "person_name": "潘顿"}]
        result = evaluate_single_video(gt_entries, pred_entries)
        assert result["horse_id"]["recall"] == 0.5
        assert result["rider"]["recall"] == 0.5

    def test_extra_prediction(self):
        gt_entries = [{"horse_id": "B123", "rider_name": "潘顿"}]
        pred_entries = [
            {"horse_id": "B123", "person_name": "潘顿"},
            {"horse_id": "C999", "person_name": "何泽尧"},
        ]
        result = evaluate_single_video(gt_entries, pred_entries)
        assert result["horse_id"]["precision"] == 0.5
        assert result["pair"]["precision"] == 0.5

    def test_other_rider_excluded_from_rider_set(self):
        """系统输出 '其他骑师' 时，骑手维度应忽略该条。"""
        gt_entries = [{"horse_id": "B123", "rider_name": "潘顿"}]
        pred_entries = [
            {"horse_id": "B123", "person_name": "潘顿"},
            {"horse_id": "A045", "person_name": "其他骑师"},
        ]
        result = evaluate_single_video(gt_entries, pred_entries)
        assert result["rider"]["precision"] == 1.0
        assert result["rider"]["recall"] == 1.0
```

**Step 2: 运行测试确认失败**

```bash
cd backend && python -m pytest tests/test_eval_runner.py::TestEvaluateSingleVideo -v
```

预期：FAIL（`ImportError`）

**Step 3: 实现 `evaluate_single_video`**

在 `backend/eval_runner.py` 中新增：

```python
PLACEHOLDER_RIDER = "其他骑师"


def evaluate_single_video(
    gt_entries: list[dict[str, str]],
    pred_entries: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """
    @param gt_entries Ground Truth 列表，每项含 horse_id, rider_name
    @param pred_entries 系统预测列表，每项含 horse_id, person_name
    @return 三个维度 (horse_id, rider, pair) 的 P/R/F1 字典
    """
    gt_horses = {e["horse_id"] for e in gt_entries}
    pred_horses = {e["horse_id"] for e in pred_entries}

    gt_riders = {e["rider_name"] for e in gt_entries if e["rider_name"] != PLACEHOLDER_RIDER}
    pred_riders = {e["person_name"] for e in pred_entries if e.get("person_name") != PLACEHOLDER_RIDER}

    gt_pairs = {(e["horse_id"], e["rider_name"]) for e in gt_entries}
    pred_pairs = {(e["horse_id"], e.get("person_name", PLACEHOLDER_RIDER)) for e in pred_entries}

    return {
        "horse_id": compute_set_metrics(pred_horses, gt_horses),
        "rider": compute_set_metrics(pred_riders, gt_riders),
        "pair": compute_set_metrics(pred_pairs, gt_pairs),
    }
```

**Step 4: 运行测试确认通过**

```bash
cd backend && python -m pytest tests/test_eval_runner.py -v
```

预期：全部 PASS

**Step 5: Commit**

```bash
git add backend/eval_runner.py backend/tests/test_eval_runner.py
git commit -m "feat(eval): 实现马号/骑手/配对三维度单视频评测"
```

---

## Task 4: 实现跨视频汇总指标 — 宏平均 / 微平均 / 完全匹配率

**Files:**
- Modify: `backend/eval_runner.py`
- Modify: `backend/tests/test_eval_runner.py`

**Step 1: 编写失败测试 — 汇总指标**

在 `backend/tests/test_eval_runner.py` 中新增：

```python
from eval_runner import aggregate_metrics


class TestAggregateMetrics:
    """Test macro/micro averaging and exact match rate."""

    def test_two_perfect_videos(self):
        per_video = [
            {
                "video_file": "v1.mp4",
                "horse_id": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0},
                "rider":    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0},
                "pair":     {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0},
            },
            {
                "video_file": "v2.mp4",
                "horse_id": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 3, "fp": 0, "fn": 0},
                "rider":    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 3, "fp": 0, "fn": 0},
                "pair":     {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 3, "fp": 0, "fn": 0},
            },
        ]
        agg = aggregate_metrics(per_video)
        assert agg["exact_match_rate"] == 1.0
        for dim in ("horse_id", "rider", "pair"):
            assert agg[dim]["macro"]["f1"] == 1.0
            assert agg[dim]["micro"]["f1"] == 1.0

    def test_one_perfect_one_bad(self):
        per_video = [
            {
                "video_file": "v1.mp4",
                "horse_id": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0},
                "rider":    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0},
                "pair":     {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0},
            },
            {
                "video_file": "v2.mp4",
                "horse_id": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 2, "fn": 1},
                "rider":    {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 2, "fn": 1},
                "pair":     {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 2, "fn": 1},
            },
        ]
        agg = aggregate_metrics(per_video)
        assert agg["exact_match_rate"] == 0.5
        for dim in ("horse_id", "rider", "pair"):
            assert agg[dim]["macro"]["f1"] == pytest.approx(0.5)
```

**Step 2: 运行测试确认失败**

```bash
cd backend && python -m pytest tests/test_eval_runner.py::TestAggregateMetrics -v
```

预期：FAIL

**Step 3: 实现 `aggregate_metrics`**

在 `backend/eval_runner.py` 中新增：

```python
def aggregate_metrics(per_video: list[dict[str, Any]]) -> dict[str, Any]:
    """
    @param per_video 每个视频的三维度指标列表
    @return 汇总指标：宏平均、微平均、完全匹配率
    """
    n = len(per_video)
    if n == 0:
        return {}

    dimensions = ("horse_id", "rider", "pair")
    result: dict[str, Any] = {}

    exact_match_count = sum(
        1 for v in per_video
        if all(v[d]["fp"] == 0 and v[d]["fn"] == 0 for d in dimensions)
    )
    result["exact_match_rate"] = exact_match_count / n

    for dim in dimensions:
        macro_p = sum(v[dim]["precision"] for v in per_video) / n
        macro_r = sum(v[dim]["recall"] for v in per_video) / n
        macro_f1 = sum(v[dim]["f1"] for v in per_video) / n

        total_tp = sum(v[dim]["tp"] for v in per_video)
        total_fp = sum(v[dim]["fp"] for v in per_video)
        total_fn = sum(v[dim]["fn"] for v in per_video)
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        result[dim] = {
            "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
            "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        }

    return result
```

**Step 4: 运行测试确认通过**

```bash
cd backend && python -m pytest tests/test_eval_runner.py -v
```

预期：全部 PASS

**Step 5: Commit**

```bash
git add backend/eval_runner.py backend/tests/test_eval_runner.py
git commit -m "feat(eval): 实现跨视频宏平均/微平均/完全匹配率汇总"
```

---

## Task 5: 实现错误归因分析

**Files:**
- Modify: `backend/eval_runner.py`
- Modify: `backend/tests/test_eval_runner.py`

**Step 1: 编写失败测试 — 错误归因**

在 `backend/tests/test_eval_runner.py` 中新增：

```python
from eval_runner import classify_errors


class TestClassifyErrors:
    """Test error attribution for a single video."""

    def test_no_errors(self):
        gt = [{"horse_id": "B123", "rider_name": "潘顿"}]
        pred = [{"horse_id": "B123", "person_name": "潘顿"}]
        errors = classify_errors(gt, pred)
        assert len(errors) == 0

    def test_miss_horse(self):
        gt = [{"horse_id": "B123", "rider_name": "潘顿"}]
        pred = []
        errors = classify_errors(gt, pred)
        assert any(e["type"] == "Miss-Horse" and e["horse_id"] == "B123" for e in errors)

    def test_fp_horse(self):
        gt = []
        pred = [{"horse_id": "C999", "person_name": "何泽尧"}]
        errors = classify_errors(gt, pred)
        assert any(e["type"] == "FP-Horse" and e["horse_id"] == "C999" for e in errors)

    def test_miss_rider(self):
        gt = [{"horse_id": "B123", "rider_name": "潘顿"}]
        pred = [{"horse_id": "B123", "person_name": "其他骑师"}]
        errors = classify_errors(gt, pred)
        assert any(e["type"] == "Miss-Rider" and e["rider_name"] == "潘顿" for e in errors)

    def test_wrong_pair(self):
        gt = [
            {"horse_id": "B123", "rider_name": "潘顿"},
            {"horse_id": "A045", "rider_name": "莫雷拉"},
        ]
        pred = [
            {"horse_id": "B123", "person_name": "莫雷拉"},
            {"horse_id": "A045", "person_name": "潘顿"},
        ]
        errors = classify_errors(gt, pred)
        assert any(e["type"] == "Wrong-Pair" for e in errors)

    def test_fp_rider(self):
        gt = [{"horse_id": "B123", "rider_name": "潘顿"}]
        pred = [
            {"horse_id": "B123", "person_name": "潘顿"},
            {"horse_id": "C999", "person_name": "何泽尧"},
        ]
        errors = classify_errors(gt, pred)
        assert any(e["type"] == "FP-Rider" and e["rider_name"] == "何泽尧" for e in errors)
```

**Step 2: 运行测试确认失败**

```bash
cd backend && python -m pytest tests/test_eval_runner.py::TestClassifyErrors -v
```

预期：FAIL

**Step 3: 实现 `classify_errors`**

在 `backend/eval_runner.py` 中新增：

```python
def classify_errors(
    gt_entries: list[dict[str, str]],
    pred_entries: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    @param gt_entries Ground Truth 列表
    @param pred_entries 系统预测列表
    @return 错误列表，每项含 type, horse_id, rider_name, detail
    """
    errors: list[dict[str, str]] = []

    gt_horses = {e["horse_id"] for e in gt_entries}
    pred_horses = {e["horse_id"] for e in pred_entries}

    gt_riders = {e["rider_name"] for e in gt_entries if e["rider_name"] != PLACEHOLDER_RIDER}
    pred_riders = {e.get("person_name", PLACEHOLDER_RIDER) for e in pred_entries
                   if e.get("person_name") != PLACEHOLDER_RIDER}

    gt_pair_map = {e["horse_id"]: e["rider_name"] for e in gt_entries}
    pred_pair_map = {e["horse_id"]: e.get("person_name", PLACEHOLDER_RIDER) for e in pred_entries}

    for h in gt_horses - pred_horses:
        errors.append({
            "type": "Miss-Horse",
            "horse_id": h,
            "rider_name": gt_pair_map.get(h, ""),
            "detail": f"GT 中的马号 {h} 未被系统检出",
        })

    for h in pred_horses - gt_horses:
        errors.append({
            "type": "FP-Horse",
            "horse_id": h,
            "rider_name": pred_pair_map.get(h, ""),
            "detail": f"系统误检了马号 {h}",
        })

    for r in gt_riders - pred_riders:
        errors.append({
            "type": "Miss-Rider",
            "horse_id": "",
            "rider_name": r,
            "detail": f"GT 中的骑手 {r} 未被系统识别",
        })

    for r in pred_riders - gt_riders:
        errors.append({
            "type": "FP-Rider",
            "horse_id": "",
            "rider_name": r,
            "detail": f"系统误检了骑手 {r}",
        })

    matched_horses = gt_horses & pred_horses
    for h in matched_horses:
        gt_r = gt_pair_map.get(h, PLACEHOLDER_RIDER)
        pred_r = pred_pair_map.get(h, PLACEHOLDER_RIDER)
        if gt_r != pred_r and gt_r != PLACEHOLDER_RIDER:
            errors.append({
                "type": "Wrong-Pair",
                "horse_id": h,
                "rider_name": f"GT={gt_r}, Pred={pred_r}",
                "detail": f"马号 {h} 的骑手配对错误：应为 {gt_r}，实为 {pred_r}",
            })

    return errors
```

**Step 4: 运行测试确认通过**

```bash
cd backend && python -m pytest tests/test_eval_runner.py -v
```

预期：全部 PASS

**Step 5: Commit**

```bash
git add backend/eval_runner.py backend/tests/test_eval_runner.py
git commit -m "feat(eval): 实现六类错误归因分析"
```

---

## Task 6: 实现 Ground Truth 与 Summary JSON 加载

**Files:**
- Modify: `backend/eval_runner.py`
- Modify: `backend/tests/test_eval_runner.py`

**Step 1: 编写失败测试 — 文件加载**

在 `backend/tests/test_eval_runner.py` 中新增：

```python
import json
from pathlib import Path
from eval_runner import load_ground_truth, load_summary


class TestLoadFiles:
    """Test loading ground truth and summary JSON files."""

    def test_load_ground_truth(self, tmp_path):
        gt_data = {
            "version": "1.0",
            "videos": [
                {
                    "video_file": "race1.mp4",
                    "ground_truth": [
                        {"horse_id": "B123", "rider_name": "潘顿", "notes": ""}
                    ],
                }
            ],
        }
        gt_path = tmp_path / "ground_truth.json"
        gt_path.write_text(json.dumps(gt_data, ensure_ascii=False), encoding="utf-8")
        result = load_ground_truth(gt_path)
        assert "race1.mp4" in result
        assert result["race1.mp4"][0]["horse_id"] == "B123"

    def test_load_summary(self, tmp_path):
        summary_data = {
            "filename": "race1.mp4",
            "detections": [
                {"horse_id": "B123", "person_name": "潘顿", "confidence": "0.85", "timestamp": "00:05"},
            ],
        }
        summary_path = tmp_path / "processed_race1_summary.json"
        summary_path.write_text(json.dumps(summary_data, ensure_ascii=False), encoding="utf-8")
        result = load_summary(summary_path)
        assert result["filename"] == "race1.mp4"
        assert len(result["detections"]) == 1
```

**Step 2: 运行测试确认失败**

```bash
cd backend && python -m pytest tests/test_eval_runner.py::TestLoadFiles -v
```

预期：FAIL

**Step 3: 实现加载函数**

在 `backend/eval_runner.py` 中新增：

```python
import json
from pathlib import Path


def load_ground_truth(gt_path: Path) -> dict[str, list[dict[str, str]]]:
    """
    @param gt_path ground_truth.json 文件路径
    @return {video_file: [gt_entries]} 的映射
    """
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    return {
        v["video_file"]: v["ground_truth"]
        for v in data["videos"]
    }


def load_summary(summary_path: Path) -> dict[str, Any]:
    """
    @param summary_path _summary.json 文件路径
    @return 解析后的 summary 字典
    """
    return json.loads(summary_path.read_text(encoding="utf-8"))
```

**Step 4: 运行测试确认通过**

```bash
cd backend && python -m pytest tests/test_eval_runner.py -v
```

预期：全部 PASS

**Step 5: Commit**

```bash
git add backend/eval_runner.py backend/tests/test_eval_runner.py
git commit -m "feat(eval): 实现 ground_truth 与 summary JSON 加载"
```

---

## Task 7: 实现 Markdown 报告生成

**Files:**
- Modify: `backend/eval_runner.py`
- Modify: `backend/tests/test_eval_runner.py`

**Step 1: 编写失败测试 — 报告生成**

在 `backend/tests/test_eval_runner.py` 中新增：

```python
from eval_runner import generate_markdown_report


class TestGenerateMarkdownReport:
    """Test markdown report generation."""

    def test_report_contains_summary_table(self):
        agg = {
            "exact_match_rate": 0.5,
            "horse_id": {
                "macro": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
                "micro": {"precision": 0.88, "recall": 0.82, "f1": 0.85},
            },
            "rider": {
                "macro": {"precision": 0.7, "recall": 0.6, "f1": 0.65},
                "micro": {"precision": 0.72, "recall": 0.63, "f1": 0.67},
            },
            "pair": {
                "macro": {"precision": 0.6, "recall": 0.5, "f1": 0.55},
                "micro": {"precision": 0.62, "recall": 0.53, "f1": 0.57},
            },
        }
        per_video = []
        all_errors = []
        md = generate_markdown_report(agg, per_video, all_errors)
        assert "# 评测报告" in md
        assert "完全匹配率" in md
        assert "50.0%" in md

    def test_report_contains_error_section(self):
        agg = {
            "exact_match_rate": 0.0,
            "horse_id": {"macro": {"precision": 0, "recall": 0, "f1": 0}, "micro": {"precision": 0, "recall": 0, "f1": 0}},
            "rider": {"macro": {"precision": 0, "recall": 0, "f1": 0}, "micro": {"precision": 0, "recall": 0, "f1": 0}},
            "pair": {"macro": {"precision": 0, "recall": 0, "f1": 0}, "micro": {"precision": 0, "recall": 0, "f1": 0}},
        }
        all_errors = [
            {"video_file": "v1.mp4", "type": "Miss-Horse", "horse_id": "B123", "rider_name": "潘顿", "detail": "..."},
            {"video_file": "v1.mp4", "type": "Miss-Horse", "horse_id": "A045", "rider_name": "莫雷拉", "detail": "..."},
            {"video_file": "v1.mp4", "type": "FP-Rider", "horse_id": "", "rider_name": "何泽尧", "detail": "..."},
        ]
        md = generate_markdown_report(agg, [], all_errors)
        assert "错误归因统计" in md
        assert "Miss-Horse" in md
```

**Step 2: 运行测试确认失败**

```bash
cd backend && python -m pytest tests/test_eval_runner.py::TestGenerateMarkdownReport -v
```

预期：FAIL

**Step 3: 实现 `generate_markdown_report`**

在 `backend/eval_runner.py` 中新增：

```python
from collections import Counter
import time


def generate_markdown_report(
    agg: dict[str, Any],
    per_video: list[dict[str, Any]],
    all_errors: list[dict[str, str]],
) -> str:
    """
    @param agg 汇总指标
    @param per_video 每个视频的三维度指标
    @param all_errors 所有错误列表
    @return Markdown 格式的评测报告
    """
    lines: list[str] = []
    lines.append(f"# 评测报告\n")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. 总览\n")
    lines.append(f"- **完全匹配率：** {agg['exact_match_rate'] * 100:.1f}%")
    lines.append(f"- **评测视频数：** {len(per_video)}\n")

    lines.append("| 维度 | 宏平均 P | 宏平均 R | 宏平均 F1 | 微平均 P | 微平均 R | 微平均 F1 |")
    lines.append("|------|----------|----------|-----------|----------|----------|-----------|")
    for dim, label in [("horse_id", "马号"), ("rider", "骑手"), ("pair", "配对")]:
        ma = agg[dim]["macro"]
        mi = agg[dim]["micro"]
        lines.append(
            f"| {label} "
            f"| {ma['precision']:.1%} | {ma['recall']:.1%} | {ma['f1']:.1%} "
            f"| {mi['precision']:.1%} | {mi['recall']:.1%} | {mi['f1']:.1%} |"
        )
    lines.append("")

    if per_video:
        lines.append("## 2. 逐视频明细（按配对 F1 升序）\n")
        sorted_videos = sorted(per_video, key=lambda v: v["pair"]["f1"])
        lines.append("| 视频 | 马号 F1 | 骑手 F1 | 配对 F1 |")
        lines.append("|------|---------|---------|---------|")
        for v in sorted_videos:
            lines.append(
                f"| {v['video_file']} "
                f"| {v['horse_id']['f1']:.1%} "
                f"| {v['rider']['f1']:.1%} "
                f"| {v['pair']['f1']:.1%} |"
            )
        lines.append("")

    if all_errors:
        lines.append("## 3. 错误归因统计\n")
        error_counts = Counter(e["type"] for e in all_errors)
        total_errors = len(all_errors)
        lines.append("| 错误类型 | 数量 | 占比 |")
        lines.append("|----------|------|------|")
        for etype, count in error_counts.most_common():
            lines.append(f"| {etype} | {count} | {count / total_errors:.1%} |")
        lines.append("")

        lines.append("## 4. 错误详情\n")
        lines.append("| 视频 | 错误类型 | 马号 | 骑手 | 详情 |")
        lines.append("|------|----------|------|------|------|")
        for e in all_errors:
            lines.append(
                f"| {e['video_file']} | {e['type']} "
                f"| {e.get('horse_id', '')} | {e.get('rider_name', '')} "
                f"| {e.get('detail', '')} |"
            )
        lines.append("")

        lines.append("## 5. 优化建议\n")
        if error_counts:
            top_error = error_counts.most_common(1)[0]
            module_map = {
                "Miss-Horse": "马匹检测/追踪/OCR",
                "FP-Horse": "OCR 误读/追踪漂移",
                "Wrong-Horse": "OCR 字符识别",
                "Miss-Rider": "人脸检测/识别",
                "FP-Rider": "人脸匹配阈值",
                "Wrong-Pair": "追踪 ID 与人脸分配",
            }
            suggestion = module_map.get(top_error[0], "未知模块")
            lines.append(
                f"最高频错误为 **{top_error[0]}**（{top_error[1]} 次，占 {top_error[1] / total_errors:.1%}），"
                f"建议优先优化 **{suggestion}** 模块。"
            )
    else:
        lines.append("## 3. 错误归因统计\n")
        lines.append("无错误，全部匹配正确。\n")

    return "\n".join(lines)
```

**Step 4: 运行测试确认通过**

```bash
cd backend && python -m pytest tests/test_eval_runner.py -v
```

预期：全部 PASS

**Step 5: Commit**

```bash
git add backend/eval_runner.py backend/tests/test_eval_runner.py
git commit -m "feat(eval): 实现 Markdown 格式评测报告生成"
```

---

## Task 8: 实现主入口脚本 — 串联完整评测流程

**Files:**
- Modify: `backend/eval_runner.py`

**Step 1: 实现 `run_evaluation` 主函数和 CLI 入口**

在 `backend/eval_runner.py` 末尾新增：

```python
import argparse


def run_evaluation(eval_dir: Path) -> dict[str, Any]:
    """
    @param eval_dir eval/ 目录路径
    @return 完整评测结果字典
    """
    gt_path = eval_dir / "ground_truth.json"
    results_dir = eval_dir / "results"
    reports_dir = eval_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    gt_map = load_ground_truth(gt_path)

    summary_files = sorted(results_dir.glob("*_summary.json"))
    filename_to_summary: dict[str, list[dict]] = {}
    for sf in summary_files:
        s = load_summary(sf)
        filename_to_summary[s["filename"]] = s["detections"]

    per_video: list[dict[str, Any]] = []
    all_errors: list[dict[str, str]] = []

    for video_file, gt_entries in gt_map.items():
        pred_entries = filename_to_summary.get(video_file, [])
        metrics = evaluate_single_video(gt_entries, pred_entries)
        metrics["video_file"] = video_file
        per_video.append(metrics)

        errors = classify_errors(gt_entries, pred_entries)
        for e in errors:
            e["video_file"] = video_file
        all_errors.extend(errors)

    agg = aggregate_metrics(per_video)

    date_str = time.strftime("%Y-%m-%d")

    report_json = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": agg,
        "per_video": per_video,
        "errors": all_errors,
    }
    json_path = reports_dir / f"eval_report_{date_str}.json"
    json_path.write_text(
        json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_content = generate_markdown_report(agg, per_video, all_errors)
    md_path = reports_dir / f"eval_report_{date_str}.md"
    md_path.write_text(md_content, encoding="utf-8")

    print(f"[eval] JSON 报告: {json_path}")
    print(f"[eval] Markdown 报告: {md_path}")
    print(f"[eval] 完全匹配率: {agg.get('exact_match_rate', 0) * 100:.1f}%")

    return report_json


def main():
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="赛马识别系统端到端评测工具")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "eval",
        help="eval/ 目录路径（默认: 项目根目录/eval/）",
    )
    args = parser.parse_args()
    run_evaluation(args.eval_dir)


if __name__ == "__main__":
    main()
```

**Step 2: 本地验证 CLI 帮助输出**

```bash
cd backend && python eval_runner.py --help
```

预期：显示帮助信息，包括 `--eval-dir` 参数说明

**Step 3: Commit**

```bash
git add backend/eval_runner.py
git commit -m "feat(eval): 实现评测主入口脚本，支持 CLI 调用"
```

---

## Task 9: 集成测试 — 使用样例数据端到端验证

**Files:**
- Modify: `backend/tests/test_eval_runner.py`

**Step 1: 编写集成测试**

在 `backend/tests/test_eval_runner.py` 中新增：

```python
from eval_runner import run_evaluation


class TestIntegration:
    """End-to-end integration test with sample data."""

    def test_full_pipeline(self, tmp_path):
        eval_dir = tmp_path / "eval"
        (eval_dir / "results").mkdir(parents=True)
        (eval_dir / "reports").mkdir(parents=True)

        gt = {
            "version": "1.0",
            "videos": [
                {
                    "video_file": "race1.mp4",
                    "ground_truth": [
                        {"horse_id": "B123", "rider_name": "潘顿", "notes": ""},
                        {"horse_id": "A045", "rider_name": "莫雷拉", "notes": ""},
                    ],
                },
                {
                    "video_file": "race2.mp4",
                    "ground_truth": [
                        {"horse_id": "C789", "rider_name": "何泽尧", "notes": ""},
                    ],
                },
            ],
        }
        (eval_dir / "ground_truth.json").write_text(
            json.dumps(gt, ensure_ascii=False), encoding="utf-8"
        )

        s1 = {
            "filename": "race1.mp4",
            "detections": [
                {"horse_id": "B123", "person_name": "潘顿", "confidence": "0.9", "timestamp": "00:05"},
                {"horse_id": "A045", "person_name": "莫雷拉", "confidence": "0.8", "timestamp": "00:10"},
            ],
        }
        (eval_dir / "results" / "processed_race1_summary.json").write_text(
            json.dumps(s1, ensure_ascii=False), encoding="utf-8"
        )

        s2 = {
            "filename": "race2.mp4",
            "detections": [
                {"horse_id": "C789", "person_name": "其他骑师", "confidence": "0.7", "timestamp": "00:03"},
            ],
        }
        (eval_dir / "results" / "processed_race2_summary.json").write_text(
            json.dumps(s2, ensure_ascii=False), encoding="utf-8"
        )

        report = run_evaluation(eval_dir)

        assert report["summary"]["exact_match_rate"] == 0.5
        assert len(report["per_video"]) == 2

        race1 = next(v for v in report["per_video"] if v["video_file"] == "race1.mp4")
        assert race1["pair"]["f1"] == 1.0

        race2_errors = [e for e in report["errors"] if e["video_file"] == "race2.mp4"]
        assert any(e["type"] == "Miss-Rider" for e in race2_errors)

        assert (eval_dir / "reports").glob("eval_report_*.md")
        assert (eval_dir / "reports").glob("eval_report_*.json")
```

**Step 2: 运行全部测试**

```bash
cd backend && python -m pytest tests/test_eval_runner.py -v
```

预期：全部 PASS

**Step 3: Commit**

```bash
git add backend/tests/test_eval_runner.py
git commit -m "test(eval): 添加端到端集成测试"
```

---

## 使用方式

评测流程完成后，运行方式如下：

```bash
# 1. 将评测视频放入 eval/videos/

# 2. 对所有视频跑系统流水线，输出 _summary.json 到 eval/results/

# 3. 填写 eval/ground_truth.json（参考 ground_truth_template.json）

# 4. 运行评测
cd backend && python eval_runner.py --eval-dir ../eval

# 5. 查看报告
cat ../eval/reports/eval_report_YYYY-MM-DD.md
```
