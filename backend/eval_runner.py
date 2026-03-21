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
