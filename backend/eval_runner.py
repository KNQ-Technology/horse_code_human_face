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
