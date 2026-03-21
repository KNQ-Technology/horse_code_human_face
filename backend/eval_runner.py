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
