"""
评测工具：对比系统输出与 Ground Truth，计算 P/R/F1 并生成错误归因报告。
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
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
    import argparse

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
