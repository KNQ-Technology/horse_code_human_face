"""Tests for eval_runner module."""
from __future__ import annotations

import pytest
from eval_runner import compute_set_metrics, evaluate_single_video, aggregate_metrics


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
