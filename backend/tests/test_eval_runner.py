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
