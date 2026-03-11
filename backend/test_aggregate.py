"""Tests for _aggregate_detections in processor.py."""
from __future__ import annotations

import pytest
from processor import _aggregate_detections


def _det(
    track_id: int,
    conf: float = 0.85,
    stable_id: str = "",
    fused_ready: bool = False,
    status: str = "UNCONFIRMED",
    state_stable_id: str = "",
    rider_name: str = "",
    rider_score: float = 0.0,
    rider_source: str = "none",
) -> dict:
    """Build a single detection dict matching processor.py's det_with_roi layout."""
    return {
        "track_id": track_id,
        "conf": conf,
        "ocr_fused": {
            "stable_id": stable_id,
            "ready": fused_ready,
        },
        "track_state": {
            "status": status,
            "stable_id": state_stable_id,
        },
        "rider_identity": {
            "name": rider_name,
            "score": rider_score,
            "source": rider_source,
            "matched": bool(rider_name),
        },
    }


def _frame(index: int, detections: list[dict]) -> dict:
    return {"frame_index": index, "detections": detections}


class TestFilterNoiseTracks:
    """Tracks that never reach CONFIRMED state or have too few frames should be excluded."""

    def test_short_unconfirmed_track_excluded(self):
        """A track with only a few UNCONFIRMED frames and no stable_id is noise."""
        frames = [
            _frame(i, [_det(track_id=10, conf=0.5, status="UNCONFIRMED")])
            for i in range(5)
        ]
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 0

    def test_confirmed_track_with_stable_id_included(self):
        """A track that reaches CONFIRMED with a stable_id should appear."""
        frames = []
        for i in range(80):
            if i < 10:
                frames.append(_frame(i, [_det(track_id=1, conf=0.85, status="UNCONFIRMED")]))
            else:
                frames.append(_frame(i, [
                    _det(track_id=1, conf=0.85, stable_id="J079",
                         fused_ready=True, status="CONFIRMED", state_stable_id="J079"),
                ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["horse_id"] == "J079"


class TestMergeByHorseId:
    """Multiple tracks with the same horse_id should merge into one row."""

    def test_same_horse_id_across_tracks_merged(self):
        """K012 appearing in Track 17 and Track 63 produces a single row."""
        frames = []
        for i in range(40):
            frames.append(_frame(i, [
                _det(track_id=17, conf=0.80, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="鍾易禮", rider_score=0.75, rider_source="face"),
            ]))
        for i in range(40, 70):
            frames.append(_frame(i, [
                _det(track_id=63, conf=0.60, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        horse_ids = [r["horse_id"] for r in result]
        assert horse_ids.count("K012") == 1
        assert result[0]["person_name"] == "鍾易禮"


class TestMultipleHorseIdsPerTrack:
    """A track that changes stable_id (e.g. HOLD then new CONFIRMED) should use the best one."""

    def test_picks_horse_id_with_most_confirmed_frames(self):
        """Track 25: D012 confirmed for 20 frames, H294 confirmed for 80 frames → pick H294."""
        frames = []
        for i in range(20):
            frames.append(_frame(i, [
                _det(track_id=25, conf=0.82, stable_id="D012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="D012"),
            ]))
        for i in range(20, 100):
            frames.append(_frame(i, [
                _det(track_id=25, conf=0.83, stable_id="H294",
                     fused_ready=True, status="CONFIRMED", state_stable_id="H294"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["horse_id"] == "H294"


class TestMultipleRidersPerHorse:
    """When multiple riders are identified for one horse, pick the best one."""

    def test_picks_rider_with_most_face_frames(self):
        """Rider with far more face-source frames wins, even if score is similar."""
        frames = []
        # 鍾易禮: 50 frames with face source
        for i in range(50):
            frames.append(_frame(i, [
                _det(track_id=36, conf=0.87, stable_id="J301",
                     fused_ready=True, status="CONFIRMED", state_stable_id="J301",
                     rider_name="鍾易禮", rider_score=0.50, rider_source="face"),
            ]))
        # 潘頓: only 4 frames with face source, similar score
        for i in range(50, 54):
            frames.append(_frame(i, [
                _det(track_id=36, conf=0.87, stable_id="J301",
                     fused_ready=True, status="CONFIRMED", state_stable_id="J301",
                     rider_name="潘頓", rider_score=0.48, rider_source="face"),
            ]))
        # More confirmed frames to keep it above threshold
        for i in range(54, 80):
            frames.append(_frame(i, [
                _det(track_id=36, conf=0.87, stable_id="J301",
                     fused_ready=True, status="CONFIRMED", state_stable_id="J301"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["person_name"] == "鍾易禮"

    def test_color_only_rider_loses_to_face_rider(self):
        """face source beats color source regardless of score."""
        frames = []
        for i in range(30):
            frames.append(_frame(i, [
                _det(track_id=76, conf=0.87, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="何澤堯", rider_score=0.60, rider_source="color"),
            ]))
        for i in range(30, 60):
            frames.append(_frame(i, [
                _det(track_id=76, conf=0.87, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="蔡明紹", rider_score=0.45, rider_source="face"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["person_name"] == "蔡明紹"

    def test_few_face_frames_rider_treated_as_noise(self):
        """A rider identified by face in only 3 frames should be ignored as noise."""
        frames = []
        # Real rider: many face frames
        for i in range(60):
            frames.append(_frame(i, [
                _det(track_id=17, conf=0.80, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="鍾易禮", rider_score=0.50, rider_source="face"),
            ]))
        # Noise rider: only 3 face frames
        for i in range(60, 63):
            frames.append(_frame(i, [
                _det(track_id=17, conf=0.80, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="潘頓", rider_score=0.55, rider_source="face"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert result[0]["person_name"] == "鍾易禮"

    def test_face_locked_not_counted_as_independent_evidence(self):
        """face_locked is cached result, not independent face comparison.
        蔡明紹 has 38 face + 48 face_locked = 86 total, but only 38 independent.
        鍾易禮 has 46 face (all independent). 鍾易禮 should win."""
        frames = []
        # 鍾易禮: 46 face frames on K012
        for i in range(46):
            frames.append(_frame(i, [
                _det(track_id=17, conf=0.80, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="鍾易禮", rider_score=0.47, rider_source="face"),
            ]))
        # 蔡明紹: 38 face + 48 face_locked (should only count 38)
        for i in range(46, 84):
            frames.append(_frame(i, [
                _det(track_id=76, conf=0.87, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="蔡明紹", rider_score=0.36, rider_source="face"),
            ]))
        for i in range(84, 132):
            frames.append(_frame(i, [
                _det(track_id=76, conf=0.87, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="蔡明紹", rider_score=0.36, rider_source="face_locked"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["person_name"] == "鍾易禮"

    def test_tie_broken_by_avg_score(self):
        """When two riders have same face frame count, higher avg score wins."""
        frames = []
        # 鍾易禮: 46 face frames, avg score 0.47
        for i in range(46):
            frames.append(_frame(i, [
                _det(track_id=17, conf=0.80, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="鍾易禮", rider_score=0.47, rider_source="face"),
            ]))
        # 何澤堯: 46 face frames, avg score 0.26
        for i in range(46, 92):
            frames.append(_frame(i, [
                _det(track_id=76, conf=0.87, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="何澤堯", rider_score=0.26, rider_source="face"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["person_name"] == "鍾易禮"

    def test_cross_track_same_horse_picks_dominant_rider(self):
        """When same horse appears in multiple tracks, the rider with most face frames across all tracks wins."""
        frames = []
        # Track 17: K012 with 鍾易禮 (50 face frames)
        for i in range(50):
            frames.append(_frame(i, [
                _det(track_id=17, conf=0.80, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="鍾易禮", rider_score=0.47, rider_source="face"),
            ]))
        # Track 76: K012 with 何澤堯 (20 face frames) then 蔡明紹 (8 face_locked frames only)
        for i in range(50, 70):
            frames.append(_frame(i, [
                _det(track_id=76, conf=0.87, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="何澤堯", rider_score=0.45, rider_source="face"),
            ]))
        for i in range(70, 78):
            frames.append(_frame(i, [
                _det(track_id=76, conf=0.87, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012",
                     rider_name="蔡明紹", rider_score=0.46, rider_source="face_locked"),
            ]))
        # Padding to meet threshold
        for i in range(78, 100):
            frames.append(_frame(i, [
                _det(track_id=76, conf=0.87, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["person_name"] == "鍾易禮"


class TestQualityThreshold:
    """Only display results that meet minimum quality criteria."""

    def test_track_without_confirmed_state_excluded(self):
        """A track with stable_id but never CONFIRMED should not show."""
        frames = [
            _frame(i, [
                _det(track_id=27, conf=0.65, stable_id="X999",
                     fused_ready=True, status="UNCONFIRMED", state_stable_id=""),
            ])
            for i in range(30)
        ]
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 0

    def test_track_with_few_confirmed_frames_excluded(self):
        """A track confirmed for only 3 frames should be filtered as noise."""
        frames = [
            _frame(i, [_det(track_id=44, conf=0.57, status="UNCONFIRMED")])
            for i in range(20)
        ]
        for i in range(20, 23):
            frames.append(_frame(i, [
                _det(track_id=44, conf=0.57, stable_id="Z001",
                     fused_ready=True, status="CONFIRMED", state_stable_id="Z001"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 0

    def test_no_rider_still_shows_as_other(self):
        """A confirmed horse without rider should show person_name='其他骑师'."""
        frames = [
            _frame(i, [
                _det(track_id=1, conf=0.88, stable_id="J079",
                     fused_ready=True, status="CONFIRMED", state_stable_id="J079"),
            ])
            for i in range(50)
        ]
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["person_name"] == "其他骑师"


class TestHorseQualityFilter:
    """Horses should be filtered by avg confidence and duration."""

    def test_low_avg_conf_horse_excluded(self):
        """A horse with low average detection confidence should be filtered out."""
        frames = []
        for i in range(80):
            frames.append(_frame(i, [
                _det(track_id=99, conf=0.35, stable_id="X111",
                     fused_ready=True, status="CONFIRMED", state_stable_id="X111"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 0

    def test_high_conf_horse_passes(self):
        """A horse with high average detection confidence should pass."""
        frames = []
        for i in range(80):
            frames.append(_frame(i, [
                _det(track_id=99, conf=0.85, stable_id="J301",
                     fused_ready=True, status="CONFIRMED", state_stable_id="J301"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["horse_id"] == "J301"

    def test_short_duration_horse_excluded(self):
        """A horse that appears for less than 2 seconds should be filtered out."""
        frames = []
        # 25 fps, 30 frames = 1.2 sec (too short)
        for i in range(30):
            frames.append(_frame(i, [
                _det(track_id=55, conf=0.90, stable_id="Z999",
                     fused_ready=True, status="CONFIRMED", state_stable_id="Z999"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 0

    def test_long_duration_horse_passes(self):
        """A horse appearing for several seconds should pass."""
        frames = []
        # 25 fps, 100 frames = 4 sec (passes)
        for i in range(100):
            frames.append(_frame(i, [
                _det(track_id=55, conf=0.85, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1

    def test_mixed_quality_only_good_horses_pass(self):
        """Among multiple horses, only those meeting both conf and duration pass."""
        frames = []
        # Good horse: high conf, long duration (100 frames = 4s)
        for i in range(100):
            frames.append(_frame(i, [
                _det(track_id=1, conf=0.88, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012"),
                # Bad horse: low conf, long duration
                _det(track_id=2, conf=0.30, stable_id="X001",
                     fused_ready=True, status="CONFIRMED", state_stable_id="X001"),
            ]))
        # Another bad horse: high conf, short duration (20 frames = 0.8s)
        for i in range(100, 120):
            frames.append(_frame(i, [
                _det(track_id=1, conf=0.88, stable_id="K012",
                     fused_ready=True, status="CONFIRMED", state_stable_id="K012"),
                _det(track_id=3, conf=0.92, stable_id="Y002",
                     fused_ready=True, status="CONFIRMED", state_stable_id="Y002"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        horse_ids = [r["horse_id"] for r in result]
        assert "K012" in horse_ids
        assert "X001" not in horse_ids
        assert "Y002" not in horse_ids

    def test_cross_track_horse_aggregates_duration(self):
        """Duration should span from first to last frame across all tracks."""
        frames = []
        # Track 10: frames 0-30, then gap, then Track 20: frames 70-100
        for i in range(30):
            frames.append(_frame(i, [
                _det(track_id=10, conf=0.85, stable_id="H294",
                     fused_ready=True, status="CONFIRMED", state_stable_id="H294"),
            ]))
        for i in range(30, 70):
            frames.append(_frame(i, []))
        for i in range(70, 100):
            frames.append(_frame(i, [
                _det(track_id=20, conf=0.83, stable_id="H294",
                     fused_ready=True, status="CONFIRMED", state_stable_id="H294"),
            ]))
        result = _aggregate_detections(frames, fps=25.0)
        assert len(result) == 1
        assert result[0]["horse_id"] == "H294"


class TestEndToEndWithRealPattern:
    """Test with data matching the real results.json pattern."""

    def test_real_pattern_produces_expected_output(self):
        """Simulate the pattern from test_people.mp4 results."""
        frames = []
        # Track 1: J079, confirmed, no rider, 90 frames
        for i in range(90):
            st = "CONFIRMED" if i >= 10 else "UNCONFIRMED"
            sid = "J079" if i >= 10 else ""
            frames.append(_frame(i, [
                _det(track_id=1, conf=0.88, stable_id=sid,
                     fused_ready=bool(sid), status=st, state_stable_id=sid),
            ]))
        # Track 10: noise, 2 frames
        for i in range(90, 92):
            frames.append(_frame(i, [
                _det(track_id=10, conf=0.51, status="UNCONFIRMED"),
            ]))
        # Track 2: L097, confirmed, no rider, 148 frames
        for i in range(100, 248):
            st = "CONFIRMED" if i >= 115 else "UNCONFIRMED"
            sid = "L097" if i >= 115 else ""
            frames.append(_frame(i, [
                _det(track_id=2, conf=0.81, stable_id=sid,
                     fused_ready=bool(sid), status=st, state_stable_id=sid),
            ]))

        result = _aggregate_detections(frames, fps=25.0)
        horse_ids = {r["horse_id"] for r in result}
        assert "J079" in horse_ids
        assert "L097" in horse_ids
        # Track 10 should not appear
        assert all("T10" not in r["horse_id"] for r in result)
        assert len(result) == 2
