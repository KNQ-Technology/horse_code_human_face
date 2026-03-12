"""Tests for display_name correction in rider_feature_store and rider_identity.

Bug: display_name in the feature store is frozen at creation time and never
corrected, even when face recognition consistently identifies a different rider.
The cached_rider_code creates a self-reinforcing loop, and the face_lock fallback
only fires when the store returns an empty name (which never happens).
"""
from __future__ import annotations

import numpy as np
import pytest

from horse_id.rider_feature_store import RiderFeatureStore
from horse_id.rider_identity import RiderIdentityConfig, RiderIdentityModule
from horse_id.types import HorseDetection


def _random_color(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.rand(576).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def store(tmp_path):
    known = {"班德禮", "田泰安", "巫显东", "艾道拿"}
    return RiderFeatureStore(str(tmp_path / "riders.sqlite"), known_names=known)


# ---------------------------------------------------------------------------
# Fix #1: Feature Store — display_name correction
# ---------------------------------------------------------------------------

class TestStoreDisplayNameCorrection:
    """_update_row should correct display_name when face evidence identifies
    a different known rider with sufficient confidence."""

    def test_corrects_when_face_score_exceeds_threshold(self, store):
        color = _random_color()
        created = store._create_rider(
            display_name="田泰安", face_name="田泰安",
            face_score=0.40, color_feat=color,
            status="known", horse_hint="", match_type="face",
        )
        code = created["rider_code"]

        result = store.observe(
            face_name="班德禮", face_score=0.65,
            color_feat=color, horse_hint="H303", horse_map_name="",
            cached_rider_code=code,
        )
        assert result["display_name"] == "班德禮"

    def test_corrects_from_accumulated_evidence(self, store):
        """Stored face_name + high score should trigger correction even
        when the current frame provides no new face observation."""
        color = _random_color()
        store._create_rider(
            display_name="田泰安", face_name="班德禮",
            face_score=0.65, color_feat=color,
            status="known", horse_hint="H303", match_type="face",
        )
        code = store.conn.execute(
            "SELECT rider_code FROM riders LIMIT 1"
        ).fetchone()["rider_code"]

        result = store.observe(
            face_name="", face_score=0.0,
            color_feat=color, horse_hint="", horse_map_name="",
            cached_rider_code=code,
        )
        assert result["display_name"] == "班德禮"

    def test_no_correction_when_face_score_below_threshold(self, store):
        color = _random_color()
        created = store._create_rider(
            display_name="田泰安", face_name="田泰安",
            face_score=0.20, color_feat=color,
            status="known", horse_hint="", match_type="face",
        )
        code = created["rider_code"]

        result = store.observe(
            face_name="班德禮", face_score=0.30,
            color_feat=color, horse_hint="", horse_map_name="",
            cached_rider_code=code,
        )
        assert result["display_name"] == "田泰安"

    def test_no_correction_for_unknown_name(self, tmp_path):
        s = RiderFeatureStore(
            str(tmp_path / "limited.sqlite"), known_names={"田泰安"},
        )
        color = _random_color()
        created = s._create_rider(
            display_name="田泰安", face_name="田泰安",
            face_score=0.40, color_feat=color,
            status="known", horse_hint="", match_type="face",
        )
        code = created["rider_code"]

        result = s.observe(
            face_name="未注册骑手", face_score=0.80,
            color_feat=color, horse_hint="", horse_map_name="",
            cached_rider_code=code,
        )
        assert result["display_name"] == "田泰安"

    def test_no_correction_when_names_already_match(self, store):
        color = _random_color()
        created = store._create_rider(
            display_name="田泰安", face_name="田泰安",
            face_score=0.40, color_feat=color,
            status="known", horse_hint="", match_type="face",
        )
        code = created["rider_code"]

        result = store.observe(
            face_name="田泰安", face_score=0.70,
            color_feat=color, horse_hint="", horse_map_name="",
            cached_rider_code=code,
        )
        assert result["display_name"] == "田泰安"


# ---------------------------------------------------------------------------
# Fix #2: Rider Identity — face_lock override
# ---------------------------------------------------------------------------

class TestFaceLockOverridesStore:
    """When face_lock confidently identifies a different person than the store,
    the locked name should win — even when the face_score is too low for
    the store to self-correct (fix #1)."""

    def test_face_lock_overrides_stale_store_name(self, tmp_path):
        db_path = str(tmp_path / "riders.sqlite")
        known = {"班德禮", "田泰安"}

        config = RiderIdentityConfig(
            face_db_uri="",
            feature_store_path=db_path,
        )
        module = RiderIdentityModule(config)
        module._feature_store._known_names = known
        module._known_face_names = known

        color = _random_color()
        created = module._feature_store._create_rider(
            display_name="田泰安", face_name="田泰安",
            face_score=0.40, color_feat=color,
            status="known", horse_hint="", match_type="color",
        )
        rider_code = created["rider_code"]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        module.begin_frame(frame)

        track_id = 5
        state = module._get_track_state(track_id)
        state.locked_name = "班德禮"
        state.locked_score = 0.40
        state.lock_streak = 5
        state.lock_candidate = "班德禮"
        state.frames_since_face = 0

        module._track_rider_cache[track_id] = rider_code

        det = HorseDetection(
            x=640, y=360, w=200, h=400,
            conf=0.9, cls_name="horse", track_id=track_id,
        )
        result = module.identify(det, ocr_fused_payload=None)
        assert result["name"] == "班德禮"
        assert result["source"] == "face_locked"
