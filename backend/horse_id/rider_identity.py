from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from horse_id.types import HorseDetection
from horse_id.rider_feature_store import RiderFeatureStore
from save_rider_faces_to_milvus import _is_lite_uri, init_face_detector, l2_normalize


@dataclass
class RiderIdentityConfig:
    """Configuration for rider identity module."""

    face_db_uri: str = ""
    face_collection: str = "rider_faces"
    face_dim: int = 512
    face_min_score: float = 0.3
    face_device: str = "cpu"
    face_models_dir: str = ""
    horse_rider_map_path: str = ""
    vote_window_size: int = 18
    face_weight: float = 0.7
    color_weight: float = 0.2
    horse_weight: float = 0.1
    min_accept_score: float = 0.20
    min_color_similarity: float = 0.70
    feature_store_path: str = "outputs/rider_identity.sqlite"
    face_lock_threshold: int = 3
    face_lock_max_hold: int = 90
    face_lock_challenge: int = 5
    face_interval: int = 3
    use_feature_store: bool = True


@dataclass
class _TrackIdentityState:
    votes: deque[tuple[str, float]]
    stable_name: str = ""
    stable_score: float = 0.0
    color_ema: np.ndarray | None = None
    locked_name: str = ""
    locked_score: float = 0.0
    lock_streak: int = 0
    lock_candidate: str = ""
    challenge_name: str = ""
    challenge_count: int = 0
    frames_since_face: int = 0
    last_face_bbox: list[int] = field(default_factory=list)


class _FaceIdentityBackend:
    """Face retrieval backend for rider name search in Milvus."""

    def __init__(
        self,
        db_uri: str,
        min_score: float,
        collection_name: str,
        dim: int,
        device: str,
        models_dir: str = "",
    ) -> None:
        self.db_uri = db_uri
        self.min_score = float(min_score)
        self.collection_name = collection_name
        self.dim = int(dim)
        self.use_lite = _is_lite_uri(db_uri)
        self.face_detector = init_face_detector(
            device=device,
            models_dir=Path(models_dir).resolve() if models_dir else None,
        )
        self.client = None
        self.collection = None
        if self.use_lite:
            from pymilvus import MilvusClient

            self.client = MilvusClient(str(Path(db_uri).expanduser().resolve()))
        else:
            from pymilvus import Collection, connections

            connections.connect(uri=db_uri)
            self.collection = Collection(name=collection_name)
            self.collection.load()

    def _query_name(self, embedding: list[float]) -> tuple[str, float] | None:
        if self.use_lite and self.client is not None:
            res = self.client.search(
                collection_name=self.collection_name,
                data=[embedding],
                limit=1,
                output_fields=["name"],
            )
            if not res or len(res) == 0 or len(res[0]) == 0:
                return None
            hit = res[0][0]
            score = float(hit.get("distance") or hit.get("score") or 0.0)
            if score < self.min_score:
                return None
            name = hit.get("name")
            if name is None:
                return None
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            return str(name), score

        if (not self.use_lite) and self.collection is not None:
            results = self.collection.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 128}},
                limit=1,
                output_fields=["name"],
            )
            for hits in results:
                if not hits or len(hits) == 0:
                    return None
                hit = hits[0]
                score = float(hit.distance)
                if score < self.min_score:
                    return None
                name = hit.get("name")
                if name is None and hasattr(hit, "entity") and hit.entity:
                    name = getattr(hit.entity, "name", None)
                if name is None:
                    return None
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                return str(name), score
        return None

    def get_all_known_names(self) -> set[str]:
        """Return all distinct rider names registered in the face database."""
        names: set[str] = set()
        if self.use_lite and self.client is not None:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["name"],
                limit=16384,
            )
            for row in results:
                name = row.get("name", "")
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                name = str(name).strip()
                if name:
                    names.add(name)
        elif (not self.use_lite) and self.collection is not None:
            results = self.collection.query(
                expr="id > 0",
                output_fields=["name"],
                limit=16384,
            )
            for row in results:
                name = row.get("name", "")
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                name = str(name).strip()
                if name:
                    names.add(name)
        return names

    def recognize_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        bboxes, kpss = self.face_detector.detect_with_keypoints(frame)
        if len(bboxes) == 0:
            return []
        frame_h, frame_w = frame.shape[:2]
        recognized: list[dict[str, Any]] = []
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            x1, y1, x2, y2, score = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])
            if score < 0.5:
                continue
            x1_i = max(0, int(x1))
            y1_i = max(0, int(y1))
            x2_i = min(frame_w, int(x2))
            y2_i = min(frame_h, int(y2))
            if x2_i - x1_i < 20 or y2_i - y1_i < 20:
                continue
            face_entry: dict[str, Any] = {
                "name": "",
                "score": 0.0,
                "det_score": score,
                "bbox": [x1_i, y1_i, x2_i, y2_i],
                "center": [(x1_i + x2_i) // 2, (y1_i + y2_i) // 2],
            }
            if kpss is None:
                recognized.append(face_entry)
                continue
            kps = kpss[i]
            feat = self.face_detector.get_embedding_from_kps(frame, kps)
            if feat.size == 0:
                recognized.append(face_entry)
                continue
            emb = l2_normalize(feat)
            if emb.size != self.dim:
                recognized.append(face_entry)
                continue
            query_res = self._query_name(emb.tolist())
            if query_res is None:
                recognized.append(face_entry)
                continue
            name, sim = query_res
            face_entry["name"] = name
            face_entry["score"] = float(sim)
            recognized.append(face_entry)
        return recognized


class RiderIdentityModule:
    """
    Unified rider identity module.

    Features:
    - face retrieval from Milvus
    - rider clothing color feature
    - optional horse-number -> rider mapping
    - temporal vote smoothing per track
    """

    def __init__(self, config: RiderIdentityConfig) -> None:
        self.config = config
        self._use_feature_store = config.use_feature_store
        self._face_backend: _FaceIdentityBackend | None = None
        self._track_states: dict[int, _TrackIdentityState] = {}
        self._name_color_proto: dict[str, np.ndarray] = {}
        self._track_rider_cache: dict[int, str] = {}

        if self._use_feature_store:
            self._horse_to_riders, self._rider_to_horse = self._load_horse_rider_relations(config.horse_rider_map_path)
        else:
            self._horse_to_riders, self._rider_to_horse = {}, {}

        self._frame_faces: list[dict[str, Any]] = []
        self._frame_used_face_indices: set[int] = set()
        self._frame: np.ndarray | None = None
        self._enabled = True

        uri = (config.face_db_uri or "").strip()
        if uri:
            try:
                if _is_lite_uri(uri) and not Path(uri).expanduser().resolve().exists():
                    raise FileNotFoundError(f"face db not found: {uri}")
                self._face_backend = _FaceIdentityBackend(
                    db_uri=uri,
                    min_score=config.face_min_score,
                    collection_name=config.face_collection,
                    dim=config.face_dim,
                    device=config.face_device,
                    models_dir=config.face_models_dir,
                )
            except Exception:
                self._face_backend = None

        self._known_face_names: set[str] = set()
        if self._face_backend is not None:
            self._known_face_names = self._face_backend.get_all_known_names()

        if self._use_feature_store:
            self._feature_store = RiderFeatureStore(config.feature_store_path, known_names=self._known_face_names)
        else:
            self._feature_store = None  # type: ignore[assignment]

        self._face_frame_counter: int = 0
        self._cached_frame_faces: list[dict[str, Any]] = []

        self._enabled = self._face_backend is not None or bool(self._horse_to_riders)
        if not self._use_feature_store:
            print("[rider_identity] feature_store & horse_rider_map DISABLED by config")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _load_horse_rider_relations(path: str) -> tuple[dict[str, set[str]], dict[str, str]]:
        p = (path or "").strip()
        if not p:
            return {}, {}
        file_path = Path(p).expanduser().resolve()
        if not file_path.is_file():
            return {}, {}
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return {}, {}
        if not isinstance(data, dict):
            return {}, {}

        horse_to_riders: dict[str, set[str]] = {}
        rider_to_horse: dict[str, str] = {}

        has_nested = "horse_to_riders" in data or "rider_to_horse" in data
        if has_nested:
            h2r = data.get("horse_to_riders", {})
            r2h = data.get("rider_to_horse", {})
            if isinstance(h2r, dict):
                for horse, riders in h2r.items():
                    horse_id = str(horse).strip()
                    if not horse_id:
                        continue
                    rider_set: set[str] = set()
                    if isinstance(riders, list):
                        for name in riders:
                            rider = str(name).strip()
                            if rider:
                                rider_set.add(rider)
                    elif isinstance(riders, str):
                        rider = riders.strip()
                        if rider:
                            rider_set.add(rider)
                    if rider_set:
                        horse_to_riders[horse_id] = rider_set
            if isinstance(r2h, dict):
                for rider, horse in r2h.items():
                    rider_name = str(rider).strip()
                    horse_id = str(horse).strip()
                    if rider_name and horse_id:
                        rider_to_horse[rider_name] = horse_id
                        horse_to_riders.setdefault(horse_id, set()).add(rider_name)
        else:
            for horse, riders in data.items():
                horse_id = str(horse).strip()
                if not horse_id:
                    continue
                rider_set: set[str] = set()
                if isinstance(riders, list):
                    for name in riders:
                        rider = str(name).strip()
                        if rider:
                            rider_set.add(rider)
                elif isinstance(riders, str):
                    rider = str(riders).strip()
                    if rider:
                        rider_set.add(rider)
                if rider_set:
                    horse_to_riders[horse_id] = rider_set

        for horse_id, riders in horse_to_riders.items():
            for rider_name in riders:
                if rider_name not in rider_to_horse:
                    rider_to_horse[rider_name] = horse_id
        return horse_to_riders, rider_to_horse

    def _allowed_riders_by_horse(self, horse_id: str) -> set[str]:
        hid = (horse_id or "").strip()
        if not hid:
            return set()
        return set(self._horse_to_riders.get(hid, set()))

    def _is_rider_allowed(self, rider_name: str, horse_id: str) -> bool:
        name = (rider_name or "").strip()
        hid = (horse_id or "").strip()
        if not name:
            return False
        if not hid:
            return True
        allowed = self._allowed_riders_by_horse(hid)
        if allowed:
            return name in allowed
        bound_horse = self._rider_to_horse.get(name, "")
        if bound_horse:
            return bound_horse == hid
        return True

    def begin_frame(self, frame: np.ndarray, frame_idx: int = -1) -> None:
        self._frame = frame
        self._frame_used_face_indices = set()
        if self._face_backend is None:
            self._frame_faces = []
            return
        self._face_frame_counter += 1
        if self._face_frame_counter % self.config.face_interval == 1 or self.config.face_interval <= 1:
            self._frame_faces = self._face_backend.recognize_faces(frame)
            self._cached_frame_faces = list(self._frame_faces)
        else:
            self._frame_faces = list(self._cached_frame_faces)

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-9) -> float:
        a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < eps or nb < eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _extract_rider_color_feature(self, frame: np.ndarray, det: HorseDetection) -> np.ndarray | None:
        x1 = int(round(det.x - det.w / 2.0))
        y1 = int(round(det.y - det.h / 2.0))
        x2 = int(round(det.x + det.w / 2.0))
        y2 = int(round(det.y + det.h / 2.0))
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        horse_h = max(1, y2 - y1)
        rider_top = max(0, y1 - int(0.50 * horse_h))
        rider_bottom = min(h, y1 + int(0.28 * horse_h))
        if rider_bottom <= rider_top:
            return None

        rider_patch = frame[rider_top:rider_bottom, x1:x2]
        if rider_patch.size == 0:
            return None

        hsv = cv2.cvtColor(rider_patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 6], [0, 180, 0, 256, 0, 256]).astype(np.float32)
        hist = hist.flatten()
        norm = float(np.linalg.norm(hist))
        if norm <= 1e-9:
            return None
        return hist / norm

    def _match_face_for_horse(self, det: HorseDetection) -> dict[str, Any] | None:
        x1 = int(round(det.x - det.w / 2.0))
        y1 = int(round(det.y - det.h / 2.0))
        x2 = int(round(det.x + det.w / 2.0))
        y2 = int(round(det.y + det.h / 2.0))
        horse_h = max(1, y2 - y1)
        rider_y_min = y1 - int(0.45 * horse_h)
        rider_y_max = y1 + int(0.45 * horse_h)

        best: dict[str, Any] | None = None
        best_idx: int = -1
        best_priority: float = -1.0
        for i, face in enumerate(self._frame_faces):
            center = face.get("center")
            if not isinstance(center, list) or len(center) != 2:
                continue
            cx, cy = int(center[0]), int(center[1])
            if not (x1 <= cx <= x2):
                continue
            if not (rider_y_min <= cy <= rider_y_max):
                continue
            priority = float(face.get("score", 0.0)) or float(face.get("det_score", 0.0))
            if best is None or priority > best_priority:
                best = face
                best_idx = i
                best_priority = priority
        if best_idx >= 0:
            self._frame_used_face_indices.add(best_idx)
        return best

    @property
    def unmatched_faces(self) -> list[dict[str, Any]]:
        """Return faces detected in this frame but not matched to any horse."""
        return [
            f for i, f in enumerate(self._frame_faces)
            if i not in self._frame_used_face_indices
        ]

    def _get_track_state(self, track_id: int) -> _TrackIdentityState:
        state = self._track_states.get(track_id)
        if state is not None:
            return state
        state = _TrackIdentityState(votes=deque(maxlen=max(6, int(self.config.vote_window_size))))
        self._track_states[track_id] = state
        return state

    def _update_color_proto(self, name: str, color_feat: np.ndarray) -> None:
        if not name:
            return
        old = self._name_color_proto.get(name)
        if old is None:
            self._name_color_proto[name] = color_feat
            return
        merged = 0.82 * old + 0.18 * color_feat
        norm = float(np.linalg.norm(merged))
        if norm > 1e-9:
            self._name_color_proto[name] = merged / norm

    def identify(
        self,
        det: HorseDetection,
        ocr_fused_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self._frame is None:
            return {"name": "", "score": 0.0, "matched": False, "source": "none", "face_bbox": []}

        color_feat = self._extract_rider_color_feature(self._frame, det)
        face_hit = self._match_face_for_horse(det) if self._face_backend is not None else None
        face_detected_in_frame = face_hit is not None

        horse_id = ""
        horse_rider_names: list[str] = []
        if ocr_fused_payload:
            if bool(ocr_fused_payload.get("ready", False)):
                horse_id = str(ocr_fused_payload.get("stable_id", "")).strip()
            if horse_id and self._use_feature_store:
                horse_rider_names = sorted(self._allowed_riders_by_horse(horse_id))

        allowed_names = set(horse_rider_names)

        candidate_scores: dict[str, float] = {}
        source_by_name: dict[str, str] = {}
        face_bbox: list[int] = []
        face_score = 0.0
        color_sim = 0.0

        if face_hit is not None:
            face_name = str(face_hit.get("name", ""))
            face_score = float(face_hit.get("score", 0.0))
            bbox = face_hit.get("bbox", [])
            if isinstance(bbox, list) and len(bbox) >= 4:
                face_bbox = [int(v) for v in bbox[:4]]
            if face_name and self._is_rider_allowed(face_name, horse_id):
                candidate_scores[face_name] = candidate_scores.get(face_name, 0.0) + self.config.face_weight * face_score
                source_by_name[face_name] = "face"

        if horse_rider_names:
            each_weight = self.config.horse_weight / max(1, len(horse_rider_names))
            for rider_name in horse_rider_names:
                candidate_scores[rider_name] = candidate_scores.get(rider_name, 0.0) + each_weight
                if rider_name not in source_by_name:
                    source_by_name[rider_name] = "horse_map"

        if color_feat is not None:
            best_name = ""
            best_sim = 0.0
            for name, proto in self._name_color_proto.items():
                if not self._is_rider_allowed(name, horse_id):
                    continue
                sim = self._cosine_similarity(color_feat, proto)
                if sim > best_sim:
                    best_name = name
                    best_sim = sim
            color_sim = best_sim
            if best_name and best_sim >= self.config.min_color_similarity:
                candidate_scores[best_name] = candidate_scores.get(best_name, 0.0) + self.config.color_weight * best_sim
                if best_name not in source_by_name:
                    source_by_name[best_name] = "color"

        if not candidate_scores:
            best_name = ""
            best_score = 0.0
            source = "none"
            matched = False
        else:
            best_name, best_score = max(candidate_scores.items(), key=lambda x: x[1])
            source = source_by_name.get(best_name, "mixed")
            matched = best_score >= self.config.min_accept_score
            if not matched:
                best_name = ""

        color_for_store = color_feat if color_feat is not None else None
        track_id = det.track_id
        cached_code = ""
        face_is_locked = False
        face_lock_name_out = ""

        if track_id is not None:
            state = self._get_track_state(int(track_id))
            cached_code = self._track_rider_cache.get(int(track_id), "")

            # -- Step A: face detection counter --------------------------------
            if face_bbox:
                state.frames_since_face = 0
                state.last_face_bbox = list(face_bbox)
            else:
                state.frames_since_face += 1

            # -- color EMA -----------------------------------------------------
            if color_feat is not None:
                if state.color_ema is None:
                    state.color_ema = color_feat
                else:
                    state.color_ema = 0.8 * state.color_ema + 0.2 * color_feat
                    norm = float(np.linalg.norm(state.color_ema))
                    if norm > 1e-9:
                        state.color_ema = state.color_ema / norm
                if best_name:
                    self._update_color_proto(best_name, state.color_ema)

            # -- temporal voting (unchanged) -----------------------------------
            if best_name:
                state.votes.append((best_name, best_score))

            if state.votes:
                agg: dict[str, float] = {}
                for name, score in state.votes:
                    agg[name] = agg.get(name, 0.0) + float(score)
                stable_name, stable_score_sum = max(agg.items(), key=lambda x: x[1])
                stable_score = stable_score_sum / max(1, len(state.votes))
                if stable_score >= self.config.min_accept_score:
                    state.stable_name = stable_name
                    state.stable_score = stable_score
                    best_name = stable_name
                    best_score = stable_score
                    matched = True
                    if source == "none":
                        source = "temporal"

            if state.color_ema is not None:
                color_for_store = state.color_ema

            # -- Step B: lock streak accumulation ------------------------------
            if best_name:
                if best_name == state.lock_candidate:
                    state.lock_streak += 1
                else:
                    state.lock_candidate = best_name
                    state.lock_streak = 1
                if (
                    state.lock_streak >= self.config.face_lock_threshold
                    and not state.locked_name
                ):
                    state.locked_name = state.lock_candidate
                    state.locked_score = best_score

            # -- Step C: locked-state output logic -----------------------------
            if state.locked_name:
                face_is_locked = True
                face_lock_name_out = state.locked_name

                if best_name == state.locked_name:
                    state.challenge_name = ""
                    state.challenge_count = 0
                    if best_score > state.locked_score:
                        state.locked_score = best_score
                elif best_name and best_name != state.locked_name:
                    if best_name == state.challenge_name:
                        state.challenge_count += 1
                    else:
                        state.challenge_name = best_name
                        state.challenge_count = 1
                    if state.challenge_count >= self.config.face_lock_challenge:
                        state.locked_name = state.challenge_name
                        state.locked_score = best_score
                        state.lock_candidate = state.challenge_name
                        state.lock_streak = state.challenge_count
                        state.challenge_name = ""
                        state.challenge_count = 0
                        face_lock_name_out = state.locked_name
                else:
                    state.challenge_name = ""
                    state.challenge_count = 0

                if not matched and state.frames_since_face < self.config.face_lock_max_hold:
                    best_name = state.locked_name
                    best_score = state.locked_score
                    matched = True
                    source = "face_locked"

            # -- Step D: timeout unlock ----------------------------------------
            if state.frames_since_face >= self.config.face_lock_max_hold and state.locked_name:
                state.locked_name = ""
                state.locked_score = 0.0
                state.lock_streak = 0
                state.lock_candidate = ""
                state.challenge_name = ""
                state.challenge_count = 0
                face_is_locked = False
                face_lock_name_out = ""

        if self._use_feature_store and self._feature_store is not None:
            store_identity = self._feature_store.observe(
                face_name=best_name if matched else (face_hit.get("name", "") if face_hit else ""),
                face_score=best_score if matched else face_score,
                color_feat=color_for_store,
                horse_hint=horse_id,
                horse_map_name=horse_rider_names[0] if len(horse_rider_names) == 1 else "",
                cached_rider_code=cached_code,
                color_match_threshold=self.config.min_color_similarity,
                allowed_display_names=sorted(allowed_names),
            )
            final_name = str(store_identity.get("display_name", "")) or str(store_identity.get("rider_code", ""))
            final_code = str(store_identity.get("rider_code", ""))
            final_status = str(store_identity.get("status", "unknown"))
            final_match_type = str(store_identity.get("match_type", "none"))
        else:
            store_identity = {}
            final_name = best_name if matched else ""
            final_code = ""
            final_status = "known" if matched else "unknown"
            final_match_type = source if matched else "none"

        if face_is_locked and face_lock_name_out and face_lock_name_out != final_name:
            final_name = face_lock_name_out
            final_status = "known"
            final_match_type = "face_lock"
            if not matched:
                matched = True
                best_score = (
                    state.locked_score
                    if track_id is not None
                    else best_score
                )
            source = "face_locked"

        if track_id is not None and final_code:
            self._track_rider_cache[int(track_id)] = final_code
        if final_name and color_for_store is not None:
            self._update_color_proto(final_name, color_for_store)

        return {
            "name": final_name,
            "score": float(best_score if matched else face_score),
            "matched": bool(final_name),
            "source": source if matched else ("store" if self._use_feature_store else "none"),
            "rider_code": final_code,
            "rider_status": final_status,
            "store_match_type": final_match_type,
            "is_new_rider": bool(store_identity.get("is_new", False)),
            "face_bbox": face_bbox,
            "face_score": float(face_score),
            "face_detected": face_detected_in_frame,
            "color_similarity": float(color_sim),
            "horse_id": horse_id,
            "horse_map_name": ",".join(horse_rider_names),
            "face_locked": face_is_locked,
            "face_lock_name": face_lock_name_out,
        }

