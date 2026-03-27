from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from horse_id.config import load_config, PipelineConfig
from horse_id.detector import HorseDetector
from horse_id.direction_filter import DirectionFilter
from horse_id.types import HorseDetection, ROIBox
from horse_id.enhancer import ROIEnhancer
from horse_id.ocr_engine import OCREngine
from horse_id.rider_identity import RiderIdentityConfig, RiderIdentityModule
from horse_id.roi_extractor import ROIExtractor
from horse_id.track_fusion import OCRTrackFuser
from horse_id.track_state import TrackStateMachine
from horse_id.visualizer import ResultVisualizer
from horse_id.vlm_fallback import VLMFallback


class _FFmpegWriter:
    """Video writer that pipes raw frames to FFmpeg for H.264 encoding."""

    def __init__(self, proc: subprocess.Popen[bytes], output_path: str) -> None:
        self._proc = proc
        self._output_path = output_path

    def isOpened(self) -> bool:
        return self._proc.stdin is not None and self._proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def release(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0:
            stderr_tail = ""
            if self._proc.stderr:
                stderr_tail = self._proc.stderr.read().decode(errors="replace")[-500:]
            print(f"[video] FFmpeg exited with code {self._proc.returncode}: {stderr_tail}")


def _create_video_writer(
    output_path: str, fps: float, width: int, height: int
) -> cv2.VideoWriter | _FFmpegWriter:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            probe = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5,
            )
            if "libx264" in probe.stdout:
                encoder = "libx264"
            elif "libopenh264" in probe.stdout:
                encoder = "libopenh264"
            else:
                raise RuntimeError("No H.264 encoder found in FFmpeg")

            cmd: list[str] = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", encoder,
                "-pix_fmt", "yuv420p",
            ]
            if encoder == "libx264":
                cmd += ["-preset", "medium", "-crf", "23"]
            cmd.append(output_path)

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            print(f"[video] Using FFmpeg ({encoder}) for H.264 output encoding.")
            return _FFmpegWriter(proc, output_path)
        except Exception as exc:
            print(f"[video] FFmpeg init failed ({exc}), falling back to OpenCV mp4v.")

    print("[video] FFmpeg not found, using OpenCV mp4v (output file may be larger).")
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def _frame_to_timestamp(frame_idx: int, fps: float) -> str:
    seconds = frame_idx / max(fps, 1.0)
    mm = int(seconds) // 60
    ss = int(seconds) % 60
    return f"{mm:02d}:{ss:02d}"


_TRUSTED_FACE_SOURCES = {"face", "face_locked"}

MIN_CONFIRMED_FRAMES = 10
MIN_FACE_FRAMES_FOR_RIDER = 5
MIN_AVG_CONF = 0.50
MIN_DURATION_SEC = 2.0

# Simple (VLM-only) mode: IDs shown on video / in API summary after this many frames.
MIN_SIMPLE_DISPLAY_FRAMES = 25


def _aggregate_detections(
    frame_results: list[dict[str, Any]], fps: float
) -> list[dict[str, str]]:
    """Aggregate per-frame detections into a summary for the frontend.

    Strategy:
    1. Per track, count confirmed frames per stable_id → pick the best horse_id.
    2. Filter: only keep tracks that reached CONFIRMED with ≥ MIN_CONFIRMED_FRAMES.
    3. Group by horse_id across all tracks → one row per horse.
    4. Pick best rider by source priority then score.
    """
    track_info: dict[int, dict[str, Any]] = {}

    for fr in frame_results:
        for det in fr.get("detections", []):
            track_id = det.get("track_id")
            if track_id is None:
                continue

            state = det.get("track_state", {})
            fused = det.get("ocr_fused", {})
            rider = det.get("rider_identity", {})

            status = str(state.get("status", "UNCONFIRMED"))
            stable_id = str(fused.get("stable_id", ""))
            rider_name = str(rider.get("name", ""))
            rider_score = float(rider.get("score", 0))
            rider_source = str(rider.get("source", "none"))
            horse_conf = float(det.get("conf", 0))

            if track_id not in track_info:
                track_info[track_id] = {
                    "confirmed_by_id": {},
                    "confs_by_id": {},
                    "frames_by_id": {},
                    "total_frames": 0,
                    "riders_by_id": {},
                    "best_conf": 0.0,
                    "best_frame": 0,
                }
            ti = track_info[track_id]
            ti["total_frames"] += 1
            frame_idx_val = fr["frame_index"]

            if status == "CONFIRMED" and stable_id:
                ti["confirmed_by_id"][stable_id] = ti["confirmed_by_id"].get(stable_id, 0) + 1
                if stable_id not in ti["confs_by_id"]:
                    ti["confs_by_id"][stable_id] = []
                ti["confs_by_id"][stable_id].append(horse_conf)
                if stable_id not in ti["frames_by_id"]:
                    ti["frames_by_id"][stable_id] = [frame_idx_val, frame_idx_val]
                else:
                    rng = ti["frames_by_id"][stable_id]
                    rng[0] = min(rng[0], frame_idx_val)
                    rng[1] = max(rng[1], frame_idx_val)

            if rider_name and status == "CONFIRMED" and stable_id:
                if stable_id not in ti["riders_by_id"]:
                    ti["riders_by_id"][stable_id] = []
                ti["riders_by_id"][stable_id].append((rider_name, rider_score, rider_source))

            if horse_conf > ti["best_conf"]:
                ti["best_conf"] = horse_conf
                ti["best_frame"] = frame_idx_val

    horse_groups: dict[str, dict[str, Any]] = {}

    for _tid, ti in track_info.items():
        if not ti["confirmed_by_id"]:
            continue

        best_id = max(ti["confirmed_by_id"], key=ti["confirmed_by_id"].get)  # type: ignore[arg-type]
        confirmed_count = ti["confirmed_by_id"][best_id]

        if confirmed_count < MIN_CONFIRMED_FRAMES:
            continue

        if best_id not in horse_groups:
            horse_groups[best_id] = {
                "confirmed_frames": 0,
                "riders": [],
                "confs": [],
                "first_frame": 999999999,
                "last_frame": 0,
                "best_conf": 0.0,
                "best_frame": 0,
            }
        hg = horse_groups[best_id]
        hg["confirmed_frames"] += confirmed_count
        hg["riders"].extend(ti["riders_by_id"].get(best_id, []))
        hg["confs"].extend(ti["confs_by_id"].get(best_id, []))
        if best_id in ti["frames_by_id"]:
            rng = ti["frames_by_id"][best_id]
            hg["first_frame"] = min(hg["first_frame"], rng[0])
            hg["last_frame"] = max(hg["last_frame"], rng[1])
        if ti["best_conf"] > hg["best_conf"]:
            hg["best_conf"] = ti["best_conf"]
            hg["best_frame"] = ti["best_frame"]

    horse_rider_picks: dict[str, tuple[str, int, float]] = {}

    for horse_id, hg in horse_groups.items():
        avg_conf = sum(hg["confs"]) / len(hg["confs"]) if hg["confs"] else 0.0
        if avg_conf < MIN_AVG_CONF:
            continue
        duration_frames = hg["last_frame"] - hg["first_frame"] + 1
        duration_sec = duration_frames / max(fps, 1.0)
        if duration_sec < MIN_DURATION_SEC:
            continue

        rider_name = ""
        rider_face_cnt = 0
        rider_score = 0.0
        if hg["riders"]:
            face_count: dict[str, int] = {}
            score_sum: dict[str, float] = {}
            for rname, rscore, rsource in hg["riders"]:
                if rsource in _TRUSTED_FACE_SOURCES:
                    face_count[rname] = face_count.get(rname, 0) + 1
                    score_sum[rname] = score_sum.get(rname, 0.0) + rscore

            candidates = {
                name: cnt for name, cnt in face_count.items()
                if cnt >= MIN_FACE_FRAMES_FOR_RIDER
            }
            if candidates:
                rider_name = max(
                    candidates,
                    key=lambda n: (candidates[n], score_sum[n] / face_count[n]),
                )
                rider_face_cnt = candidates[rider_name]
                rider_score = score_sum[rider_name] / face_count[rider_name]

        horse_rider_picks[horse_id] = (rider_name, rider_face_cnt, rider_score)

    claimed: dict[str, str] = {}
    for horse_id in sorted(
        horse_rider_picks,
        key=lambda h: -horse_rider_picks[h][1],
    ):
        rname, cnt, _ = horse_rider_picks[horse_id]
        if rname and rname not in claimed:
            claimed[rname] = horse_id

    results = []
    for horse_id in sorted(horse_rider_picks.keys()):
        rname, _, rscore = horse_rider_picks[horse_id]
        if rname and claimed.get(rname) != horse_id:
            continue

        hg = horse_groups[horse_id]
        person_name = rname if rname else "其他骑师"
        conf_str = f"{hg['best_conf']:.2f}_{rscore:.2f}"
        timestamp = _frame_to_timestamp(hg["best_frame"], fps)
        results.append({
            "timestamp": timestamp,
            "horse_id": horse_id,
            "person_name": person_name,
            "confidence": conf_str,
        })

    return results


def _aggregate_detections_simple(
    frame_results: list[dict[str, Any]], fps: float
) -> list[dict[str, str]]:
    """Aggregate per-frame VLM detections for simple mode.

    No temporal voting — directly counts VLM-detected IDs per track,
    picks the most frequent ID per track, then deduplicates across tracks.
    """
    track_votes: dict[int, dict[str, int]] = {}

    for fr in frame_results:
        for det in fr.get("detections", []):
            track_id = det.get("track_id")
            if track_id is None:
                continue
            vlm_id = str(det.get("vlm_id", ""))
            if not vlm_id:
                continue
            if track_id not in track_votes:
                track_votes[track_id] = {}
            track_votes[track_id][vlm_id] = track_votes[track_id].get(vlm_id, 0) + 1

    horse_ids: dict[str, int] = {}
    for _tid, votes in track_votes.items():
        best_id = max(votes, key=votes.get)  # type: ignore[arg-type]
        count = votes[best_id]
        if best_id not in horse_ids or count > horse_ids[best_id]:
            horse_ids[best_id] = count

    return [
        {"horse_id": hid, "confidence": f"{count}"}
        for hid, count in sorted(horse_ids.items())
        if count >= MIN_SIMPLE_DISPLAY_FRAMES
    ]


def process_video(
    task_id: str,
    video_path: str,
    output_video_path: str,
    tasks: dict,
    config_path: str = "config/pipeline.yaml",
    mode: str = "full",
) -> dict[str, Any]:
    """Run the detection pipeline on an uploaded video.

    Args:
        mode: "full" for complete pipeline, "simple" for VLM saddle-pad only.

    Returns a result dict matching the frontend API format.
    """
    is_simple = mode == "simple"
    config = load_config(config_path)
    config.runtime.input_video = video_path
    config.runtime.output_video = output_video_path
    config.runtime.output_json = ""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    detector = HorseDetector(config.detector)

    roi_extractor: ROIExtractor | None = None
    enhancer: ROIEnhancer | None = None
    ocr_engine: OCREngine | None = None
    direction_filter: DirectionFilter | None = None
    rider_identity: RiderIdentityModule | None = None

    if not is_simple:
        roi_extractor = ROIExtractor(config.roi)
        enhancer = ROIEnhancer(config.enhance)
        ocr_engine = OCREngine(config.ocr, device="gpu:0")

    vlm_fallback: VLMFallback | None = None
    if config.vlm_fallback and config.vlm_fallback.enabled and config.vlm_fallback.api_key:
        vlm_cfg = config.vlm_fallback
        if is_simple:
            # In simple mode VLM is the sole detector; reduce cooldown so the
            # next call fires as soon as the previous response arrives (~0.5s gap).
            from dataclasses import replace as _dc_replace
            vlm_cfg = _dc_replace(vlm_cfg, cooldown_frames=15)
            print(f"[vlm_fallback] simple mode: cooldown_frames overridden to {vlm_cfg.cooldown_frames}")
        vlm_fallback = VLMFallback(vlm_cfg)
        print(f"[vlm_fallback] enabled  cooldown_frames={vlm_cfg.cooldown_frames}")
    else:
        print("[vlm_fallback] disabled")

    if is_simple and vlm_fallback is None:
        raise ValueError("simple mode requires vlm_fallback to be enabled with a valid api_key")

    viz_mode = config.runtime.viz_mode if config.runtime.viz_mode else "display"
    print(f"[viz] mode={viz_mode}  pipeline_mode={mode}")
    track_fuser = OCRTrackFuser(config.fusion)

    if not is_simple:
        direction_filter = DirectionFilter(
            target_direction=config.runtime.target_direction,
            min_frames=config.runtime.direction_min_frames,
            min_displacement_px=config.runtime.direction_min_displacement,
        )
        if direction_filter.target_direction != "both":
            print(f"[direction] filter active: target={direction_filter.target_direction}")

    visualizer = ResultVisualizer(viz_mode=viz_mode)

    if not is_simple:
        ri = config.rider_identity_settings
        rider_identity = RiderIdentityModule(
            RiderIdentityConfig(
                face_db_uri=ri.face_db_uri if ri else "",
                face_collection=ri.face_collection if ri else "rider_faces",
                face_dim=ri.face_dim if ri else 512,
                face_min_score=ri.face_min_score if ri else 0.3,
                face_device=ri.face_device if ri else "cuda",
                face_models_dir=ri.face_models_dir if ri else "",
                horse_rider_map_path=ri.horse_rider_map if ri else "",
                feature_store_path=ri.feature_store_path if ri else "outputs/rider_identity.sqlite",
                use_feature_store=ri.use_feature_store if ri else True,
            )
        )
        if rider_identity.enabled:
            print("[rider_identity] enabled")
        else:
            print("[rider_identity] disabled")

    ocr_interval_frames = max(1, int(config.ocr.interval_frames))
    ocr_cache_by_track: dict[int, dict[str, object]] = {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("Cannot read frame size from input video.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 0

    resolution_str = f"{frame_w}x{frame_h}"

    track_state = TrackStateMachine(config=config.fusion, fps=fps)

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = _create_video_writer(output_video_path, fps, frame_w, frame_h)
    if not writer.isOpened():
        raise ValueError(f"Cannot create output video: {output_video_path}")

    frame_results: list[dict[str, Any]] = []
    frame_idx = 0
    start_ts = time.perf_counter()
    # Per-track VLM id vote counts (simple mode) for display gating + footer overlay.
    simple_track_votes: dict[int, dict[str, int]] = {}

    _t_accum: dict[str, float] = {
        "read": 0.0, "yolo": 0.0, "face_det": 0.0, "enhance": 0.0,
        "ocr": 0.0, "vlm": 0.0, "rider_id": 0.0, "viz": 0.0, "write": 0.0,
    }

    tasks[task_id]["message"] = "正在初始化 AI 模型..."
    tasks[task_id]["progress"] = 1

    while True:
        _t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        _t_accum["read"] += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        if not is_simple:
            track_state.begin_frame()
        detections = detector.detect(frame)
        _t_accum["yolo"] += time.perf_counter() - _t0

        if not is_simple and rider_identity is not None:
            _t0 = time.perf_counter()
            rider_identity.begin_frame(frame, frame_idx=frame_idx)
            _t_accum["face_det"] += time.perf_counter() - _t0

        cur_h, cur_w = frame.shape[:2]
        det_with_roi = []
        rois = []
        vis_detections: list[HorseDetection] = []
        vis_rois: list[ROIBox] = []
        ocr_infos: list[dict[str, object]] = []
        first_roi_raw = None
        first_roi_enh = None
        first_roi_bin = None
        first_quality_score = None
        first_ocr_text = None
        first_ocr_conf = None
        first_ocr_valid = None

        for det in detections:
            if is_simple:
                vis_detections.append(det)
                vis_rois.append(ROIBox(x1=0, y1=0, x2=0, y2=0))

                _tv0 = time.perf_counter()
                vlm_detail: dict[str, str] = {}
                if vlm_fallback is not None:
                    vlm_result = vlm_fallback.infer(
                        track_id=det.track_id, frame=frame, det=det, frame_idx=frame_idx,
                    )
                    vlm_detail = vlm_fallback.get_detail(det.track_id)
                _t_accum["vlm"] += time.perf_counter() - _tv0

                vlm_full_id = vlm_detail.get("full_id", "")
                vlm_conf = 0.85 if vlm_full_id else 0.0

                tid = det.track_id
                display_vlm_id = ""
                if tid is not None and vlm_full_id:
                    if tid not in simple_track_votes:
                        simple_track_votes[tid] = {}
                    tv = simple_track_votes[tid]
                    tv[vlm_full_id] = tv.get(vlm_full_id, 0) + 1
                    best_for_track = max(tv, key=tv.get)  # type: ignore[arg-type]
                    if tv[best_for_track] >= MIN_SIMPLE_DISPLAY_FRAMES:
                        display_vlm_id = best_for_track

                ocr_infos.append({
                    "track_id": det.track_id,
                    "text": display_vlm_id,
                    "conf": vlm_conf,
                    "valid": bool(display_vlm_id),
                    "ocr_source": "vlm" if vlm_full_id else "empty",
                    "stable_id": display_vlm_id,
                    "stable_ready": bool(display_vlm_id),
                    "state": "CONFIRMED" if display_vlm_id else "UNCONFIRMED",
                    "state_id": display_vlm_id,
                    "rider_name": "",
                    "rider_score": 0.0,
                    "rider_matched": False,
                    "rider_source": "none",
                    "rider_code": "",
                    "rider_status": "",
                    "rider_new": False,
                    "face_bbox": [],
                    "face_score": 0.0,
                    "face_detected": False,
                    "horse_id": "",
                    "vlm_bg": vlm_detail.get("bg_color", ""),
                    "vlm_font": vlm_detail.get("font_color", ""),
                    "vlm_number": vlm_detail.get("number", ""),
                    "vlm_prefix": vlm_detail.get("prefix", ""),
                    "vlm_full_id": vlm_full_id,
                })

                item = {
                    **asdict(det),
                    "vlm_id": vlm_full_id,
                    "vlm_conf": vlm_conf,
                    "vlm_detail": vlm_detail,
                }
                det_with_roi.append(item)
                continue

            # --- full mode: direction filter → ROI → enhance → OCR → VLM fallback → rider ---
            direction_filter.update(det.track_id, det.x)
            if not direction_filter.should_process(det.track_id):
                continue

            roi = roi_extractor.build_roi(det=det, frame_w=cur_w, frame_h=cur_h)
            rois.append(roi)
            vis_detections.append(det)
            vis_rois.append(roi)
            roi_crop = frame[roi.y1:roi.y2, roi.x1:roi.x2]

            _te0 = time.perf_counter()
            if roi_crop.size == 0:
                enhanced_gray = None
                binary_img = None
                quality_payload = {"score": 0.0, "sharpness": 0.0, "brightness": 0.0, "contrast": 0.0}
            else:
                enhanced_gray, binary_img = enhancer.enhance(roi_crop)
                quality = enhancer.quality(enhanced_gray)
                quality_payload = enhancer.serialize_quality(quality)
            _t_accum["enhance"] += time.perf_counter() - _te0

            _to0 = time.perf_counter()
            ocr_source = "empty"
            if roi_crop.size == 0:
                ocr_payload = {"text": "", "conf": 0.0, "valid": False, "raw_text": ""}
            else:
                use_cached = False
                cached_payload: dict[str, object] | None = None
                if det.track_id is not None and ocr_interval_frames > 1:
                    cache_entry = ocr_cache_by_track.get(det.track_id)
                    if cache_entry is not None:
                        last_fi = int(cache_entry["frame_index"])
                        if frame_idx - last_fi < ocr_interval_frames:
                            use_cached = True
                            cached_payload = dict(cache_entry["ocr_payload"])

                if use_cached and cached_payload is not None:
                    ocr_payload = cached_payload
                    ocr_source = "cached"
                else:
                    ocr_result = ocr_engine.infer(enhanced_gray)
                    ocr_payload = ocr_engine.serialize(ocr_result)
                    ocr_source = "fresh"
                    if det.track_id is not None:
                        ocr_cache_by_track[det.track_id] = {
                            "frame_index": frame_idx,
                            "ocr_payload": dict(ocr_payload),
                        }

            _t_accum["ocr"] += time.perf_counter() - _to0

            _tv0 = time.perf_counter()
            vlm_detail = {}
            track_already_locked = det.track_id is not None and det.track_id in track_fuser._locked_ids
            if not ocr_payload["valid"] and vlm_fallback is not None and not track_already_locked:
                vlm_result = vlm_fallback.infer(
                    track_id=det.track_id, frame=frame, det=det, frame_idx=frame_idx,
                )
                vlm_detail = vlm_fallback.get_detail(det.track_id)
                if vlm_result.valid:
                    ocr_payload = OCREngine.serialize(vlm_result)
                    ocr_source = "vlm_fallback"
            _t_accum["vlm"] += time.perf_counter() - _tv0

            fused = track_fuser.update(
                track_id=det.track_id,
                text=str(ocr_payload["text"]),
                conf=float(ocr_payload["conf"]),
                valid=bool(ocr_payload["valid"]),
                source=ocr_source,
            )
            fused_payload = fused.to_dict()
            state_result = track_state.update(
                track_id=det.track_id,
                fused_ready=bool(fused_payload["ready"]),
                fused_id=str(fused_payload["stable_id"]),
            )
            state_payload = state_result.to_dict()
            _tr0 = time.perf_counter()
            rider_payload = rider_identity.identify(det=det, ocr_fused_payload=fused_payload)
            _t_accum["rider_id"] += time.perf_counter() - _tr0

            if viz_mode == "debug" and first_roi_raw is None and roi_crop.size != 0 and enhanced_gray is not None and binary_img is not None:
                first_roi_raw = roi_crop.copy()
                first_roi_enh = enhanced_gray.copy()
                first_roi_bin = binary_img.copy()
                first_quality_score = float(quality_payload["score"])
                first_ocr_text = str(ocr_payload["text"])
                first_ocr_conf = float(ocr_payload["conf"])
                first_ocr_valid = bool(ocr_payload["valid"])

            ocr_infos.append({
                "track_id": det.track_id,
                "text": str(ocr_payload["text"]),
                "conf": float(ocr_payload["conf"]),
                "valid": bool(ocr_payload["valid"]),
                "ocr_source": ocr_source,
                "stable_id": str(fused_payload["stable_id"]),
                "stable_ready": bool(fused_payload["ready"]),
                "state": str(state_payload["status"]),
                "state_id": str(state_payload["stable_id"]),
                "rider_name": rider_payload["name"],
                "rider_score": rider_payload["score"],
                "rider_matched": rider_payload["matched"],
                "rider_source": rider_payload.get("source", "none"),
                "rider_code": rider_payload.get("rider_code", ""),
                "rider_status": rider_payload.get("rider_status", ""),
                "rider_new": rider_payload.get("is_new_rider", False),
                "face_bbox": rider_payload.get("face_bbox", []),
                "face_score": rider_payload.get("face_score", 0.0),
                "face_detected": rider_payload.get("face_detected", False),
                "horse_id": rider_payload.get("horse_id", ""),
                "vlm_bg": vlm_detail.get("bg_color", ""),
                "vlm_font": vlm_detail.get("font_color", ""),
                "vlm_number": vlm_detail.get("number", ""),
                "vlm_prefix": vlm_detail.get("prefix", ""),
                "vlm_full_id": vlm_detail.get("full_id", ""),
            })

            item = {
                **asdict(det),
                "roi_bbox": roi_extractor.serialize(roi),
                "roi_quality": quality_payload,
                "ocr": ocr_payload,
                "ocr_runtime": {"source": ocr_source, "interval_frames": ocr_interval_frames},
                "ocr_fused": fused_payload,
                "track_state": state_payload,
                "rider_identity": rider_payload,
            }
            det_with_roi.append(item)

        _tv0 = time.perf_counter()
        show_unmatched = (
            not is_simple
            and rider_identity is not None
            and rider_identity.enabled
        )
        vis_frame = visualizer.draw_frame(
            frame=frame, detections=vis_detections, rois=vis_rois, ocr_infos=ocr_infos,
            unmatched_faces=rider_identity.unmatched_faces if show_unmatched else None,
        )
        if is_simple:
            footer_dedup: dict[str, int] = {}
            for _tid, votes in simple_track_votes.items():
                if not votes:
                    continue
                bid, cnt = max(votes.items(), key=lambda kv: kv[1])
                if cnt >= MIN_SIMPLE_DISPLAY_FRAMES:
                    footer_dedup[bid] = max(footer_dedup.get(bid, 0), cnt)
            simple_footer = sorted(footer_dedup.items())
            vis_frame = visualizer.draw_simple_saddle_footer(
                vis_frame, simple_footer, MIN_SIMPLE_DISPLAY_FRAMES,
            )
        if viz_mode == "debug" and not is_simple:
            vis_frame = visualizer.draw_roi_comparison_panel(
                frame=vis_frame,
                roi_original_bgr=first_roi_raw,
                roi_enhanced_gray=first_roi_enh,
                roi_binary=first_roi_bin,
                quality_score=first_quality_score,
                ocr_text=first_ocr_text,
                ocr_conf=first_ocr_conf,
                ocr_valid=first_ocr_valid,
            )
        _t_accum["viz"] += time.perf_counter() - _tv0
        _tw0 = time.perf_counter()
        writer.write(vis_frame)
        _t_accum["write"] += time.perf_counter() - _tw0

        frame_results.append({"frame_index": frame_idx, "detections": det_with_roi})
        if not is_simple:
            track_state.end_frame()
        frame_idx += 1

        if total_frames > 0:
            progress = min(99, int(frame_idx / total_frames * 100))
        else:
            progress = min(99, frame_idx)
        tasks[task_id]["progress"] = progress
        tasks[task_id]["message"] = f"正在进行 AI 识别... {progress}%"

    cap.release()
    writer.release()

    total_elapsed = max(1e-6, time.perf_counter() - start_ts)
    print(f"[processor] frames={frame_idx} elapsed={total_elapsed:.2f}s avg_fps={frame_idx / total_elapsed:.2f}")

    print("\n" + "=" * 60)
    print("[profiling] 各模块累计耗时 (秒 / 占比)")
    print("=" * 60)
    _t_total = sum(_t_accum.values())
    _t_other = total_elapsed - _t_total
    _labels = {
        "read": "视频读取",
        "yolo": "YOLO检测+ByteTrack",
        "face_det": "人脸检测(SCRFD)",
        "enhance": "ROI增强",
        "ocr": "OCR识别(PaddleOCR)",
        "vlm": "VLM回退(Qwen-API)",
        "rider_id": "骑手身份识别",
        "viz": "可视化绘制",
        "write": "视频写入",
    }
    for key in ["read", "yolo", "face_det", "enhance", "ocr", "vlm", "rider_id", "viz", "write"]:
        secs = _t_accum[key]
        pct = secs / total_elapsed * 100.0 if total_elapsed > 0 else 0.0
        avg_ms = secs / max(1, frame_idx) * 1000.0
        print(f"  {_labels[key]:<22s} {secs:8.2f}s  ({pct:5.1f}%)  avg {avg_ms:6.1f}ms/帧")
    if _t_other > 0:
        pct = _t_other / total_elapsed * 100.0
        avg_ms = _t_other / max(1, frame_idx) * 1000.0
        print(f"  {'其他(融合/状态/IO)':<22s} {_t_other:8.2f}s  ({pct:5.1f}%)  avg {avg_ms:6.1f}ms/帧")
    print(f"  {'总计':<22s} {total_elapsed:8.2f}s  (100.0%)")
    print("=" * 60)

    duration_sec = frame_idx / max(fps, 1.0)
    mm = int(duration_sec) // 60
    ss = int(duration_sec) % 60
    duration_str = f"{mm:02d}:{ss:02d}"

    if is_simple:
        summary_detections = _aggregate_detections_simple(frame_results, fps)
    else:
        summary_detections = _aggregate_detections(frame_results, fps)

    result = {
        "filename": os.path.basename(video_path),
        "duration": duration_str,
        "resolution": resolution_str,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detections": summary_detections,
        "mode": mode,
    }

    output_stem = Path(output_video_path).stem
    output_dir = Path(output_video_path).parent
    frames_json_path = output_dir / f"{output_stem}_frames.json"
    summary_json_path = output_dir / f"{output_stem}_summary.json"
    try:
        frames_json_path.write_text(
            json.dumps(frame_results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        summary_json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[processor] saved {frames_json_path}  ({frames_json_path.stat().st_size / 1024:.0f} KB)")
        print(f"[processor] saved {summary_json_path}")
    except Exception as exc:
        print(f"[processor] failed to save JSON: {exc}")

    return result
