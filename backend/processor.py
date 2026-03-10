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


def _aggregate_detections(
    frame_results: list[dict[str, Any]], fps: float
) -> list[dict[str, str]]:
    """Aggregate per-frame detections into a summary for the frontend.

    Groups by track_id and picks the best-confidence snapshot for each
    confirmed horse/rider pair.
    """
    best_by_track: dict[int, dict[str, Any]] = {}

    for fr in frame_results:
        for det in fr.get("detections", []):
            track_id = det.get("track_id")
            if track_id is None:
                continue

            state = det.get("track_state", {})
            fused = det.get("ocr_fused", {})
            rider = det.get("rider_identity", {})

            stable_id = str(fused.get("stable_id", ""))
            rider_name = str(rider.get("name", ""))
            horse_conf = float(det.get("conf", 0))
            rider_score = float(rider.get("score", 0))

            if not stable_id and not rider_name:
                continue

            prev = best_by_track.get(track_id)
            if prev is None or horse_conf > prev["horse_conf"]:
                best_by_track[track_id] = {
                    "frame_index": fr["frame_index"],
                    "stable_id": stable_id,
                    "rider_name": rider_name,
                    "horse_conf": horse_conf,
                    "rider_score": rider_score,
                    "status": state.get("status", ""),
                }

    results = []
    for _tid, info in sorted(best_by_track.items()):
        horse_id = info["stable_id"] if info["stable_id"] else f"T{_tid}"
        person_name = info["rider_name"] if info["rider_name"] else "未识别"
        conf_str = f"{info['horse_conf']:.2f}_{info['rider_score']:.2f}"
        timestamp = _frame_to_timestamp(info["frame_index"], fps)
        results.append({
            "timestamp": timestamp,
            "horse_id": horse_id,
            "person_name": person_name,
            "confidence": conf_str,
        })

    return results


def process_video(
    task_id: str,
    video_path: str,
    output_video_path: str,
    tasks: dict,
    config_path: str = "config/pipeline.yaml",
    face_db_uri: str = "",
    face_models_dir: str = "",
    horse_rider_map: str = "",
) -> dict[str, Any]:
    """Run the full detection pipeline on an uploaded video.

    Returns a result dict matching the frontend API format.
    """
    config = load_config(config_path)
    config.runtime.input_video = video_path
    config.runtime.output_video = output_video_path
    config.runtime.output_json = ""

    vlm_api_key = os.environ.get("VLM_API_KEY", "")
    if vlm_api_key and config.vlm_fallback:
        config.vlm_fallback.api_key = vlm_api_key
        config.vlm_fallback.enabled = True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    detector = HorseDetector(config.detector)
    roi_extractor = ROIExtractor(config.roi)
    enhancer = ROIEnhancer(config.enhance)
    ocr_engine = OCREngine(config.ocr)

    vlm_fallback: VLMFallback | None = None
    if config.vlm_fallback and config.vlm_fallback.enabled and config.vlm_fallback.api_key:
        vlm_fallback = VLMFallback(config.vlm_fallback)
        print("[vlm_fallback] enabled")
    else:
        print("[vlm_fallback] disabled")

    track_fuser = OCRTrackFuser(config.fusion)
    visualizer = ResultVisualizer()

    rider_identity = RiderIdentityModule(
        RiderIdentityConfig(
            face_db_uri=face_db_uri,
            face_models_dir=face_models_dir,
            horse_rider_map_path=horse_rider_map,
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

    tasks[task_id]["message"] = "正在初始化 AI 模型..."
    tasks[task_id]["progress"] = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        track_state.begin_frame()
        detections = detector.detect(frame)
        rider_identity.begin_frame(frame)
        cur_h, cur_w = frame.shape[:2]
        det_with_roi = []
        rois = []
        ocr_infos: list[dict[str, object]] = []
        first_roi_raw = None
        first_roi_enh = None
        first_roi_bin = None
        first_quality_score = None
        first_ocr_text = None
        first_ocr_conf = None
        first_ocr_valid = None

        for det in detections:
            roi = roi_extractor.build_roi(det=det, frame_w=cur_w, frame_h=cur_h)
            rois.append(roi)
            roi_crop = frame[roi.y1:roi.y2, roi.x1:roi.x2]

            if roi_crop.size == 0:
                enhanced_gray = None
                binary_img = None
                quality_payload = {"score": 0.0, "sharpness": 0.0, "brightness": 0.0, "contrast": 0.0}
            else:
                enhanced_gray, binary_img = enhancer.enhance(roi_crop)
                quality = enhancer.quality(enhanced_gray)
                quality_payload = enhancer.serialize_quality(quality)

            ocr_source = "empty"
            if roi_crop.size == 0:
                ocr_payload: dict[str, object] = {"text": "", "conf": 0.0, "valid": False, "raw_text": ""}
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

            vlm_detail: dict[str, str] = {}
            if not ocr_payload["valid"] and vlm_fallback is not None:
                vlm_result = vlm_fallback.infer(
                    track_id=det.track_id, frame=frame, det=det, frame_idx=frame_idx,
                )
                vlm_detail = vlm_fallback.get_detail(det.track_id)
                if vlm_result.valid:
                    ocr_payload = OCREngine.serialize(vlm_result)
                    ocr_source = "vlm_fallback"

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
            rider_payload = rider_identity.identify(det=det, ocr_fused_payload=fused_payload)

            if first_roi_raw is None and roi_crop.size != 0 and enhanced_gray is not None and binary_img is not None:
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

        vis_frame = visualizer.draw_frame(
            frame=frame, detections=detections, rois=rois, ocr_infos=ocr_infos,
            unmatched_faces=rider_identity.unmatched_faces if rider_identity.enabled else None,
        )
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
        writer.write(vis_frame)

        frame_results.append({"frame_index": frame_idx, "detections": det_with_roi})
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

    duration_sec = frame_idx / max(fps, 1.0)
    mm = int(duration_sec) // 60
    ss = int(duration_sec) % 60
    duration_str = f"{mm:02d}:{ss:02d}"

    summary_detections = _aggregate_detections(frame_results, fps)

    return {
        "filename": os.path.basename(video_path),
        "duration": duration_str,
        "resolution": resolution_str,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detections": summary_detections,
    }
