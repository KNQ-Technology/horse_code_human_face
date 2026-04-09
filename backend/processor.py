from __future__ import annotations

import _nvidia_dll_fix  # noqa: F401 — must precede paddle/torch imports

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import Counter
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
from horse_id.vlm_video import VLMVideoIdentifier, VLMVideoConfig


def _step_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class _FFmpegWriter:
    """Video writer that pipes raw frames to FFmpeg for H.264 encoding.

    FFmpeg 会把大量进度与统计写到 stderr。若使用 stderr=PIPE 却不持续读取，
    管道缓冲区满后 FFmpeg 会阻塞，进而无法再读 stdin，表现为 release()/wait()
    永远挂起。Windows 管道缓冲通常更小，更容易触发；Linux 长视频同样可能踩中。
    因此必须在子进程运行期间排空 stderr（后台线程），而不是等到 wait() 后再读。
    """

    def __init__(self, proc: subprocess.Popen[bytes], output_path: str) -> None:
        self._proc = proc
        self._output_path = output_path
        self._stderr_tail = bytearray()
        self._stderr_thread: threading.Thread | None = None
        if proc.stderr is not None:
            err = proc.stderr

            def _drain_stderr() -> None:
                try:
                    while True:
                        chunk = err.read(65536)
                        if not chunk:
                            break
                        if len(self._stderr_tail) > 16384:
                            del self._stderr_tail[:-8192]
                        self._stderr_tail.extend(chunk)
                except Exception:
                    pass

            self._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            self._stderr_thread.start()

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
        _wait0 = time.perf_counter()
        self._proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=30.0)
        _wait_dt = time.perf_counter() - _wait0
        if _wait_dt > 0.5:
            print(f"[STEP] FFmpeg 子进程 wait 结束 | {_step_ts()} | 耗时={_wait_dt:.2f}s (编码收尾)")
        if self._proc.returncode != 0:
            stderr_tail = bytes(self._stderr_tail)[-800:].decode(errors="replace")
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
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-nostats",
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


def _aggregate_detections_face_only(
    frame_results: list[dict[str, Any]], fps: float
) -> list[dict[str, str]]:
    """按轨迹汇总人脸匹配结果（无 OCR / 马号）。"""
    track_votes: dict[int, list[tuple[str, float]]] = {}
    track_best_frame: dict[int, int] = {}
    track_best_conf: dict[int, float] = {}

    for fr in frame_results:
        fi = int(fr.get("frame_index", 0))
        for det in fr.get("detections", []):
            tid = det.get("track_id")
            if tid is None:
                continue
            tid = int(tid)
            rider = det.get("rider_identity", {})
            name = str(rider.get("name", "")).strip()
            matched = bool(rider.get("matched", False))
            score = float(rider.get("score", 0.0))
            hconf = float(det.get("conf", 0.0))
            if tid not in track_votes:
                track_votes[tid] = []
                track_best_conf[tid] = 0.0
                track_best_frame[tid] = fi
            if name and matched:
                track_votes[tid].append((name, score))
            if hconf > track_best_conf[tid]:
                track_best_conf[tid] = hconf
                track_best_frame[tid] = fi

    results: list[dict[str, str]] = []
    for tid in sorted(track_votes.keys()):
        votes = track_votes[tid]
        if len(votes) < MIN_FACE_FRAMES_FOR_RIDER:
            continue
        top_name, _cnt = Counter(n for n, _ in votes).most_common(1)[0]
        sub = [s for n, s in votes if n == top_name]
        avg_score = sum(sub) / len(sub) if sub else 0.0
        bf = track_best_frame[tid]
        results.append({
            "timestamp": _frame_to_timestamp(bf, fps),
            "horse_id": "",
            "person_name": top_name,
            "confidence": f"{track_best_conf[tid]:.2f}_{avg_score:.2f}",
        })
    return results


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
        mode: "full" 全流程；"simple" 仅 VLM 鞍垫号码；"face" 仅检测+追踪+人脸（无 OCR/VLM）。

    Returns a result dict matching the frontend API format.
    """
    is_simple = mode == "simple"
    is_face_only = mode == "face"
    _init_t0 = time.perf_counter()
    print(
        f"[STEP] 开始: 初始化管线(配置/检测器/OCR/人脸等) | {_step_ts()} | "
        f"task={task_id[:8]} mode={mode} file={os.path.basename(video_path)}"
    )
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

    if not is_simple and not is_face_only:
        roi_extractor = ROIExtractor(config.roi)
        enhancer = ROIEnhancer(config.enhance)
        ocr_engine = OCREngine(config.ocr, device="gpu:0")

    vlm_fallback: VLMFallback | None = None
    if not is_simple and not is_face_only:
        if config.vlm_fallback and config.vlm_fallback.enabled and config.vlm_fallback.api_key:
            vlm_cfg = config.vlm_fallback
            vlm_fallback = VLMFallback(vlm_cfg)
            print(f"[vlm_fallback] enabled  cooldown_frames={vlm_cfg.cooldown_frames}")
        else:
            print("[vlm_fallback] disabled")

    viz_mode = config.runtime.viz_mode if config.runtime.viz_mode else "display"
    print(f"[viz] mode={viz_mode}  pipeline_mode={mode}")
    track_fuser = OCRTrackFuser(config.fusion)
    hide_numbers = mode in ("full", "face")

    if not is_simple and not is_face_only:
        direction_filter = DirectionFilter(
            target_direction=config.runtime.target_direction,
            min_frames=config.runtime.direction_min_frames,
            min_displacement_px=config.runtime.direction_min_displacement,
        )
        if direction_filter.target_direction != "both":
            print(f"[direction] filter active: target={direction_filter.target_direction}")

    visualizer = ResultVisualizer(viz_mode=viz_mode, hide_numbers=hide_numbers)

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

    print(
        f"[STEP] 结束: 初始化管线 | {_step_ts()} | 耗时={time.perf_counter() - _init_t0:.2f}s | "
        f"resolution={resolution_str} total_frames≈{total_frames}"
    )

    frame_results: list[dict[str, Any]] = []
    frame_idx = 0
    start_ts = time.perf_counter()
    best_crops: dict[int, tuple["np.ndarray", float]] = {}


    _t_accum: dict[str, float] = {
        "read": 0.0, "yolo": 0.0, "face_det": 0.0, "enhance": 0.0,
        "ocr": 0.0, "vlm": 0.0, "rider_id": 0.0, "viz": 0.0, "write": 0.0,
    }

    # 记录各阶段运行设备
    _yolo_dev = str(getattr(config.detector, "device", "cpu")).lower()
    _yolo_device = "GPU" if any(k in _yolo_dev for k in ("cuda", "gpu", "0", "1", "2", "3")) else "CPU"
    _ocr_device = "GPU" if ocr_engine is not None else "CPU"  # OCREngine 用 gpu:0
    _face_dev = str(getattr(config.rider_identity_settings, "face_device", "cpu") if config.rider_identity_settings else "cpu").lower()
    _face_device = "GPU" if any(k in _face_dev for k in ("cuda", "gpu", "0", "1", "2", "3")) else "CPU"
    _stage_devices: dict[str, str] = {
        "read": "CPU", "yolo": _yolo_device, "face_det": _face_device,
        "enhance": "CPU", "ocr": _ocr_device, "vlm": "API",
        "rider_id": _face_device, "viz": "CPU", "write": "CPU",
    }

    tasks[task_id]["message"] = "正在初始化 AI 模型..."
    tasks[task_id]["progress"] = 1
    tasks[task_id]["stage_devices"] = _stage_devices

    _loop_t0 = time.perf_counter()
    print(
        f"[STEP] 开始: 逐帧处理主循环 | {_step_ts()} | task={task_id[:8]} "
        f"(全功能进度条 0–97% 对应该阶段)"
    )

    # --- Pipeline stage 1: prefetch read thread ---
    _READ_QUEUE_SIZE = 4
    _read_q: queue.Queue[tuple[bool, np.ndarray | None]] = queue.Queue(maxsize=_READ_QUEUE_SIZE)

    def _reader_thread() -> None:
        while True:
            ok, frame = cap.read()
            _read_q.put((ok, frame))
            if not ok:
                break

    _reader = threading.Thread(target=_reader_thread, daemon=True)
    _reader.start()

    # --- Pipeline stage 3: async viz + write thread ---
    _WRITE_QUEUE_SIZE = 4
    # Queue items: None (sentinel) or dict with viz params + frame
    _write_q: queue.Queue[dict | None] = queue.Queue(maxsize=_WRITE_QUEUE_SIZE)
    _write_t_accum = {"viz": 0.0, "write": 0.0}

    def _writer_thread() -> None:
        while True:
            item = _write_q.get()
            if item is None:
                break
            _tv0 = time.perf_counter()
            viz_func = item["viz_func"]
            vis_frame = viz_func()
            _write_t_accum["viz"] += time.perf_counter() - _tv0
            _tw0 = time.perf_counter()
            writer.write(vis_frame)
            _write_t_accum["write"] += time.perf_counter() - _tw0

    _writer = threading.Thread(target=_writer_thread, daemon=True)
    _writer.start()

    while True:
        _t0 = time.perf_counter()
        ok, frame = _read_q.get()
        if not ok:
            break
        _t_accum["read"] += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        if not is_simple and not is_face_only:
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
                item = {**asdict(det)}
                det_with_roi.append(item)
                if det.track_id is not None:
                    x1 = max(0, int(round(det.x - det.w / 2)))
                    y1 = max(0, int(round(det.y - det.h / 2)))
                    x2 = min(cur_w, int(round(det.x + det.w / 2)))
                    y2 = min(cur_h, int(round(det.y + det.h / 2)))
                    area = det.w * det.h
                    if det.track_id not in best_crops or area > best_crops[det.track_id][1]:
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size > 0:
                            best_crops[det.track_id] = (crop, area)
                continue

            if is_face_only:
                assert rider_identity is not None
                vis_detections.append(det)
                _tr0 = time.perf_counter()
                rider_payload = rider_identity.identify(det=det, ocr_fused_payload=None)
                _t_accum["rider_id"] += time.perf_counter() - _tr0

                horse_x1 = int(round(det.x - det.w / 2.0))
                horse_y1 = int(round(det.y - det.h / 2.0))
                horse_x2 = int(round(det.x + det.w / 2.0))
                horse_y2 = int(round(det.y + det.h / 2.0))

                ocr_infos.append({
                    "track_id": det.track_id,
                    "rider_name": rider_payload["name"],
                    "rider_score": rider_payload["score"],
                    "rider_matched": rider_payload["matched"],
                    "rider_source": rider_payload.get("source", "none"),
                    "face_bbox": rider_payload.get("face_bbox", []),
                    "face_score": rider_payload.get("face_score", 0.0),
                    "face_detected": rider_payload.get("face_detected", False),
                    "horse_bbox": [horse_x1, horse_y1, horse_x2, horse_y2],
                })
                item = {
                    **asdict(det),
                    "rider_identity": rider_payload,
                }
                det_with_roi.append(item)
                continue

            # --- full mode: direction filter → ROI → enhance → OCR → VLM fallback → rider ---
            assert direction_filter is not None
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

        if is_face_only and rider_identity is not None:
            for uface in rider_identity.unmatched_faces:
                ubbox = uface.get("bbox", [])
                if not isinstance(ubbox, list) or len(ubbox) < 4:
                    continue
                ocr_infos.append({
                    "track_id": None,
                    "rider_name": str(uface.get("name", "")),
                    "rider_score": float(uface.get("score", 0.0)),
                    "rider_matched": bool(uface.get("name") and float(uface.get("score", 0)) > 0),
                    "rider_source": "face",
                    "face_bbox": [int(v) for v in ubbox[:4]],
                    "face_score": float(uface.get("det_score", 0.0)),
                    "face_detected": True,
                    "horse_bbox": [],
                })

        # Capture viz params for async viz+write thread
        _viz_frame = frame
        _viz_detections = vis_detections
        _viz_rois = vis_rois
        _viz_ocr_infos = ocr_infos
        _viz_unmatched = (
            list(rider_identity.unmatched_faces)
            if (not is_simple and rider_identity is not None and rider_identity.enabled)
            else None
        )
        _viz_roi_raw = first_roi_raw
        _viz_roi_enh = first_roi_enh
        _viz_roi_bin = first_roi_bin
        _viz_qs = first_quality_score
        _viz_ot = first_ocr_text
        _viz_oc = first_ocr_conf
        _viz_ov = first_ocr_valid

        def _do_viz(
            _f=_viz_frame, _d=_viz_detections, _r=_viz_rois, _o=_viz_ocr_infos,
            _u=_viz_unmatched, _rr=_viz_roi_raw, _re=_viz_roi_enh, _rb=_viz_roi_bin,
            _qs=_viz_qs, _ot=_viz_ot, _oc=_viz_oc, _ov=_viz_ov,
        ):
            if is_simple:
                vf = visualizer.draw_frame_boxes_only(frame=_f, detections=_d)
            elif is_face_only:
                vf = visualizer.draw_frame_face_only(frame=_f, face_infos=_o)
            else:
                vf = visualizer.draw_frame(
                    frame=_f, detections=_d, rois=_r, ocr_infos=_o,
                    unmatched_faces=_u,
                )
            if viz_mode == "debug" and not is_simple and not is_face_only and not hide_numbers:
                vf = visualizer.draw_roi_comparison_panel(
                    frame=vf, roi_original_bgr=_rr, roi_enhanced_gray=_re,
                    roi_binary=_rb, quality_score=_qs, ocr_text=_ot,
                    ocr_conf=_oc, ocr_valid=_ov,
                )
            return vf

        _write_q.put({"viz_func": _do_viz})

        frame_results.append({"frame_index": frame_idx, "detections": det_with_roi})
        if not is_simple and not is_face_only:
            track_state.end_frame()
        frame_idx += 1

        # 简化模式 Phase1 只占 0–55%，避免结束后跳回 60% 造成进度倒退；全功能模式占 0–97%，为收尾与保存留出区间
        if total_frames > 0:
            if is_simple:
                progress = min(55, int(frame_idx / total_frames * 55))
            else:
                progress = min(97, int(frame_idx / total_frames * 97))
        else:
            progress = min(55 if is_simple else 97, frame_idx)
        tasks[task_id]["progress"] = progress
        tasks[task_id]["message"] = f"正在进行 AI 识别... {progress}%"

        # 每 10 帧更新一次 profiling 数据供前端实时展示
        if frame_idx % 10 == 0:
            _elapsed = time.perf_counter() - start_ts
            if _elapsed > 0:
                _prof = {}
                for _pk, _pv in _t_accum.items():
                    _pv_merged = _pv + _write_t_accum.get(_pk, 0.0)
                    _prof[_pk] = round(_pv_merged / _elapsed * 100, 1)
                _prof_sum = sum(_t_accum.values()) + sum(_write_t_accum.values())
                _prof["other"] = round(max(0, _elapsed - _prof_sum) / _elapsed * 100, 1)
                tasks[task_id]["profiling"] = _prof

    # Signal writer thread to finish and wait
    _write_q.put(None)
    _writer.join()
    _reader.join(timeout=5.0)
    _t_accum["write"] += _write_t_accum["write"]
    _t_accum["viz"] += _write_t_accum.get("viz", 0.0)

    # Phase: 帧循环结束，开始收尾
    print(
        f"[STEP] 结束: 逐帧处理主循环 | {_step_ts()} | "
        f"耗时={time.perf_counter() - _loop_t0:.2f}s | frames={frame_idx}"
    )

    _cap_t0 = time.perf_counter()
    print(f"[STEP] 开始: cap.release | {_step_ts()}")
    cap.release()
    print(f"[STEP] 结束: cap.release | {_step_ts()} | 耗时={time.perf_counter() - _cap_t0:.3f}s")

    _wr_t0 = time.perf_counter()
    print(f"[STEP] 开始: writer.release (含 FFmpeg 管道收尾，可能较慢) | {_step_ts()}")
    writer.release()
    print(f"[STEP] 结束: writer.release | {_step_ts()} | 耗时={time.perf_counter() - _wr_t0:.2f}s")

    if is_simple:
        tasks[task_id]["message"] = "正在筛选轨迹与准备 VLM..."
        tasks[task_id]["progress"] = 55
        print(f"[STEP] 开始: 简化模式 Phase1 后筛选轨迹 | {_step_ts()}")
    else:
        tasks[task_id]["message"] = "视频帧处理完成，正在统计与汇总..."
        tasks[task_id]["progress"] = 98
        print(f"[STEP] 阶段: profiling 与后续汇总/保存 | {_step_ts()} (全功能进度 98%→100%)")

    total_elapsed = max(1e-6, time.perf_counter() - start_ts)
    print(f"[processor] frames={frame_idx} elapsed={total_elapsed:.2f}s avg_fps={frame_idx / total_elapsed:.2f}")

    # 最终 profiling 写入 task
    _final_prof = {}
    for _pk, _pv in _t_accum.items():
        _final_prof[_pk] = round(_pv / total_elapsed * 100, 1)
    _prof_sum = sum(_t_accum.values())
    _final_prof["other"] = round(max(0, total_elapsed - _prof_sum) / total_elapsed * 100, 1)
    tasks[task_id]["profiling"] = _final_prof

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

    track_labels: dict[int, dict[str, str]] = {}
    if is_simple:
        _MIN_TRACK_FRAMES = 10
        track_frame_counts: dict[int, int] = {}
        for fr in frame_results:
            for det in fr.get("detections", []):
                tid = det.get("track_id")
                if tid is not None:
                    track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1
        before = len(best_crops)
        for tid in list(best_crops.keys()):
            cnt = track_frame_counts.get(tid, 0)
            if cnt < _MIN_TRACK_FRAMES:
                print(f"[processor] filtering track {tid}: only {cnt} frames (< {_MIN_TRACK_FRAMES})")
                del best_crops[tid]
        num_horses = len(best_crops)
        print(f"[processor] Phase 1 done: {before} tracks detected, {num_horses} kept after filter (>= {_MIN_TRACK_FRAMES} frames)")
        print(f"[STEP] 结束: 简化模式轨迹筛选 | {_step_ts()} | 保留马匹数={num_horses}")

        tasks[task_id]["message"] = "YOLO 检测完成，开始逐马 VLM 识别..."
        tasks[task_id]["progress"] = 60
        _vlm_t0 = time.perf_counter()
        print(f"[STEP] 开始: Phase2 VLM 逐马识别 | {_step_ts()} | crops={len(best_crops)}")
        vlm_video_cfg_data = config.vlm_video
        if vlm_video_cfg_data is None:
            vlm_video_cfg_data = VLMVideoConfig()
        if not vlm_video_cfg_data.api_key and config.vlm_fallback and config.vlm_fallback.api_key:
            vlm_video_cfg_data.api_key = config.vlm_fallback.api_key
        if vlm_video_cfg_data.api_key:
            vlm_video_identifier = VLMVideoIdentifier(vlm_video_cfg_data)
            track_labels = vlm_video_identifier.identify_crops(
                best_crops, tasks=tasks, task_id=task_id,
            )
            print(f"[processor] VLM per-track results: {track_labels}")
            if num_horses > 0 and len(track_labels) == 0:
                print(
                    "[processor] WARNING: VLM 未识别到任何鞍垫号码(画面将显示「?」)."
                    " 请检查: 1) api_key 与网络 2) vlm_video.model 是否为多模态(如 qwen-vl-max)"
                    " 3) 后端日志中 [vlm_video] track N API 调用失败 行"
                )
        else:
            print("[processor] WARNING: no api_key for VLM, skipping Phase 2")
            track_labels = {}
        print(
            f"[STEP] 结束: Phase2 VLM | {_step_ts()} | 耗时={time.perf_counter() - _vlm_t0:.2f}s | "
            f"labels={len(track_labels)}"
        )

        # Phase 3: re-render video with labels on bounding boxes
        tasks[task_id]["message"] = "正在重新渲染视频..."
        tasks[task_id]["progress"] = 80
        _rerender_t0 = time.perf_counter()
        print(f"[STEP] 开始: Phase3 重渲染输出视频 | {_step_ts()} | frames≈{len(frame_results)}")
        cap2 = cv2.VideoCapture(video_path)
        writer2 = _create_video_writer(output_video_path, fps, frame_w, frame_h)
        n_rerender = len(frame_results)
        for fi, fr_data in enumerate(frame_results):
            ok2, frame2 = cap2.read()
            if not ok2:
                break
            dets_raw = fr_data.get("detections", [])
            vis_frame2 = visualizer.draw_frame_boxes_with_labels(frame2, dets_raw, track_labels)
            writer2.write(vis_frame2)
            if n_rerender > 0:
                tasks[task_id]["progress"] = 80 + int(15 * (fi + 1) / n_rerender)
        cap2.release()
        writer2.release()
        print(f"[processor] Phase 3 done: re-rendered {len(frame_results)} frames")
        print(
            f"[STEP] 结束: Phase3 重渲染 | {_step_ts()} | 耗时={time.perf_counter() - _rerender_t0:.2f}s"
        )

        tasks[task_id]["message"] = "正在生成汇总数据..."
        tasks[task_id]["progress"] = 96

        summary_detections = []
        for tid, lbl in track_labels.items():
            entry: dict[str, str] = {"track_id": str(tid)}
            num = lbl.get("number", "")
            bg = lbl.get("bg_color", "")
            prefix = lbl.get("color_prefix", "")
            horse_id = (prefix + num) if prefix and num else num
            entry["horse_id"] = horse_id
            entry["number"] = num
            entry["bg_color"] = bg
            summary_detections.append(entry)
    elif is_face_only:
        tasks[task_id]["message"] = "正在汇总人脸识别结果..."
        tasks[task_id]["progress"] = 99
        _agg_start = time.perf_counter()
        print(
            f"[STEP] 开始: _aggregate_detections_face_only | {_step_ts()} | "
            f"task={task_id[:8]} 帧条目={len(frame_results)}"
        )
        summary_detections = _aggregate_detections_face_only(frame_results, fps)
        _agg_elapsed = time.perf_counter() - _agg_start
        print(
            f"[STEP] 结束: _aggregate_detections_face_only | {_step_ts()} | "
            f"耗时={_agg_elapsed:.2f}s | 汇总条数={len(summary_detections)}"
        )
    else:
        tasks[task_id]["message"] = "正在汇总轨迹与识别结果..."
        tasks[task_id]["progress"] = 99
        _agg_start = time.perf_counter()
        print(
            f"[STEP] 开始: _aggregate_detections | {_step_ts()} | "
            f"task={task_id[:8]} 帧条目={len(frame_results)}"
        )
        summary_detections = _aggregate_detections(frame_results, fps)
        _agg_elapsed = time.perf_counter() - _agg_start
        print(
            f"[STEP] 结束: _aggregate_detections | {_step_ts()} | 耗时={_agg_elapsed:.2f}s | "
            f"汇总条数={len(summary_detections)}"
        )

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
    tasks[task_id]["message"] = "正在保存结果文件（大文件可能需数十秒）..."
    tasks[task_id]["progress"] = 99
    _save_bundle_t0 = time.perf_counter()
    try:
        _fj_t0 = time.perf_counter()
        print(f"[STEP] 开始: 写入 frames_json (可能很大) | {_step_ts()} | task={task_id[:8]}")
        frames_json_path.write_text(
            json.dumps(frame_results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        _fj_kb = frames_json_path.stat().st_size / 1024
        print(
            f"[STEP] 结束: 写入 frames_json | {_step_ts()} | 耗时={time.perf_counter() - _fj_t0:.2f}s | {_fj_kb:.0f} KB"
        )

        _sj_t0 = time.perf_counter()
        print(f"[STEP] 开始: 写入 summary_json | {_step_ts()} | task={task_id[:8]}")
        summary_json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[STEP] 结束: 写入 summary_json | {_step_ts()} | 耗时={time.perf_counter() - _sj_t0:.3f}s | "
            f"path={summary_json_path.name}"
        )
        print(
            f"[STEP] JSON 落盘合计 | {_step_ts()} | "
            f"总耗时={time.perf_counter() - _save_bundle_t0:.2f}s"
        )
    except Exception as exc:
        print(
            f"[STEP] 写入 JSON 失败 | {_step_ts()} | 已耗时={time.perf_counter() - _save_bundle_t0:.2f}s | {exc}"
        )
        print(f"[processor] failed to save JSON: {exc}")

    _total_elapsed = time.perf_counter() - start_ts
    print(
        f"[STEP] 完成: process_video 全流程 | {_step_ts()} | "
        f"总耗时={_total_elapsed:.2f}s | 进度=100%"
    )
    tasks[task_id]["progress"] = 100
    tasks[task_id]["message"] = "处理完成"
    return result
