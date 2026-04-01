from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "请仔细观察这段赛马视频，识别所有马匹身上鞍垫（saddle cloth）上的数字号码。\n\n"
    "本视频中大约有 {num_horses} 匹马。\n\n"
    "请列出你能识别到的每匹马的鞍垫号码（纯数字）。\n"
    "注意：号码通常是 1~3 位数字，请务必识别完整，不要遗漏任何一位数字。\n"
    "请严格以JSON数组格式回答，不要包含其他任何内容：\n"
    '[{{"number": "数字"}}, ...]'
)

_CROP_PROMPT = (
    "这是一匹赛马身上鞍垫（saddle cloth）的近景图片。\n"
    "请识别：\n"
    "1. 鞍垫上的数字号码（1~3位纯数字，务必识别完整）\n"
    "2. 鞍垫底色（只能从以下选项中选一个：橙、白、黑、黄、绿、蓝、棕）\n\n"
    "重要：如果图片模糊、数字太小无法确认，请将 number 设为空字符串。\n"
    "宁可不识别，也不要猜测不确定的数字。\n\n"
    "请严格以JSON格式回答，不要包含其他任何内容：\n"
    '{"number": "数字或空字符串", "bg_color": "底色"}'
)

_COLOR_MAP: list[dict[str, str]] = [
    {"bg": "橙", "prefix": "L"},
    {"bg": "白", "prefix": "K"},
    {"bg": "黑", "prefix": "J"},
    {"bg": "黄", "prefix": "H"},
    {"bg": "绿", "prefix": "G"},
    {"bg": "蓝", "prefix": "D"},
    {"bg": "棕", "prefix": "C"},
]

_NUMBER_RE = re.compile(r"\d+")
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _map_bg_to_prefix(bg_color: str) -> str:
    for entry in _COLOR_MAP:
        if entry["bg"] in bg_color:
            return entry["prefix"]
    return ""


def _encode_image_base64(img: "np.ndarray") -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class VLMVideoConfig:
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.5-plus"
    timeout: float = 300.0
    max_retries: int = 1
    segment_seconds: int = 15
    overlap_seconds: int = 3
    compress_crf: int = 28
    compress_scale: str = "672:380"
    fps: float = 2.0


def _normalize_number(raw: str) -> str:
    m = _NUMBER_RE.search(raw)
    if not m:
        return ""
    digits = m.group().lstrip("0") or "0"
    return digits.zfill(3)[-3:]


def _get_video_duration(path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def segment_video(
    video_path: str,
    segment_sec: int = 15,
    overlap_sec: int = 3,
    crf: int = 28,
    scale: str = "672:380",
) -> list[str]:
    """Split video into compressed segments using FFmpeg."""
    duration = _get_video_duration(video_path)
    if duration <= 0:
        logger.warning("Cannot determine video duration: %s", video_path)
        return []

    tmp_dir = tempfile.mkdtemp(prefix="vlm_seg_")
    segments: list[str] = []

    if duration <= segment_sec:
        seg_path = os.path.join(tmp_dir, "seg_000.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264", "-crf", str(crf),
            "-vf", f"scale={scale}",
            "-an", "-pix_fmt", "yuv420p",
            seg_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                segments.append(seg_path)
        except Exception as exc:
            logger.warning("ffmpeg compress failed: %s", exc)
        print(f"[vlm_video] short video ({duration:.1f}s), no split. 1 segment, tmp_dir={tmp_dir}")
        return segments

    step = max(1, segment_sec - overlap_sec)
    start = 0.0

    while start < duration:
        seg_duration = min(segment_sec, duration - start)
        if seg_duration < 2.0 and segments:
            break

        seg_path = os.path.join(tmp_dir, f"seg_{len(segments):03d}.mp4")
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(seg_duration),
            "-i", video_path,
            "-c:v", "libx264", "-crf", str(crf),
            "-vf", f"scale={scale}",
            "-an", "-pix_fmt", "yuv420p",
            seg_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                segments.append(seg_path)
                logger.info("segment %d: start=%.1fs dur=%.1fs size=%.1fKB",
                            len(segments) - 1, start, seg_duration,
                            os.path.getsize(seg_path) / 1024)
        except Exception as exc:
            logger.warning("ffmpeg segment failed at %.1fs: %s", start, exc)

        start += step

    print(f"[vlm_video] split into {len(segments)} segment(s), tmp_dir={tmp_dir}")
    return segments


def _encode_video_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


class VLMVideoIdentifier:
    """Identify horse saddle-pad numbers by sending video segments to Qwen VL."""

    def __init__(self, config: VLMVideoConfig) -> None:
        self.config = config
        from openai import OpenAI
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def identify_segment(self, segment_path: str, num_horses: int = 0) -> list[str]:
        """Return list of number strings detected in this segment."""
        b64 = _encode_video_base64(segment_path)
        size_mb = len(b64) / (1024 * 1024)
        if size_mb > 10.0:
            logger.warning("segment base64 too large: %.1fMB > 10MB, skipping", size_mb)
            return []

        prompt_text = _PROMPT_TEMPLATE.format(num_horses=num_horses if num_horses > 0 else "若干")

        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                                "fps": self.config.fps,
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                extra_body={"enable_thinking": False},
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception:
            logger.warning("VLM video API call failed for %s", segment_path, exc_info=True)
            return []

        print(f"[vlm_video] segment={os.path.basename(segment_path)} raw={raw_text[:300]}")
        return self._parse_response(raw_text)

    def identify_video(
        self,
        video_path: str,
        tasks: dict | None = None,
        task_id: str = "",
        num_horses: int = 0,
    ) -> list[dict[str, str]]:
        """Full pipeline: segment -> VLM calls -> merge."""
        cfg = self.config
        segments = segment_video(
            video_path,
            segment_sec=cfg.segment_seconds,
            overlap_sec=cfg.overlap_seconds,
            crf=cfg.compress_crf,
            scale=cfg.compress_scale,
        )
        if not segments:
            logger.warning("No segments produced for %s", video_path)
            return []

        all_numbers: list[str] = []
        for i, seg_path in enumerate(segments):
            if tasks and task_id:
                tasks[task_id]["message"] = f"VLM 识别中... ({i + 1}/{len(segments)})"
            numbers = self.identify_segment(seg_path, num_horses=num_horses)
            all_numbers.extend(numbers)
            print(f"[vlm_video] segment {i}: {len(numbers)} number(s) detected")

        tmp_dir = os.path.dirname(segments[0]) if segments else ""
        if tmp_dir and tmp_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return self._merge_results(all_numbers)

    @staticmethod
    def _parse_response(text: str) -> list[str]:
        m = _JSON_ARRAY_RE.search(text)
        if not m:
            return []
        try:
            arr = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(arr, list):
            return []

        results: list[str] = []
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            number = _normalize_number(str(obj.get("number", "")))
            if number:
                results.append(number)
        return results

    @staticmethod
    def _merge_results(numbers: list[str]) -> list[dict[str, str]]:
        counts: dict[str, int] = {}
        for n in numbers:
            counts[n] = counts.get(n, 0) + 1

        return [
            {"horse_id": num, "confidence": str(cnt)}
            for num, cnt in counts.items()
        ]

    _MIN_CROP_PX = 50

    def identify_crop(self, crop: "np.ndarray", track_id: int) -> dict[str, str]:
        """Identify number and color from a single cropped horse image."""
        h, w = crop.shape[:2]
        if h < self._MIN_CROP_PX or w < self._MIN_CROP_PX:
            print(f"[vlm_video] track {track_id}: crop too small ({w}x{h}), skipping")
            return {}

        b64 = _encode_image_base64(crop)
        if not b64:
            return {}

        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": _CROP_PROMPT},
                        ],
                    }
                ],
                extra_body={"enable_thinking": False},
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception:
            logger.warning("VLM crop API call failed for track %d", track_id, exc_info=True)
            return {}

        print(f"[vlm_video] crop track={track_id} raw={raw_text[:200]}")
        return self._parse_crop_response(raw_text)

    @staticmethod
    def _parse_crop_response(text: str) -> dict[str, str]:
        m = _JSON_OBJ_RE.search(text)
        if not m:
            return {}
        try:
            obj = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(obj, dict):
            return {}

        number = _normalize_number(str(obj.get("number", "")))
        bg_color = str(obj.get("bg_color", "")).strip()
        if not number:
            return {}
        prefix = _map_bg_to_prefix(bg_color) if bg_color else ""
        return {"number": number, "bg_color": bg_color, "color_prefix": prefix}

    def identify_crops(
        self,
        best_crops: dict[int, tuple[Any, float]],
        tasks: dict | None = None,
        task_id: str = "",
    ) -> dict[int, dict[str, str]]:
        """Identify each tracked horse from its best crop image (parallel)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(best_crops)
        if tasks and task_id:
            tasks[task_id]["message"] = f"VLM 并行识别 {total} 匹马..."

        track_labels: dict[int, dict[str, str]] = {}
        futures: dict[Any, int] = {}

        with ThreadPoolExecutor(max_workers=min(total, 5)) as pool:
            for tid, (crop, _area) in best_crops.items():
                fut = pool.submit(self.identify_crop, crop, tid)
                futures[fut] = tid

            done_count = 0
            for fut in as_completed(futures):
                tid = futures[fut]
                done_count += 1
                if tasks and task_id:
                    tasks[task_id]["message"] = f"VLM 识别中... ({done_count}/{total})"
                try:
                    result = fut.result()
                except Exception:
                    logger.warning("VLM crop failed for track %d", tid, exc_info=True)
                    result = {}
                if result:
                    track_labels[tid] = result
                print(f"[vlm_video] track {tid}: {result}")

        ordered: dict[int, dict[str, str]] = {}
        for tid in best_crops:
            if tid in track_labels:
                ordered[tid] = track_labels[tid]
        return ordered
