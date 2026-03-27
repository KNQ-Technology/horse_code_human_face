from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

import cv2
import numpy as np

from horse_id.config import VLMFallbackConfig
from horse_id.types import HorseDetection, OCRResult

logger = logging.getLogger(__name__)

_COLOR_MAP: list[dict[str, str]] = [
    {"bg": "橙", "font": "黑", "prefix": "L"},
    {"bg": "白", "font": "深蓝", "prefix": "K"},
    {"bg": "黑", "font": "粉红", "prefix": "J"},
    {"bg": "黄", "font": "黑", "prefix": "H"},
    {"bg": "绿", "font": "白", "prefix": "G"},
    {"bg": "蓝", "font": "白", "prefix": "D"},
    {"bg": "棕", "font": "黄", "prefix": "C"},
]

_PROMPT = (
    "请观察这匹赛马身上的鞍垫（saddle cloth），告诉我：\n"
    "1. 鞍垫的底色（只能从以下选项中选一个：橙、白、黑、黄、绿、蓝、棕）\n"
    "2. 鞍垫上数字的字体颜色（只能从以下选项中选一个：黑、深蓝、粉红、白、黄）\n"
    "3. 鞍垫上的数字号码（纯数字，如 001、23、12）\n\n"
    '请严格以JSON格式回答，不要包含其他任何内容：\n'
    '{"bg_color": "底色", "font_color": "字色", "number": "数字"}'
)

_EMPTY = OCRResult(text="", conf=0.0, valid=False, raw_text="")

_NUMBER_RE = re.compile(r"\d+")
_JSON_RE = re.compile(r"\{[^}]+\}")


class VLMFallback:
    """Fallback horse number recognition via Qwen VL multimodal API."""

    def __init__(self, config: VLMFallbackConfig) -> None:
        self.config = config
        self._last_call_frame: dict[int, int] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._last_details: dict[int | None, dict[str, str]] = {}
        self._last_valid_result: dict[int, OCRResult] = {}
        self._pending_tracks: set[int] = set()
        self._lock = __import__("threading").Lock()

        from openai import OpenAI
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def get_detail(self, track_id: int | None) -> dict[str, str]:
        return self._last_details.get(track_id, {})

    @staticmethod
    def _encode_image(crop_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _parse_response(text: str) -> tuple[str, str, str]:
        """Return (bg_color, font_color, number) or empty strings on failure."""
        m = _JSON_RE.search(text)
        if not m:
            return "", "", ""
        try:
            obj: dict[str, Any] = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return "", "", ""
        bg = str(obj.get("bg_color", "")).strip()
        font = str(obj.get("font_color", "")).strip()
        number = str(obj.get("number", "")).strip()
        nm = _NUMBER_RE.search(number)
        if nm:
            digits = nm.group().lstrip("0") or "0"
            number = digits.zfill(3)[-3:]
        else:
            number = ""
        return bg, font, number

    @staticmethod
    def _map_color_to_prefix(bg_color: str, font_color: str) -> str:
        for entry in _COLOR_MAP:
            if entry["bg"] in bg_color and entry["font"] in font_color:
                return entry["prefix"]
        for entry in _COLOR_MAP:
            if entry["bg"] in bg_color:
                return entry["prefix"]
        return ""

    def _crop_horse(self, frame: np.ndarray, det: HorseDetection) -> np.ndarray:
        fh, fw = frame.shape[:2]
        x1 = max(0, int(det.x - det.w / 2))
        y1 = max(0, int(det.y - det.h / 2))
        x2 = min(fw, int(det.x + det.w / 2))
        y2 = min(fh, int(det.y + det.h / 2))
        return frame[y1:y2, x1:x2]

    def _call_api_sync(
        self,
        track_id: int,
        crop_bgr: np.ndarray,
        frame_idx: int,
    ) -> None:
        """Run VLM API call in background thread. Writes result to caches."""
        b64 = self._encode_image(crop_bgr)
        if not b64:
            self._pending_tracks.discard(track_id)
            return
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
                            {"type": "text", "text": _PROMPT},
                        ],
                    }
                ],
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception:
            logger.warning("VLM API call failed for track %s frame %d", track_id, frame_idx, exc_info=True)
            self._pending_tracks.discard(track_id)
            return

        with self._lock:
            self._last_call_frame[track_id] = frame_idx

        bg, font, number = self._parse_response(raw_text)
        prefix = self._map_color_to_prefix(bg, font) if bg else ""
        full_id = (prefix + number) if prefix and number else ""

        with self._lock:
            self._last_details[track_id] = {
                "bg_color": bg,
                "font_color": font,
                "number": number,
                "prefix": prefix,
                "full_id": full_id,
                "raw": raw_text,
            }

        if number and prefix:
            result = OCRResult(text=full_id, conf=0.85, valid=True, raw_text=raw_text)
            with self._lock:
                self._last_valid_result[track_id] = result
            logger.info("VLM async done: track=%s => %s", track_id, full_id)

        self._pending_tracks.discard(track_id)

    def _reset_track(self, track_id: int) -> None:
        """Clear all cached data for a track (called when track_id is reused)."""
        self._last_call_frame.pop(track_id, None)
        self._last_seen_frame.pop(track_id, None)
        self._last_details.pop(track_id, None)
        self._last_valid_result.pop(track_id, None)
        self._pending_tracks.discard(track_id)

    def infer(
        self,
        track_id: int | None,
        frame: np.ndarray,
        det: HorseDetection,
        frame_idx: int,
    ) -> OCRResult:
        if track_id is None:
            return _EMPTY

        with self._lock:
            last_seen = self._last_seen_frame.get(track_id, -9999)
            gap = frame_idx - last_seen

            # If track was absent for longer than cooldown, the ID was likely
            # reused for a new horse — purge stale cache to avoid misidentification.
            if gap > self.config.cooldown_frames and last_seen != -9999:
                logger.debug(
                    "VLM: track %d reuse detected (gap=%d frames), clearing stale cache",
                    track_id, gap,
                )
                self._reset_track(track_id)

            self._last_seen_frame[track_id] = frame_idx

            last = self._last_call_frame.get(track_id, -9999)
            if frame_idx - last < self.config.cooldown_frames:
                cached = self._last_valid_result.get(track_id)
                if cached is not None:
                    return cached
                return _EMPTY

        if track_id in self._pending_tracks:
            cached = self._last_valid_result.get(track_id)
            return cached if cached is not None else _EMPTY

        crop = self._crop_horse(frame, det)
        if crop.size == 0:
            return _EMPTY

        crop_copy = crop.copy()
        self._pending_tracks.add(track_id)
        with self._lock:
            self._last_call_frame[track_id] = frame_idx

        import threading
        t = threading.Thread(
            target=self._call_api_sync,
            args=(track_id, crop_copy, frame_idx),
            daemon=True,
        )
        t.start()

        cached = self._last_valid_result.get(track_id)
        return cached if cached is not None else _EMPTY
