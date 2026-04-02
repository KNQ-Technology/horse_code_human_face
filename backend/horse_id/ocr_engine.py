from __future__ import annotations

import inspect
import logging
import os
import re
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR
import paddle

from horse_id.config import OCRConfig
from horse_id.types import OCRResult

logging.getLogger("ppocr").setLevel(logging.WARNING)


class OCREngine:
    """OCR wrapper for ROI-level number recognition."""

    def __init__(self, config: OCRConfig, device: str = "gpu:0") -> None:
        self.config = config
        resolved_device = self._resolve_device(device)
        self.ocr, self._uses_predict_api = self._build_ocr(resolved_device)
        self._regex = re.compile(config.regex_pattern)
        self._whitelist = set(config.whitelist)

    @staticmethod
    def _gpu_available() -> bool:
        try:
            if not paddle.is_compiled_with_cuda():
                return False
        except Exception:
            return False

        try:
            return paddle.device.cuda.device_count() > 0
        except Exception:
            return True

    @classmethod
    def _resolve_device(cls, device: str) -> str:
        requested = (device or "").strip().lower()
        if requested in {"", "auto"}:
            return "gpu:0" if cls._gpu_available() else "cpu"
        if requested.startswith(("gpu", "cuda")) and not cls._gpu_available():
            return "cpu"
        return device

    @staticmethod
    def _build_candidates(device: str) -> list[dict[str, Any]]:
        signature = inspect.signature(PaddleOCR)
        params = signature.parameters
        candidates: list[dict[str, Any]] = []

        modern_kwargs: dict[str, Any] = {"lang": "en"}
        if "lang" in params:
            modern_kwargs["lang"] = None
        if "ocr_version" in params:
            modern_kwargs["ocr_version"] = "PP-OCRv4"
        if "use_textline_orientation" in params:
            modern_kwargs["use_textline_orientation"] = False
        if "use_angle_cls" in params:
            modern_kwargs["use_angle_cls"] = False
        if "device" in params:
            modern_kwargs["device"] = device
        elif "use_gpu" in params:
            modern_kwargs["use_gpu"] = device.startswith(("gpu", "cuda"))
        candidates.append(modern_kwargs)

        if "ocr_version" in modern_kwargs:
            without_version = dict(modern_kwargs)
            without_version.pop("ocr_version", None)
            candidates.append(without_version)

        if device.startswith(("gpu", "cuda")):
            cpu_fallback = dict(candidates[-1])
            if "device" in cpu_fallback:
                cpu_fallback["device"] = "cpu"
            if "use_gpu" in cpu_fallback:
                cpu_fallback["use_gpu"] = False
            candidates.append(cpu_fallback)

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for item in candidates:
            key = tuple(sorted((k, str(v)) for k, v in item.items()))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _build_ocr(self, device: str) -> tuple[PaddleOCR, bool]:
        last_error: Exception | None = None
        for candidate in self._build_candidates(device):
            kwargs = dict(candidate)
            kwargs["lang"] = self.config.lang
            try:
                ocr = PaddleOCR(**kwargs)
                return ocr, hasattr(ocr, "predict")
            except Exception as exc:
                last_error = exc

        if last_error is None:
            raise RuntimeError("Failed to initialize PaddleOCR without an explicit error.")
        raise last_error

    def _normalize_text(self, text: str) -> str:
        upper = text.upper()
        return "".join(ch for ch in upper if ch in self._whitelist)

    def _is_valid(self, text: str) -> bool:
        if not (self.config.expected_min_len <= len(text) <= self.config.expected_max_len):
            return False
        return bool(self._regex.match(text))

    def infer(self, roi_gray: np.ndarray) -> OCRResult:
        """Run OCR on enhanced ROI and return normalized result."""
        if roi_gray.size == 0:
            return OCRResult(text="", conf=0.0, valid=False, raw_text="")

        roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        if self._uses_predict_api:
            results = self.ocr.predict(roi_bgr)
            best_text, best_conf = self._extract_from_predict_results(results)
        else:
            results = self.ocr.ocr(roi_bgr, cls=False)
            best_text, best_conf = self._extract_from_legacy_results(results)

        normalized = self._normalize_text(best_text)
        valid = self._is_valid(normalized) and best_conf >= self.config.conf_threshold
        return OCRResult(
            text=normalized if valid else "",
            conf=float(best_conf),
            valid=bool(valid),
            raw_text=best_text,
        )

    @staticmethod
    def _extract_from_predict_results(results: Any) -> tuple[str, float]:
        best_text = ""
        best_conf = 0.0
        if not results:
            return best_text, best_conf

        res = results[0]
        rec_texts = res.get("rec_texts", []) if isinstance(res, dict) else getattr(res, "rec_texts", [])
        rec_scores = res.get("rec_scores", []) if isinstance(res, dict) else getattr(res, "rec_scores", [])
        for text, conf in zip(rec_texts, rec_scores):
            conf = float(conf)
            if conf > best_conf:
                best_text = str(text)
                best_conf = conf
        return best_text, best_conf

    @staticmethod
    def _extract_from_legacy_results(results: Any) -> tuple[str, float]:
        best_text = ""
        best_conf = 0.0
        if not isinstance(results, list) or not results:
            return best_text, best_conf

        first_batch = results[0]
        if not isinstance(first_batch, list):
            return best_text, best_conf

        for item in first_batch:
            if not isinstance(item, list) or len(item) < 2:
                continue
            rec = item[1]
            if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                continue
            text, conf = rec[0], rec[1]
            conf = float(conf)
            if conf > best_conf:
                best_text = str(text)
                best_conf = conf
        return best_text, best_conf

    @staticmethod
    def serialize(result: OCRResult) -> dict[str, Any]:
        """Serialize OCR dataclass for JSON output."""
        return asdict(result)

