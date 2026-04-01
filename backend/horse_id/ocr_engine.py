from __future__ import annotations

import logging
import re
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np
from paddleocr import PaddleOCR

from horse_id.config import OCRConfig
from horse_id.types import OCRResult

logging.getLogger("ppocr").setLevel(logging.WARNING)


class OCREngine:
    """OCR wrapper for ROI-level number recognition."""

    def __init__(self, config: OCRConfig, device: str = "gpu:0") -> None:
        self.config = config
        self.ocr = PaddleOCR(
            ocr_version="PP-OCRv4",
            use_textline_orientation=False,
            lang=config.lang,
            device=device,
        )
        self._regex = re.compile(config.regex_pattern)
        self._whitelist = set(config.whitelist)

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
        results = self.ocr.predict(roi_bgr)

        best_text = ""
        best_conf = 0.0

        if results:
            res = results[0]
            rec_texts = res.get("rec_texts", []) if isinstance(res, dict) else getattr(res, "rec_texts", [])
            rec_scores = res.get("rec_scores", []) if isinstance(res, dict) else getattr(res, "rec_scores", [])
            for text, conf in zip(rec_texts, rec_scores):
                conf = float(conf)
                if conf > best_conf:
                    best_text = str(text)
                    best_conf = conf

        normalized = self._normalize_text(best_text)
        valid = self._is_valid(normalized) and best_conf >= self.config.conf_threshold
        return OCRResult(
            text=normalized if valid else "",
            conf=float(best_conf),
            valid=bool(valid),
            raw_text=best_text,
        )

    @staticmethod
    def serialize(result: OCRResult) -> dict[str, Any]:
        """Serialize OCR dataclass for JSON output."""
        return asdict(result)

