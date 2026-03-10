from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HorseDetection:
    """Horse detection box in xywh format with confidence."""

    x: float
    y: float
    w: float
    h: float
    conf: float
    cls_name: str = "horse"
    track_id: int | None = None


@dataclass
class ROIBox:
    """ROI box in absolute pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class OCRResult:
    """Frame-level OCR result for a single ROI."""

    text: str
    conf: float
    valid: bool
    raw_text: str

