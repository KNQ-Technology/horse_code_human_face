from __future__ import annotations

from dataclasses import asdict
from typing import Any

from horse_id.config import ROIConfig
from horse_id.types import HorseDetection, ROIBox


class ROIExtractor:
    """Extract left shoulder ROI from horse detection box by ratio rules."""

    def __init__(self, config: ROIConfig) -> None:
        self.config = config

    def build_roi(self, det: HorseDetection, frame_w: int, frame_h: int) -> ROIBox:
        """Build a clipped ROI box for one horse detection."""
        horse_x1 = det.x - det.w / 2.0
        horse_y1 = det.y - det.h / 2.0

        x1 = horse_x1 + self.config.x_min_ratio * det.w
        x2 = horse_x1 + self.config.x_max_ratio * det.w
        y1 = horse_y1 + self.config.y_min_ratio * det.h
        y2 = horse_y1 + self.config.y_max_ratio * det.h

        # Keep ROI valid inside frame bounds.
        ix1 = max(0, min(frame_w - 1, int(round(x1))))
        iy1 = max(0, min(frame_h - 1, int(round(y1))))
        ix2 = max(ix1 + 1, min(frame_w, int(round(x2))))
        iy2 = max(iy1 + 1, min(frame_h, int(round(y2))))

        return ROIBox(x1=ix1, y1=iy1, x2=ix2, y2=iy2)

    @staticmethod
    def serialize(roi: ROIBox) -> dict[str, Any]:
        """Serialize ROI dataclass for JSON output."""
        return asdict(roi)

