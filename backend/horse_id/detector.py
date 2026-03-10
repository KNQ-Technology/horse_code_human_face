from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from ultralytics import YOLO

from horse_id.config import DetectorConfig
from horse_id.types import HorseDetection


class HorseDetector:
    """YOLO horse detector wrapper for frame-level inference."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.model = YOLO(config.model_path)

    def detect(self, frame: np.ndarray) -> list[HorseDetection]:
        """Run detector and return filtered horse detections."""
        if self.config.track_enabled:
            results = self.model.track(
                source=frame,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                tracker=self.config.tracker_name,
                persist=self.config.track_persist,
                verbose=False,
            )
        else:
            results = self.model.predict(
                source=frame,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )
        if not results:
            return []

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None or boxes.xywh is None:
            return []

        detections: list[HorseDetection] = []
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        if boxes.id is not None:
            track_ids: list[int | None] = [int(v) for v in boxes.id.cpu().numpy().tolist()]
        else:
            track_ids = [None] * len(xywh)

        for (x, y, w, h), conf, cls_id, track_id in zip(xywh, confs, clss, track_ids):
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            if cls_name != "horse":
                continue
            detections.append(
                HorseDetection(
                    x=float(x),
                    y=float(y),
                    w=float(w),
                    h=float(h),
                    conf=float(conf),
                    cls_name=cls_name,
                    track_id=track_id,
                )
            )
        return detections

    @staticmethod
    def serialize(detections: list[HorseDetection]) -> list[dict[str, Any]]:
        """Serialize dataclass detections for JSON output."""
        return [asdict(item) for item in detections]

