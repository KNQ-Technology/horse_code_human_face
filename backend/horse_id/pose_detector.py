from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class PoseResult:
    """
    单个人体的姿态结果。

    - person_id: 此人的 ID（优先使用跟踪 ID，否则为索引）
    - keypoints: 关键点坐标，shape = (K, 3)，每个关键点为 (x, y, conf)
    - bbox: 人体框 (x1, y1, x2, y2)，像素坐标
    - score: 人体框置信度
    """

    person_id: int
    keypoints: np.ndarray  # (K, 3): x, y, conf
    bbox: Optional[Tuple[float, float, float, float]]
    score: float


class PoseDetector:
    """
    使用 YOLO pose 模型的人体姿态骨骼点检测器。

    支持自定义置信度阈值：仅当人体框置信度 score >= conf_threshold 时才返回该结果。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        if model_path is None:
            model_path = Path(__file__).with_suffix("").parent / "models" / "yolo11n-pose.pt"
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.conf_threshold = float(conf_threshold)
        self.device = device if device else "cpu"

    def detect(self, image: np.ndarray) -> List[PoseResult]:
        """
        对输入图像进行人体姿态检测，返回每个人的关键点和框信息。

        :param image: 输入图像，BGR 格式的 numpy 数组 (H, W, 3)
        :return: 每个人对应一个 PoseResult
        """
        # Ultralytics YOLO 接口：单张图像预测时返回一个长度为 1 的列表
        results = self.model.predict(source=image, verbose=False, device=self.device)
        if not results:
            return []

        result = results[0]

        # keypoints: [num_person, num_kpts, 2]，像素坐标
        if result.keypoints is None or result.keypoints.xy is None:
            return []

        kpts_xy = result.keypoints.xy.cpu().numpy()  # (N, K, 2)

        # 如果模型有关键点置信度（部分版本为 .xyc 或 .data），尽量取出来；否则设为 1.0
        # 兼容处理：优先尝试 .data，失败则退化为全 1.0
        try:
            raw_kpts = result.keypoints.data.cpu().numpy()  # (N, K, 3) or (N, K, 2)
            if raw_kpts.shape[-1] == 3:
                kpts_conf = raw_kpts[..., 2]
            else:
                kpts_conf = np.ones(raw_kpts.shape[:2], dtype=np.float32)
        except Exception:
            kpts_conf = np.ones(kpts_xy.shape[:2], dtype=np.float32)

        boxes = result.boxes
        poses: List[PoseResult] = []

        if boxes is None or boxes.xyxy is None or boxes.conf is None:
            # 无框时 score=0，低于阈值，不返回
            return []

        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        scores = boxes.conf.cpu().numpy()  # (N,)
        # 如果模型/跟踪器提供了 id，则优先作为 person_id，否则使用索引
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        else:
            track_ids = list(range(len(xyxy)))

        num_person = kpts_xy.shape[0]
        for i in range(num_person):
            xy = kpts_xy[i]  # (K, 2)
            conf = kpts_conf[i][..., None]  # (K, 1)
            keypoints = np.concatenate([xy, conf], axis=-1)  # (K, 3)

            if i < xyxy.shape[0] and i < scores.shape[0]:
                x1, y1, x2, y2 = xyxy[i].tolist()
                score = float(scores[i])
                bbox: Optional[Tuple[float, float, float, float]] = (x1, y1, x2, y2)
            else:
                bbox = None
                score = 0.0

            if score < self.conf_threshold:
                continue
            person_id = track_ids[i] if i < len(track_ids) else i
            poses.append(PoseResult(person_id=person_id, keypoints=keypoints, bbox=bbox, score=score))

        return poses

