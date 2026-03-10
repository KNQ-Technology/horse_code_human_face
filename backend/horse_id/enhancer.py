from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from horse_id.config import EnhanceConfig


@dataclass
class ROIQuality:
    """Quality metrics for enhanced ROI."""

    score: float
    sharpness: float
    brightness: float
    contrast: float


class ROIEnhancer:
    """ROI enhancement pipeline with quality scoring."""

    def __init__(self, config: EnhanceConfig) -> None:
        self.config = config
        self._gamma_lut = self._build_gamma_lut(config.gamma)
        self._clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
        )

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        safe_gamma = max(gamma, 0.01)
        lut = np.array([((i / 255.0) ** (1.0 / safe_gamma)) * 255.0 for i in range(256)]).astype(
            "uint8"
        )
        return lut

    def enhance(self, roi_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return enhanced grayscale and adaptive-binary branches."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        clahe_img = self._clahe.apply(gray)
        gamma_img = cv2.LUT(clahe_img, self._gamma_lut)
        denoise_img = cv2.bilateralFilter(
            gamma_img,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space,
        )

        blur = cv2.GaussianBlur(denoise_img, (0, 0), self.config.unsharp_sigma)
        sharpened = cv2.addWeighted(
            denoise_img,
            1.0 + self.config.unsharp_amount,
            blur,
            -self.config.unsharp_amount,
            0,
        )

        block_size = self.config.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)

        binary = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.config.adaptive_c,
        )
        return sharpened, binary

    def quality(self, enhanced_gray: np.ndarray) -> ROIQuality:
        """Calculate a [0,1] ROI quality score."""
        sharpness = float(cv2.Laplacian(enhanced_gray, cv2.CV_64F).var())
        brightness = float(np.mean(enhanced_gray) / 255.0)
        contrast = float(np.std(enhanced_gray) / 128.0)

        sharpness_norm = min(sharpness / 400.0, 1.0)
        brightness_score = max(0.0, 1.0 - abs(brightness - 0.5) / 0.5)
        contrast_norm = min(contrast, 1.0)

        score = 0.6 * sharpness_norm + 0.2 * brightness_score + 0.2 * contrast_norm
        return ROIQuality(
            score=float(score),
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
        )

    @staticmethod
    def serialize_quality(quality: ROIQuality) -> dict[str, Any]:
        """Serialize quality dataclass for JSON output."""
        return asdict(quality)

