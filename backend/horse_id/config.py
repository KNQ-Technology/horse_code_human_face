from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    model_path: str
    conf_threshold: float
    iou_threshold: float
    track_enabled: bool = True
    tracker_name: str = "bytetrack.yaml"
    track_persist: bool = True
    device: str = "0"


@dataclass
class ROIConfig:
    x_min_ratio: float
    x_max_ratio: float
    y_min_ratio: float
    y_max_ratio: float
    orientation_mode: str


@dataclass
class OCRConfig:
    lang: str
    conf_threshold: float
    whitelist: str
    expected_min_len: int
    expected_max_len: int
    regex_pattern: str
    interval_frames: int = 1


@dataclass
class VLMFallbackConfig:
    enabled: bool = True
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-vl-max"
    timeout: float = 15.0
    max_retries: int = 1
    cooldown_frames: int = 10


@dataclass
class EnhanceConfig:
    clahe_clip_limit: float
    clahe_tile_grid_size: int
    gamma: float
    bilateral_d: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    unsharp_sigma: float
    unsharp_amount: float
    adaptive_block_size: int
    adaptive_c: int


@dataclass
class FusionConfig:
    window_size: int
    vote_threshold: float
    hold_seconds: float
    lost_frame_limit: int
    bad_frame_trigger: int


@dataclass
class RuntimeConfig:
    input_video: str
    output_video: str
    output_json: str
    progress_interval_frames: int = 20
    viz_mode: str = "debug"
    target_direction: str = "both"
    direction_min_frames: int = 5
    direction_min_displacement: float = 30.0


@dataclass
class RiderIdentitySettingsConfig:
    face_db_uri: str = ""
    face_collection: str = "rider_faces"
    face_dim: int = 512
    face_min_score: float = 0.3
    face_device: str = "cuda"
    face_models_dir: str = ""
    horse_rider_map: str = ""
    feature_store_path: str = "outputs/rider_identity.sqlite"


@dataclass
class PipelineConfig:
    detector: DetectorConfig
    roi: ROIConfig
    enhance: EnhanceConfig
    ocr: OCRConfig
    fusion: FusionConfig
    runtime: RuntimeConfig
    rider_identity_settings: RiderIdentitySettingsConfig | None = None
    vlm_fallback: VLMFallbackConfig | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml root object: {path}")
    return data


def load_config(config_path: str | Path) -> PipelineConfig:
    data = _load_yaml(Path(config_path))
    vlm_data = data.get("vlm_fallback", {})
    vlm_fallback = VLMFallbackConfig(**vlm_data) if vlm_data else None
    ri_data = data.get("rider_identity", {})
    rider_identity_settings = RiderIdentitySettingsConfig(**ri_data) if ri_data else None
    return PipelineConfig(
        detector=DetectorConfig(**data["detector"]),
        roi=ROIConfig(**data["roi"]),
        enhance=EnhanceConfig(**data["enhance"]),
        ocr=OCRConfig(**data["ocr"]),
        fusion=FusionConfig(**data["fusion"]),
        runtime=RuntimeConfig(**data["runtime"]),
        rider_identity_settings=rider_identity_settings,
        vlm_fallback=vlm_fallback,
    )

