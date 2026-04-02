from __future__ import annotations

from dataclasses import dataclass
import os
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
    use_feature_store: bool = True


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
    vlm_video: VLMVideoConfig | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml root object: {path}")
    return data


def _resolve_local_path(value: str, base_dir: Path) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw
    if raw.startswith(("http://", "https://", "tcp://", "udp://")):
        return raw
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((base_dir / candidate).resolve())


def _apply_env_overrides(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    config = dict(data)

    detector_data = dict(config.get("detector", {}))
    if detector_data.get("model_path"):
        detector_data["model_path"] = _resolve_local_path(detector_data["model_path"], base_dir)
    detector_device = os.getenv("DETECTOR_DEVICE")
    if detector_device:
        detector_data["device"] = detector_device
    config["detector"] = detector_data

    ocr_data = dict(config.get("ocr", {}))
    ocr_lang = os.getenv("OCR_LANG")
    if ocr_lang:
        ocr_data["lang"] = ocr_lang
    config["ocr"] = ocr_data

    runtime_data = dict(config.get("runtime", {}))
    viz_mode = os.getenv("VIZ_MODE")
    if viz_mode:
        runtime_data["viz_mode"] = viz_mode
    config["runtime"] = runtime_data

    ri_data = dict(config.get("rider_identity", {}))
    face_db_uri = os.getenv("FACE_DB_URI")
    if face_db_uri:
        ri_data["face_db_uri"] = face_db_uri
    if ri_data.get("face_db_uri"):
        ri_data["face_db_uri"] = _resolve_local_path(ri_data["face_db_uri"], base_dir)
    face_models_dir = os.getenv("FACE_MODELS_DIR")
    if face_models_dir:
        ri_data["face_models_dir"] = face_models_dir
    if ri_data.get("face_models_dir"):
        ri_data["face_models_dir"] = _resolve_local_path(ri_data["face_models_dir"], base_dir)
    horse_rider_map = os.getenv("HORSE_RIDER_MAP")
    if horse_rider_map:
        ri_data["horse_rider_map"] = horse_rider_map
    if ri_data.get("horse_rider_map"):
        ri_data["horse_rider_map"] = _resolve_local_path(ri_data["horse_rider_map"], base_dir)
    feature_store_path = os.getenv("RIDER_FEATURE_STORE_PATH")
    if feature_store_path:
        ri_data["feature_store_path"] = feature_store_path
    if ri_data.get("feature_store_path"):
        ri_data["feature_store_path"] = _resolve_local_path(ri_data["feature_store_path"], base_dir)
    face_device = os.getenv("FACE_DEVICE")
    if face_device:
        ri_data["face_device"] = face_device
    config["rider_identity"] = ri_data

    vlm_data = dict(config.get("vlm_fallback", {}))
    vlm_api_key = os.getenv("VLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if vlm_api_key:
        vlm_data["api_key"] = vlm_api_key
    vlm_base_url = os.getenv("VLM_BASE_URL")
    if vlm_base_url:
        vlm_data["base_url"] = vlm_base_url
    vlm_model = os.getenv("VLM_MODEL")
    if vlm_model:
        vlm_data["model"] = vlm_model
    config["vlm_fallback"] = vlm_data

    vlm_video_data = dict(config.get("vlm_video", {}))
    if vlm_api_key:
        vlm_video_data["api_key"] = vlm_api_key
    if vlm_base_url:
        vlm_video_data["base_url"] = vlm_base_url
    if vlm_model:
        vlm_video_data["model"] = vlm_model
    config["vlm_video"] = vlm_video_data

    return config


def load_config(config_path: str | Path) -> PipelineConfig:
    config_file = Path(config_path).expanduser().resolve()
    project_dir = config_file.parent.parent
    data = _apply_env_overrides(_load_yaml(config_file), project_dir)
    vlm_data = data.get("vlm_fallback", {})
    vlm_fallback = VLMFallbackConfig(**vlm_data) if vlm_data else None
    ri_data = data.get("rider_identity", {})
    rider_identity_settings = RiderIdentitySettingsConfig(**ri_data) if ri_data else None
    vlm_video_data = data.get("vlm_video", {})
    vlm_video_cfg = VLMVideoConfig(**vlm_video_data) if vlm_video_data else None
    if vlm_video_cfg and vlm_fallback and vlm_fallback.api_key and not vlm_video_cfg.api_key:
        vlm_video_cfg.api_key = vlm_fallback.api_key
    return PipelineConfig(
        detector=DetectorConfig(**data["detector"]),
        roi=ROIConfig(**data["roi"]),
        enhance=EnhanceConfig(**data["enhance"]),
        ocr=OCRConfig(**data["ocr"]),
        fusion=FusionConfig(**data["fusion"]),
        runtime=RuntimeConfig(**data["runtime"]),
        rider_identity_settings=rider_identity_settings,
        vlm_fallback=vlm_fallback,
        vlm_video=vlm_video_cfg,
    )

