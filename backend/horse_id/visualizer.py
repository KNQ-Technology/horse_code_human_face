from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from horse_id.types import HorseDetection, ROIBox

_COLOR_GREEN = (0, 255, 0)
_COLOR_CYAN = (255, 255, 0)
_COLOR_YELLOW = (0, 255, 255)
_COLOR_ORANGE = (0, 165, 255)
_COLOR_WHITE = (255, 255, 255)
_COLOR_RED = (80, 80, 255)
_COLOR_MAGENTA = (255, 0, 255)
_COLOR_FACE_BOX = (255, 128, 0)
_COLOR_GOLD = (0, 215, 255)
_COLOR_RIDER_MATCHED = (0, 230, 0)
_COLOR_RIDER_UNMATCHED = (120, 120, 120)

_SOURCE_COLORS: dict[str, tuple[int, int, int]] = {
    "face": (255, 128, 0),
    "face_locked": (0, 200, 255),
    "color": (0, 200, 200),
    "horse_map": (200, 200, 0),
    "temporal": (200, 100, 255),
    "store": (180, 180, 180),
    "none": (120, 120, 120),
}

_CJK_FONT_PATH = "/home/hshan/fonts/STHeiti.ttc"
_pil_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _pil_font_cache:
        try:
            _pil_font_cache[size] = ImageFont.truetype(_CJK_FONT_PATH, size)
        except (OSError, IOError):
            _pil_font_cache[size] = ImageFont.load_default()
    return _pil_font_cache[size]


def _text_size_pil(text: str, font_size: int) -> tuple[int, int]:
    """Return (width, height) of rendered text using PIL font."""
    font = _get_font(font_size)
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def _put_text_pil(
    canvas: np.ndarray,
    text: str,
    xy: tuple[int, int],
    font_size: int,
    color_bgr: tuple[int, int, int],
) -> None:
    """Render text with CJK support via PIL. Modifies canvas in-place."""
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(xy, text, font=font, fill=rgb)
    canvas[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


class PilBatchRenderer:
    """Context manager for batched PIL text rendering.

    Converts canvas BGR->RGB once on enter, accumulates draw calls,
    then converts back RGB->BGR once on exit.
    """

    def __init__(self, canvas: np.ndarray) -> None:
        self._canvas = canvas
        self._pil_img: Image.Image | None = None
        self._draw: ImageDraw.ImageDraw | None = None

    def __enter__(self) -> "PilBatchRenderer":
        self._pil_img = Image.fromarray(cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB))
        self._draw = ImageDraw.Draw(self._pil_img)
        return self

    def text(
        self,
        xy: tuple[int, int],
        text: str,
        font_size: int,
        color_bgr: tuple[int, int, int],
    ) -> None:
        if self._draw is None:
            return
        font = _get_font(font_size)
        rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        self._draw.text(xy, text, font=font, fill=rgb)

    def __exit__(self, *_: object) -> None:
        if self._pil_img is not None:
            self._canvas[:] = cv2.cvtColor(np.asarray(self._pil_img), cv2.COLOR_RGB2BGR)
        self._pil_img = None
        self._draw = None


class ResultVisualizer:
    """Draw detection and ROI overlays for quick visual inspection."""

    def __init__(self, viz_mode: str = "debug", hide_numbers: bool = False):
        self._viz_mode = viz_mode
        self._hide_numbers = hide_numbers

    def draw_frame(
        self,
        frame: np.ndarray,
        detections: list[HorseDetection],
        rois: list[ROIBox],
        ocr_infos: list[dict[str, object]] | None = None,
        unmatched_faces: list[dict[str, object]] | None = None,
    ) -> np.ndarray:
        """Draw horse box, left-shoulder ROI, face bbox, and rider identity labels."""
        canvas = frame.copy()
        for idx, (det, roi) in enumerate(zip(detections, rois), start=1):
            horse_x1 = int(round(det.x - det.w / 2.0))
            horse_y1 = int(round(det.y - det.h / 2.0))
            horse_x2 = int(round(det.x + det.w / 2.0))
            horse_y2 = int(round(det.y + det.h / 2.0))

            track_label = "--" if det.track_id is None else f"T{det.track_id}"
            stable_label = "--"
            state_label = "UNCONFIRMED"
            rider_name = ""
            rider_matched = False
            rider_source = "none"
            rider_score = 0.0
            face_bbox: list[int] = []
            face_score = 0.0
            face_detected = False

            if ocr_infos is not None and idx - 1 < len(ocr_infos):
                info = ocr_infos[idx - 1]
                stable_candidate = str(info.get("state_id", ""))
                if stable_candidate:
                    stable_label = stable_candidate
                state_label = str(info.get("state", "UNCONFIRMED"))
                rider_name = str(info.get("rider_name", ""))
                rider_matched = bool(info.get("rider_matched", False))
                rider_source = str(info.get("rider_source", "none"))
                rider_score = float(info.get("rider_score", 0.0))
                raw_bbox = info.get("face_bbox", [])
                if isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
                    face_bbox = [int(v) for v in raw_bbox[:4]]
                face_score = float(info.get("face_score", 0.0))
                face_detected = bool(info.get("face_detected", False))

            if self._viz_mode == "display":
                self._draw_display_mode(
                    canvas, idx, det, horse_x1, horse_y1, horse_x2, horse_y2,
                    stable_label, rider_name, rider_matched,
                    rider_source, rider_score,
                    hide_numbers=self._hide_numbers,
                )
                continue

            if rider_matched and rider_name:
                cv2.rectangle(canvas, (horse_x1, horse_y1), (horse_x2, horse_y2), _COLOR_GOLD, 3)
                self._draw_corner_brackets(
                    canvas, horse_x1 - 6, horse_y1 - 6, horse_x2 + 6, horse_y2 + 6,
                    _COLOR_GOLD, 2,
                )
            else:
                cv2.rectangle(canvas, (horse_x1, horse_y1), (horse_x2, horse_y2), _COLOR_GREEN, 2)
            cv2.rectangle(canvas, (roi.x1, roi.y1), (roi.x2, roi.y2), _COLOR_ORANGE, 2)

            label_main = f"H{idx} {track_label} conf={det.conf:.2f}"
            text_y = max(30, horse_y1 - 8)
            cv2.putText(
                canvas,
                label_main,
                (horse_x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                _COLOR_GREEN,
                2,
                cv2.LINE_AA,
            )
            if not self._hide_numbers:
                label_state = f"ID={stable_label} state={state_label}"
                cv2.putText(
                    canvas,
                    label_state,
                    (horse_x1, text_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    _COLOR_YELLOW,
                    2,
                    cv2.LINE_AA,
                )

            self._draw_rider_label(canvas, horse_x1, text_y + 44,
                                   rider_name, rider_matched, rider_source, rider_score,
                                   face_detected=face_detected)

            if face_bbox:
                face_label_name = rider_name if face_score > 0 else ""
                self._draw_face_box(canvas, face_bbox, face_label_name, face_score)

            if (not self._hide_numbers) and ocr_infos is not None and idx - 1 < len(ocr_infos):
                ocr_info = ocr_infos[idx - 1]
                text = str(ocr_info.get("text", ""))
                conf = float(ocr_info.get("conf", 0.0))
                valid = bool(ocr_info.get("valid", False))
                source = str(ocr_info.get("ocr_source", ""))
                if valid:
                    src_tag = f"[{source}]" if source else ""
                    ocr_label = f"OCR {text} ({conf:.2f}) {src_tag}"
                    bg_color = (35, 130, 35)
                else:
                    ocr_label = f"OCR -- ({conf:.2f})"
                    bg_color = (35, 35, 160)

                (tw, th), _ = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tx = max(0, roi.x1)
                ty = max(th + 8, roi.y1 - 6)
                cv2.rectangle(canvas, (tx - 3, ty - th - 6), (tx + tw + 6, ty + 3), bg_color, -1)
                cv2.putText(canvas, ocr_label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_WHITE, 2, cv2.LINE_AA)

                vlm_bg = str(ocr_info.get("vlm_bg", ""))
                vlm_font = str(ocr_info.get("vlm_font", ""))
                vlm_number = str(ocr_info.get("vlm_number", ""))
                vlm_full = str(ocr_info.get("vlm_full_id", ""))
                if vlm_bg or vlm_number:
                    vlm_label = f"VLM: bg={vlm_bg} font={vlm_font} num={vlm_number}"
                    if vlm_full:
                        vlm_label += f" => {vlm_full}"
                    (vw, vh), _ = cv2.getTextSize(vlm_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                    vy = ty + th + 10
                    vlm_bg_color = (130, 60, 20) if vlm_full else (35, 35, 130)
                    cv2.rectangle(canvas, (tx - 3, vy - vh - 4), (tx + vw + 6, vy + 3), vlm_bg_color, -1)
                    cv2.putText(canvas, vlm_label, (tx, vy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, _COLOR_CYAN, 2, cv2.LINE_AA)

        if self._viz_mode == "debug":
            if unmatched_faces:
                for face in unmatched_faces:
                    bbox = face.get("bbox", [])
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        face_bbox_u = [int(v) for v in bbox[:4]]
                        score = float(face.get("score", 0.0))
                        name = str(face.get("name", "")) if score > 0 else ""
                        self._draw_face_box(canvas, face_bbox_u, name, score)

            if ocr_infos and not self._hide_numbers:
                self._draw_ocr_summary(canvas, ocr_infos)
        return canvas

    _DISPLAY_TRUSTED_SOURCES = {"face", "face_locked"}
    _DISPLAY_MIN_RIDER_SCORE = 0.35

    @staticmethod
    def _draw_display_mode(
        canvas: np.ndarray,
        idx: int,
        det: HorseDetection,
        x1: int, y1: int, x2: int, y2: int,
        stable_id: str,
        rider_name: str,
        rider_matched: bool,
        rider_source: str,
        rider_score: float,
        hide_numbers: bool = False,
    ) -> None:
        """Display mode: minimal overlay with horse number and rider name only."""
        track_tag = "" if det.track_id is None else f"T{det.track_id}"
        if hide_numbers:
            # In full pipeline mode we intentionally hide the recognized numbering.
            horse_label = "马匹"
        else:
            horse_label = f"#{stable_id}" if stable_id != "--" else (track_tag or f"#{idx}")

        show_rider = (
            rider_matched
            and rider_name
            and rider_source in ResultVisualizer._DISPLAY_TRUSTED_SOURCES
            and rider_score >= ResultVisualizer._DISPLAY_MIN_RIDER_SCORE
        )

        if show_rider:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), _COLOR_GOLD, 3)
            ResultVisualizer._draw_corner_brackets(
                canvas, x1 - 6, y1 - 6, x2 + 6, y2 + 6, _COLOR_GOLD, 2,
            )
            rider_label = f"★ 骑手: {rider_name}"
            rider_color = _COLOR_GOLD
            horse_color = _COLOR_GOLD
            label_bg = (0, 40, 70)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), _COLOR_WHITE, 2)
            rider_label = "骑手: 其他骑师"
            rider_color = (180, 180, 180)
            horse_color = _COLOR_WHITE
            label_bg = (0, 0, 0)

        fs_horse = 24
        tw, th = _text_size_pil(horse_label, fs_horse)
        label_x = x1
        label_y = max(0, y1 - th - 28)
        fs_rider = 20
        rw, rh = _text_size_pil(rider_label, fs_rider)
        ry = label_y + th + 8

        overlay = canvas.copy()
        cv2.rectangle(overlay, (label_x - 4, label_y - 4),
                      (label_x + tw + 8, label_y + th + 4), label_bg, -1)
        cv2.rectangle(overlay, (label_x - 4, ry - 2),
                      (label_x + rw + 8, ry + rh + 2), label_bg, -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        if show_rider:
            cv2.rectangle(canvas, (label_x - 4, label_y - 4),
                          (label_x - 1, ry + rh + 2), _COLOR_GOLD, -1)

        with PilBatchRenderer(canvas) as pil:
            pil.text((label_x, label_y), horse_label, fs_horse, horse_color)
            pil.text((label_x, ry), rider_label, fs_rider, rider_color)

    @staticmethod
    def _draw_corner_brackets(
        canvas: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: tuple[int, int, int],
        thickness: int = 2,
        ratio: float = 0.25,
    ) -> None:
        """Draw L-shaped corner bracket decorations around a rectangle."""
        w, h = x2 - x1, y2 - y1
        arm = max(8, int(min(w, h) * ratio))
        cv2.line(canvas, (x1, y1), (x1 + arm, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1), (x1, y1 + arm), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1), (x2 - arm, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1), (x2, y1 + arm), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y2), (x1 + arm, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y2), (x1, y2 - arm), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y2), (x2 - arm, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y2), (x2, y2 - arm), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_face_box(
        canvas: np.ndarray,
        bbox: list[int],
        name: str,
        score: float,
    ) -> None:
        """Draw face bounding box with name label. Known riders get gold highlight."""
        x1, y1, x2, y2 = bbox
        h_img, w_img = canvas.shape[:2]

        if name:
            glow_pad = 6
            gx1 = max(0, x1 - glow_pad)
            gy1 = max(0, y1 - glow_pad)
            gx2 = min(w_img, x2 + glow_pad)
            gy2 = min(h_img, y2 + glow_pad)
            if gy2 > gy1 and gx2 > gx1:
                region = canvas[gy1:gy2, gx1:gx2].copy()
                cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), _COLOR_GOLD, -1)
                cv2.addWeighted(
                    canvas[gy1:gy2, gx1:gx2], 0.20, region, 0.80, 0,
                    canvas[gy1:gy2, gx1:gx2],
                )

            cv2.rectangle(canvas, (x1, y1), (x2, y2), _COLOR_GOLD, 3)
            ResultVisualizer._draw_corner_brackets(
                canvas, x1 - 5, y1 - 5, x2 + 5, y2 + 5, _COLOR_GOLD, 2,
            )

            box_color = _COLOR_GOLD
            label = f"★ {name} {score:.2f}"
        else:
            box_color = _COLOR_YELLOW
            label = "someone"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        fs = 18
        tw, th = _text_size_pil(label, fs)
        label_y_top = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, label_y_top), (x1 + tw + 6, label_y_top + th + 4), box_color, -1)
        _put_text_pil(canvas, label, (x1 + 3, label_y_top + 1), fs, _COLOR_WHITE)

    @staticmethod
    def _draw_rider_label(
        canvas: np.ndarray,
        x: int,
        y: int,
        name: str,
        matched: bool,
        source: str,
        score: float,
        face_detected: bool = False,
    ) -> None:
        """Draw rider identity tag with source-dependent color and background."""
        is_known = matched and bool(name)
        if is_known:
            display = f"★ Rider: {name} ({score:.2f}) [{source}]"
            fg_color = _COLOR_GOLD
            bg_color = (0, 40, 70)
        elif face_detected:
            display = "Rider: someone"
            fg_color = _COLOR_YELLOW
            bg_color = (40, 40, 60)
        else:
            display = "Rider: --"
            fg_color = _COLOR_RIDER_UNMATCHED
            bg_color = (40, 40, 40)

        fs = 19
        tw, th = _text_size_pil(display, fs)
        cv2.rectangle(canvas, (x - 2, y - 2), (x + tw + 4, y + th + 2), bg_color, -1)
        if is_known:
            cv2.rectangle(canvas, (x - 2, y - 2), (x + 1, y + th + 2), _COLOR_GOLD, -1)
        _put_text_pil(canvas, display, (x, y), fs, fg_color)

    def _draw_ocr_summary(self, canvas: np.ndarray, ocr_infos: list[dict[str, object]]) -> None:
        """Draw compact OCR + rider identity + VLM summary list in top-left corner."""
        x0, y0 = 12, 16
        line_h = 24
        has_vlm = any(info.get("vlm_bg") or info.get("vlm_number") for info in ocr_infos)
        extra_lines = len(ocr_infos) if has_vlm else 0
        box_w = 820
        box_h = 8 + line_h * (len(ocr_infos) + extra_lines + 1)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (200, 200, 200), 1)

        with PilBatchRenderer(canvas) as pil:
            pil.text((x0 + 8, y0 + 4), "Frame Summary (OCR + Rider Identity)", 18, (240, 240, 240))

            cur_line = 0
            for i, info in enumerate(ocr_infos, start=1):
                cur_line += 1
                track_id = info.get("track_id")
                text = str(info.get("text", ""))
                conf = float(info.get("conf", 0.0))
                valid = bool(info.get("valid", False))
                source = str(info.get("ocr_source", ""))
                stable_id = str(info.get("stable_id", ""))
                stable_ready = bool(info.get("stable_ready", False))
                state = str(info.get("state", "UNCONFIRMED"))
                state_id = str(info.get("state_id", ""))
                rider_name = str(info.get("rider_name", ""))
                rider_score = float(info.get("rider_score", 0.0))
                rider_matched = bool(info.get("rider_matched", False))
                rider_source = str(info.get("rider_source", "none"))

                status = "OK" if valid else "NG"
                text_show = text if text else "--"
                stable_show = stable_id if stable_ready and stable_id else "--"
                state_id_show = state_id if state_id else "--"
                track_str = f"T{track_id}" if track_id is not None else "T-"
                src_tag = f"<{source}>" if source and source not in ("empty", "fresh") else ""

                ocr_part = f"H{i}/{track_str}: {text_show} c={conf:.2f} id={stable_show} st={state}/{state_id_show} [{status}]{src_tag}"

                face_detected_s = bool(info.get("face_detected", False))
                if rider_matched and rider_name:
                    rider_part = f" | {rider_name}({rider_score:.2f})[{rider_source}]"
                    rider_color = _SOURCE_COLORS.get(rider_source, (180, 180, 180))
                elif face_detected_s:
                    rider_part = " | someone"
                    rider_color = _COLOR_YELLOW
                else:
                    rider_part = " | rider:--"
                    rider_color = _COLOR_RIDER_UNMATCHED

                fs_summary = 16
                line_y_top = y0 + 4 + line_h * cur_line
                ocr_color = (120, 255, 120) if valid else (120, 120, 255)
                pil.text((x0 + 8, line_y_top), ocr_part, fs_summary, ocr_color)

                ocr_tw, _ = _text_size_pil(ocr_part, fs_summary)
                pil.text((x0 + 8 + ocr_tw, line_y_top), rider_part, fs_summary, rider_color)

                vlm_bg = str(info.get("vlm_bg", ""))
                vlm_font = str(info.get("vlm_font", ""))
                vlm_number = str(info.get("vlm_number", ""))
                vlm_full = str(info.get("vlm_full_id", ""))
                if vlm_bg or vlm_number:
                    cur_line += 1
                    vlm_line = f"    VLM: bg={vlm_bg} font={vlm_font} num={vlm_number}"
                    if vlm_full:
                        vlm_line += f" => {vlm_full}"
                    vlm_y_top = y0 + 4 + line_h * cur_line
                    vlm_color = (255, 220, 100) if vlm_full else (100, 100, 255)
                    pil.text((x0 + 8, vlm_y_top), vlm_line, 15, vlm_color)

    def draw_roi_comparison_panel(
        self,
        frame: np.ndarray,
        roi_original_bgr: np.ndarray | None,
        roi_enhanced_gray: np.ndarray | None,
        roi_binary: np.ndarray | None,
        quality_score: float | None,
        ocr_text: str | None = None,
        ocr_conf: float | None = None,
        ocr_valid: bool | None = None,
    ) -> np.ndarray:
        """Draw ROI before/after comparison panel at top-right corner."""
        if (
            roi_original_bgr is None
            or roi_enhanced_gray is None
            or roi_binary is None
            or roi_original_bgr.size == 0
        ):
            return frame

        panel_h = 150
        panel_w_each = 190
        gap = 8
        margin = 12
        total_w = panel_w_each * 3 + gap * 2

        frame_h, frame_w = frame.shape[:2]
        x0 = max(0, frame_w - total_w - margin)
        y0 = margin

        roi_raw = cv2.resize(roi_original_bgr, (panel_w_each, panel_h), interpolation=cv2.INTER_AREA)
        roi_enh = cv2.resize(roi_enhanced_gray, (panel_w_each, panel_h), interpolation=cv2.INTER_AREA)
        roi_bin = cv2.resize(roi_binary, (panel_w_each, panel_h), interpolation=cv2.INTER_NEAREST)

        roi_enh_bgr = cv2.cvtColor(roi_enh, cv2.COLOR_GRAY2BGR)
        roi_bin_bgr = cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)

        canvas = frame.copy()
        canvas[y0 : y0 + panel_h, x0 : x0 + panel_w_each] = roi_raw
        canvas[y0 : y0 + panel_h, x0 + panel_w_each + gap : x0 + panel_w_each * 2 + gap] = roi_enh_bgr
        canvas[
            y0 : y0 + panel_h,
            x0 + panel_w_each * 2 + gap * 2 : x0 + panel_w_each * 3 + gap * 2,
        ] = roi_bin_bgr

        cv2.rectangle(canvas, (x0, y0), (x0 + total_w, y0 + panel_h), _COLOR_WHITE, 2)
        cv2.putText(canvas, "ROI Raw", (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_WHITE, 2)
        cv2.putText(canvas, "Enhanced", (x0 + panel_w_each + gap + 6, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_WHITE, 2)
        cv2.putText(canvas, "Binary", (x0 + panel_w_each * 2 + gap * 2 + 6, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_WHITE, 2)

        if quality_score is not None:
            cv2.putText(canvas, f"Q={quality_score:.2f}", (x0, y0 + panel_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, _COLOR_YELLOW, 2)
        if ocr_conf is not None:
            text_show = ocr_text if ocr_text else "--"
            status = "OK" if ocr_valid else "NG"
            cv2.putText(canvas, f"OCR={text_show} conf={ocr_conf:.2f} [{status}]",
                        (x0 + 120, y0 + panel_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, _COLOR_WHITE, 2)
        return canvas

    def draw_simple_saddle_footer(
        self,
        frame: np.ndarray,
        confirmed: list[tuple[str, int]],
        min_frames: int,
    ) -> np.ndarray:
        """Draw saddle-pad number summary at the bottom-right (simple / VLM-only mode).

        Args:
            frame: Input BGR frame (not modified in place; a copy is returned).
            confirmed: Pairs of ``(horse_id, frame_count)`` that already meet ``min_frames``.
            min_frames: Threshold label for the waiting hint when nothing is confirmed yet.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        margin = 14
        pad = 10
        fs = 20
        if confirmed:
            body = "  ".join(f"{hid}（{cnt}帧）" for hid, cnt in confirmed)
            main = f"鞍垫 {body}"
        else:
            main = f"鞍垫号码：累计 ≥{min_frames} 帧后显示"
        tw, th = _text_size_pil(main, fs)
        bx2 = w - margin
        by2 = h - margin
        tx = max(margin, bx2 - tw - pad)
        ty = max(margin, by2 - th - pad)
        bx1 = max(0, tx - pad)
        by1 = max(0, ty - pad)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (18, 18, 28), -1)
        cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), _COLOR_WHITE, 2)
        with PilBatchRenderer(canvas) as pil:
            pil.text((tx, ty), main, fs, _COLOR_WHITE)
        return canvas
