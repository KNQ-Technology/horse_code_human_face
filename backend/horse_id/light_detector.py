"""
夜间/低照度人脸图像增强：基于 LAB 亮度统计、去噪、CLAHE、gamma 提亮（高光保护）。
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _lab_L01(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    return L

def night_brightness_metrics(bgr: np.ndarray):
    L = _lab_L01(bgr)
    p5 = float(np.percentile(L, 5))
    p50 = float(np.percentile(L, 50))
    p90 = float(np.percentile(L, 90))
    p95 = float(np.percentile(L, 95))
    shadow_ratio = float(np.mean(L < 0.20))
    highlight_ratio = float(np.mean(L > 0.90))
    dr = p95 - p5
    return {
        "p5": p5, "p50": p50, "p90": p90, "p95": p95,
        "shadow_ratio": shadow_ratio,
        "highlight_ratio": highlight_ratio,
        "dr": dr
    }

def is_night_scene_and_needs_enhance(m: dict[str, float]) -> bool:
    """
    判断是否为夜间/低照度且需要增强。
    仅当同时满足「整体偏暗」且「阴影面积大」时才返回 True；
    白天光照充足时（p50 较高或阴影少）返回 False，不进行增强。
    """
    # 整体很暗：L 通道中位数 < 0.30；阴影多：L<0.2 的像素占比 > 35%
    return (m["p50"] < 0.30) and (m["shadow_ratio"] > 0.35)

def denoise_night(bgr: np.ndarray):
    # 夜间典型：降噪强度比白天略高，但不要过猛（否则涂抹）
    return cv2.fastNlMeansDenoisingColored(
        bgr, None,
        h=8, hColor=8,
        templateWindowSize=7,
        searchWindowSize=21
    )

def clahe_on_L(bgr: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(L)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def gamma_on_L_with_highlight_protection(bgr: np.ndarray, target_p50=0.45):
    """
    夜间：用自适应gamma抬亮度，但对高亮区域做保护，避免路灯炸白。
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0

    p50 = float(np.percentile(L, 50))
    p50 = max(p50, 1e-6)

    # 让中位数趋近 target_p50（夜间目标别太高，避免“白昼化”）
    gamma = np.log(target_p50) / np.log(p50)
    gamma = float(np.clip(gamma, 0.55, 1.0))  # 夜间通常在 0.6~0.9

    # 基础gamma提亮
    Lg = np.power(L, gamma)

    # 高光保护：L越接近1，增益越小
    # protect in [0,1]：暗区≈1，亮区→0
    protect = 1.0 - np.clip((L - 0.75) / 0.25, 0.0, 1.0)
    # 混合：暗区用Lg，亮区更多保留原L
    Lout = protect * Lg + (1.0 - protect) * L

    lab[:, :, 0] = (Lout * 255.0).clip(0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out, gamma

def auto_night_enhance(bgr: np.ndarray):
    """
    夜间专用：判定 + 去噪 + CLAHE + gamma(带高光保护)
    返回：out, enhanced(bool), debug_info(dict)
    """
    m0 = night_brightness_metrics(bgr)
    need = is_night_scene_and_needs_enhance(m0)
    if not need:
        return bgr, False, {"metrics": m0, "method": "none"}

    # 1) 去噪
    dn = denoise_night(bgr)

    # 2) 若动态范围很小（灰、闷），先CLAHE把阴影细节“拉出来”
    # 夜间dr常偏小，这里阈值略放宽
    if m0["dr"] < 0.45:
        mid = clahe_on_L(dn, clip_limit=2.0, tile_grid_size=(8, 8))
        method2 = "denoise+clahe"
    else:
        mid = dn
        method2 = "denoise"

    # 3) gamma抬阴影并保护高光
    out, gamma = gamma_on_L_with_highlight_protection(mid, target_p50=0.45)

    m1 = night_brightness_metrics(out)
    dbg = {
        "method": method2 + "+gamma_protected",
        "gamma": gamma,
        "metrics_before": m0,
        "metrics_after": m1
    }
    return out, True, dbg


class LightEnhancer:
    """
    低照度/夜间图像增强器：判断是否需要增强，并对人脸裁剪等小图做去噪 + CLAHE + gamma 提亮。
    适用于根据鼻子截取的人脸区域，提升暗光下人脸检测与识别的效果。

    判定规则：仅当 p50 < 0.30 且 shadow_ratio > 0.35 时才增强；
    白天光照充足时（中位数亮或阴影少）不增强，直接返回原图。
    """

    def __init__(
        self,
        target_p50: float = 0.45,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self.target_p50 = target_p50
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

    def get_metrics(self, bgr: np.ndarray) -> dict[str, float]:
        """返回当前图像的亮度相关指标（p5/p50/p90/p95、阴影/高光比例、动态范围等）。"""
        return night_brightness_metrics(bgr)

    def needs_enhance(self, bgr: np.ndarray) -> bool:
        """根据亮度指标判断是否为夜间/偏暗场景且需要增强。"""
        m = night_brightness_metrics(bgr)
        return is_night_scene_and_needs_enhance(m)

    def enhance(
        self,
        bgr: np.ndarray,
        force: bool = False,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        """
        对 BGR 图像做自动夜间增强（判定 + 去噪 + 可选 CLAHE + gamma 带高光保护）。

        :param bgr: 输入 BGR 图像（如根据鼻子截取的人脸区域）。
        :param force: 若 True，不判断是否夜间，强制走一遍增强流程。
        :return: (out, enhanced, debug_info)，enhanced 表示是否进行了增强，debug_info 含 metrics/method 等。
        """
        if bgr is None or bgr.size == 0:
            return bgr, False, {}
        if force:
            m0 = night_brightness_metrics(bgr)
            need = True
        else:
            m0 = night_brightness_metrics(bgr)
            need = is_night_scene_and_needs_enhance(m0)
        if not need:
            return bgr, False, {"metrics": m0, "method": "none"}

        dn = denoise_night(bgr)
        if m0["dr"] < 0.45:
            mid = clahe_on_L(dn, clip_limit=self.clahe_clip_limit, tile_grid_size=self.clahe_tile_grid_size)
            method2 = "denoise+clahe"
        else:
            mid = dn
            method2 = "denoise"

        out, gamma = gamma_on_L_with_highlight_protection(mid, target_p50=self.target_p50)
        m1 = night_brightness_metrics(out)
        dbg = {
            "method": method2 + "+gamma_protected",
            "gamma": gamma,
            "metrics_before": m0,
            "metrics_after": m1,
        }
        return out, True, dbg


def _to_gray01(bgr: np.ndarray) -> np.ndarray:
    """BGR -> Gray (0~1)"""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def tan_triggs_normalization(
    gray01: np.ndarray,
    gamma: float = 0.2,
    sigma0: float = 1.0,
    sigma1: float = 2.0,
    alpha: float = 0.1,
    tau: float = 10.0,
) -> np.ndarray:
    """
    Tan & Triggs normalization（识别友好）：
    - gamma 压缩动态范围
    - DoG 去除缓慢变化的光照
    - 两次对比度归一化 + 截断，抑制极端值
    输出：0~1 灰度
    """
    x = np.power(np.maximum(gray01, 1e-6), gamma)

    g0 = cv2.GaussianBlur(x, (0, 0), sigma0)
    g1 = cv2.GaussianBlur(x, (0, 0), sigma1)
    dog = g0 - g1

    denom = np.power(np.mean(np.abs(dog) ** alpha), 1.0 / alpha) + 1e-6
    y = dog / denom

    y = np.clip(y, -tau, tau)
    denom2 = np.power(np.mean(np.abs(y) ** alpha), 1.0 / alpha) + 1e-6
    y = y / denom2

    # 归一化到 0~1
    y = y / (np.max(np.abs(y)) + 1e-6)
    y = 0.5 + 0.5 * y
    return np.clip(y, 0.0, 1.0)


def mild_detail_boost_gray(gray01: np.ndarray, amount: float = 0.15, radius: float = 1.0) -> np.ndarray:
    """
    极保守细节增强（识别优先）：在灰度域做轻微 unsharp，避免噪声伪纹理。
    amount: 0.10~0.25 之间通常更安全
    """
    g = (gray01 * 255.0).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), radius)
    sharp = cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp / 255.0, 0.0, 1.0)


def gray01_to_bgr(gray01: np.ndarray) -> np.ndarray:
    g8 = (gray01 * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)


class FaceRecEnhancer:
    """
    200x200 暗光人脸 ROI：识别准确率优先增强器（不改脸、分布更稳）。

    Pipeline（仅灰度/亮度域）：
    1) 轻去噪
    2) 灰度 CLAHE（轻）
    3) Tan–Triggs 光照归一化（核心）
    4) 可选：极轻量细节增强（默认开，但很弱）

    输出默认是“灰度三通道 BGR”（方便对接仍要求3通道的模型/后处理）。
    """

    def __init__(
        self,
        # 判定阈值：对ROI可以比整图更宽松
        p50_th: float = 0.32,
        shadow_ratio_th: float = 0.30,

        # 去噪强度（ROI建议比整图更轻）
        denoise_h: int = 6,
        denoise_hColor: int = 6,

        # CLAHE：识别优先不要太猛
        clahe_clip_limit: float = 1.8,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),

        # Tan-Triggs 参数
        tt_gamma: float = 0.2,
        tt_sigma0: float = 1.0,
        tt_sigma1: float = 2.0,
        tt_alpha: float = 0.1,
        tt_tau: float = 10.0,

        # 细节增强（识别优先：轻）
        detail_amount: float = 0.15,
        detail_radius: float = 1.0,

        # 输出格式
        output_gray_bgr: bool = True,
    ) -> None:
        self.p50_th = float(p50_th)
        self.shadow_ratio_th = float(shadow_ratio_th)

        self.denoise_h = int(denoise_h)
        self.denoise_hColor = int(denoise_hColor)

        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = clahe_tile_grid_size

        self.tt_gamma = float(tt_gamma)
        self.tt_sigma0 = float(tt_sigma0)
        self.tt_sigma1 = float(tt_sigma1)
        self.tt_alpha = float(tt_alpha)
        self.tt_tau = float(tt_tau)

        self.detail_amount = float(detail_amount)
        self.detail_radius = float(detail_radius)

        self.output_gray_bgr = bool(output_gray_bgr)

    def _metrics_gray(self, bgr: np.ndarray) -> Dict[str, float]:
        gray01 = _to_gray01(bgr)
        p50 = float(np.percentile(gray01, 50))
        shadow_ratio = float(np.mean(gray01 < 0.20))
        return {"p50_gray": p50, "shadow_ratio_gray": shadow_ratio}

    def needs_enhance(self, bgr: np.ndarray) -> bool:
        m = self._metrics_gray(bgr)
        return (m["p50_gray"] < self.p50_th) and (m["shadow_ratio_gray"] > self.shadow_ratio_th)

    def enhance(self, bgr: np.ndarray, force: bool = False) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        if bgr is None or bgr.size == 0:
            return bgr, False, {}

        m0 = self._metrics_gray(bgr)
        need = True if force else ((m0["p50_gray"] < self.p50_th) and (m0["shadow_ratio_gray"] > self.shadow_ratio_th))
        if not need:
            return bgr, False, {"method": "none", "metrics": m0}

        # 1) 轻去噪（避免涂抹脸部关键梯度）
        dn = cv2.fastNlMeansDenoisingColored(
            bgr, None,
            h=self.denoise_h, hColor=self.denoise_hColor,
            templateWindowSize=7,
            searchWindowSize=21
        )

        # 2) 灰度 + 轻 CLAHE（提升局部对比，强度受控）
        gray = cv2.cvtColor(dn, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)
        gray = clahe.apply(gray)
        gray01 = gray.astype(np.float32) / 255.0

        # 3) Tan–Triggs（核心：光照归一化，让特征更稳定）
        tt = tan_triggs_normalization(
            gray01,
            gamma=self.tt_gamma,
            sigma0=self.tt_sigma0,
            sigma1=self.tt_sigma1,
            alpha=self.tt_alpha,
            tau=self.tt_tau,
        )

        # 4) 极轻量细节增强（可关闭：detail_amount=0）
        if self.detail_amount > 1e-6:
            tt = mild_detail_boost_gray(tt, amount=self.detail_amount, radius=self.detail_radius)

        out = gray01_to_bgr(tt) if self.output_gray_bgr else (tt * 255.0).clip(0, 255).astype(np.uint8)

        dbg = {
            "method": "denoise(light)+clahe(gray)+tan_triggs+detail(light)",
            "metrics_before": m0,
            "params": {
                "denoise": {"h": self.denoise_h, "hColor": self.denoise_hColor},
                "clahe": {"clip_limit": self.clahe_clip_limit, "tile_grid_size": self.clahe_tile_grid_size},
                "tan_triggs": {
                    "gamma": self.tt_gamma,
                    "sigma0": self.tt_sigma0,
                    "sigma1": self.tt_sigma1,
                    "alpha": self.tt_alpha,
                    "tau": self.tt_tau,
                },
                "detail": {"amount": self.detail_amount, "radius": self.detail_radius},
                "output_gray_bgr": self.output_gray_bgr,
            },
        }
        return out, True, dbg


if __name__ == "__main__":
    img = cv2.imread("input.jpg")
    out, enhanced, dbg = auto_night_enhance(img)
    print("enhanced:", enhanced)
    print(dbg)
    cv2.imwrite("output.jpg", out)