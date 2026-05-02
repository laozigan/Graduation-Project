#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


ENABLE_MKLDNN = _env_flag('PPS_ENABLE_MKLDNN', False)
os.environ['FLAGS_use_mkldnn'] = '1' if ENABLE_MKLDNN else '0'
os.environ['FLAGS_use_mkldnn_common_opt'] = '1' if ENABLE_MKLDNN else '0'
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'

import cv2
import numpy as np

from paddleocr import PaddleOCR
from src.utils.paddle_runtime import resolve_paddle_use_gpu


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


@dataclass
class PreprocessConfig:
    lang: str = "ch"
    use_ocr_orientation: bool = True
    enable_perspective_correction: bool = True
    enable_denoise: bool = True
    max_skew_correction: float = 15.0
    min_document_area_ratio: float = 0.50


@dataclass
class PreprocessResult:
    input_path: str
    output_path: str
    orientation_angle: float
    skew_angle: float
    orientation_method: str
    perspective_method: str


def is_supported_image(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def rotate_image(
    image: np.ndarray,
    angle: float,
    border_value: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


class ImagePreprocessor:
    """独立图像预处理器（仅图片输入）。"""

    def __init__(self, config: Optional[PreprocessConfig] = None, ocr_model: Optional[PaddleOCR] = None):
        self.config = config or PreprocessConfig()
        self._ocr_model: Optional[PaddleOCR] = ocr_model

    def _get_ocr_model(self) -> Optional[PaddleOCR]:
        if not self.config.use_ocr_orientation or PaddleOCR is None:
            return None
        if self._ocr_model is None:
            use_gpu, _ = resolve_paddle_use_gpu()
            self._ocr_model = PaddleOCR(lang=self.config.lang, use_textline_orientation=True, use_gpu=use_gpu)
        return self._ocr_model

    @staticmethod
    def _projection_score(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        horizontal = binary.sum(axis=1).astype(np.float32)
        vertical = binary.sum(axis=0).astype(np.float32)
        return float(horizontal.var() - 0.75 * vertical.var())

    def _fallback_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, float, str]:
        candidates = [0, 90, 180, 270]
        best_angle = 0.0
        best_score = float("-inf")
        best_image = image

        for angle in candidates:
            if angle == 0:
                candidate = image
            elif angle == 90:
                candidate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                candidate = cv2.rotate(image, cv2.ROTATE_180)
            else:
                candidate = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            score = self._projection_score(candidate)
            if score > best_score:
                best_score = score
                best_angle = float(angle)
                best_image = candidate

        return best_image, best_angle, "projection_fallback"

    def _detect_document_quad(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 180)

        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = float(image.shape[0] * image.shape[1])

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            area = cv2.contourArea(approx)
            ratio = area / image_area
            if ratio < self.config.min_document_area_ratio or ratio > 0.98:
                continue

            pts = approx.reshape(4, 2).astype(np.float32)
            return pts

        return None

    def correct_perspective(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        if not self.config.enable_perspective_correction:
            return image, "disabled"

        quad = self._detect_document_quad(image)
        if quad is None:
            return image, "none"

        transformed = _four_point_transform(image, quad)
        h, w = image.shape[:2]
        th, tw = transformed.shape[:2]
        # 避免误检内部表格边框导致的过度裁切。
        if tw < int(w * 0.75) or th < int(h * 0.75):
            return image, "rejected_small_warp"
        return transformed, "contour_quad"

    def correct_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, float, str]:
        ocr_model = self._get_ocr_model()
        if ocr_model is not None:
            try:
                result = ocr_model.predict(image)
                first = result[0] if isinstance(result, (list, tuple)) and result else result
                if first and hasattr(first, "get"):
                    doc_res = first.get("doc_preprocessor_res")
                    if isinstance(doc_res, dict):
                        angle = float(doc_res.get("angle", 0.0))
                        output_img = doc_res.get("output_img")
                        if isinstance(output_img, np.ndarray) and output_img.size > 0:
                            return output_img, angle, "paddleocr_doc_preprocessor"
            except Exception:
                pass

        return self._fallback_orientation(image)

    def _estimate_skew_by_hough(self, image: np.ndarray) -> Optional[float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=max(80, int(image.shape[1] * 0.2)),
            maxLineGap=15,
        )
        if lines is None:
            return None

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90

            if abs(angle) <= self.config.max_skew_correction:
                angles.append(float(angle))

        if not angles:
            return None

        return float(np.median(angles))

    def _estimate_skew_by_min_rect(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary > 0))
        if coords.size == 0:
            return 0.0

        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle

        if abs(angle) > self.config.max_skew_correction:
            return 0.0
        return float(angle)

    def estimate_skew_angle(self, image: np.ndarray) -> float:
        hough_angle = self._estimate_skew_by_hough(image)
        if hough_angle is not None:
            return hough_angle
        return self._estimate_skew_by_min_rect(image)

    def _apply_best_deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """比较原图与正负旋转结果，仅在质量提升时才执行倾斜校正。"""
        raw_angle = self.estimate_skew_angle(image)
        if abs(raw_angle) < 0.3:
            return image, 0.0

        baseline_score = self._projection_score(image)
        minus_img = rotate_image(image, -raw_angle)
        plus_img = rotate_image(image, raw_angle)

        candidates = [
            (0.0, image, baseline_score),
            (-float(raw_angle), minus_img, self._projection_score(minus_img)),
            (float(raw_angle), plus_img, self._projection_score(plus_img)),
        ]
        best_angle, best_image, best_score = max(candidates, key=lambda item: item[2])

        # 分数提升不足时视为误检，保持原图。
        min_improvement_ratio = 0.01
        denominator = max(abs(baseline_score), 1.0)
        improvement_ratio = (best_score - baseline_score) / denominator
        if best_angle == 0.0 or improvement_ratio < min_improvement_ratio:
            return image, 0.0

        return best_image, best_angle

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        working = image
        if self.config.enable_denoise:
            working = cv2.fastNlMeansDenoisingColored(working, None, 3, 3, 7, 21)

        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(enhanced, 1.35, gaussian, -0.35, 0)
        return sharpened

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, str, str]:
        perspective_img, perspective_method = self.correct_perspective(image)
        oriented_img, orientation_angle, orientation_method = self.correct_orientation(perspective_img)

        deskewed, skew_angle = self._apply_best_deskew(oriented_img)

        enhanced = self.enhance_image(deskewed)
        return enhanced, orientation_angle, skew_angle, orientation_method, perspective_method

    def preprocess_file(self, input_path: str, output_path: str) -> PreprocessResult:
        if not is_supported_image(input_path):
            raise ValueError(f"仅支持图片文件: {SUPPORTED_IMAGE_EXTENSIONS}，当前输入为 {input_path}")

        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图片: {input_path}")

        processed, orientation_angle, skew_angle, orientation_method, perspective_method = self.preprocess_image(image)

        ensure_parent_dir(output_path)
        ok = cv2.imwrite(output_path, processed)
        if not ok:
            raise RuntimeError(f"保存预处理结果失败: {output_path}")

        return PreprocessResult(
            input_path=input_path,
            output_path=output_path,
            orientation_angle=orientation_angle,
            skew_angle=skew_angle,
            orientation_method=orientation_method,
            perspective_method=perspective_method,
        )
