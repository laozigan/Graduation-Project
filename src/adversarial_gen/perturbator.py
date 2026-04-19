#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure adversarial perturbation for sensitive text regions.

This module applies constrained image-space perturbations only on sensitive text
areas and protects table lines/cell borders to preserve table structure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.image_preprocessing import ImagePreprocessor, PreprocessConfig
from src.table_extraction import convert_ocr_result_to_boxes, load_images_from_file


@dataclass
class AttackConfig:
    epsilon: float = 14.0
    alpha: float = 3.0
    steps: int = 5
    seed: int = 2026
    bbox_margin: int = 2
    text_threshold_block_size: int = 25
    text_threshold_c: int = 8
    min_text_pixels: int = 20
    line_protect_width: int = 2
    force_bbox_fallback: bool = True
    adaptive_detect_missing_cells: bool = True
    adaptive_use_nlp: bool = True
    adaptive_ocr_lang: str = "ch"


@dataclass
class AttackPageReport:
    page: int
    orientation_angle: float
    orientation_method: str
    sensitive_boxes: int
    attacked_boxes: int
    fallback_boxes: int
    attacked_pixels: int
    changed_pixels: int
    total_pixels: int
    box_source: str

    @property
    def changed_ratio(self) -> float:
        if self.total_pixels <= 0:
            return 0.0
        return float(self.changed_pixels) / float(self.total_pixels)


class AdversarialPerturbator:
    """Generate mask-constrained adversarial perturbation for OCR-sensitive cells."""

    def __init__(self, config: Optional[AttackConfig] = None):
        self.config = config or AttackConfig()
        self._adaptive_ocr_model = None
        self._adaptive_detector = None

    def apply_to_page(
        self,
        image: np.ndarray,
        cells: Sequence[Dict[str, Any]],
        return_report: bool = False,
    ) -> Any:
        if image is None or image.size == 0:
            raise ValueError("input image is empty")

        sensitive_bboxes, box_source = self._collect_sensitive_bboxes_with_adaptive(image, cells)
        report: Dict[str, int] = {
            "sensitive_boxes": len(sensitive_bboxes),
            "attacked_boxes": 0,
            "fallback_boxes": 0,
            "attacked_pixels": 0,
            "changed_pixels": 0,
            "total_pixels": int(image.shape[0] * image.shape[1]),
        }
        report["box_source"] = box_source  # type: ignore[index]
        if not sensitive_bboxes:
            out = image.copy()
            if return_report:
                return out, report
            return out

        line_mask = self._build_line_protection_mask(image, cells)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rng = np.random.default_rng(self.config.seed)
        original = image.astype(np.float32)
        delta = np.zeros_like(original, dtype=np.float32)

        for bbox in sensitive_bboxes:
            x1, y1, x2, y2 = self._sanitize_bbox(bbox, image.shape[1], image.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            roi_h = y2 - y1
            roi_w = x2 - x1
            if roi_h < 4 or roi_w < 4:
                continue

            text_mask = self._build_text_mask(gray[y1:y2, x1:x2])
            protected_roi = line_mask[y1:y2, x1:x2] > 0
            writable = np.logical_and(text_mask > 0, np.logical_not(protected_roi))

            if int(np.count_nonzero(writable)) < self.config.min_text_pixels:
                if self.config.force_bbox_fallback:
                    bbox_fallback = self._build_bbox_fallback_mask(roi_h, roi_w, protected_roi)
                    if int(np.count_nonzero(bbox_fallback)) >= self.config.min_text_pixels:
                        writable = bbox_fallback
                        report["fallback_boxes"] += 1
                    else:
                        continue
                else:
                    continue

            report["attacked_boxes"] += 1
            report["attacked_pixels"] += int(np.count_nonzero(writable))

            writable_3c = writable[..., None].astype(np.float32)
            roi_delta = delta[y1:y2, x1:x2]

            for _ in range(self.config.steps):
                noise = self._gen_attack_pattern(roi_h, roi_w, rng)
                step = (self.config.alpha * noise)[..., None]
                roi_delta = np.clip(roi_delta + step * writable_3c, -self.config.epsilon, self.config.epsilon)

            delta[y1:y2, x1:x2] = roi_delta

        attacked = np.clip(original + delta, 0.0, 255.0).astype(np.uint8)
        changed = np.any(attacked != image, axis=2)
        report["changed_pixels"] = int(np.count_nonzero(changed))

        if return_report:
            return attacked, report
        return attacked

    def _collect_sensitive_bboxes(self, cells: Sequence[Dict[str, Any]]) -> List[List[float]]:
        bboxes: List[List[float]] = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue

            is_sensitive = False
            sensitives = cell.get("sensitives")
            if isinstance(sensitives, list):
                is_sensitive = any(isinstance(item, dict) and item.get("is_sensitive") for item in sensitives)

            sensitive = cell.get("sensitive")
            if isinstance(sensitive, dict) and sensitive.get("is_sensitive"):
                is_sensitive = True

            if not is_sensitive:
                continue

            bbox = cell.get("bbox")
            if self._is_valid_bbox(bbox):
                bboxes.append([float(v) for v in bbox])
        return bboxes

    def _collect_sensitive_bboxes_with_adaptive(
        self,
        image: np.ndarray,
        cells: Sequence[Dict[str, Any]],
    ) -> Tuple[List[List[float]], str]:
        bboxes = self._collect_sensitive_bboxes(cells)
        if bboxes:
            return bboxes, "cells_sensitive"

        bboxes = self._infer_sensitive_bboxes_from_cell_text(cells)
        if bboxes:
            return bboxes, "cells_text_detected"

        if not self.config.adaptive_detect_missing_cells:
            return [], "none"

        bboxes = self._infer_sensitive_bboxes_from_ocr_lines(image)
        if bboxes:
            return bboxes, "ocr_lines_detected"

        return [], "none"

    def _get_adaptive_detector(self):
        if self._adaptive_detector is not None:
            return self._adaptive_detector
        from src.sensitive_detection import SensitiveDetector

        self._adaptive_detector = SensitiveDetector(
            use_nlp=self.config.adaptive_use_nlp,
            enable_uie=False,
        )
        return self._adaptive_detector

    def _get_adaptive_ocr_model(self):
        if self._adaptive_ocr_model is not None:
            return self._adaptive_ocr_model
        from paddleocr import PaddleOCR

        self._adaptive_ocr_model = PaddleOCR(lang=self.config.adaptive_ocr_lang, use_angle_cls=True)
        return self._adaptive_ocr_model

    def _infer_sensitive_bboxes_from_cell_text(self, cells: Sequence[Dict[str, Any]]) -> List[List[float]]:
        detector = self._get_adaptive_detector()
        bboxes: List[List[float]] = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            bbox = cell.get("bbox")
            text = str(cell.get("text", "")).strip()
            if not self._is_valid_bbox(bbox) or not text:
                continue
            if detector.detect_all(text):
                bboxes.append([float(v) for v in bbox])
        return bboxes

    def _infer_sensitive_bboxes_from_ocr_lines(self, image: np.ndarray) -> List[List[float]]:
        try:
            ocr = self._get_adaptive_ocr_model()
            detector = self._get_adaptive_detector()
            output = ocr.predict(image)
            first = output[0] if isinstance(output, (list, tuple)) and output else output
            text_boxes = convert_ocr_result_to_boxes(first)
            bboxes: List[List[float]] = []
            for item in text_boxes:
                text = str(item.get("text", "")).strip()
                bbox = item.get("bbox")
                if not text or not self._is_valid_bbox(bbox):
                    continue
                if detector.detect_all(text):
                    bboxes.append([float(v) for v in bbox])
            return bboxes
        except Exception:
            return []

    @staticmethod
    def _is_valid_bbox(bbox: Any) -> bool:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return False
        try:
            float(bbox[0])
            float(bbox[1])
            float(bbox[2])
            float(bbox[3])
            return True
        except (TypeError, ValueError):
            return False

    def _sanitize_bbox(self, bbox: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
        x1 = int(np.floor(min(bbox[0], bbox[2]))) + self.config.bbox_margin
        y1 = int(np.floor(min(bbox[1], bbox[3]))) + self.config.bbox_margin
        x2 = int(np.ceil(max(bbox[0], bbox[2]))) - self.config.bbox_margin
        y2 = int(np.ceil(max(bbox[1], bbox[3]))) - self.config.bbox_margin

        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        return x1, y1, x2, y2

    def _build_line_protection_mask(self, image: np.ndarray, cells: Sequence[Dict[str, Any]]) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            10,
        )

        h, w = gray.shape
        horiz_size = max(8, w // 30)
        vert_size = max(8, h // 30)

        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))

        horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel)
        line_mask = cv2.bitwise_or(horiz, vert)

        # Protect detected cell borders to stabilize table geometry.
        border_mask = np.zeros_like(line_mask)
        for cell in cells:
            bbox = cell.get("bbox") if isinstance(cell, dict) else None
            if not self._is_valid_bbox(bbox):
                continue
            x1, y1, x2, y2 = self._sanitize_bbox([float(v) for v in bbox], w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(border_mask, (x1, y1), (x2, y2), 255, self.config.line_protect_width)

        line_mask = cv2.bitwise_or(line_mask, border_mask)
        line_mask = cv2.dilate(line_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        return line_mask

    def _build_text_mask(self, roi_gray: np.ndarray) -> np.ndarray:
        block_size = self.config.text_threshold_block_size
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)

        text_mask = cv2.adaptiveThreshold(
            roi_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            self.config.text_threshold_c,
        )
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
        return text_mask

    @staticmethod
    def _build_bbox_fallback_mask(roi_h: int, roi_w: int, protected_roi: np.ndarray) -> np.ndarray:
        writable = np.ones((roi_h, roi_w), dtype=bool)
        if roi_h > 2:
            writable[0, :] = False
            writable[-1, :] = False
        if roi_w > 2:
            writable[:, 0] = False
            writable[:, -1] = False
        return np.logical_and(writable, np.logical_not(protected_roi))

    @staticmethod
    def _gen_attack_pattern(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
        rand = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        rand = cv2.GaussianBlur(rand, (0, 0), sigmaX=0.9)

        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        fx = rng.uniform(12.0, 24.0)
        fy = rng.uniform(12.0, 24.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        wave = np.sin(2.0 * np.pi * (fx * gx + fy * gy) + phase).astype(np.float32)

        pattern = 0.7 * rand + 0.3 * wave
        denom = float(np.max(np.abs(pattern)))
        if denom < 1e-6:
            return np.zeros((h, w), dtype=np.float32)
        return pattern / denom


def _flatten_pages(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if isinstance(item, dict) and isinstance(item.get("cells"), list):
            normalized.append({"page": int(item.get("page", idx)), "cells": item["cells"]})
        elif isinstance(item, list):
            normalized.append({"page": idx, "cells": item})
    return normalized


def _normalize_right_angle(angle: float, tolerance: float = 8.0) -> int:
    candidates = [0, 90, 180, 270]
    wrapped = float(angle) % 360.0
    best = min(candidates, key=lambda c: min(abs(wrapped - c), abs((wrapped - c) % 360.0)))
    diff = min(abs(wrapped - best), abs((wrapped - best) % 360.0))
    if diff <= tolerance:
        return int(best)
    return 0


def _rotate_image_and_cells(
    image: np.ndarray,
    cells: Sequence[Dict[str, Any]],
    right_angle: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if right_angle not in (0, 90, 180, 270):
        return image, [dict(c) for c in cells]

    h, w = image.shape[:2]
    if right_angle == 0:
        rotated_img = image
    elif right_angle == 90:
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif right_angle == 180:
        rotated_img = cv2.rotate(image, cv2.ROTATE_180)
    else:
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def rotate_bbox(bbox: Sequence[float]) -> List[float]:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        if right_angle == 90:
            rotated = np.column_stack([h - points[:, 1], points[:, 0]])
        elif right_angle == 180:
            rotated = np.column_stack([w - points[:, 0], h - points[:, 1]])
        elif right_angle == 270:
            rotated = np.column_stack([points[:, 1], w - points[:, 0]])
        else:
            rotated = points

        rx1 = float(np.min(rotated[:, 0]))
        ry1 = float(np.min(rotated[:, 1]))
        rx2 = float(np.max(rotated[:, 0]))
        ry2 = float(np.max(rotated[:, 1]))
        return [rx1, ry1, rx2, ry2]

    rotated_cells: List[Dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        new_cell = dict(cell)
        bbox = new_cell.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            new_cell["bbox"] = rotate_bbox(bbox)
        rotated_cells.append(new_cell)
    return rotated_img, rotated_cells


def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image
    scale = float(target_h) / float(max(h, 1))
    target_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _build_compare_canvas(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_title: str = "oriented",
    right_title: str = "adversarial",
) -> np.ndarray:
    target_h = max(left_image.shape[0], right_image.shape[0])
    left = _resize_to_height(left_image, target_h)
    right = _resize_to_height(right_image, target_h)

    gap = 12
    panel_h = 52
    content_w = left.shape[1] + gap + right.shape[1]
    canvas = np.full((panel_h + target_h, content_w, 3), 255, dtype=np.uint8)
    canvas[panel_h:, : left.shape[1]] = left
    canvas[panel_h:, left.shape[1] + gap :] = right

    cv2.putText(canvas, left_title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        right_title,
        (left.shape[1] + gap + 10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return canvas


def run_attack_from_files(
    input_file: str,
    detection_json: str,
    output_paths: Sequence[str],
    config: Optional[AttackConfig] = None,
    dpi: int = 200,
    poppler_path: Optional[str] = None,
    auto_orient: Optional[bool] = None,
    orient_mode: str = "upside_down",
    verbose: bool = False,
    oriented_output_paths: Optional[Sequence[str]] = None,
    compare_output_paths: Optional[Sequence[str]] = None,
) -> List[str]:
    if auto_orient is not None:
        orient_mode = "upside_down" if auto_orient else "off"
    if orient_mode not in {"off", "upside_down", "always"}:
        raise ValueError("orient_mode must be one of: off, upside_down, always")

    with open(detection_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    pages = _flatten_pages(payload)
    if not pages:
        raise ValueError("no page/cell data found in detection json")

    images = load_images_from_file(input_file, dpi=dpi, poppler_path=poppler_path)
    if not images:
        raise ValueError("no image page loaded from input file")

    if len(output_paths) != len(pages):
        raise ValueError(
            f"output path count ({len(output_paths)}) does not match page count ({len(pages)})"
        )
    if oriented_output_paths is not None and len(oriented_output_paths) != len(pages):
        raise ValueError(
            f"oriented output path count ({len(oriented_output_paths)}) does not match page count ({len(pages)})"
        )
    if compare_output_paths is not None and len(compare_output_paths) != len(pages):
        raise ValueError(
            f"compare output path count ({len(compare_output_paths)}) does not match page count ({len(pages)})"
        )

    perturbator = AdversarialPerturbator(config=config)
    orienter: Optional[ImagePreprocessor] = None
    if orient_mode != "off":
        orienter = ImagePreprocessor(
            config=PreprocessConfig(
                use_ocr_orientation=True,
                enable_perspective_correction=False,
                enable_denoise=False,
            )
        )

    saved_paths: List[str] = []

    for idx, page in enumerate(pages):
        if idx >= len(images):
            break

        current = images[idx]
        orientation_angle = 0.0
        orientation_method = "disabled"
        current_cells = page.get("cells", [])
        if orienter is not None:
            _, orientation_angle, orientation_method = orienter.correct_orientation(current)
            right_angle = _normalize_right_angle(orientation_angle)
            applied_angle = 0
            if orient_mode == "always":
                applied_angle = right_angle
            elif orient_mode == "upside_down" and right_angle == 180:
                applied_angle = 180
            current, current_cells = _rotate_image_and_cells(current, current_cells, applied_angle)

        attacked, page_report_dict = perturbator.apply_to_page(current, current_cells, return_report=True)
        page_report = AttackPageReport(
            page=idx,
            orientation_angle=orientation_angle,
            orientation_method=orientation_method,
            sensitive_boxes=page_report_dict["sensitive_boxes"],
            attacked_boxes=page_report_dict["attacked_boxes"],
            fallback_boxes=page_report_dict["fallback_boxes"],
            attacked_pixels=page_report_dict["attacked_pixels"],
            changed_pixels=page_report_dict["changed_pixels"],
            total_pixels=page_report_dict["total_pixels"],
            box_source=str(page_report_dict.get("box_source", "unknown")),
        )

        out = output_paths[idx]
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ok = cv2.imwrite(out, attacked)
        if not ok:
            raise IOError(f"failed to write output image: {out}")

        if oriented_output_paths is not None:
            oriented_out = oriented_output_paths[idx]
            oriented_dir = os.path.dirname(oriented_out)
            if oriented_dir:
                os.makedirs(oriented_dir, exist_ok=True)
            if not cv2.imwrite(oriented_out, current):
                raise IOError(f"failed to write oriented image: {oriented_out}")

        if compare_output_paths is not None:
            compare_out = compare_output_paths[idx]
            compare_dir = os.path.dirname(compare_out)
            if compare_dir:
                os.makedirs(compare_dir, exist_ok=True)
            compare_canvas = _build_compare_canvas(current, attacked)
            if not cv2.imwrite(compare_out, compare_canvas):
                raise IOError(f"failed to write compare image: {compare_out}")

        if verbose:
            print(
                f"[attack] page={page_report.page + 1} "
                f"orient={page_report.orientation_method}:{page_report.orientation_angle:.1f} "
                f"source={page_report.box_source} "
                f"boxes={page_report.attacked_boxes}/{page_report.sensitive_boxes} "
                f"fallback={page_report.fallback_boxes} "
                f"changed={page_report.changed_pixels}/{page_report.total_pixels} "
                f"({page_report.changed_ratio * 100.0:.2f}%)"
            )
            if page_report.sensitive_boxes > 0 and page_report.attacked_boxes == 0:
                print(
                    f"[attack] warning: page {page_report.page + 1} has sensitive boxes "
                    "but none were writable. Consider increasing --bbox-margin/--epsilon or checking JSON alignment."
                )
            if page_report.sensitive_boxes == 0:
                print(
                    f"[attack] warning: page {page_report.page + 1} has 0 sensitive boxes in detection JSON; "
                    "no adversarial perturbation applied."
                )

        saved_paths.append(out)

    return saved_paths
