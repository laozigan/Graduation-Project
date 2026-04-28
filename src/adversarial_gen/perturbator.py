#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure adversarial perturbation for sensitive text regions.

This module applies constrained image-space perturbations only on sensitive text
areas and protects table lines/cell borders to preserve table structure.
"""

from __future__ import annotations

import json
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .advbox_generator import AdvBoxROIAttackParams, run_advbox_roi_attack
from src.image_preprocessing import ImagePreprocessor, PreprocessConfig
from src.table_extraction import convert_ocr_result_to_boxes, load_images_from_file


@dataclass
class AttackConfig:
    epsilon: float = 14.0
    alpha: float = 3.0
    steps: int = 5
    attack_method: str = "random"
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
    pgd_align_weight: float = 1.0
    pgd_magnitude_weight: float = 0.3
    pgd_edge_weight: float = 0.05
    advbox_roi_expand: int = 8
    advbox_restarts: int = 3
    advbox_momentum: float = 0.8
    advbox_attack_name: str = "PGD"
    advbox_epsilon_steps: int = 6
    advbox_spsa_sigma: float = 2.0
    advbox_spsa_samples: int = 4
    advbox_text_change_bonus: float = 0.5
    advbox_rec_model: str = "PP-OCRv5_server_rec"
    enable_mkldnn: bool = False
    num_threads: int = 0
    image_scale: float = 1.0


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
        self._advbox_recognizer = None
        self._pgd_fallback_warned = False
        self._advbox_fallback_warned = False
        self._advbox_backend = self._detect_advbox_backend()
        self._configure_runtime()

    @staticmethod
    def _detect_advbox_backend() -> str:
        # Keep this lightweight so the class can run even when optional libs are absent.
        project_root = Path(__file__).resolve().parents[2]
        if (project_root / "AdvBox").exists():
            return "local_advbox"
        if importlib.util.find_spec("advbox") is not None:
            return "advbox"
        if importlib.util.find_spec("adversarialbox") is not None:
            return "adversarialbox"
        return "none"

    def _configure_runtime(self) -> None:
        """Configure runtime knobs before model initialization for better CPU performance."""
        if self.config.enable_mkldnn:
            os.environ["FLAGS_use_mkldnn"] = "1"
            os.environ["FLAGS_use_mkldnn_common_opt"] = "1"

        if self.config.num_threads > 0:
            thread_str = str(self.config.num_threads)
            os.environ["OMP_NUM_THREADS"] = thread_str
            os.environ["MKL_NUM_THREADS"] = thread_str
            os.environ["OPENBLAS_NUM_THREADS"] = thread_str
            try:
                cv2.setNumThreads(self.config.num_threads)
            except Exception:
                pass
        else:
            try:
                cv2.setUseOptimized(True)
            except Exception:
                pass

    def apply_to_page(
        self,
        image: np.ndarray,
        cells: Sequence[Dict[str, Any]],
        return_report: bool = False,
    ) -> Any:
        if image is None or image.size == 0:
            raise ValueError("input image is empty")

        scale = float(self.config.image_scale)
        if scale <= 0.0:
            scale = 1.0

        # Optional downscale for faster optimization; results are resized back.
        if abs(scale - 1.0) > 1e-6:
            scaled_image, scaled_cells = _scale_image_and_cells(image, cells, scale)
            attacked_scaled, report = self._apply_to_page_core(scaled_image, scaled_cells, return_report=True)
            upsampled = cv2.resize(
                attacked_scaled,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            # Only paste back truly perturbed regions to avoid full-image resampling artifacts.
            scaled_changed = np.any(attacked_scaled != scaled_image, axis=2).astype(np.uint8)
            changed_mask = cv2.resize(
                scaled_changed,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
            attacked = image.copy()
            attacked[changed_mask] = upsampled[changed_mask]
            report["total_pixels"] = int(image.shape[0] * image.shape[1])
            report["changed_pixels"] = int(np.count_nonzero(np.any(attacked != image, axis=2)))
            if return_report:
                return attacked, report
            return attacked

        return self._apply_to_page_core(image, cells, return_report=return_report)

    def _apply_to_page_core(
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
        attack_method = (self.config.attack_method or "random").strip().lower().replace("-", "_")

        for bbox in sensitive_bboxes:
            x1, y1, x2, y2 = self._sanitize_bbox(bbox, image.shape[1], image.shape[0])
            if attack_method == "advbox_roi" and int(self.config.advbox_roi_expand) > 0:
                x1, y1, x2, y2 = self._expand_bbox(
                    x1,
                    y1,
                    x2,
                    y2,
                    int(self.config.advbox_roi_expand),
                    image.shape[1],
                    image.shape[0],
                )
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

            if attack_method == "advbox_roi":
                # AdvBox-style ROI optimization with multi-restart and momentum ascent.
                roi_delta = self._optimize_delta_advbox_roi(
                    roi_delta=roi_delta,
                    original_roi=original[y1:y2, x1:x2],
                    writable_3c=writable_3c,
                    rng=rng,
                )
            elif attack_method == "pgd":
                # White-box-style PGD on differentiable surrogate objective in writable text region.
                roi_delta = self._optimize_delta_pgd(
                    roi_delta=roi_delta,
                    original_roi=original[y1:y2, x1:x2],
                    writable_3c=writable_3c,
                    rng=rng,
                )
            else:
                roi_delta = self._optimize_delta_random(
                    roi_delta=roi_delta,
                    writable_3c=writable_3c,
                    roi_h=roi_h,
                    roi_w=roi_w,
                    rng=rng,
                )

            delta[y1:y2, x1:x2] = roi_delta

        attacked = np.clip(original + delta, 0.0, 255.0).astype(np.uint8)
        changed = np.any(attacked != image, axis=2)
        report["changed_pixels"] = int(np.count_nonzero(changed))

        if return_report:
            return attacked, report
        return attacked

    def _optimize_delta_random(
        self,
        roi_delta: np.ndarray,
        writable_3c: np.ndarray,
        roi_h: int,
        roi_w: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        for _ in range(self.config.steps):
            noise = self._gen_attack_pattern(roi_h, roi_w, rng)
            step = (self.config.alpha * noise)[..., None]
            roi_delta = np.clip(roi_delta + step * writable_3c, -self.config.epsilon, self.config.epsilon)
        return roi_delta

    def _optimize_delta_pgd(
        self,
        roi_delta: np.ndarray,
        original_roi: np.ndarray,
        writable_3c: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        try:
            import paddle
        except Exception:
            if not self._pgd_fallback_warned:
                print("Warning: paddle unavailable for PGD, fallback to random attack pattern.")
                self._pgd_fallback_warned = True
            return self._optimize_delta_random(
                roi_delta=roi_delta,
                writable_3c=writable_3c,
                roi_h=roi_delta.shape[0],
                roi_w=roi_delta.shape[1],
                rng=rng,
            )

        if int(np.count_nonzero(writable_3c)) <= 0:
            return roi_delta

        eps = float(self.config.epsilon) / 255.0
        alpha = float(self.config.alpha) / 255.0

        original_norm = np.clip(original_roi / 255.0, 0.0, 1.0).astype(np.float32)
        delta_norm = np.clip(roi_delta / 255.0, -eps, eps).astype(np.float32)
        mask_np = writable_3c.astype(np.float32)

        # Fixed target pattern guides PGD direction and avoids zero-gradient starts.
        pattern = self._gen_attack_pattern(roi_delta.shape[0], roi_delta.shape[1], rng)[..., None]
        target_np = np.repeat(pattern, 3, axis=2).astype(np.float32)

        orig_t = paddle.to_tensor(original_norm, stop_gradient=True)
        mask_t = paddle.to_tensor(mask_np, stop_gradient=True)
        target_t = paddle.to_tensor(target_np, stop_gradient=True)
        delta_t = paddle.to_tensor(delta_norm)
        delta_t.stop_gradient = False

        for _ in range(max(1, int(self.config.steps))):
            adv_t = paddle.clip(orig_t + delta_t, 0.0, 1.0)
            diff_t = (adv_t - orig_t) * mask_t

            # Key PGD objectives:
            # 1) align with target pattern, 2) increase perturbation magnitude,
            # 3) raise local high-frequency to hurt OCR readability.
            align_term = paddle.mean(diff_t * target_t)
            magnitude_term = paddle.mean(paddle.abs(diff_t))

            adv_gray = 0.114 * adv_t[:, :, 0] + 0.587 * adv_t[:, :, 1] + 0.299 * adv_t[:, :, 2]
            gx = adv_gray[:, 1:] - adv_gray[:, :-1]
            gy = adv_gray[1:, :] - adv_gray[:-1, :]
            edge_term = paddle.mean(paddle.abs(gx)) + paddle.mean(paddle.abs(gy))

            loss = (
                float(self.config.pgd_align_weight) * align_term
                + float(self.config.pgd_magnitude_weight) * magnitude_term
                + float(self.config.pgd_edge_weight) * edge_term
            )
            loss.backward()
            grad = delta_t.grad
            if grad is None:
                break

            step = alpha * paddle.sign(grad) * mask_t
            next_delta = paddle.clip(delta_t + step, -eps, eps)
            next_delta = next_delta * mask_t

            delta_t = next_delta.detach()
            delta_t.stop_gradient = False

        return (delta_t.numpy() * 255.0).astype(np.float32)

    def _optimize_delta_advbox_roi(
        self,
        roi_delta: np.ndarray,
        original_roi: np.ndarray,
        writable_3c: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Prefer official adversarialbox backend when available, with seamless fallback.
        if self._advbox_backend != "none":
            advbox_params = AdvBoxROIAttackParams(
                epsilon=float(self.config.epsilon),
                alpha=float(self.config.alpha),
                steps=max(1, int(self.config.steps)),
                epsilon_steps=max(1, int(self.config.advbox_epsilon_steps)),
                attack_name=str(self.config.advbox_attack_name),
                spsa_sigma=float(self.config.advbox_spsa_sigma),
                spsa_samples=max(1, int(self.config.advbox_spsa_samples)),
                text_change_bonus=float(self.config.advbox_text_change_bonus),
            )

            recognizer = None
            try:
                recognizer = self._get_advbox_recognizer()
            except Exception:
                recognizer = None

            advbox_delta = run_advbox_roi_attack(
                roi_delta=roi_delta,
                original_roi=original_roi,
                writable_3c=writable_3c,
                params=advbox_params,
                recognizer=recognizer,
                rng=rng,
            )
            if advbox_delta is not None:
                return advbox_delta
            if not self._advbox_fallback_warned:
                print("Warning: adversarialbox backend call failed, fallback to built-in advbox_roi optimizer.")
                self._advbox_fallback_warned = True

        try:
            import paddle
        except Exception:
            if not self._advbox_fallback_warned:
                print("Warning: paddle unavailable for advbox_roi, fallback to random attack pattern.")
                self._advbox_fallback_warned = True
            return self._optimize_delta_random(
                roi_delta=roi_delta,
                writable_3c=writable_3c,
                roi_h=roi_delta.shape[0],
                roi_w=roi_delta.shape[1],
                rng=rng,
            )

        if int(np.count_nonzero(writable_3c)) <= 0:
            return roi_delta

        if self._advbox_backend == "none" and not self._advbox_fallback_warned:
            print("Warning: advbox package not found; using built-in advbox_roi optimizer.")
            self._advbox_fallback_warned = True

        eps = float(self.config.epsilon) / 255.0
        alpha = float(self.config.alpha) / 255.0

        original_norm = np.clip(original_roi / 255.0, 0.0, 1.0).astype(np.float32)
        base_delta = np.clip(roi_delta / 255.0, -eps, eps).astype(np.float32)
        mask_np = writable_3c.astype(np.float32)

        # Use a stable target texture so all restarts optimize toward the same objective.
        pattern = self._gen_attack_pattern(roi_delta.shape[0], roi_delta.shape[1], rng)[..., None]
        target_np = np.repeat(pattern, 3, axis=2).astype(np.float32)

        orig_t = paddle.to_tensor(original_norm, stop_gradient=True)
        mask_t = paddle.to_tensor(mask_np, stop_gradient=True)
        target_t = paddle.to_tensor(target_np, stop_gradient=True)

        restarts = max(1, int(self.config.advbox_restarts))
        momentum = float(np.clip(self.config.advbox_momentum, 0.0, 0.99))

        best_delta = base_delta.copy()
        best_score = -1e9

        for restart_idx in range(restarts):
            if restart_idx == 0:
                init_delta = base_delta.copy()
            else:
                init_delta = rng.uniform(-eps, eps, size=base_delta.shape).astype(np.float32)
                init_delta = init_delta * mask_np

            delta_t = paddle.to_tensor(init_delta)
            delta_t.stop_gradient = False
            velocity = paddle.zeros_like(delta_t)

            for _ in range(max(1, int(self.config.steps))):
                adv_t = paddle.clip(orig_t + delta_t, 0.0, 1.0)
                diff_t = (adv_t - orig_t) * mask_t

                align_term = paddle.mean(diff_t * target_t)
                magnitude_term = paddle.mean(paddle.abs(diff_t))

                adv_gray = 0.114 * adv_t[:, :, 0] + 0.587 * adv_t[:, :, 1] + 0.299 * adv_t[:, :, 2]
                gx = adv_gray[:, 1:] - adv_gray[:, :-1]
                gy = adv_gray[1:, :] - adv_gray[:-1, :]
                edge_term = paddle.mean(paddle.abs(gx)) + paddle.mean(paddle.abs(gy))

                score = (
                    float(self.config.pgd_align_weight) * align_term
                    + float(self.config.pgd_magnitude_weight) * magnitude_term
                    + float(self.config.pgd_edge_weight) * edge_term
                )
                score.backward()
                grad = delta_t.grad
                if grad is None:
                    break

                velocity = momentum * velocity + paddle.sign(grad) * mask_t
                next_delta = paddle.clip(delta_t + alpha * paddle.sign(velocity), -eps, eps)
                next_delta = next_delta * mask_t

                delta_t = next_delta.detach()
                delta_t.stop_gradient = False

            cand_delta = delta_t.numpy().astype(np.float32)
            cand_score = self._score_surrogate_np(
                original_norm=original_norm,
                delta_norm=cand_delta,
                mask_np=mask_np,
                target_np=target_np,
            )
            if cand_score > best_score:
                best_score = cand_score
                best_delta = cand_delta

        return (best_delta * 255.0).astype(np.float32)

    def _score_surrogate_np(
        self,
        original_norm: np.ndarray,
        delta_norm: np.ndarray,
        mask_np: np.ndarray,
        target_np: np.ndarray,
    ) -> float:
        adv_norm = np.clip(original_norm + delta_norm, 0.0, 1.0)
        diff = (adv_norm - original_norm) * mask_np

        align_term = float(np.mean(diff * target_np))
        magnitude_term = float(np.mean(np.abs(diff)))

        adv_gray = 0.114 * adv_norm[:, :, 0] + 0.587 * adv_norm[:, :, 1] + 0.299 * adv_norm[:, :, 2]
        if adv_gray.shape[0] > 1 and adv_gray.shape[1] > 1:
            gx = adv_gray[:, 1:] - adv_gray[:, :-1]
            gy = adv_gray[1:, :] - adv_gray[:-1, :]
            edge_term = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
        else:
            edge_term = 0.0

        return (
            float(self.config.pgd_align_weight) * align_term
            + float(self.config.pgd_magnitude_weight) * magnitude_term
            + float(self.config.pgd_edge_weight) * edge_term
        )

    def _collect_sensitive_bboxes(self, cells: Sequence[Dict[str, Any]]) -> List[List[float]]:
        bboxes: List[List[float]] = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue

            is_sensitive = False
            sensitives = cell.get("sensitives")
            if isinstance(sensitives, list):
                # Compatible with detector outputs that return only matched items
                # (usually no explicit "is_sensitive" field).
                is_sensitive = bool(sensitives) or any(
                    isinstance(item, dict) and item.get("is_sensitive") for item in sensitives
                )

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

        self._adaptive_ocr_model = PaddleOCR(lang=self.config.adaptive_ocr_lang, use_textline_orientation=True)
        return self._adaptive_ocr_model

    def _get_advbox_recognizer(self):
        if self._advbox_recognizer is not None:
            return self._advbox_recognizer
        from paddleocr import TextRecognition

        self._advbox_recognizer = TextRecognition(model_name=self.config.advbox_rec_model)
        return self._advbox_recognizer

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

    @staticmethod
    def _expand_bbox(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        extra: int,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        x1 = max(0, x1 - extra)
        y1 = max(0, y1 - extra)
        x2 = min(width, x2 + extra)
        y2 = min(height, y2 + extra)
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


def _scale_image_and_cells(
    image: np.ndarray,
    cells: Sequence[Dict[str, Any]],
    scale: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if abs(scale - 1.0) <= 1e-6:
        return image, [dict(c) for c in cells if isinstance(c, dict)]

    h, w = image.shape[:2]
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled_image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    scaled_cells: List[Dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        new_cell = dict(cell)
        bbox = new_cell.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            new_cell["bbox"] = [
                float(bbox[0]) * scale,
                float(bbox[1]) * scale,
                float(bbox[2]) * scale,
                float(bbox[3]) * scale,
            ]
        scaled_cells.append(new_cell)

    return scaled_image, scaled_cells


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
