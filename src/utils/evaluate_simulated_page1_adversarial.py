#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from paddleocr import TextRecognition

from src.adversarial_gen.perturbator import (
    AttackConfig,
    _normalize_right_angle,
    _rotate_image_and_cells,
    run_attack_from_files,
)
from src.image_preprocessing import ImagePreprocessor, PreprocessConfig
from src.sensitive_detection import SensitiveDetector
from src.utils.evaluation import SensitiveDetectionEvaluator


@dataclass
class EvalPaths:
    base_dir: Path = Path("outputs/ocr_result")
    source_pdf: Path = Path("data/dataset/sample/resume_sample_20200120/pdf/simulated_data.pdf")
    gt_full: Path = Path("data/annotations/simulated_data_ground_truth_template.json")
    page1_image: Path = Path("outputs/ocr_result/simulated_data_page1_original.jpg")
    page1_gt: Path = Path("outputs/ocr_result/simulated_data_page1_ground_truth.json")
    page1_oriented_image: Path = Path("outputs/ocr_result/simulated_data_page1_oriented_eval.jpg")
    page1_oriented_gt: Path = Path("outputs/ocr_result/simulated_data_page1_oriented_ground_truth.json")
    before_pred: Path = Path("outputs/ocr_result/simulated_data_page1_before_pred_by_gtcrop.json")
    after_pred: Path = Path("outputs/ocr_result/simulated_data_page1_after_pred_by_gtcrop.json")
    adv_image: Path = Path("outputs/ocr_result/simulated_data_page1_adv_eval.jpg")
    compare_image: Path = Path("outputs/ocr_result/simulated_data_page1_compare_eval.jpg")
    report_txt: Path = Path("outputs/ocr_result/simulated_data_page1_eval_report.txt")
    report_json: Path = Path("outputs/ocr_result/simulated_data_page1_eval_summary.json")


def _extract_page1_image_and_gt(paths: EvalPaths, dpi: int) -> None:
    from src.table_extraction import load_images_from_file

    paths.base_dir.mkdir(parents=True, exist_ok=True)

    images = load_images_from_file(
        str(paths.source_pdf),
        dpi=dpi,
        poppler_path="D:/PrivacyProtectionSystem/poppler/Library/bin",
    )
    if not images:
        raise RuntimeError("No page loaded from simulated_data.pdf")

    if not cv2.imwrite(str(paths.page1_image), images[0]):
        raise RuntimeError(f"Failed to save page1 image: {paths.page1_image}")

    payload = json.loads(paths.gt_full.read_text(encoding="utf-8"))
    page0 = None
    for item in payload:
        if isinstance(item, dict) and int(item.get("page", -1)) == 0:
            page0 = item
            break
    if page0 is None:
        raise RuntimeError("page=0 not found in ground truth template")

    paths.page1_gt.write_text(json.dumps([page0], ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_oriented_page_and_gt(paths: EvalPaths, orient_mode: str) -> Tuple[float, str, int]:
    image = cv2.imread(str(paths.page1_image))
    if image is None:
        raise RuntimeError(f"Cannot read image: {paths.page1_image}")

    payload = json.loads(paths.page1_gt.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError("Invalid page1 GT payload")

    page = payload[0]
    cells = page.get("cells", [])
    if not isinstance(cells, list):
        raise RuntimeError("GT cells must be a list")

    if orient_mode == "off":
        if not cv2.imwrite(str(paths.page1_oriented_image), image):
            raise RuntimeError(f"Failed to save oriented image: {paths.page1_oriented_image}")
        paths.page1_oriented_gt.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0.0, "disabled", 0

    if orient_mode not in {"upside_down", "always"}:
        raise ValueError("orient_mode must be one of: off, upside_down, always")

    orienter = ImagePreprocessor(
        config=PreprocessConfig(
            use_ocr_orientation=True,
            enable_perspective_correction=False,
            enable_denoise=False,
        )
    )
    _, orientation_angle, orientation_method = orienter.correct_orientation(image)
    right_angle = _normalize_right_angle(orientation_angle)
    apply_angle = right_angle if orient_mode == "always" else (180 if right_angle == 180 else 0)

    oriented_img, oriented_cells = _rotate_image_and_cells(image, cells, apply_angle)
    if not cv2.imwrite(str(paths.page1_oriented_image), oriented_img):
        raise RuntimeError(f"Failed to save oriented image: {paths.page1_oriented_image}")

    oriented_page = {"page": 0, "cells": oriented_cells}
    paths.page1_oriented_gt.write_text(
        json.dumps([oriented_page], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return orientation_angle, orientation_method, apply_angle


def _parse_rec_texts(pred_output: Any) -> str:
    if not isinstance(pred_output, (list, tuple)):
        return ""
    cleaned: List[str] = []
    for item in pred_output:
        if isinstance(item, dict):
            txt = str(item.get("rec_text", "")).strip()
            if txt:
                cleaned.append(txt)
    return " ".join(cleaned)


def _predict_cells_by_gt_bbox(image_path: Path, gt_json: Path, output_json: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    payload = json.loads(gt_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError("Invalid page1 GT json")

    page = payload[0]
    gt_cells = page.get("cells", [])
    if not isinstance(gt_cells, list):
        raise RuntimeError("GT page has no cells list")

    ocr = TextRecognition(model_name="PP-OCRv5_server_rec")
    detector = SensitiveDetector(use_nlp=True, enable_uie=False)

    h, w = image.shape[:2]
    pred_cells: List[Dict[str, Any]] = []

    for cell in gt_cells:
        bbox = cell.get("bbox", [0, 0, 0, 0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            bbox = [0, 0, 0, 0]

        x1 = max(0, min(w, int(bbox[0]) - 2))
        y1 = max(0, min(h, int(bbox[1]) - 2))
        x2 = max(0, min(w, int(bbox[2]) + 2))
        y2 = max(0, min(h, int(bbox[3]) + 2))

        text = ""
        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]
            try:
                pred = ocr.predict(crop)
                text = _parse_rec_texts(pred)
            except Exception:
                text = ""

        sensitives = detector.detect_all(text)
        sensitive = detector._summarize_results(sensitives)

        pred_cells.append(
            {
                "bbox": [float(v) for v in bbox[:4]],
                "text": text,
                "sensitives": sensitives,
                "sensitive": sensitive,
            }
        )

    output_json.write_text(
        json.dumps([{"page": 0, "cells": pred_cells}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _metrics_pair(result: Dict[str, Any]) -> Tuple[float, float, float]:
    overall = result.get("overall_metrics", {}) if isinstance(result, dict) else {}
    return (
        float(overall.get("precision", 0.0)),
        float(overall.get("recall", 0.0)),
        float(overall.get("f1_score", 0.0)),
    )


def _write_report(
    before: Dict[str, Any],
    after: Dict[str, Any],
    paths: EvalPaths,
    attack_cfg: AttackConfig,
    orientation_angle: float,
    orientation_method: str,
    right_angle: int,
) -> None:
    b_p, b_r, b_f1 = _metrics_pair(before)
    a_p, a_r, a_f1 = _metrics_pair(after)

    d_p = a_p - b_p
    d_r = a_r - b_r
    d_f1 = a_f1 - b_f1

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("simulated_data 第1页 对抗扰动前后敏感识别评估报告")
    lines.append("=" * 72)
    lines.append("")
    lines.append("[实验设置]")
    lines.append(f"- 输入PDF: {paths.source_pdf.as_posix()}")
    lines.append(f"- 第1页原图: {paths.page1_image.as_posix()}")
    lines.append(f"- 第1页转正图: {paths.page1_oriented_image.as_posix()}")
    lines.append(f"- 第1页转正GT: {paths.page1_oriented_gt.as_posix()}")
    lines.append(
        f"- 方向校正: method={orientation_method}, angle={orientation_angle:.1f}, applied_right_angle={right_angle}"
    )
    lines.append("- OCR方式: TextRecognition 基于GT bbox逐格识别")
    lines.append("- 敏感检测: SensitiveDetector(use_nlp=True, enable_uie=False)")
    lines.append(
        f"- 扰动参数: epsilon={attack_cfg.epsilon}, alpha={attack_cfg.alpha}, steps={attack_cfg.steps}, "
        f"bbox_margin={attack_cfg.bbox_margin}, line_protect_width={attack_cfg.line_protect_width}"
    )
    lines.append("")
    lines.append("[总体指标对比]")
    lines.append(f"- 扰动前 Precision: {b_p:.4f}")
    lines.append(f"- 扰动后 Precision: {a_p:.4f}")
    lines.append(f"- Precision 变化: {d_p:+.4f}")
    lines.append("")
    lines.append(f"- 扰动前 Recall: {b_r:.4f}")
    lines.append(f"- 扰动后 Recall: {a_r:.4f}")
    lines.append(f"- Recall 变化: {d_r:+.4f}")
    lines.append("")
    lines.append(f"- 扰动前 F1: {b_f1:.4f}")
    lines.append(f"- 扰动后 F1: {a_f1:.4f}")
    lines.append(f"- F1 变化: {d_f1:+.4f}")
    lines.append("")

    lines.append("[样本统计]")
    b_summary = before.get("summary", {}) if isinstance(before, dict) else {}
    a_summary = after.get("summary", {}) if isinstance(after, dict) else {}
    lines.append(
        "- 扰动前: "
        f"samples={b_summary.get('total_samples', 0)}, "
        f"pred_sensitive={b_summary.get('total_sensitive_predicted', 0)}, "
        f"actual_sensitive={b_summary.get('total_sensitive_actual', 0)}, "
        f"correct={b_summary.get('correct_sensitive_predictions', 0)}"
    )
    lines.append(
        "- 扰动后: "
        f"samples={a_summary.get('total_samples', 0)}, "
        f"pred_sensitive={a_summary.get('total_sensitive_predicted', 0)}, "
        f"actual_sensitive={a_summary.get('total_sensitive_actual', 0)}, "
        f"correct={a_summary.get('correct_sensitive_predictions', 0)}"
    )
    lines.append("")

    lines.append("[关键输出文件]")
    lines.append(f"- 扰动前预测: {paths.before_pred.as_posix()}")
    lines.append(f"- 扰动后预测: {paths.after_pred.as_posix()}")
    lines.append(f"- 扰动图: {paths.adv_image.as_posix()}")
    lines.append(f"- 转正图: {paths.page1_oriented_image.as_posix()}")
    lines.append(f"- 对比图: {paths.compare_image.as_posix()}")

    paths.report_txt.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "before": before,
        "after": after,
        "delta": {
            "precision": round(d_p, 4),
            "recall": round(d_r, 4),
            "f1_score": round(d_f1, 4),
        },
        "files": {
            "before_pred": paths.before_pred.as_posix(),
            "after_pred": paths.after_pred.as_posix(),
            "adv_image": paths.adv_image.as_posix(),
            "oriented_image": paths.page1_oriented_image.as_posix(),
            "compare_image": paths.compare_image.as_posix(),
            "report_txt": paths.report_txt.as_posix(),
        },
        "orientation": {
            "method": orientation_method,
            "angle": round(float(orientation_angle), 2),
            "applied_right_angle": int(right_angle),
        },
    }
    paths.report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simulated_data page1 before/after adversarial perturbation")
    parser.add_argument("--dpi", type=int, default=72, help="PDF to image DPI; default 72 to match existing GT scale")
    parser.add_argument(
        "--orient-mode",
        choices=["off", "upside_down", "always"],
        default="off",
        help="Orientation strategy for evaluation inputs",
    )
    args = parser.parse_args()

    paths = EvalPaths()

    attack_cfg = AttackConfig(
        epsilon=36.0,
        alpha=9.0,
        steps=12,
        seed=2026,
        bbox_margin=0,
        line_protect_width=1,
        force_bbox_fallback=True,
        adaptive_detect_missing_cells=True,
    )

    _extract_page1_image_and_gt(paths, dpi=args.dpi)
    orientation_angle, orientation_method, right_angle = _prepare_oriented_page_and_gt(paths, orient_mode=args.orient_mode)

    _predict_cells_by_gt_bbox(paths.page1_oriented_image, paths.page1_oriented_gt, paths.before_pred)

    run_attack_from_files(
        input_file=str(paths.page1_oriented_image),
        detection_json=str(paths.page1_oriented_gt),
        output_paths=[str(paths.adv_image)],
        config=attack_cfg,
        dpi=200,
        orient_mode="off",
        verbose=True,
        compare_output_paths=[str(paths.compare_image)],
    )

    _predict_cells_by_gt_bbox(paths.adv_image, paths.page1_oriented_gt, paths.after_pred)

    evaluator = SensitiveDetectionEvaluator()
    before_eval = evaluator.evaluate_from_files(str(paths.before_pred), str(paths.page1_oriented_gt))
    after_eval = evaluator.evaluate_from_files(str(paths.after_pred), str(paths.page1_oriented_gt))

    _write_report(before_eval, after_eval, paths, attack_cfg, orientation_angle, orientation_method, right_angle)

    print("[done] report:", paths.report_txt.as_posix())
    print("[done] summary:", paths.report_json.as_posix())


if __name__ == "__main__":
    main()
