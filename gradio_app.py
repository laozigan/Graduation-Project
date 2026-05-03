#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr

from src.adversarial_gen.perturbator import AttackConfig, run_attack_from_files
from src.document_processor import processor as processor_module
from src.sensitive_detection import SensitiveDetector
from src.table_extraction import load_images_from_file
from src.utils.adversarial_evaluation import AdversarialEvaluator


_RUNTIME_CACHE: Dict[str, Any] = {
    "key": None,
    "ocr": None,
    "detector": None,
    "structure_engine": None,
}


def _ensure_dirs() -> None:
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/ocr_result").mkdir(parents=True, exist_ok=True)
    Path("outputs/preprocessed").mkdir(parents=True, exist_ok=True)
    Path("outputs/adversarial").mkdir(parents=True, exist_ok=True)


def _resolve_file_path(file_input: Any) -> Optional[str]:
    if file_input is None:
        return None
    if isinstance(file_input, str):
        return file_input
    if isinstance(file_input, dict):
        name = file_input.get("name")
        if isinstance(name, str):
            return name
    name = getattr(file_input, "name", None)
    if isinstance(name, str):
        return name
    return None


def _safe_stem(file_path: str) -> str:
    stem = Path(file_path).stem.strip()
    stem = stem.replace(" ", "_")
    return stem or "document"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _find_free_port(preferred_port: int, host: str, max_tries: int = 50) -> int:
    port = max(1, int(preferred_port))
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                port += 1
                continue
        return port
    raise OSError(f"No free port found starting from {preferred_port}")


def _load_page_count(det_json_path: str) -> int:
    with open(det_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        return 0

    count = 0
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("cells"), list):
            count += 1
        elif isinstance(item, list):
            count += 1
    return count


def _build_output_paths(output: str, page_count: int) -> List[str]:
    if page_count <= 1:
        return [output]

    stem, ext = os.path.splitext(output)
    ext = ext or ".jpg"
    return [f"{stem}_p{index + 1}{ext}" for index in range(page_count)]


def _build_empty_detection_pages(page_count: int) -> List[Dict[str, Any]]:
    return [{"page": idx, "cells": []} for idx in range(page_count)]


def _get_runtime(
    lang: str,
    use_nlp: bool,
    use_uie: bool,
    uie_model: str,
    use_ppstructure: bool,
) -> Tuple[Any, SensitiveDetector, Any]:
    key = (lang, use_nlp, use_uie, uie_model, use_ppstructure)
    if _RUNTIME_CACHE.get("key") != key:
        print("Initializing OCR and sensitive detector...")
        use_gpu = processor_module.resolve_use_gpu()
        try:
            ocr = processor_module.PaddleOCR(lang=lang, use_textline_orientation=True, use_gpu=use_gpu)
        except Exception:
            # PaddleOCR version in this environment may not accept `use_gpu` kwarg.
            ocr = processor_module.PaddleOCR(lang=lang, use_textline_orientation=True)
        detector = SensitiveDetector(
            use_nlp=use_nlp,
            enable_uie=use_uie,
            uie_model=uie_model,
        )
        structure_engine = processor_module._build_structure_engine(lang, use_gpu=use_gpu) if use_ppstructure else None

        _RUNTIME_CACHE["key"] = key
        _RUNTIME_CACHE["ocr"] = ocr
        _RUNTIME_CACHE["detector"] = detector
        _RUNTIME_CACHE["structure_engine"] = structure_engine

    return (
        _RUNTIME_CACHE["ocr"],
        _RUNTIME_CACHE["detector"],
        _RUNTIME_CACHE["structure_engine"],
    )


def _summarize_detection_pages(pages: Sequence[Dict[str, Any]]) -> Tuple[List[List[Any]], Dict[str, int]]:
    rows: List[List[Any]] = []
    total_pages = 0
    total_cells = 0
    sensitive_cells = 0
    total_matches = 0

    for page in pages:
        if not isinstance(page, dict):
            continue
        total_pages += 1
        page_number = int(page.get("page", total_pages - 1)) + 1
        cells = page.get("cells", [])
        if not isinstance(cells, list):
            continue

        total_cells += len(cells)
        for cell_index, cell in enumerate(cells, start=1):
            if not isinstance(cell, dict):
                continue
            sensitive_items = cell.get("sensitives") or []
            if not sensitive_items:
                continue

            sensitive_cells += 1
            pii_types: List[str] = []
            match_counter = 0
            for item in sensitive_items:
                if not isinstance(item, dict):
                    continue
                pii_type = str(item.get("type", "unknown"))
                if pii_type not in pii_types:
                    pii_types.append(pii_type)
                details = item.get("match_details")
                if isinstance(details, list):
                    match_counter += len(details)
                else:
                    match_counter += 1

            total_matches += max(1, match_counter)
            text = str(cell.get("text", "")).replace("\n", " ").strip()
            if len(text) > 80:
                text = f"{text[:77]}..."

            rows.append(
                [
                    page_number,
                    cell_index,
                    ", ".join(pii_types) if pii_types else "unknown",
                    text,
                    max(1, match_counter),
                ]
            )

    stats = {
        "pages": total_pages,
        "cells": total_cells,
        "sensitive_cells": sensitive_cells,
        "matches": total_matches,
    }
    return rows, stats


def run_end_to_end(
    upload_file: Any,
    lang: str,
    dpi: int,
    use_nlp: bool,
    use_uie: bool,
    uie_model: str,
    use_ppstructure: bool,
    attack_method: str,
    epsilon: float,
    steps: int,
    alpha: float,
    seed: int,
    bbox_margin: int,
    line_protect_width: int,
    force_bbox_fallback: bool,
    adaptive_detect_missing_cells: bool,
    text_threshold_block_size: int,
    text_threshold_c: int,
    min_text_pixels: int,
    enable_mkldnn: bool,
    num_threads: int,
    image_scale: float,
    advbox_roi_expand: int,
    advbox_restarts: int,
    advbox_momentum: float,
    advbox_attack_name: str,
    advbox_epsilon_steps: int,
    advbox_spsa_sigma: float,
    advbox_spsa_samples: int,
    advbox_text_change_bonus: float,
    advbox_rec_model: str,
):
    input_path = _resolve_file_path(upload_file)
    if not input_path:
        return "请先上传图片或 PDF 文件。", None, [], None, [], []

    if not os.path.exists(input_path):
        return f"未找到输入文件：{input_path}", None, [], None, [], []

    try:
        _ensure_dirs()
        stamp = _timestamp()
        stem = _safe_stem(input_path)

        det_json_path = str(Path("outputs/ocr_result") / f"{stem}_{stamp}_detection.json")
        viz_path = str(Path("outputs/ocr_result") / f"{stem}_{stamp}_viz.jpg")

        ocr, detector, structure_engine = _get_runtime(
            lang=lang,
            use_nlp=use_nlp,
            use_uie=use_uie,
            uie_model=uie_model,
            use_ppstructure=use_ppstructure,
        )

        pipeline_input = processor_module._maybe_preprocess_input(input_path, lang, ocr_model=ocr)
        pages = processor_module.process_document(
            pipeline_input,
            ocr,
            detector,
            output_json=det_json_path,
            output_viz=viz_path,
            dpi=int(dpi),
            structure_engine=structure_engine,
        )

        rows, stats = _summarize_detection_pages(pages)

        page_count = _load_page_count(det_json_path)
        if page_count <= 0:
            return "检测完成，但未发现有效页面。", None, rows, det_json_path, [], []

        base_adv = str(Path("outputs/adversarial") / f"{stem}_{stamp}_adv.jpg")
        base_compare = str(Path("outputs/adversarial") / f"{stem}_{stamp}_compare.jpg")
        adv_paths = _build_output_paths(base_adv, page_count)
        compare_paths = _build_output_paths(base_compare, page_count)

        cfg = AttackConfig(
            epsilon=float(epsilon),
            alpha=float(alpha),
            steps=int(steps),
            attack_method=attack_method,
            seed=int(seed),
            bbox_margin=int(bbox_margin),
            text_threshold_block_size=int(text_threshold_block_size),
            text_threshold_c=int(text_threshold_c),
            min_text_pixels=int(min_text_pixels),
            line_protect_width=int(line_protect_width),
            force_bbox_fallback=bool(force_bbox_fallback),
            adaptive_detect_missing_cells=bool(adaptive_detect_missing_cells),
            adaptive_use_nlp=bool(use_nlp),
            enable_mkldnn=bool(enable_mkldnn),
            num_threads=int(num_threads),
            image_scale=float(image_scale),
            advbox_roi_expand=int(advbox_roi_expand),
            advbox_restarts=int(advbox_restarts),
            advbox_momentum=float(advbox_momentum),
            advbox_attack_name=advbox_attack_name,
            advbox_epsilon_steps=int(advbox_epsilon_steps),
            advbox_spsa_sigma=float(advbox_spsa_sigma),
            advbox_spsa_samples=int(advbox_spsa_samples),
            advbox_text_change_bonus=float(advbox_text_change_bonus),
            advbox_rec_model=advbox_rec_model,
        )

        saved: List[str] = []
        attack_error: Optional[str] = None
        try:
            saved = run_attack_from_files(
                input_file=input_path,
                detection_json=det_json_path,
                output_paths=adv_paths,
                config=cfg,
                dpi=int(dpi),
                orient_mode="off",
                verbose=True,
                compare_output_paths=compare_paths,
            )
        except Exception as exc:
            attack_error = str(exc)
            print(f"Warning: adversarial generation failed, keep detection outputs only. Detail: {exc}")

        gallery = [path for path in compare_paths if os.path.exists(path)]
        if not gallery:
            gallery = [path for path in saved if os.path.exists(path)]

        downloadable_files = [path for path in saved if os.path.exists(path)]

        # 评估对抗样本（若有生成结果）
        eval_report = None
        if downloadable_files and pages:
            try:
                import cv2 as _cv2
                import numpy as _np
                from src.table_extraction import TableExtractor as _TableExtractor

                imgs = load_images_from_file(input_path, dpi=int(dpi), poppler_path=getattr(processor_module, "poppler_path", None))
                extractor = _TableExtractor(ocr, poppler_path=getattr(processor_module, "poppler_path", None))

                per_page = []
                detect_comparisons = []
                for idx, adv_path in enumerate(downloadable_files):
                    try:
                        adv_img = _cv2.imread(adv_path)
                        orig_img = imgs[idx] if idx < len(imgs) else None
                        orig_cells = pages[idx].get("cells", []) if idx < len(pages) else []
                        adv_cells = extractor.extract_from_image(adv_img) if adv_img is not None else []
                        metrics = AdversarialEvaluator.evaluate_page(orig_img, adv_img, orig_cells, adv_cells)
                        per_page.append(metrics)
                        # compute detection comparison: run sensitive detector on adv_cells
                        adv_cells_with_sens = []
                        for a in adv_cells:
                            text = a.get('text', '')
                            sens = detector.detect_all(text)
                            new = dict(a)
                            new['sensitives'] = sens
                            adv_cells_with_sens.append(new)
                        detect_comp = AdversarialEvaluator.detection_consistency(orig_cells, adv_cells_with_sens)
                        detect_comparisons.append(detect_comp)
                    except Exception as e:
                        per_page.append({"error": str(e)})
                        detect_comparisons.append({"error": str(e)})

                # 汇总数值型指标
                agg = {}
                if per_page:
                    numeric_keys = [k for k, v in per_page[0].items() if isinstance(v, (int, float))]
                    for k in numeric_keys:
                        vals = [p[k] for p in per_page if isinstance(p.get(k), (int, float))]
                        agg[k] = float(_np.mean(vals)) if vals else None

                eval_report = {"per_page": per_page, "aggregate": agg}
                # 构建表格行: [页码, PSNR, SSIM, 敏感CER, 表格完整度]
                eval_table_rows = []
                detection_comparison_rows = []
                for i, p in enumerate(per_page):
                    if isinstance(p, dict) and p.get("error"):
                        eval_table_rows.append([i + 1, None, None, None, None])
                        detection_comparison_rows.append([i + 1] + [None] * 9)
                    else:
                        eval_table_rows.append([
                            i + 1,
                            p.get("psnr"),
                            p.get("ssim"),
                            p.get("sensitive_cer"),
                            p.get("table_integrity_iou"),
                        ])
                        # detection comparison
                        dc = detect_comparisons[i] if i < len(detect_comparisons) else None
                        if isinstance(dc, dict) and dc.get("post"):
                            pre = dc.get("pre", {})
                            post = dc.get("post", {})
                            delta = dc.get("delta", {})
                            detection_comparison_rows.append([
                                i + 1,
                                pre.get("precision"),
                                post.get("precision"),
                                delta.get("precision"),
                                pre.get("recall"),
                                post.get("recall"),
                                delta.get("recall"),
                                pre.get("f1"),
                                post.get("f1"),
                                delta.get("f1"),
                            ])
                        else:
                            detection_comparison_rows.append([i + 1] + [None] * 9)
                # 构建表格行: [页码, PSNR, SSIM, 敏感CER, 表格完整度]
                eval_table_rows = []
                for i, p in enumerate(per_page):
                    if isinstance(p, dict) and p.get("error"):
                        eval_table_rows.append([i + 1, None, None, None, None])
                    else:
                        eval_table_rows.append([
                            i + 1,
                            p.get("psnr"),
                            p.get("ssim"),
                            p.get("sensitive_cer"),
                            p.get("table_integrity_iou"),
                        ])
            except Exception as exc:
                eval_report = {"error": str(exc)}
                eval_table_rows = []
            else:
                eval_report = {"note": "no adversarial files generated to evaluate"}
                eval_table_rows = []
                detection_comparison_rows = []

        status_prefix = "处理完成。"
        if attack_error:
            status_prefix = f"检测已完成，但对抗样本生成失败：{attack_error}。"
        status = (
            f"{status_prefix}页数={stats['pages']}，文本块={stats['cells']}，"
            f"敏感文本块={stats['sensitive_cells']}，命中项={stats['matches']}，"
            f"对抗样本文件数={len(downloadable_files)}"
        )

        # 构建前端展示的汇总 HTML（格式化并高亮 delta）
        def fmt(val, ndigits=4):
            try:
                if val is None:
                    return "-"
                if isinstance(val, float):
                    return f"{val:.{ndigits}f}"
                return str(val)
            except Exception:
                return str(val)

        if isinstance(eval_report, dict) and eval_report.get("aggregate"):
            agg = eval_report.get("aggregate", {})
            psnr = agg.get("psnr")
            ssim_v = agg.get("ssim")
            cer = agg.get("sensitive_cer")
            table_iou = agg.get("table_integrity_iou")
        else:
            psnr = ssim_v = cer = table_iou = None

        # 从 detection_comparison_rows 计算整体的 post 与 delta 的平均值
        def avg_of_column(rows, idx):
            vals = [r[idx] for r in rows if isinstance(r, list) and r[idx] is not None]
            return float(_np.mean(vals)) if vals else None

        det_rows = detection_comparison_rows if 'detection_comparison_rows' in locals() else []
        post_prec = avg_of_column(det_rows, 2)
        delta_prec = avg_of_column(det_rows, 3)
        post_rec = avg_of_column(det_rows, 5)
        delta_rec = avg_of_column(det_rows, 6)
        post_f1 = avg_of_column(det_rows, 8)
        delta_f1 = avg_of_column(det_rows, 9)

        def color_span(value):
            if value is None:
                return "-"
            try:
                v = float(value)
            except Exception:
                return str(value)
            color = "black"
            if v < 0:
                color = "red"
            elif v > 0:
                color = "green"
            return f"<span style=\"color:{color};font-weight:600\">{v:+.4f}</span>"

        html = []
        html.append("<div style='font-family:Arial, Helvetica, sans-serif'>")
        html.append("<h3>视觉与保真度（聚合）</h3>")
        html.append("<table style='border-collapse:collapse'>")
        html.append("<tr><td style='padding:6px'>PSNR</td><td style='padding:6px'>%s</td></tr>" % fmt(psnr, 2))
        html.append("<tr><td style='padding:6px'>SSIM</td><td style='padding:6px'>%s</td></tr>" % fmt(ssim_v, 4))
        html.append("<tr><td style='padding:6px'>敏感文本 CER</td><td style='padding:6px'>%s</td></tr>" % fmt(cer, 4))
        html.append("<tr><td style='padding:6px'>表格完整度 (IoU)</td><td style='padding:6px'>%s</td></tr>" % fmt(table_iou, 4))
        html.append("</table>")

        html.append("<h3>识别效果对比（整体）</h3>")
        html.append("<table style='border-collapse:collapse'>")
        html.append("<tr><th style='padding:6px;text-align:left'>指标</th><th style='padding:6px'>Pre</th><th style='padding:6px'>Post</th><th style='padding:6px'>Delta</th></tr>")
        # Precision
        html.append("<tr><td style='padding:6px'>Precision</td><td style='padding:6px'>%s</td><td style='padding:6px'>%s</td><td style='padding:6px'>%s</td></tr>" % (fmt(1.0,4), fmt(post_prec,4), color_span(delta_prec)))
        html.append("<tr><td style='padding:6px'>Recall</td><td style='padding:6px'>%s</td><td style='padding:6px'>%s</td><td style='padding:6px'>%s</td></tr>" % (fmt(1.0,4), fmt(post_rec,4), color_span(delta_rec)))
        html.append("<tr><td style='padding:6px'>F1</td><td style='padding:6px'>%s</td><td style='padding:6px'>%s</td><td style='padding:6px'>%s</td></tr>" % (fmt(1.0,4), fmt(post_f1,4), color_span(delta_f1)))
        html.append("</table>")

        if isinstance(eval_report, dict) and eval_report.get("note"):
            html.append("<p style='color:gray'>%s</p>" % eval_report.get("note"))

        html.append("</div>")
        evaluation_html = "\n".join(html)

        return status, det_json_path, gallery, downloadable_files, evaluation_html, eval_report
    except Exception as exc:
        return f"流程执行失败：{exc}", None, [], None, [], []


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="基于对抗样本的表格图像隐私保护系统") as demo:
        gr.Markdown(
            "# 基于对抗样本的表格图像隐私保护系统\n"
            "上传一张图片或 PDF，完成端到端处理：图像预处理 -> 表格结构提取 -> 敏感信息检测 -> 对抗样本生成。"
        )

        with gr.Tab("一键处理流程"):
            input_file = gr.File(label="上传文件（图片/PDF）", type="filepath")

            with gr.Row():
                lang = gr.Dropdown(choices=["ch", "en"], value="ch", label="OCR 语言")
                dpi = gr.Slider(120, 400, value=200, step=10, label="分辨率（DPI）")

            with gr.Row():
                use_nlp = gr.Checkbox(value=True, label="启用 NLP")
                use_uie = gr.Checkbox(value=True, label="启用 UIE")
                use_ppstructure = gr.Checkbox(value=True, label="启用 PPStructure")
                uie_model = gr.Textbox(value="uie-x-base", label="UIE 模型")

            with gr.Row():
                attack_method = gr.Dropdown(
                    choices=["random", "pgd", "advbox_roi"],
                    value="pgd",
                    label="攻击方法",
                )
                epsilon = gr.Slider(1, 64, value=24, step=1, label="扰动强度（epsilon）")
                steps = gr.Slider(1, 30, value=9, step=1, label="迭代步数（steps）")

            with gr.Accordion("高级参数", open=False):
                with gr.Row():
                    alpha = gr.Slider(1, 24, value=6, step=1, label="步长（alpha）")
                    seed = gr.Number(value=2026, precision=0, label="随机种子")
                    bbox_margin = gr.Slider(0, 12, value=2, step=1, label="边界留白（bbox_margin）")
                    line_protect_width = gr.Slider(0, 8, value=2, step=1, label="线条保护宽度")

                with gr.Row():
                    force_bbox_fallback = gr.Checkbox(value=True, label="启用 bbox 兜底")
                    adaptive_detect_missing_cells = gr.Checkbox(value=True, label="启用缺失单元格自适应")
                    text_threshold_block_size = gr.Slider(3, 61, value=25, step=2, label="文本阈值块大小")
                    text_threshold_c = gr.Slider(0, 32, value=8, step=1, label="文本阈值偏移")

                with gr.Row():
                    min_text_pixels = gr.Slider(1, 200, value=20, step=1, label="最小可扰动像素数")
                    enable_mkldnn = gr.Checkbox(value=False, label="启用 MKLDNN")
                    num_threads = gr.Slider(0, 16, value=0, step=1, label="CPU 线程数")
                    image_scale = gr.Slider(0.5, 1.0, value=1.0, step=0.05, label="攻击缩放比例")

                with gr.Row():
                    advbox_roi_expand = gr.Slider(0, 32, value=8, step=1, label="AdvBox ROI 扩展")
                    advbox_restarts = gr.Slider(1, 10, value=3, step=1, label="AdvBox 重启次数")
                    advbox_momentum = gr.Slider(0.0, 0.95, value=0.8, step=0.05, label="AdvBox 动量")
                    advbox_attack_name = gr.Dropdown(choices=["PGD", "FGSM", "BIM", "MIFGSM"], value="PGD", label="AdvBox 攻击类型")

                with gr.Row():
                    advbox_epsilon_steps = gr.Slider(1, 12, value=6, step=1, label="AdvBox epsilon 步数")
                    advbox_spsa_sigma = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label="AdvBox SPSA sigma")
                    advbox_spsa_samples = gr.Slider(1, 16, value=4, step=1, label="AdvBox SPSA 样本数")
                    advbox_text_change_bonus = gr.Slider(0.0, 2.0, value=0.5, step=0.1, label="文本变化奖励")

                advbox_rec_model = gr.Textbox(value="PP-OCRv5_server_rec", label="AdvBox 识别模型")

            run_pipeline_btn = gr.Button("运行端到端流程", variant="primary")

            pipeline_status = gr.Textbox(label="运行状态", interactive=False)
            detection_json_file = gr.File(label="检测结果 JSON")
            attack_gallery = gr.Gallery(label="预览图", show_label=True, columns=2, height=360)
            attack_files = gr.Files(label="对抗样本输出文件")
            evaluation_summary = gr.HTML(label="评估汇总")
            evaluation_report = gr.JSON(label="评估结果")

            run_pipeline_btn.click(
                fn=run_end_to_end,
                inputs=[
                    input_file,
                    lang,
                    dpi,
                    use_nlp,
                    use_uie,
                    uie_model,
                    use_ppstructure,
                    attack_method,
                    epsilon,
                    steps,
                    alpha,
                    seed,
                    bbox_margin,
                    line_protect_width,
                    force_bbox_fallback,
                    adaptive_detect_missing_cells,
                    text_threshold_block_size,
                    text_threshold_c,
                    min_text_pixels,
                    enable_mkldnn,
                    num_threads,
                    image_scale,
                    advbox_roi_expand,
                    advbox_restarts,
                    advbox_momentum,
                    advbox_attack_name,
                    advbox_epsilon_steps,
                    advbox_spsa_sigma,
                    advbox_spsa_samples,
                    advbox_text_change_bonus,
                    advbox_rec_model,
                ],
                outputs=[
                    pipeline_status,
                    detection_json_file,
                    attack_gallery,
                    attack_files,
                    evaluation_summary,
                    evaluation_report,
                ],
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gradio frontend for PrivacyProtectionSystem")
    parser.add_argument("--host", default="0.0.0.0", help="Host for Gradio server")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo()
    demo.queue(max_size=16)
    launch_port = _find_free_port(args.port, args.host)
    if launch_port != args.port:
        print(f"Port {args.port} is busy, falling back to {launch_port}.")
    demo.launch(server_name=args.host, server_port=launch_port, share=args.share)


if __name__ == "__main__":
    main()
