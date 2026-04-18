#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档信息提取工具（支持 PDF / 图片）
功能：
1. 若输入为图片，先执行图像预处理（透视/方向/倾斜/增强）
2. 读取 PDF 或图片文件
3. 对每页进行 OCR 和表格结构分析
4. 提取单元格内容
5. 检测其中的敏感信息（身份证、手机号、银行卡、邮箱、姓名、地址等）
6. 输出 JSON 结果并可选生成标注图像
"""

import os
import re
import json
import argparse
import sys
from typing import List, Dict, Tuple, Optional

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


ENABLE_MKLDNN = _env_flag('PPS_ENABLE_MKLDNN', False)
os.environ['FLAGS_use_mkldnn'] = '1' if ENABLE_MKLDNN else '0'
os.environ['FLAGS_use_mkldnn_common_opt'] = '1' if ENABLE_MKLDNN else '0'
os.environ['FLAGS_enable_pir_api'] = '0'  # 禁用 PIR API 以避免兼容性问题
os.environ['FLAGS_enable_pir_in_executor'] = '0'  # 关闭 PIR executor，兼容 Taskflow UIE 静态导出

import cv2
import numpy as np
from paddleocr import PaddleOCR
try:
    from paddleocr import PPStructure
except Exception:
    PPStructure = None
try:
    from paddleocr import PPStructureV3
except Exception:
    PPStructureV3 = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.sensitive_detection import SensitiveDetector
from src.table_extraction import TableExtractor, load_images_from_file
from src.image_preprocessing import ImagePreprocessor, PreprocessConfig, SUPPORTED_IMAGE_EXTENSIONS

# 检查poppler路径是否存在
poppler_path = "D:/PrivacyProtectionSystem/poppler/Library/bin"
if not os.path.exists(poppler_path):
    poppler_path = None  # 使用系统默认路径

# ========== 1. 敏感信息检测模块（内嵌，也可独立导入） ==========

# ========== 2. 表格信息提取模块 ==========
# 该功能已移到 src.table_extraction.extractor 模块。

# ========== 3. 文件加载（图片 / PDF） ==========
# 文件加载逻辑同样由 src.table_extraction.extractor 管理。

# ========== 4. 主处理流程 ==========
def _is_image_file(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def _build_preprocessed_path(input_path: str) -> str:
    stem, _ = os.path.splitext(os.path.basename(input_path))
    return os.path.join("outputs", "preprocessed", f"{stem}_pipeline_preprocessed.jpg")


def _maybe_preprocess_input(file_path: str, lang: str, ocr_model: Optional[PaddleOCR] = None) -> str:
    """图片输入时先做预处理，PDF直接返回原路径。"""
    if not _is_image_file(file_path):
        return file_path

    output_path = _build_preprocessed_path(file_path)
    print(f"检测到图片输入，开始预处理: {file_path}")
    config = PreprocessConfig(lang=lang, use_ocr_orientation=True)
    preprocessor = ImagePreprocessor(config=config, ocr_model=ocr_model)
    result = preprocessor.preprocess_file(file_path, output_path)
    print(
        "图片预处理完成: "
        f"output={result.output_path}, perspective={result.perspective_method}, "
        f"orientation={result.orientation_angle}, skew={result.skew_angle}"
    )
    return result.output_path


def _build_structure_engine(lang: str):
    """Build PPStructure engine if available; return None on failure."""
    if PPStructure is None and PPStructureV3 is None:
        print("Warning: PPStructure is unavailable. Continue with OCR clustering only.")
        return None

    if PPStructureV3 is not None:
        try:
            return PPStructureV3(
                lang=lang,
                use_chart_recognition=False,
                use_formula_recognition=False,
                use_seal_recognition=False,
            )
        except Exception as exc:
            print(f"Warning: PPStructureV3 init failed. Try legacy PPStructure. Detail: {exc}")

    if PPStructure is None:
        print("Warning: Legacy PPStructure is unavailable. Continue with OCR clustering only.")
        return None
    try:
        return PPStructure(show_log=False, lang=lang, ocr=True, layout=True, table=True)
    except TypeError:
        # Compatibility fallback for older constructor signatures.
        return PPStructure(lang=lang)
    except Exception as exc:
        print(f"Warning: PPStructure init failed. Continue without it. Detail: {exc}")
        return None


def _normalize_bbox(raw_bbox) -> Optional[List[float]]:
    if raw_bbox is None:
        return None
    arr = np.array(raw_bbox)
    if arr.size == 0:
        return None
    if arr.ndim == 2 and arr.shape[1] == 2:
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        return [float(x_min), float(y_min), float(x_max), float(y_max)]
    flat = arr.flatten()
    if flat.size >= 4:
        x1, y1, x2, y2 = flat[:4]
        return [float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]
    return None


def _collect_structure_cells(node, sink: List[Dict]) -> None:
    if isinstance(node, dict):
        text = node.get("text") or node.get("transcription")
        bbox = _normalize_bbox(node.get("bbox") or node.get("box") or node.get("points"))
        if isinstance(text, str) and text.strip() and bbox is not None:
            sink.append({"bbox": bbox, "text": text.strip(), "source": "ppstructure"})
        for value in node.values():
            _collect_structure_cells(value, sink)
        return
    if isinstance(node, list):
        for item in node:
            _collect_structure_cells(item, sink)


def _extract_ppstructure_cells(img: np.ndarray, structure_engine) -> List[Dict]:
    if structure_engine is None:
        return []
    try:
        result = structure_engine(img)
    except Exception as exc:
        print(f"Warning: PPStructure inference failed on page. Detail: {exc}")
        return []

    raw_cells: List[Dict] = []
    _collect_structure_cells(result, raw_cells)

    unique_cells: List[Dict] = []
    seen = set()
    for cell in raw_cells:
        key = (
            cell["text"],
            round(cell["bbox"][0], 1),
            round(cell["bbox"][1], 1),
            round(cell["bbox"][2], 1),
            round(cell["bbox"][3], 1),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_cells.append(cell)
    return unique_cells


def _bbox_iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _merge_cells_with_structure(ocr_cells: List[Dict], structure_cells: List[Dict], iou_threshold: float = 0.65) -> List[Dict]:
    merged = [dict(cell) for cell in ocr_cells]
    for s_cell in structure_cells:
        best_idx = -1
        best_iou = 0.0
        for idx, base_cell in enumerate(merged):
            iou = _bbox_iou(base_cell["bbox"], s_cell["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            base_text = str(merged[best_idx].get("text", "")).strip()
            struct_text = str(s_cell.get("text", "")).strip()
            if struct_text and len(struct_text) > len(base_text):
                merged[best_idx]["text"] = struct_text
                merged[best_idx]["source"] = "fusion"
            continue

        merged.append(s_cell)
    return merged


def process_document(file_path: str,
                     ocr_model: PaddleOCR,
                     detector: SensitiveDetector,
                     output_json: str = None,
                     output_viz: str = None,
                     dpi: int = 200,
                     structure_engine=None) -> List[Dict]:
    """
    处理文档，返回每页的提取结果。
    结果格式：
    [
        {
            "page": 0,
            "cells": [
                {"bbox": [x1,y1,x2,y2], "text": "...", "sensitives": [...]},
                ...
            ]
        },
        ...
    ]
    """
    print(f"正在加载文件: {file_path}")
    images = load_images_from_file(file_path, dpi=dpi, poppler_path=poppler_path)
    print(f"共 {len(images)} 页")

    all_pages = []
    extractor = TableExtractor(ocr_model, poppler_path=poppler_path)
    for page_idx, img in enumerate(images):
        print(f"处理第 {page_idx+1} 页...")

        # 1. 表格单元格提取
        print("开始表格单元格提取...")
        cells = extractor.extract_from_image(img)
        print(f"提取到 {len(cells)} 个候选单元格")

        if structure_engine is not None:
            structured_cells = _extract_ppstructure_cells(img, structure_engine)
            if structured_cells:
                before_merge = len(cells)
                cells = _merge_cells_with_structure(cells, structured_cells)
                print(f"PPStructure 增强后单元格数: {before_merge} -> {len(cells)}")

        if not cells:
            print("OCR与PPStructure均无结果，跳过此页")
            all_pages.append({"page": page_idx, "cells": []})
            continue

        # 2. 检测敏感信息
        for cell in cells:
            sensitives = detector.detect_all(cell['text'])
            cell['sensitives'] = sensitives

        # 5. 可选：生成标注图像
        if output_viz and page_idx == 0:  # 仅第一页示例
            viz = img.copy()
            for cell in cells:
                summary = detector._summarize_results(cell.get('sensitives', []))
                x1,y1,x2,y2 = map(int, cell['bbox'])
                color = (0,0,255) if summary['is_sensitive'] else (0,255,0)
                cv2.rectangle(viz, (x1,y1), (x2,y2), color, 2)
                # 添加标签
                if summary['is_sensitive']:
                    label = summary.get('type', 'sensitive')
                    cv2.putText(viz, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.imwrite(output_viz, viz)
            print(f"标注图像已保存: {output_viz}")

        all_pages.append({
            "page": page_idx,
            "cells": cells
        })

    # 输出 JSON
    if output_json:
        # 将 numpy 类型转换为原生 Python 类型
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(convert(all_pages), f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {output_json}")

    return all_pages

# ========== 5. 主入口 ==========
def main():
    parser = argparse.ArgumentParser(description="文档表格信息提取与敏感信息检测")
    parser.add_argument("input", help="输入文件路径（图片或PDF）")
    parser.add_argument("--json", "-j", default="output.json", help="输出JSON文件路径")
    parser.add_argument("--viz", "-v", default=None, help="输出标注图像路径（可选，仅第一页）")
    parser.add_argument("--dpi", type=int, default=200, help="PDF转换分辨率（默认200）")
    parser.add_argument("--lang", default="ch", help="OCR语言（ch/en等，默认ch）")
    parser.add_argument("--use-nlp", dest="use_nlp", action="store_true", help="启用基于jieba的NLP敏感信息检测增强")
    parser.add_argument("--no-nlp", dest="use_nlp", action="store_false", help="禁用NLP增强，仅使用规则检测")
    parser.add_argument("--use-uie", dest="use_uie", action="store_true", help="启用 UIE-X 语义抽取（姓名等）")
    parser.add_argument("--no-uie", dest="use_uie", action="store_false", help="禁用 UIE-X，仅使用规则/NLP")
    parser.add_argument("--uie-model", default="uie-x-base", help="UIE 模型名，默认 uie-x-base")
    parser.add_argument("--use-ppstructure", dest="use_ppstructure", action="store_true", help="启用 PPStructure 结构增强")
    parser.add_argument("--no-ppstructure", dest="use_ppstructure", action="store_false", help="禁用 PPStructure 结构增强")
    parser.set_defaults(use_nlp=True)
    parser.set_defaults(use_uie=True)
    parser.set_defaults(use_ppstructure=True)
    args = parser.parse_args()

    # 初始化 OCR（CPU模式）
    print("初始化 PaddleOCR...")
    print(f"MKLDNN: {'enabled' if ENABLE_MKLDNN else 'disabled'} (set PPS_ENABLE_MKLDNN=1 to enable)")
    ocr = PaddleOCR(lang=args.lang, use_angle_cls=True)
    detector = SensitiveDetector(
        use_nlp=args.use_nlp,
        enable_uie=args.use_uie,
        uie_model=args.uie_model,
    )
    structure_engine = _build_structure_engine(args.lang) if args.use_ppstructure else None

    # 若输入是图片，先进行图像预处理，再进入后续主流程
    pipeline_input = _maybe_preprocess_input(args.input, args.lang, ocr_model=ocr)

    # 处理文档
    results = process_document(
        pipeline_input,
        ocr,
        detector,
        args.json,
        args.viz,
        args.dpi,
        structure_engine=structure_engine,
    )

    # 控制台打印摘要
    print("\n===== 检测摘要 =====")
    for page in results:
        sensitive_cells = []
        for cell in page['cells']:
            summary = detector._summarize_results(cell.get('sensitives', []))
            if summary['is_sensitive']:
                sensitive_cells.append((cell, summary))
        print(f"第 {page['page']+1} 页: 共 {len(page['cells'])} 个单元格，其中敏感单元格 {len(sensitive_cells)} 个")
        for cell, summary in sensitive_cells:
            print(f"  - {cell['text'][:50]} -> {summary.get('type', 'unknown')} (置信度 {summary.get('confidence', 0.0)})")

if __name__ == "__main__":
    main()