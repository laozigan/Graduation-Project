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

import cv2
import numpy as np
from paddleocr import PaddleOCR

os.environ['FLAGS_use_mkldnn'] = '0'  # 禁用 OneDNN 加速
os.environ['FLAGS_use_mkldnn_common_opt'] = '0'  # 禁用通用优化
os.environ['FLAGS_enable_pir_api'] = '0'  # 禁用 PIR API 以避免兼容性问题

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


def process_document(file_path: str,
                     ocr_model: PaddleOCR,
                     detector: SensitiveDetector,
                     output_json: str = None,
                     output_viz: str = None,
                     dpi: int = 200) -> List[Dict]:
    """
    处理文档，返回每页的提取结果。
    结果格式：
    [
        {
            "page": 0,
            "cells": [
                {"bbox": [x1,y1,x2,y2], "text": "...", "sensitive": {...}},
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
        if not cells:
            print("OCR无结果，跳过此页")
            all_pages.append({"page": page_idx, "cells": []})
            continue

        # 2. 检测敏感信息
        for cell in cells:
            sens = detector.detect(cell['text'])
            cell['sensitive'] = sens

        # 5. 可选：生成标注图像
        if output_viz and page_idx == 0:  # 仅第一页示例
            viz = img.copy()
            for cell in cells:
                x1,y1,x2,y2 = map(int, cell['bbox'])
                color = (0,0,255) if cell['sensitive']['is_sensitive'] else (0,255,0)
                cv2.rectangle(viz, (x1,y1), (x2,y2), color, 2)
                # 添加标签
                if cell['sensitive']['is_sensitive']:
                    label = cell['sensitive']['type']
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
    parser.set_defaults(use_nlp=True)
    args = parser.parse_args()

    # 初始化 OCR（CPU模式）
    print("初始化 PaddleOCR...")
    ocr = PaddleOCR(lang=args.lang, use_angle_cls=True)
    detector = SensitiveDetector(use_nlp=args.use_nlp)

    # 若输入是图片，先进行图像预处理，再进入后续主流程
    pipeline_input = _maybe_preprocess_input(args.input, args.lang, ocr_model=ocr)

    # 处理文档
    results = process_document(pipeline_input, ocr, detector, args.json, args.viz, args.dpi)

    # 控制台打印摘要
    print("\n===== 检测摘要 =====")
    for page in results:
        sensitive_cells = [c for c in page['cells'] if c['sensitive']['is_sensitive']]
        print(f"第 {page['page']+1} 页: 共 {len(page['cells'])} 个单元格，其中敏感单元格 {len(sensitive_cells)} 个")
        for cell in sensitive_cells:
            print(f"  - {cell['text'][:50]} -> {cell['sensitive']['type']} (置信度 {cell['sensitive']['confidence']})")

if __name__ == "__main__":
    main()