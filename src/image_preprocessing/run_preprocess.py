#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.image_preprocessing import ImagePreprocessor, PreprocessConfig


def build_default_output_path(input_path: str) -> str:
    stem, _ = os.path.splitext(os.path.basename(input_path))
    return os.path.join("outputs", "preprocessed", f"{stem}_preprocessed.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description="独立图像预处理器：透视+方向+倾斜+增强")
    parser.add_argument("input", help="输入图片路径（jpg/jpeg/png/bmp/tif/tiff）")
    parser.add_argument("--output", "-o", default=None, help="输出图片路径")
    parser.add_argument("--report", default=None, help="可选：输出预处理报告 JSON")
    parser.add_argument("--lang", default="ch", help="PaddleOCR 语言参数（默认 ch）")
    parser.add_argument("--no-ocr-orientation", action="store_true", help="禁用 OCR 方向校正")
    parser.add_argument("--no-perspective", action="store_true", help="禁用透视校正")
    parser.add_argument("--no-denoise", action="store_true", help="禁用去噪")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or build_default_output_path(input_path)

    config = PreprocessConfig(
        lang=args.lang,
        use_ocr_orientation=not args.no_ocr_orientation,
        enable_perspective_correction=not args.no_perspective,
        enable_denoise=not args.no_denoise,
    )
    preprocessor = ImagePreprocessor(config=config)
    result = preprocessor.preprocess_file(input_path, output_path)

    print("预处理完成")
    print(f"输入文件: {result.input_path}")
    print(f"输出文件: {result.output_path}")
    print(f"透视校正方式: {result.perspective_method}")
    print(f"方向校正角度: {result.orientation_angle}")
    print(f"方向校正方法: {result.orientation_method}")
    print(f"倾斜角估计: {result.skew_angle}")

    if args.report:
        report = {
            "input_path": result.input_path,
            "output_path": result.output_path,
            "perspective_method": result.perspective_method,
            "orientation_angle": result.orientation_angle,
            "orientation_method": result.orientation_method,
            "skew_angle": result.skew_angle,
        }
        report_parent = os.path.dirname(args.report)
        if report_parent:
            os.makedirs(report_parent, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告文件: {args.report}")


if __name__ == "__main__":
    main()
