#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from .perturbator import AttackConfig, run_attack_from_files


def _build_output_paths(output: str, page_count: int) -> List[str]:
    if page_count <= 1:
        return [output]

    stem, ext = os.path.splitext(output)
    ext = ext or ".jpg"
    return [f"{stem}_p{idx + 1}{ext}" for idx in range(page_count)]


def _prepare_optional_output_paths(path: Optional[str], page_count: int) -> Optional[List[str]]:
    if not path:
        return None
    return _build_output_paths(path, page_count)


def _load_page_count(det_json: str) -> int:
    import json

    with open(det_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return 0

    count = 0
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("cells"), list):
            count += 1
        elif isinstance(item, list):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pure adversarial perturbation on sensitive cells while preserving table lines"
    )
    parser.add_argument("input", help="Input image/PDF path")
    parser.add_argument("--det-json", required=True, help="Detection result JSON from processor.py")
    parser.add_argument("--output", "-o", required=True, help="Output image path (multi-page auto-suffixed)")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF conversion")
    parser.add_argument("--epsilon", type=float, default=24.0, help="L_inf perturbation budget")
    parser.add_argument("--alpha", type=float, default=6.0, help="Step size per iteration")
    parser.add_argument("--steps", type=int, default=9, help="Iteration count")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--bbox-margin", type=int, default=2, help="Skip bbox borders by N pixels")
    parser.add_argument("--line-protect-width", type=int, default=2, help="Protected border thickness")
    parser.add_argument(
        "--orient-mode",
        choices=["off", "upside_down", "always"],
        default="upside_down",
        help="Orientation strategy: off=no rotation, upside_down=only rotate 180deg pages, always=apply detected right-angle rotation",
    )
    parser.add_argument("--auto-orient", dest="auto_orient", action="store_true", help="Compatibility alias: same as --orient-mode upside_down")
    parser.add_argument("--no-auto-orient", dest="auto_orient", action="store_false", help="Compatibility alias: same as --orient-mode off")
    parser.add_argument("--force-bbox-fallback", dest="force_bbox_fallback", action="store_true", help="Fallback to bbox-interior perturbation when text mask is too small")
    parser.add_argument("--no-force-bbox-fallback", dest="force_bbox_fallback", action="store_false", help="Disable bbox fallback")
    parser.add_argument("--adaptive-missing-cells", dest="adaptive_missing_cells", action="store_true", help="When no sensitive cells are found, infer sensitive text regions from OCR lines")
    parser.add_argument("--no-adaptive-missing-cells", dest="adaptive_missing_cells", action="store_false", help="Disable OCR-line fallback when table cells are unclear")
    parser.add_argument("--oriented-output", default=None, help="Optional path to save auto-oriented image")
    parser.add_argument("--compare-output", default=None, help="Optional path to save oriented-vs-adversarial comparison image")
    parser.set_defaults(auto_orient=None)
    parser.set_defaults(force_bbox_fallback=True)
    parser.set_defaults(adaptive_missing_cells=True)
    args = parser.parse_args()

    page_count = _load_page_count(args.det_json)
    if page_count <= 0:
        raise ValueError("No pages found in detection JSON")

    output_paths = _build_output_paths(args.output, page_count)
    oriented_output_paths = _prepare_optional_output_paths(args.oriented_output, page_count)
    compare_output_paths = _prepare_optional_output_paths(args.compare_output, page_count)

    for out in output_paths:
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    if oriented_output_paths is not None:
        for out in oriented_output_paths:
            out_dir = os.path.dirname(out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
    if compare_output_paths is not None:
        for out in compare_output_paths:
            out_dir = os.path.dirname(out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

    config = AttackConfig(
        epsilon=args.epsilon,
        alpha=args.alpha,
        steps=args.steps,
        seed=args.seed,
        bbox_margin=args.bbox_margin,
        line_protect_width=args.line_protect_width,
        force_bbox_fallback=args.force_bbox_fallback,
        adaptive_detect_missing_cells=args.adaptive_missing_cells,
    )

    saved = run_attack_from_files(
        input_file=args.input,
        detection_json=args.det_json,
        output_paths=output_paths,
        config=config,
        dpi=args.dpi,
        auto_orient=args.auto_orient,
        orient_mode=args.orient_mode,
        verbose=True,
        oriented_output_paths=oriented_output_paths,
        compare_output_paths=compare_output_paths,
    )

    print("Adversarial outputs:")
    for path in saved:
        print(path)

    if oriented_output_paths is not None:
        print("Oriented outputs:")
        for path in oriented_output_paths:
            print(path)

    if compare_output_paths is not None:
        print("Compare outputs:")
        for path in compare_output_paths:
            print(path)


if __name__ == "__main__":
    main()
