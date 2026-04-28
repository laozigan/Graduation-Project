#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List


def _run_processor_cli(argv: List[str]) -> None:
    from src.document_processor import processor as processor_module

    old_argv = sys.argv
    try:
        sys.argv = ["processor.py"] + argv
        processor_module.main()
    finally:
        sys.argv = old_argv


def _run_attack_cli(argv: List[str]) -> None:
    from src.adversarial_gen import run_attack as attack_module

    old_argv = sys.argv
    try:
        sys.argv = ["run_attack.py"] + argv
        attack_module.main()
    finally:
        sys.argv = old_argv


def _load_page_count(det_json: str) -> int:
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


def _build_output_paths(output: str, page_count: int) -> List[str]:
    if page_count <= 1:
        return [output]

    stem, ext = os.path.splitext(output)
    ext = ext or ".jpg"
    return [f"{stem}_p{idx + 1}{ext}" for idx in range(page_count)]


def _run_pipeline(args: argparse.Namespace) -> None:
    from src.document_processor import processor as processor_module
    from src.sensitive_detection import SensitiveDetector
    from src.adversarial_gen.perturbator import AttackConfig, run_attack_from_files

    print("初始化 PaddleOCR...")
    print(
        f"MKLDNN: {'enabled' if processor_module.ENABLE_MKLDNN else 'disabled'} "
        "(set PPS_ENABLE_MKLDNN=1 to enable)"
    )
    ocr = processor_module.PaddleOCR(lang=args.lang, use_textline_orientation=True)
    detector = SensitiveDetector(
        use_nlp=args.use_nlp,
        enable_uie=args.use_uie,
        uie_model=args.uie_model,
    )
    structure_engine = processor_module._build_structure_engine(args.lang) if args.use_ppstructure else None

    pipeline_input = processor_module._maybe_preprocess_input(args.input, args.lang, ocr_model=ocr)

    det_json = args.det_json
    if not det_json:
        stem = Path(args.input).stem
        det_json = str(Path("outputs") / "ocr_result" / f"{stem}_detection.json")
    det_dir = os.path.dirname(det_json)
    if det_dir:
        os.makedirs(det_dir, exist_ok=True)

    processor_module.process_document(
        pipeline_input,
        ocr,
        detector,
        output_json=det_json,
        output_viz=args.viz,
        dpi=args.dpi,
        structure_engine=structure_engine,
    )

    page_count = _load_page_count(det_json)
    if page_count <= 0:
        raise ValueError(f"No pages found in detection JSON: {det_json}")

    output_paths = _build_output_paths(args.output, page_count)
    compare_paths = _build_output_paths(args.compare_output, page_count) if args.compare_output else None

    attack_cfg = AttackConfig(
        epsilon=args.epsilon,
        alpha=args.alpha,
        steps=args.steps,
        attack_method=args.attack_method,
        seed=args.seed,
        bbox_margin=args.bbox_margin,
        line_protect_width=args.line_protect_width,
        force_bbox_fallback=args.force_bbox_fallback,
        adaptive_detect_missing_cells=args.adaptive_missing_cells,
        advbox_attack_name=args.advbox_attack_name,
        advbox_spsa_sigma=args.advbox_spsa_sigma,
        advbox_spsa_samples=args.advbox_spsa_samples,
        advbox_text_change_bonus=args.advbox_text_change_bonus,
        image_scale=args.image_scale,
    )

    saved = run_attack_from_files(
        input_file=args.input,
        detection_json=det_json,
        output_paths=output_paths,
        config=attack_cfg,
        dpi=args.dpi,
        orient_mode="off",
        verbose=True,
        compare_output_paths=compare_paths,
    )

    print("Pipeline completed. Adversarial outputs:")
    for path in saved:
        print(path)
    print(f"Detection JSON: {det_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified entry for process, attack, and end-to-end pipeline")
    subparsers = parser.add_subparsers(dest="command")

    process_parser = subparsers.add_parser("process", help="Run document processing and sensitive labeling")
    process_parser.add_argument("processor_args", nargs=argparse.REMAINDER, help="Arguments forwarded to processor")

    attack_parser = subparsers.add_parser("attack", help="Run adversarial sample generation")
    attack_parser.add_argument("attack_args", nargs=argparse.REMAINDER, help="Arguments forwarded to run_attack")

    pipeline_parser = subparsers.add_parser("pipeline", help="Run process and attack in one command")
    pipeline_parser.add_argument("input", help="Input file path (image/pdf)")
    pipeline_parser.add_argument("--det-json", default=None, help="Detection json path. Defaults to outputs/ocr_result/<stem>_detection.json")
    pipeline_parser.add_argument("--viz", default=None, help="Optional visualization image path")
    pipeline_parser.add_argument("--output", "-o", required=True, help="Output adversarial image path (multi-page auto-suffixed)")
    pipeline_parser.add_argument("--compare-output", default=None, help="Optional compare image path (multi-page auto-suffixed)")
    pipeline_parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF conversion")
    pipeline_parser.add_argument("--lang", default="ch", help="OCR language")
    pipeline_parser.add_argument("--use-nlp", dest="use_nlp", action="store_true")
    pipeline_parser.add_argument("--no-nlp", dest="use_nlp", action="store_false")
    pipeline_parser.add_argument("--use-uie", dest="use_uie", action="store_true")
    pipeline_parser.add_argument("--no-uie", dest="use_uie", action="store_false")
    pipeline_parser.add_argument("--uie-model", default="uie-x-base")
    pipeline_parser.add_argument("--use-ppstructure", dest="use_ppstructure", action="store_true")
    pipeline_parser.add_argument("--no-ppstructure", dest="use_ppstructure", action="store_false")

    pipeline_parser.add_argument("--attack-method", choices=["random", "pgd", "advbox_roi"], default="pgd")
    pipeline_parser.add_argument("--epsilon", type=float, default=24.0)
    pipeline_parser.add_argument("--alpha", type=float, default=6.0)
    pipeline_parser.add_argument("--steps", type=int, default=9)
    pipeline_parser.add_argument("--seed", type=int, default=2026)
    pipeline_parser.add_argument("--bbox-margin", type=int, default=2)
    pipeline_parser.add_argument("--line-protect-width", type=int, default=2)
    pipeline_parser.add_argument("--force-bbox-fallback", dest="force_bbox_fallback", action="store_true")
    pipeline_parser.add_argument("--no-force-bbox-fallback", dest="force_bbox_fallback", action="store_false")
    pipeline_parser.add_argument("--adaptive-missing-cells", dest="adaptive_missing_cells", action="store_true")
    pipeline_parser.add_argument("--no-adaptive-missing-cells", dest="adaptive_missing_cells", action="store_false")
    pipeline_parser.add_argument("--advbox-attack-name", choices=["PGD", "FGSM", "BIM", "MIFGSM"], default="PGD")
    pipeline_parser.add_argument("--advbox-spsa-sigma", type=float, default=2.0)
    pipeline_parser.add_argument("--advbox-spsa-samples", type=int, default=4)
    pipeline_parser.add_argument("--advbox-text-change-bonus", type=float, default=0.5)
    pipeline_parser.add_argument("--image-scale", type=float, default=1.0)

    pipeline_parser.set_defaults(use_nlp=True)
    pipeline_parser.set_defaults(use_uie=True)
    pipeline_parser.set_defaults(use_ppstructure=True)
    pipeline_parser.set_defaults(force_bbox_fallback=True)
    pipeline_parser.set_defaults(adaptive_missing_cells=True)

    args = parser.parse_args()

    # Backward compatibility: no subcommand -> behave like old processor entry.
    if args.command is None:
        _run_processor_cli(sys.argv[1:])
        return

    if args.command == "process":
        forwarded = args.processor_args
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        _run_processor_cli(forwarded)
        return

    if args.command == "attack":
        forwarded = args.attack_args
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        _run_attack_cli(forwarded)
        return

    if args.command == "pipeline":
        _run_pipeline(args)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
