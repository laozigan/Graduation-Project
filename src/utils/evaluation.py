#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""敏感信息检测评估工具。"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Set


class SensitiveDetectionEvaluator:
    """敏感信息检测评估器。"""

    def __init__(self):
        self.sensitive_types = [
            "name",
            "phone",
            "email",
            "bank_card",
            "id_card",
            "address",
            "id",
            "medical",
            "social_security",
            "passport",
            "birth_date",
        ]

    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, Any]:
        """评估预测结果与真实标签的匹配情况。"""
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"预测结果数量({len(predictions)})与真实标签数量({len(ground_truth)})不匹配"
            )

        total_predictions = len(predictions)
        total_sensitive_pred = 0
        total_sensitive_true = 0
        correct_sensitive = 0

        type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        detailed_results = []

        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            pred_types = self._extract_types(pred)
            true_types = self._extract_types(true)

            pred_sensitive = bool(pred_types)
            true_sensitive = bool(true_types)
            pred_type = next(iter(pred_types)) if pred_sensitive else "none"
            true_type = next(iter(true_types)) if true_sensitive else "none"

            total_sensitive_pred += int(pred_sensitive)
            total_sensitive_true += int(true_sensitive)

            if pred_sensitive and true_sensitive:
                common_types = pred_types & true_types
                if common_types:
                    correct_sensitive += 1
                    for t in common_types:
                        type_stats[t]["tp"] += 1
                else:
                    for t in pred_types:
                        type_stats[t]["fp"] += 1
                    for t in true_types:
                        type_stats[t]["fn"] += 1
            elif pred_sensitive and not true_sensitive:
                for t in pred_types:
                    type_stats[t]["fp"] += 1
            elif not pred_sensitive and true_sensitive:
                for t in true_types:
                    type_stats[t]["fn"] += 1
            else:
                for t in self.sensitive_types:
                    type_stats[t]["tn"] += 1

            detailed_results.append(
                {
                    "index": i,
                    "text": pred.get("text", ""),
                    "predicted": {
                        "is_sensitive": pred_sensitive,
                        "type": pred_type,
                        "types": sorted(pred_types),
                    },
                    "ground_truth": {
                        "is_sensitive": true_sensitive,
                        "type": true_type,
                        "types": sorted(true_types),
                    },
                    "match": (pred_sensitive == true_sensitive)
                    and (not pred_sensitive or bool(pred_types & true_types)),
                }
            )

        overall_metrics = self._calculate_metrics(total_sensitive_pred, total_sensitive_true, correct_sensitive)

        type_metrics = {}
        for sens_type in self.sensitive_types:
            stats = type_stats[sens_type]
            type_metrics[sens_type] = self._calculate_metrics(
                stats["tp"] + stats["fp"],
                stats["tp"] + stats["fn"],
                stats["tp"],
            )

        return {
            "summary": {
                "total_samples": total_predictions,
                "total_sensitive_predicted": total_sensitive_pred,
                "total_sensitive_actual": total_sensitive_true,
                "correct_sensitive_predictions": correct_sensitive,
            },
            "overall_metrics": overall_metrics,
            "type_metrics": type_metrics,
            "detailed_results": detailed_results,
        }

    def _calculate_metrics(
        self, predicted_positive: int, actual_positive: int, true_positive: int
    ) -> Dict[str, float]:
        """计算准确率、召回率、F1分数等指标。"""
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
        recall = true_positive / actual_positive if actual_positive > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "predicted_positive": predicted_positive,
            "actual_positive": actual_positive,
            "true_positive": true_positive,
        }

    def _extract_types(self, item: Dict) -> Set[str]:
        """从单条结果中提取所有敏感类型，兼容 sensitive/sensitives 两种结构。"""
        types: Set[str] = set()

        sensitives = item.get("sensitives")
        if isinstance(sensitives, list):
            for sens in sensitives:
                sens_type = sens.get("type")
                if sens_type:
                    if sens_type == "chinese_name":
                        sens_type = "name"
                    types.add(sens_type)

        sensitive = item.get("sensitive", {})
        if isinstance(sensitive, dict) and sensitive.get("is_sensitive"):
            sens_type = sensitive.get("type")
            if sens_type:
                if sens_type == "chinese_name":
                    sens_type = "name"
                types.add(sens_type)

        return types

    def _flatten_items(self, payload: Any) -> List[Dict]:
        """将多种文件结构拍平为 cell 列表。"""
        if not isinstance(payload, list):
            return []

        flattened: List[Dict] = []
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("cells"), list):
                flattened.extend(item["cells"])
            elif isinstance(item, list):
                flattened.extend([x for x in item if isinstance(x, dict)])
            elif isinstance(item, dict):
                flattened.append(item)
        return flattened

    def evaluate_from_files(self, prediction_file: str, ground_truth_file: str) -> Dict[str, Any]:
        """从 JSON 文件加载预测结果和真实标签并进行评估。"""
        with open(prediction_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        with open(ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        flat_predictions = self._flatten_items(predictions)
        flat_ground_truth = self._flatten_items(ground_truth)
        return self.evaluate_predictions(flat_predictions, flat_ground_truth)

    def print_report(self, evaluation_results: Dict[str, Any], output_file: str = None):
        """打印评估报告。"""
        report_lines: List[str] = []

        def add_line(line: str):
            report_lines.append(line)
            print(line)

        add_line("=" * 60)
        add_line("敏感信息检测评估报告")
        add_line("=" * 60)

        summary = evaluation_results["summary"]
        add_line(f"总样本数: {summary['total_samples']}")
        add_line(f"预测为敏感: {summary['total_sensitive_predicted']}")
        add_line(f"实际敏感数: {summary['total_sensitive_actual']}")
        add_line(f"正确预测敏感: {summary['correct_sensitive_predictions']}")
        add_line("")

        overall = evaluation_results["overall_metrics"]
        add_line("总体指标:")
        add_line(f"  精确率 (Precision): {overall['precision']:.4f}")
        add_line(f"  召回率 (Recall): {overall['recall']:.4f}")
        add_line(f"  F1分数: {overall['f1_score']:.4f}")
        add_line("")

        add_line("各类型指标:")
        type_metrics = evaluation_results["type_metrics"]
        for sens_type, metrics in type_metrics.items():
            if metrics["predicted_positive"] > 0 or metrics["actual_positive"] > 0:
                add_line(f"  {sens_type}:")
                add_line(
                    f"    精确率: {metrics['precision']:.4f} "
                    f"(预测:{metrics['predicted_positive']}, 正确:{metrics['true_positive']})"
                )
                add_line(
                    f"    召回率: {metrics['recall']:.4f} "
                    f"(实际:{metrics['actual_positive']}, 正确:{metrics['true_positive']})"
                )
                add_line(f"    F1分数: {metrics['f1_score']:.4f}")
        add_line("")

        detailed = evaluation_results["detailed_results"]
        false_positives = [
            r
            for r in detailed
            if r["predicted"]["is_sensitive"] and not r["ground_truth"]["is_sensitive"]
        ]
        false_negatives = [
            r
            for r in detailed
            if not r["predicted"]["is_sensitive"] and r["ground_truth"]["is_sensitive"]
        ]
        type_mismatches = [
            r
            for r in detailed
            if r["predicted"]["is_sensitive"]
            and r["ground_truth"]["is_sensitive"]
            and not (set(r["predicted"].get("types", [])) & set(r["ground_truth"].get("types", [])))
        ]

        add_line("错误分析:")
        add_line(f"  误报 (False Positive): {len(false_positives)} 个")
        add_line(f"  漏报 (False Negative): {len(false_negatives)} 个")
        add_line(f"  类型错误 (Type Mismatch): {len(type_mismatches)} 个")
        add_line("")

        if false_positives:
            add_line("误报示例 (前5个):")
            for fp in false_positives[:5]:
                add_line(
                    f"  '{fp['text'][:50]}...' -> "
                    f"预测:{fp['predicted']['type']}, 实际:非敏感"
                )

        if false_negatives:
            add_line("漏报示例 (前5个):")
            for fn in false_negatives[:5]:
                add_line(
                    f"  '{fn['text'][:50]}...' -> "
                    f"实际:{fn['ground_truth']['type']}, 未检测"
                )

        if type_mismatches:
            add_line("类型错误示例 (前5个):")
            for tm in type_mismatches[:5]:
                add_line(
                    f"  '{tm['text'][:50]}...' -> "
                    f"预测:{tm['predicted']['type']}, 实际:{tm['ground_truth']['type']}"
                )

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            add_line(f"\n报告已保存到: {output_file}")


def create_sample_ground_truth(predictions_file: str, output_file: str):
    """基于预测结果创建 ground truth 模板。"""
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if isinstance(predictions, list):
        template = predictions
    else:
        template = []

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"Ground truth模板已创建: {output_file}")
    print("请手动编辑此文件来修正真实标签，然后用于评估。")


if __name__ == "__main__":
    evaluator = SensitiveDetectionEvaluator()

    sample_predictions = [
        {"text": "姓名：张三", "sensitives": [{"is_sensitive": True, "type": "name"}]},
        {"text": "电话：13800138000", "sensitives": [{"is_sensitive": True, "type": "phone"}]},
        {"text": "这是一段普通文本", "sensitives": []},
    ]

    sample_ground_truth = [
        {"text": "姓名：张三", "sensitives": [{"is_sensitive": True, "type": "name"}]},
        {"text": "电话：13800138000", "sensitives": [{"is_sensitive": True, "type": "phone"}]},
        {"text": "这是一段普通文本", "sensitives": []},
    ]

    results = evaluator.evaluate_predictions(sample_predictions, sample_ground_truth)
    evaluator.print_report(results)