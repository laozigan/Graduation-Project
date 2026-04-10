#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敏感信息检测评估工具
用于验证敏感信息识别的准确率、召回率等指标
"""

import os
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import numpy as np


class SensitiveDetectionEvaluator:
    """敏感信息检测评估器"""

    def __init__(self):
        self.sensitive_types = [
            'name', 'phone', 'email', 'bank_card', 'id_card',
            'address', 'id', 'medical'
        ]

    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, Any]:
        """
        评估预测结果与真实标签的匹配情况

        Args:
            predictions: 预测结果列表，每个dict包含 'text', 'sensitive' 字段
            ground_truth: 真实标签列表，每个dict包含 'text', 'sensitive' 字段

        Returns:
            评估结果字典
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(f"预测结果数量({len(predictions)})与真实标签数量({len(ground_truth)})不匹配")

        # 初始化统计
        total_predictions = len(predictions)
        total_sensitive_pred = 0
        total_sensitive_true = 0
        correct_sensitive = 0

        # 按类型统计
        type_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})

        # 详细匹配记录
        detailed_results = []

        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            pred_sensitive = pred.get('sensitive', {}).get('is_sensitive', False)
            true_sensitive = true.get('sensitive', {}).get('is_sensitive', False)

            pred_type = pred.get('sensitive', {}).get('type', 'unknown') if pred_sensitive else 'none'
            true_type = true.get('sensitive', {}).get('type', 'unknown') if true_sensitive else 'none'

            # 总体统计
            total_sensitive_pred += int(pred_sensitive)
            total_sensitive_true += int(true_sensitive)

            # 类型统计
            if pred_sensitive and true_sensitive:
                if pred_type == true_type:
                    correct_sensitive += 1
                    type_stats[pred_type]['tp'] += 1
                else:
                    # 类型不匹配，算作FP和FN
                    type_stats[pred_type]['fp'] += 1
                    type_stats[true_type]['fn'] += 1
            elif pred_sensitive and not true_sensitive:
                type_stats[pred_type]['fp'] += 1
            elif not pred_sensitive and true_sensitive:
                type_stats[true_type]['fn'] += 1
            else:
                # 都为非敏感，算作TN（按类型统计为tn）
                for t in self.sensitive_types:
                    type_stats[t]['tn'] += 1

            # 详细记录
            detailed_results.append({
                'index': i,
                'text': pred.get('text', ''),
                'predicted': {
                    'is_sensitive': pred_sensitive,
                    'type': pred_type
                },
                'ground_truth': {
                    'is_sensitive': true_sensitive,
                    'type': true_type
                },
                'match': (pred_sensitive == true_sensitive) and (not pred_sensitive or pred_type == true_type)
            })

        # 计算总体指标
        overall_metrics = self._calculate_metrics(total_sensitive_pred, total_sensitive_true, correct_sensitive)

        # 计算各类型指标
        type_metrics = {}
        for sens_type in self.sensitive_types:
            stats = type_stats[sens_type]
            type_metrics[sens_type] = self._calculate_metrics(
                stats['tp'] + stats['fp'],  # predicted positive
                stats['tp'] + stats['fn'],  # actual positive
                stats['tp']  # correct positive
            )

        return {
            'summary': {
                'total_samples': total_predictions,
                'total_sensitive_predicted': total_sensitive_pred,
                'total_sensitive_actual': total_sensitive_true,
                'correct_sensitive_predictions': correct_sensitive
            },
            'overall_metrics': overall_metrics,
            'type_metrics': type_metrics,
            'detailed_results': detailed_results
        }

    def _calculate_metrics(self, predicted_positive: int, actual_positive: int, true_positive: int) -> Dict[str, float]:
        """计算准确率、召回率、F1分数等指标"""
        if predicted_positive == 0:
            precision = 0.0
        else:
            precision = true_positive / predicted_positive

        if actual_positive == 0:
            recall = 0.0
        else:
            recall = true_positive / actual_positive

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # 准确率 = (TP + TN) / (TP + TN + FP + FN)
        # 这里我们只计算正类的指标，因为TN的数量很大且不重要

        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'predicted_positive': predicted_positive,
            'actual_positive': actual_positive,
            'true_positive': true_positive
        }

    def evaluate_from_files(self, prediction_file: str, ground_truth_file: str) -> Dict[str, Any]:
        """
        从JSON文件加载预测结果和真实标签进行评估

        Args:
            prediction_file: 预测结果JSON文件路径
            ground_truth_file: 真实标签JSON文件路径

        Returns:
            评估结果
        """
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        # 处理嵌套结构（每页一个列表）
        if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], list):
            # 展平嵌套结构
            flat_predictions = []
            flat_ground_truth = []
            for page_pred, page_gt in zip(predictions, ground_truth):
                flat_predictions.extend(page_pred)
                flat_ground_truth.extend(page_gt)
            predictions = flat_predictions
            ground_truth = flat_ground_truth

        return self.evaluate_predictions(predictions, ground_truth)

    def print_report(self, evaluation_results: Dict[str, Any], output_file: str = None):
        """打印评估报告"""
        report_lines = []

        def add_line(line):
            report_lines.append(line)
            print(line)

        add_line("=" * 60)
        add_line("敏感信息检测评估报告")
        add_line("=" * 60)

        summary = evaluation_results['summary']
        add_line(f"总样本数: {summary['total_samples']}")
        add_line(f"预测为敏感: {summary['total_sensitive_predicted']}")
        add_line(f"实际敏感数: {summary['total_sensitive_actual']}")
        add_line(f"正确预测敏感: {summary['correct_sensitive_predictions']}")
        add_line("")

        overall = evaluation_results['overall_metrics']
        add_line("总体指标:")
        add_line(f"  精确率 (Precision): {overall['precision']:.4f}")
        add_line(f"  召回率 (Recall): {overall['recall']:.4f}")
        add_line(f"  F1分数: {overall['f1_score']:.4f}")
        add_line("")

        add_line("各类型指标:")
        type_metrics = evaluation_results['type_metrics']
        for sens_type, metrics in type_metrics.items():
            if metrics['predicted_positive'] > 0 or metrics['actual_positive'] > 0:
                add_line(f"  {sens_type}:")
                add_line(f"    精确率: {metrics['precision']:.4f} (预测:{metrics['predicted_positive']}, 正确:{metrics['true_positive']})")
                add_line(f"    召回率: {metrics['recall']:.4f} (实际:{metrics['actual_positive']}, 正确:{metrics['true_positive']})")
                add_line(f"    F1分数: {metrics['f1_score']:.4f}")
        add_line("")

        # 错误分析
        detailed = evaluation_results['detailed_results']
        false_positives = [r for r in detailed if r['predicted']['is_sensitive'] and not r['ground_truth']['is_sensitive']]
        false_negatives = [r for r in detailed if not r['predicted']['is_sensitive'] and r['ground_truth']['is_sensitive']]
        type_mismatches = [r for r in detailed if r['predicted']['is_sensitive'] and r['ground_truth']['is_sensitive'] and r['predicted']['type'] != r['ground_truth']['type']]

        add_line("错误分析:")
        add_line(f"  误报 (False Positive): {len(false_positives)} 个")
        add_line(f"  漏报 (False Negative): {len(false_negatives)} 个")
        add_line(f"  类型错误 (Type Mismatch): {len(type_mismatches)} 个")
        add_line("")

        if false_positives:
            add_line("误报示例 (前5个):")
            for fp in false_positives[:5]:
                add_line(f"  '{fp['text'][:50]}...' -> 预测:{fp['predicted']['type']}, 实际:非敏感")

        if false_negatives:
            add_line("漏报示例 (前5个):")
            for fn in false_negatives[:5]:
                add_line(f"  '{fn['text'][:50]}...' -> 实际:{fn['ground_truth']['type']}, 未检测")

        if type_mismatches:
            add_line("类型错误示例 (前5个):")
            for tm in type_mismatches[:5]:
                add_line(f"  '{tm['text'][:50]}...' -> 预测:{tm['predicted']['type']}, 实际:{tm['ground_truth']['type']}")

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            add_line(f"\n报告已保存到: {output_file}")


def create_sample_ground_truth(predictions_file: str, output_file: str):
    """
    基于预测结果创建样本真实标签文件模板
    用户可以手动编辑这个文件来创建ground truth

    Args:
        predictions_file: 预测结果文件
        output_file: 输出模板文件
    """
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # 创建模板，将所有预测结果作为初始ground truth
    ground_truth = []
    for pred in predictions:
        gt_item = pred.copy()
        # 可以在这里调整真实标签
        ground_truth.append(gt_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"Ground truth模板已创建: {output_file}")
    print("请手动编辑此文件来修正真实标签，然后用于评估。")


if __name__ == "__main__":
    # 示例用法
    evaluator = SensitiveDetectionEvaluator()

    # 示例数据
    sample_predictions = [
        {'text': '姓名：张三', 'sensitive': {'is_sensitive': True, 'type': 'name'}},
        {'text': '电话：13800138000', 'sensitive': {'is_sensitive': True, 'type': 'phone'}},
        {'text': '这是一段普通文本', 'sensitive': {'is_sensitive': False}},
    ]

    sample_ground_truth = [
        {'text': '姓名：张三', 'sensitive': {'is_sensitive': True, 'type': 'name'}},
        {'text': '电话：13800138000', 'sensitive': {'is_sensitive': True, 'type': 'phone'}},
        {'text': '这是一段普通文本', 'sensitive': {'is_sensitive': False}},
    ]

    results = evaluator.evaluate_predictions(sample_predictions, sample_ground_truth)
    evaluator.print_report(results)