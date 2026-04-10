# 敏感信息检测评估工具使用指南

## 概述

`SensitiveDetectionEvaluator` 是一个用于验证敏感信息识别成功率的工具函数。它可以计算准确率、召回率、F1分数等指标，并提供详细的错误分析。

## 主要功能

- **总体评估**: 计算敏感信息检测的整体性能指标
- **类型评估**: 按敏感信息类型（如姓名、电话、邮箱等）分别评估
- **错误分析**: 识别误报、漏报和类型错误
- **报告生成**: 生成详细的评估报告和JSON结果

## 使用方法

### 1. 基本使用

```python
from src.utils import SensitiveDetectionEvaluator

# 初始化评估器
evaluator = SensitiveDetectionEvaluator()

# 准备预测结果和真实标签
predictions = [
    {'text': '姓名：张三', 'sensitive': {'is_sensitive': True, 'type': 'name'}},
    {'text': '电话：13800138000', 'sensitive': {'is_sensitive': True, 'type': 'phone'}},
    {'text': '普通文本', 'sensitive': {'is_sensitive': False}},
]

ground_truth = [
    {'text': '姓名：张三', 'sensitive': {'is_sensitive': True, 'type': 'name'}},
    {'text': '电话：13800138000', 'sensitive': {'is_sensitive': True, 'type': 'phone'}},
    {'text': '普通文本', 'sensitive': {'is_sensitive': False}},
]

# 执行评估
results = evaluator.evaluate_predictions(predictions, ground_truth)

# 打印报告
evaluator.print_report(results)
```

### 2. 从文件评估

```python
# 从JSON文件加载数据进行评估
results = evaluator.evaluate_from_files('predictions.json', 'ground_truth.json')
evaluator.print_report(results, 'evaluation_report.txt')
```

### 3. 创建Ground Truth模板

```python
from src.utils import create_sample_ground_truth

# 基于预测结果创建模板，用户手动标注真实标签
create_sample_ground_truth('predictions.json', 'ground_truth_template.json')
```

## 评估指标说明

- **精确率 (Precision)**: 预测为敏感的信息中，实际为敏感的比例
- **召回率 (Recall)**: 实际为敏感的信息中，被正确识别的比例
- **F1分数**: 精确率和召回率的调和平均值

## 支持的敏感信息类型

- `name`: 姓名
- `phone`: 电话号码
- `email`: 邮箱地址
- `bank_card`: 银行卡号
- `id_card`: 身份证号
- `address`: 地址
- `id`: 身份证相关
- `medical`: 医疗信息

## 示例输出

```
============================================================
敏感信息检测评估报告
============================================================
总样本数: 9
预测为敏感: 6
实际敏感数: 6
正确预测敏感: 4

总体指标:
  精确率 (Precision): 0.6667
  召回率 (Recall): 0.6667
  F1分数: 0.6667

各类型指标:
  name:
    精确率: 1.0000 (预测:1, 正确:1)
    召回率: 1.0000 (实际:1, 正确:1)
    F1分数: 1.0000
  ...

错误分析:
  误报 (False Positive): 1 个
  漏报 (False Negative): 1 个
  类型错误 (Type Mismatch): 1 个
```

## 运行测试

运行 `evaluation_test.py` 来查看完整示例：

```bash
python evaluation_test.py
```

这将生成示例评估报告和相关文件到 `test_outputs/` 目录中。