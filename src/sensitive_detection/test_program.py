import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.sensitive_detection.detector import SensitiveDetector

detector = SensitiveDetector()

test_cases = [
    # 身份证号（18 位，最后一位可能是数字或 X/x）
    ("11010119900307663X", "id_card"),
    ("11010119900307663x", "id_card"),
    ("123456199001011234", "id_card"),
    # 手机号（以 1 开头，第二位 3-9，后面 9 位数字）
    ("13812345678", "phone"),
    ("15912345678", "phone"),
    ("18812345678", "phone"),
    # 银行卡号（16-19 位数字，以 1-9 开头）
    ("6222021234567890123", "bank_card"),
    ("6217850000001234567", "bank_card"),
    # 邮箱
    ("test@example.com", "email"),
    ("user.name@domain.co.uk", "email"),
    # 关键词
    ("我的姓名是张三", "name"),
    ("家庭住址: 北京市朝阳区", "address"),
    ("身份证号：11010119900307663X", "id_card"),
    ("联系电话：13812345678", "phone"),
    ("今天天气不错", None),
    ("邮箱 test@example.com", "email"),
    ("表格内容: 二氧化碳排放量", None),
    ("11010119900307663X", "id_card"),
    ("身份证号: 11010119900307663x", "id_card"),
    ("手机: 13812345678", "phone"),
    ("联系电话 15912345678", "phone"),
    ("姓名: 张三", "name"),
    ("地址: 北京市朝阳区", "address"),
    ("普通文本没有敏感信息", None),
    ("身份证 123456199001011234", "id_card"),
    ("我的电话是18888888888", "phone"),
    ("", None),
    (None, None),
]

total = len(test_cases)
correct = 0

print("敏感信息检测测试结果：")
print("-" * 60)
for text, expected_type in test_cases:
    if text is None:
        result = detector.detect("")
    else:
        result = detector.detect(text)

    if result['is_sensitive']:
        detected_type = result['type']
    else:
        detected_type = None

    is_correct = (detected_type == expected_type)
    if is_correct:
        correct += 1

    status = "✓" if is_correct else "✗"
    print(f"{status} 输入: {str(text):30} -> 预期: {expected_type}, 实际: {detected_type}")

print("-" * 60)
success_rate = correct / total * 100
print(f"匹配成功率: {correct}/{total} = {success_rate:.2f}%")

