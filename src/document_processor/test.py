import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))# 确保能找到模块

from processor import process_document
from paddleocr import PaddleOCR
from src.sensitive_detection.detector import SensitiveDetector   # 如果内部类未导出，可重新导入或直接使用

# 初始化 OCR（只需一次，可以复用）
ocr = PaddleOCR(lang='ch', use_textline_orientation=True)
detector = SensitiveDetector(use_nlp=True)

# 调用处理函数
pdf_path = "D:/PrivacyProtectionSystem/data/raw/扫描件_身体健康情况说明.pdf"

results = process_document(
    file_path=pdf_path,
    ocr_model=ocr,
    detector=detector,
    output_json="output.json",      # 可选，不想要可以设为 None
    output_viz="annotated.jpg",     # 可选，不想要设为 None
    dpi=200
)

# results 是一个列表，每个元素对应一页，包含 cells 信息
for page in results:
    print(f"第 {page['page']+1} 页有 {len(page['cells'])} 个单元格")
    for cell in page['cells']:
        sensitives = cell.get('sensitives', [])
        if sensitives:
            main_type = sensitives[0].get('type', 'unknown')
            print(f"敏感内容: {cell['text']} -> {main_type}")