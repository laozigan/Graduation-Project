import os
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

def parse_xml_structure(xml_path: str) -> Tuple[List[List[float]], List[List[float]]]:
    """
    解析 XML 文件，提取所有列和行的边界框（归一化坐标，但本数据集坐标为绝对像素）。
    返回:
        columns: list of [xmin, ymin, xmax, ymax] 每个列边界框
        rows:    list of [xmin, ymin, xmax, ymax] 每个行边界框
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    columns = []
    rows = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        bbox = [xmin, ymin, xmax, ymax]
        
        if name == 'table column':
            columns.append(bbox)
        elif name == 'table row':
            rows.append(bbox)
    
    # 按 xmin 排序列，按 ymin 排序行
    columns.sort(key=lambda b: b[0])
    rows.sort(key=lambda b: b[1])
    return columns, rows

def load_words_from_json(json_path: str) -> List[Dict]:
    """加载 JSON 文件中的单词列表，每个单词包含 bbox 和 text"""
    with open(json_path, 'r', encoding='utf-8') as f:
        words = json.load(f)
    # 确保 bbox 是 [xmin, ymin, xmax, ymax] 格式
    for w in words:
        bbox = w['bbox']
        # bbox [x1,y1,x2,y2] 格式
        if len(bbox) == 4:
            w['bbox'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    return words

def assign_words_to_cells(columns: List[List[float]], rows: List[List[float]], 
                          words: List[Dict]) -> List[Dict]:
    """
    将单词分配到单元格（列 x 行）中。
    每个单元格由其列边界框和行边界框的交集定义。
    返回单元格列表，每个单元格包含:
        - 'bbox': 单元格区域 [xmin, ymin, xmax, ymax]
        - 'text': 合并后的文本
    """
    cells = []
    # 创建列和行的所有组合
    for col_idx, col_bbox in enumerate(columns):
        for row_idx, row_bbox in enumerate(rows):
            # 计算单元格边界框（列与行的交集）
            cell_xmin = max(col_bbox[0], row_bbox[0])
            cell_ymin = max(col_bbox[1], row_bbox[1])
            cell_xmax = min(col_bbox[2], row_bbox[2])
            cell_ymax = min(col_bbox[3], row_bbox[3])
            if cell_xmin >= cell_xmax or cell_ymin >= cell_ymax:
                continue  # 无效单元格（可能由于边界对齐问题，跳过）
            
            cell_bbox = [cell_xmin, cell_ymin, cell_xmax, cell_ymax]
            # 找到位于该单元格内的所有单词
            cell_words = []
            for w in words:
                w_bbox = w['bbox']
                # 计算单词的中心点
                center_x = (w_bbox[0] + w_bbox[2]) / 2.0
                center_y = (w_bbox[1] + w_bbox[3]) / 2.0
                if (cell_xmin -5 <= center_x <= cell_xmax +5) and (cell_ymin -5 <= center_y <= cell_ymax +5):
                    cell_words.append(w)
            if not cell_words:
                continue
            # 按 x 坐标排序（从左到右）
            cell_words.sort(key=lambda w: (w['bbox'][0] + w['bbox'][2]) / 2.0)
            # 合并文本，用空格连接
            merged_text = ' '.join([w['text'] for w in cell_words])
            cells.append({
                'bbox': cell_bbox,
                'text': merged_text,
                'col_idx': col_idx,
                'row_idx': row_idx
            })
    return cells

def load_dataset_icdar2013c(images_dir: str, xml_dir: str, words_dir: str) -> List[Dict]:
    """
    加载整个数据集，返回样本列表。
    每个样本: {'image_path': str, 'cells': list of {'bbox':..., 'text':...}, 'image_id': str}
    """
    samples = []
    # 获取所有 XML 文件
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    for xml_file in xml_files:
        base_name = xml_file.replace('.xml', '')
        # 图像路径
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"警告: 未找到图像 {base_name}")
            continue
        
        json_path = os.path.join(words_dir, base_name + '_words.json')
        if not os.path.exists(json_path):
            print(f"警告: 未找到 JSON 文件 {base_name}.json")
            continue
        
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            columns, rows = parse_xml_structure(xml_path)
            words = load_words_from_json(json_path)
            cells = assign_words_to_cells(columns, rows, words)
            samples.append({
                'image_path': img_path,
                'cells': cells,
                'image_id': base_name
            })
        except Exception as e:
            print(f"处理 {base_name} 时出错: {e}")
    return samples

def split_samples(samples: List[Dict], train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):

    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(samples, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
    return train, val, test

# 简单测试
if __name__ == '__main__':
    # 文件路径
    images_dir = "D:/PrivacyProtectionSystem/data/ICDAR-2013/images"
    test_xml_dir = "D:/PrivacyProtectionSystem/data/ICDAR-2013/test"   
    val_xml_dir = "D:/PrivacyProtectionSystem/data/ICDAR-2013/val"   
    words_dir = "D:/PrivacyProtectionSystem/data/ICDAR-2013/words"
    
    samples_test = load_dataset_icdar2013c(images_dir, test_xml_dir, words_dir)
    samples_val = load_dataset_icdar2013c(images_dir, val_xml_dir, words_dir)
    all_samples = samples_test + samples_val
    
    print(f"加载了 {len(all_samples)} 个样本")
    if all_samples:
        sample = all_samples[0]
        print(f"示例图像: {sample['image_id']}")
        print(f"单元格数量: {len(sample['cells'])}")
        for i, cell in enumerate(sample['cells'][:30]):  # 显示前3个
            print(f"  单元格 {i}: text='{cell['text']}', bbox={cell['bbox']}")