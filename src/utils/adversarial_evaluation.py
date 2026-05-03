import os
import cv2
import numpy as np
from math import log10, sqrt
from rapidfuzz.distance import Levenshtein
from skimage.metrics import structural_similarity as ssim

class AdversarialEvaluator:
    """评估对抗样本效果的客观指标计算工具类"""
    
    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """计算峰值信噪比（PSNR），越高越保真"""
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return cv2.PSNR(img1, img2)

    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """计算结构相似性（SSIM），范围0~1，越高说明结构保留越完整"""
        # 转为灰度图进行计算
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 确保窗口大小不大于图像边长
        win_size = min(7, gray1.shape[0], gray1.shape[1])
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            return 1.0 # 如果图像太小，默认完全一致
            
        score, _ = ssim(gray1, gray2, win_size=win_size, full=True)
        return float(score)

    @staticmethod
    def calculate_cer(text_orig: str, text_adv: str) -> float:
        """计算字符错误率（CER），反映攻击成功率"""
        if not text_orig:
            return 0.0 if not text_adv else 1.0
        dist = Levenshtein.distance(text_orig, text_adv)
        return dist / len(text_orig)

    @staticmethod
    def _bbox_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[2], box1[3]
        x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[2], box2[3]

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
        box2_area = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)

        union_area = box1_area + box2_area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @classmethod
    def evaluate_page(cls, orig_img: np.ndarray, adv_img: np.ndarray, orig_cells: list, adv_cells: list) -> dict:
        """
        全量评估一张图片及其 OCR 结果
        :param orig_img: 原图数组
        :param adv_img: 对抗图数组
        :param orig_cells: 原图 OCR/提取 单元格列表，包含 'bbox', 'text', 'sensitives'
        :param adv_cells: 对抗图 OCR/提取 单元格列表
        :return: 指标汇总字典
        """
        # 1. 视觉保真度 (Visual Fidelity)
        # 如果尺寸不一致，将其对齐（理论上应该一致）
        if orig_img.shape != adv_img.shape:
            adv_img = cv2.resize(adv_img, (orig_img.shape[1], orig_img.shape[0]))
            
        psnr_val = cls.calculate_psnr(orig_img, adv_img)
        ssim_val = cls.calculate_ssim(orig_img, adv_img)
        
        # 2. 对抗效果与表格完整性 (Adversarial Effectiveness & Table Integrity)
        sensitive_orig_texts = []
        sensitive_adv_texts = []
        
        non_sensitive_iou_sum = 0
        non_sensitive_count = 0
        
        # 匹配单元格
        for o_cell in orig_cells:
            o_bbox = o_cell.get('bbox', [0, 0, 0, 0])
            o_text = o_cell.get('text', '')
            sensitives = o_cell.get('sensitives', [])
            
            # 判断是否为敏感区域（只要匹配到了任何敏感正则/Uie即可被当做敏感块处理）
            is_sensitive = False
            for s in sensitives:
                if s.get('is_sensitive') or s.get('type') != 'unknown': # According to your system's output
                    is_sensitive = True
                    break
            # Fallback if the detector marks the whole cell by just having the 'sensitives' list non-empty
            if sensitives:
                is_sensitive = True
                
            # 找到在 adv_cells 中 IOU 最大的单元格及对应文本
            best_iou = 0
            best_adv_text = ""
            for a_cell in adv_cells:
                a_bbox = a_cell.get('bbox', [0, 0, 0, 0])
                iou = cls._bbox_iou(o_bbox, a_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_adv_text = a_cell.get('text', '')
                    
            if is_sensitive:
                sensitive_orig_texts.append(o_text)
                sensitive_adv_texts.append(best_adv_text)
            else:
                non_sensitive_iou_sum += best_iou
                non_sensitive_count += 1
                
        # 计算总体字符错误率
        total_cer = 0.0
        if sensitive_orig_texts:
            cers = [cls.calculate_cer(o, a) for o, a in zip(sensitive_orig_texts, sensitive_adv_texts)]
            total_cer = sum(cers) / len(cers)
            
        # 计算结构框保留度（作为表格完整性的代理指标）
        table_integrity = (non_sensitive_iou_sum / non_sensitive_count) if non_sensitive_count > 0 else 1.0

        return {
            "psnr": float(psnr_val),
            "ssim": float(ssim_val),
            "sensitive_cer": float(total_cer),
            "table_integrity_iou": float(table_integrity),
            "num_sensitive_cells": len(sensitive_orig_texts)
        }

    @classmethod
    def detection_consistency(cls, orig_cells: list, adv_cells: list, iou_threshold: float = 0.5) -> dict:
        """Compare sensitive detection between original and adversarial cells.

        Treat `orig_cells` as the reference (ground truth) and compute precision/recall/f1
        of the adversarial detection relative to it. Returns dict with `pre` and `post` metrics
        where `pre` is the reference (all ones when reference non-empty), `post` are measured,
        and `delta` is post - pre.
        """
        def best_types(cell):
            types = set()
            sens = cell.get('sensitives') or []
            for s in sens:
                t = s.get('type')
                if t:
                    if t == 'chinese_name':
                        t = 'name'
                    types.add(t)
            return types

        # match adv cells to orig by bbox IoU
        matched = []  # list of (orig, adv)
        used_adv = set()
        for o in orig_cells:
            o_bbox = o.get('bbox', [0, 0, 0, 0])
            best_idx = -1
            best_iou = 0.0
            for idx, a in enumerate(adv_cells):
                if idx in used_adv:
                    continue
                a_bbox = a.get('bbox', [0, 0, 0, 0])
                iou = cls._bbox_iou(o_bbox, a_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                matched.append((o, adv_cells[best_idx]))
                used_adv.add(best_idx)

        tp = 0
        fp = 0
        fn = 0
        # count by types
        for o, a in matched:
            o_types = best_types(o)
            a_types = best_types(a)
            # true positives: types present in both
            for t in o_types:
                if t in a_types:
                    tp += 1
                else:
                    fn += 1
            # false positives: adv types not in orig
            for t in a_types:
                if t not in o_types:
                    fp += 1

        # Unmatched orig cells count as FN for their types
        matched_orig = {id(o) for o, _ in matched}
        for o in orig_cells:
            if id(o) not in matched_orig:
                o_types = best_types(o)
                fn += len(o_types)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if tp == 0 and fn == 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if tp == 0 and fp == 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        pre = {"precision": 1.0 if (tp + fp + fn) >= 0 else None, "recall": 1.0, "f1": 1.0}
        post = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        delta = {k: round(post[k] - pre[k], 4) for k in pre}
        return {"pre": pre, "post": post, "delta": delta}
