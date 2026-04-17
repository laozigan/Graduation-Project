import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from typing import List, Dict, Optional


def get_poppler_path(default_path: Optional[str] = None) -> Optional[str]:
    """返回 Poppler 可执行文件路径，可从环境变量读取。"""
    if default_path and os.path.exists(default_path):
        return default_path
    return os.getenv("POPPLER_PATH")


def convert_to_opencv(pil_image) -> np.ndarray:
    """将 PIL.Image 转换为 OpenCV BGR 图像。"""
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(img)


def load_images_from_file(file_path: str, dpi: int = 200, poppler_path: Optional[str] = None) -> List[np.ndarray]:
    """从图片或 PDF 文件加载 OpenCV 图像列表。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"无法读取图片: {file_path}")
        return [img]
    if ext == ".pdf":
        if poppler_path:
            pil_imgs = convert_from_path(file_path, dpi=dpi, poppler_path=poppler_path)
        else:
            pil_imgs = convert_from_path(file_path, dpi=dpi)
        return [convert_to_opencv(pil) for pil in pil_imgs]
    raise ValueError(f"不支持的文件格式: {ext}")


def convert_ocr_result_to_boxes(ocr_result) -> List[Dict]:
    """将 PaddleOCR 返回结果转成标准 bbox/text 字典。"""
    boxes = []
    if not ocr_result:
        return boxes

    # Handle new PaddleOCR format (OCRResult object or dict-like object)
    if hasattr(ocr_result, 'get') or hasattr(ocr_result, '__getitem__'):
        try:
            rec_texts = ocr_result.get('rec_texts', None) if hasattr(ocr_result, 'get') else None
        except Exception:
            rec_texts = None
        if rec_texts is None:
            try:
                rec_texts = ocr_result['rec_texts']
            except Exception:
                rec_texts = []

        try:
            rec_polys = ocr_result.get('rec_polys', None) if hasattr(ocr_result, 'get') else None
        except Exception:
            rec_polys = None
        if rec_polys is None:
            try:
                rec_polys = ocr_result['rec_polys']
            except Exception:
                rec_polys = []

        for i, text in enumerate(rec_texts):
            if isinstance(text, tuple):
                text = text[0]
            if isinstance(text, str) and text.strip():
                if i >= len(rec_polys):
                    continue
                poly = rec_polys[i]
                poly_array = np.array(poly)
                if poly_array.size == 0:
                    continue
                x_min, y_min = poly_array.min(axis=0)
                x_max, y_max = poly_array.max(axis=0)
                bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
                boxes.append({"bbox": bbox, "text": text})
        return boxes

    # Handle old PaddleOCR format (list of [bbox_points, (text, confidence)])
    for line in ocr_result:
        if len(line) >= 2:
            bbox_points = line[0]
            text_conf = line[1]
            if isinstance(text_conf, tuple) and len(text_conf) >= 1:
                text = text_conf[0].strip()
                if text:
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    boxes.append({"bbox": bbox, "text": text})

    return boxes


def cluster_text_boxes(ocr_results: List[Dict], vertical_threshold: int = 15, horizontal_threshold: int = 30) -> List[Dict]:
    """将 OCR 文本框聚类成表格单元格。"""
    if not ocr_results:
        return []

    heights = [r["bbox"][3] - r["bbox"][1] for r in ocr_results if r.get("bbox")]
    median_height = float(np.median(heights)) if heights else 0.0
    row_threshold = max(5.0, min(float(vertical_threshold), median_height * 0.6))

    def overlap_ratio(a, b):
        top = max(a[1], b[1])
        bottom = min(a[3], b[3])
        overlap = max(0.0, bottom - top)
        min_h = max(1.0, min(a[3] - a[1], b[3] - b[1]))
        return overlap / min_h

    for r in ocr_results:
        r["cy"] = (r["bbox"][1] + r["bbox"][3]) / 2

    sorted_by_y = sorted(ocr_results, key=lambda x: x["cy"])
    rows = []
    current_row = [sorted_by_y[0]]
    for box in sorted_by_y[1:]:
        if abs(box["cy"] - current_row[0]["cy"]) <= row_threshold and overlap_ratio(box["bbox"], current_row[0]["bbox"]) >= 0.2:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)

    cells = []
    for row in rows:
        row.sort(key=lambda x: x["bbox"][0])
        merged = []
        current = row[0]
        for next_box in row[1:]:
            gap = next_box["bbox"][0] - current["bbox"][2]
            if gap <= horizontal_threshold and overlap_ratio(current["bbox"], next_box["bbox"]) >= 0.3:
                new_bbox = [
                    min(current["bbox"][0], next_box["bbox"][0]),
                    min(current["bbox"][1], next_box["bbox"][1]),
                    max(current["bbox"][2], next_box["bbox"][2]),
                    max(current["bbox"][3], next_box["bbox"][3]),
                ]
                new_text = current["text"] + " " + next_box["text"]
                current = {"bbox": new_bbox, "text": new_text}
            else:
                merged.append(current)
                current = next_box
        merged.append(current)
        cells.extend(merged)

    for c in cells:
        c.pop("cy", None)
    return cells


def run_ocr_on_image(img: np.ndarray,
                      ocr_model,
                      use_textline_orientation: bool = False,
                      **predict_kwargs):
    """Run OCR on a single image using PaddleOCR predict or ocr API."""
    if hasattr(ocr_model, 'predict'):
        try:
            return ocr_model.predict(
                img,
                use_textline_orientation=use_textline_orientation,
                **predict_kwargs,
            )
        except TypeError:
            # Some PaddleOCR versions accept a single img argument only
            return ocr_model.predict(img, **predict_kwargs)
    if hasattr(ocr_model, 'ocr'):
        return ocr_model.ocr(img)
    raise AttributeError("OCR model does not support predict or ocr methods")


def extract_cells_from_image(img: np.ndarray,
                             ocr_model,
                             vertical_threshold: int = 15,
                             horizontal_threshold: int = 30,
                             ocr_predict_kwargs: Optional[Dict] = None) -> List[Dict]:
    """对单张图像进行 OCR 并提取聚类后的单元格。"""
    ocr_predict_kwargs = ocr_predict_kwargs or {}
    result = run_ocr_on_image(
        img,
        ocr_model,
        use_textline_orientation=ocr_predict_kwargs.pop('use_textline_orientation', False),
        **ocr_predict_kwargs,
    )
    if not result:
        return []
    ocr_result = result[0] if isinstance(result, (list, tuple)) else result
    raw_boxes = convert_ocr_result_to_boxes(ocr_result)
    return cluster_text_boxes(raw_boxes, vertical_threshold, horizontal_threshold)


def draw_boxes(img: np.ndarray,
               boxes: List[Dict],
               color: tuple = (0, 255, 0),
               label_prefix: str = None,
               show_text: bool = False,
               text_scale: float = 0.4,
               text_thickness: int = 1,
               box_thickness: int = 2) -> np.ndarray:
    """在图像上绘制边界框，并可选显示标签。"""
    canvas = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box['bbox'])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)
        if show_text:
            display_text = box.get('text', '')[:50].replace('\n', ' ')
            if display_text:
                label = display_text if label_prefix is None else f"{label_prefix}: {display_text}"
                cv2.putText(canvas, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale, color, text_thickness, cv2.LINE_AA)
    return canvas


def draw_cells_on_image(img: np.ndarray,
                        cells: List[Dict],
                        sensitive_key: str = None,
                        show_text: bool = False,
                        text_scale: float = 0.4,
                        text_thickness: int = 1,
                        box_thickness: int = 2) -> np.ndarray:
    """在图像上绘制提取到的单元格框，并可选显示文本标签。"""
    canvas = img.copy()
    for cell in cells:
        x1, y1, x2, y2 = map(int, cell['bbox'])
        is_sensitive = False
        label = None
        if sensitive_key and isinstance(cell.get(sensitive_key), dict):
            sensitive = cell[sensitive_key]
            is_sensitive = sensitive.get('is_sensitive', False)
            label = sensitive.get('type') if is_sensitive else None
        color = (0, 0, 255) if is_sensitive else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)
        if show_text:
            display_text = cell.get('text', '')[:50].replace('\n', ' ')
            if display_text:
                text = display_text if label is None else f"{label}: {display_text}"
                cv2.putText(canvas, text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale, color, text_thickness, cv2.LINE_AA)
        elif label:
            cv2.putText(canvas, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale, color, text_thickness, cv2.LINE_AA)
    return canvas


def draw_gt_vs_ocr(img: np.ndarray,
                   gt_cells: List[Dict],
                   ocr_cells: List[Dict],
                   show_text: bool = False,
                   text_scale: float = 0.4,
                   text_thickness: int = 1,
                   box_thickness: int = 2) -> np.ndarray:
    """在图像上绘制 GT 和 OCR 结果对比图。"""
    canvas = img.copy()
    canvas = draw_boxes(canvas, gt_cells, color=(0, 255, 0), label_prefix='GT', show_text=show_text,
                        text_scale=text_scale, text_thickness=text_thickness, box_thickness=box_thickness)
    canvas = draw_boxes(canvas, ocr_cells, color=(0, 0, 255), label_prefix='OCR', show_text=show_text,
                        text_scale=text_scale, text_thickness=text_thickness, box_thickness=1)
    return canvas


def visualize_cells_on_file(file_path: str,
                            cells_by_page: List[List[Dict]],
                            output_dir: str,
                            dpi: int = 200,
                            poppler_path: Optional[str] = None,
                            sensitive_key: str = None,
                            show_text: bool = False) -> List[str]:
    """可视化每页提取结果并保存图像。"""
    os.makedirs(output_dir, exist_ok=True)
    images = load_images_from_file(file_path, dpi=dpi, poppler_path=poppler_path)
    output_paths = []
    for page_idx, img in enumerate(images):
        cells = cells_by_page[page_idx] if page_idx < len(cells_by_page) else []
        viz = draw_cells_on_image(img, cells, sensitive_key=sensitive_key, show_text=show_text)
        output_path = os.path.join(output_dir, f"visualized_page_{page_idx+1}.jpg")
        cv2.imwrite(output_path, viz)
        output_paths.append(output_path)
    return output_paths


def extract_cells_from_file(file_path: str,
                            ocr_model,
                            dpi: int = 200,
                            poppler_path: Optional[str] = None,
                            vertical_threshold: int = 15,
                            horizontal_threshold: int = 30,
                            ocr_predict_kwargs: Optional[Dict] = None) -> List[List[Dict]]:
    """从文件读取图像并提取每页的表格单元格。"""
    images = load_images_from_file(file_path, dpi=dpi, poppler_path=poppler_path)
    all_pages = []
    for img in images:
        cells = extract_cells_from_image(
            img,
            ocr_model,
            vertical_threshold,
            horizontal_threshold,
            ocr_predict_kwargs=ocr_predict_kwargs,
        )
        all_pages.append(cells)
    return all_pages


class TableExtractor:
    """表格信息提取器。"""

    def __init__(self,
                 ocr_model,
                 vertical_threshold: int = 15,
                 horizontal_threshold: int = 30,
                 poppler_path: Optional[str] = None,
                 ocr_predict_kwargs: Optional[Dict] = None):
        self.ocr_model = ocr_model
        self.vertical_threshold = vertical_threshold
        self.horizontal_threshold = horizontal_threshold
        self.poppler_path = poppler_path
        self.ocr_predict_kwargs = ocr_predict_kwargs or {}

    def extract_from_image(self, img: np.ndarray) -> List[Dict]:
        return extract_cells_from_image(
            img,
            self.ocr_model,
            self.vertical_threshold,
            self.horizontal_threshold,
            ocr_predict_kwargs=self.ocr_predict_kwargs,
        )

    def extract_from_file(self, file_path: str, dpi: int = 200) -> List[List[Dict]]:
        return extract_cells_from_file(
            file_path,
            self.ocr_model,
            dpi=dpi,
            poppler_path=self.poppler_path,
            vertical_threshold=self.vertical_threshold,
            horizontal_threshold=self.horizontal_threshold,
            ocr_predict_kwargs=self.ocr_predict_kwargs,
        )

    def visualize_image(self,
                        img: np.ndarray,
                        output_path: str,
                        cells: List[Dict],
                        sensitive_field: str = 'sensitive',
                        show_text: bool = False) -> None:
        """保存带标注的图像。"""
        viz = draw_cells_on_image(img, cells, sensitive_key=sensitive_field, show_text=show_text)
        cv2.imwrite(output_path, viz)

    def visualize_gt_vs_ocr(self,
                             img: np.ndarray,
                             gt_cells: List[Dict],
                             ocr_cells: List[Dict],
                             output_path: str,
                             show_text: bool = False) -> None:
        """保存 GT 与 OCR 结果对比图。"""
        viz = draw_gt_vs_ocr(img, gt_cells, ocr_cells, show_text=show_text)
        cv2.imwrite(output_path, viz)
