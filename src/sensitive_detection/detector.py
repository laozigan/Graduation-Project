import re
from .patterns import REGEX_PATTERNS, KEYWORD_CATEGORIES

class SensitiveDetector:
    def __init__(self, use_nlp=False):
        self.regex_patterns = REGEX_PATTERNS
        self.keyword_categories = KEYWORD_CATEGORIES
        self.use_nlp = use_nlp
        if use_nlp:
            self._init_nlp()

    def _init_nlp(self):
        """初始化轻量级NLP（例如基于jieba分词+规则）"""
        try:
            import jieba
            self.jieba = jieba
        except ImportError:
            print("Warning: jieba not installed, NLP disabled")
            self.use_nlp = False

    def detect(self, text: str):
        """
        检测单个文本中的敏感信息。
        返回: {
            'is_sensitive': bool,
            'type': str,          # 敏感类型
            'confidence': float,  # 0-1
            'match_details': [...] # 匹配到的具体内容（可选）
        }
        若无敏感信息，返回 {'is_sensitive': False}
        """
        if not text or not text.strip():
            return {'is_sensitive': False}

        text_clean = text.strip()
        # 1. 正则匹配（最高优先级）
        for sens_type, pattern in self.regex_patterns.items():
            matches = re.findall(pattern, text_clean)
            if matches:
                return {
                    'is_sensitive': True,
                    'type': sens_type,
                    'confidence': 1.0,
                    'match_details': matches
                }

        # 2. 关键词匹配
        for cat, kw_list in self.keyword_categories.items():
            for kw in kw_list:
                if kw in text_clean:
                    return {
                        'is_sensitive': True,
                        'type': cat,
                        'confidence': 0.8,
                        'match_details': [kw]
                    }

        # 3. 轻量级NLP
        if self.use_nlp:
            nlp_result = self._nlp_detect(text_clean)
            if nlp_result:
                return nlp_result

        return {'is_sensitive': False}

    def _nlp_detect(self, text):
        """基于简单规则或分词的检测"""
        # 示例：检测“身份证号：”后面跟数字的模式
        pattern = r'(身份证|id|ID)[\s:：]*([1-9]\d{5,})'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                'is_sensitive': True,
                'type': 'id_card',
                'confidence': 0.9,
                'match_details': [match.group(2)]
            }
        return None

    def detect_cells(self, cells):
        """
        批量检测单元格列表。
        cells: list of dict, 每个包含 'text' 和可选的 'bbox'
        返回: 每个单元格增加 'sensitive' 字段
        """
        for cell in cells:
            result = self.detect(cell['text'])
            cell['sensitive'] = result
        return cells