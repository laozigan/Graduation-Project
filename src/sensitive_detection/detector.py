import re
from .patterns import REGEX_PATTERNS, KEYWORD_CATEGORIES, CHINESE_NAME_REGEX

class SensitiveDetector:
    def __init__(self, use_nlp=False):
        self.regex_patterns = REGEX_PATTERNS
        self.keyword_categories = KEYWORD_CATEGORIES
        self.chinese_name_pattern = re.compile(CHINESE_NAME_REGEX)
        self.name_blacklist = {
            '联系方式', '联系人', '联系电话', '用户名', '姓名', '名字', '名称',
            '电话', '手机', '地址', '住址', '公司', '单位', '部门', '职位',
            '籍贯', '经验', '工作经验', '项目经验'
        }
        self.use_nlp = use_nlp
        self.has_jieba = False
        self.name_context_patterns = [
            re.compile(r'(?:我叫|我名叫|我姓|姓名(?:是|：|:)?|名字(?:是|：|:)?|名叫|称呼(?:是|：|:)?)\s*([\u4e00-\u9fa5]{2,3})'),
            re.compile(r'([\u4e00-\u9fa5]{2,3})\s*(?:是|为)\s*(?:我的)?(?:名字|姓名)'),
        ]
        if use_nlp:
            self._init_nlp()

    def _init_nlp(self):
        """初始化轻量级NLP（例如基于jieba分词+规则）"""
        try:
            import jieba
            self.jieba = jieba
            self.has_jieba = True
        except ImportError:
            print("Warning: jieba not installed, jieba tokenization disabled")
            self.has_jieba = False

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

        # 0. 强语境优先：社保号标签下的数值优先归类为 social_security，避免被身份证正则抢先命中
        social_label_match = re.search(r'(?:社保号|社保编号|社会保障号|社会保险号|医保号|养老保险号)\s*[：:]\s*([A-Za-z0-9\-]{8,25})', text_clean)
        if social_label_match:
            value = social_label_match.group(1).strip()
            ssn_matches = re.findall(self.regex_patterns['social_security'], value)
            if ssn_matches:
                return {
                    'is_sensitive': True,
                    'type': 'social_security',
                    'confidence': 0.9,
                    'match_details': ssn_matches
                }

        # 1. 正则匹配（最高优先级）
        for sens_type, pattern in self.regex_patterns.items():
            # 出生日期属于高频普通信息，避免全局日期误报，仅在语境规则里检测
            if sens_type in ('birth_date', 'social_security', 'passport'):
                continue
            matches = re.findall(pattern, text_clean)
            if matches:
                return {
                    'is_sensitive': True,
                    'type': sens_type,
                    'confidence': 1.0,
                    'match_details': matches
                }

        # 2. 键值对标签提取：仅识别 label:value 中的值为敏感信息
        label_value_patterns = {
            'name': r'(?:姓名|名字|名称|病人姓名|客户姓名|学生姓名|员工姓名)\s*[：:]\s*([^\s,;，。]+)',
            'phone': r'(?:手机|电话|联系电话|手机号)\s*[：:]\s*([0-9\-\s]{7,20})',
            'id': r'(?:身份证|身份证号|公民身份号码|居民身份证号)\s*[：:]\s*([0-9Xx\-\s]{10,20})',
            'address': r'(?:地址|住址|家庭住址|通讯地址|现居住地)\s*[：:]\s*([^\s,;，。]+)',
            'birth_date': r'(?:出生日期|出生年月|生日|出生时间)\s*[：:]\s*([0-9０-９]{4}\s*[年./-]\s*[0-9０-９]{1,2}(?:\s*[月./-]\s*[0-9０-９]{1,2}\s*日?)?)',
            'social_security': r'(?:社保号|社保编号|社会保障号|社会保险号|医保号|养老保险号)\s*[：:]\s*([A-Za-z0-9\-]{8,25})',
            'passport': r'(?:护照号|护照号码|passport(?:\s*number)?)\s*[：:]\s*([A-Za-z0-9]{8,12})',
        }
        for label, pattern in label_value_patterns.items():
            match = re.search(pattern, text_clean)
            if match:
                value = match.group(1).strip()
                if label == 'name':
                    name_matches = self.chinese_name_pattern.findall(value)
                    if name_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'name',  # 键值对中的姓名类型
                            'confidence': 0.9,
                            'match_details': name_matches
                        }
                elif label == 'phone':
                    phone_matches = re.findall(self.regex_patterns['phone'], value)
                    if phone_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'phone',
                            'confidence': 1.0,
                            'match_details': phone_matches
                        }
                elif label == 'id':
                    id_matches = re.findall(self.regex_patterns['id_card'], value)
                    if id_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'id_card',
                            'confidence': 1.0,
                            'match_details': id_matches
                        }
                elif label == 'address':
                    # 地址值本身视为敏感（可扩展为具体地址验证）
                    return {
                        'is_sensitive': True,
                        'type': 'address',
                        'confidence': 0.8,
                        'match_details': [value]
                    }
                elif label == 'birth_date':
                    date_matches = re.findall(self.regex_patterns['birth_date'], value)
                    if date_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'birth_date',
                            'confidence': 0.9,
                            'match_details': date_matches
                        }
                elif label == 'social_security':
                    ssn_matches = re.findall(self.regex_patterns['social_security'], value)
                    if ssn_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'social_security',
                            'confidence': 0.9,
                            'match_details': ssn_matches
                        }
                elif label == 'passport':
                    passport_matches = re.findall(self.regex_patterns['passport'], value)
                    if passport_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'passport',
                            'confidence': 0.9,
                            'match_details': passport_matches
                        }

        # 3. 轻量级NLP优先检测姓名上下文
        if self.use_nlp:
            nlp_result = self._nlp_detect(text_clean)
            if nlp_result:
                return nlp_result

        # 4. 中文姓名正则检测（纯值，仅当不含标签且文本较短时）
        if not any(kw in text_clean for kw_list in self.keyword_categories.values() for kw in kw_list):
            if len(text_clean) <= 4 or self._is_name_context(text_clean):
                name_matches = self._filter_name_matches(self.chinese_name_pattern.findall(text_clean))
                if name_matches:
                    return {
                        'is_sensitive': True,
                        'type': 'chinese_name',
                        'confidence': 0.9,
                        'match_details': name_matches
                    }

        # 5. 关键词匹配（只作为辅助标签检测，不单独判定为敏感）
        # 关键词存在本身不足以判定敏感信息，必须依赖具体值或正则匹配。
        for cat, kw_list in self.keyword_categories.items():
            for kw in kw_list:
                if kw in text_clean:
                    continue

        return {'is_sensitive': False}

    def _is_name_context(self, text):
        """判断文本是否包含明确的人名语境"""
        for pattern in self.name_context_patterns:
            match = pattern.search(text)
            if match:
                candidate = match.group(1).strip()
                if self.chinese_name_pattern.fullmatch(candidate) and not self._is_blacklisted_name(candidate):
                    return True
        return False

    def _is_blacklisted_name(self, candidate):
        """过滤明显不是姓名的词，避免短词误识别。"""
        if not candidate:
            return True
        if candidate in self.name_blacklist:
            return True
        # 带有明显语义词后缀，通常不是人名
        for suffix in ('公司', '电话', '地址', '方式', '信息', '号码', '部门', '职位', '姓名', '经验', '籍贯'):
            if candidate.endswith(suffix):
                return True
        return False

    def _filter_name_matches(self, matches):
        """过滤中文姓名正则匹配结果中的误报。"""
        return [m for m in matches if not self._is_blacklisted_name(m)]

    def _extract_following_candidate(self, tokens, index, max_window=3):
        """从关键词后抽取可能的姓名值，兼容“李 雷”被分成两个 token 的情况。"""
        candidate_parts = []
        for offset in range(1, max_window + 1):
            pos = index + offset
            if pos >= len(tokens):
                break
            token = tokens[pos].strip()
            if not token or token in {'：', ':', '是', '为', '叫', '名叫'}:
                continue
            if re.fullmatch(r'[\u4e00-\u9fa5]{1,2}', token):
                candidate_parts.append(token)
                combined = ''.join(candidate_parts)
                if 2 <= len(combined) <= 3:
                    return combined
                if len(combined) > 3:
                    return None
            else:
                break
        return None

    def _nlp_detect(self, text):
        """基于简单规则或分词的检测"""
        # 先检测更加明确的姓名语境
        for pattern in self.name_context_patterns:
            match = pattern.search(text)
            if match:
                candidate = match.group(1).strip()
                if self.chinese_name_pattern.fullmatch(candidate) and not self._is_blacklisted_name(candidate):
                    return {
                        'is_sensitive': True,
                        'type': 'name',
                        'confidence': 0.9,
                        'match_details': [candidate]
                    }

        # 使用 jieba 分词进一步辅助判断，避免将句子中的片段误识别为人名
        if self.has_jieba:
            tokens = self.jieba.lcut(text)

            # 关键词 + 邻近值识别（姓名）
            name_keywords = {'姓名', '名字', '名叫', '称呼', '我叫', '我名叫'}
            for index, token in enumerate(tokens):
                if token in name_keywords:
                    candidate = self._extract_following_candidate(tokens, index)
                    if candidate and self.chinese_name_pattern.fullmatch(candidate) and not self._is_blacklisted_name(candidate):
                        return {
                            'is_sensitive': True,
                            'type': 'name',
                            'confidence': 0.9,
                            'match_details': [candidate]
                        }

            # 关键词 + 邻近值识别（电话/身份证/出生日期/社保号/护照号）
            phone_keywords = {'电话', '手机', '手机号', '联系电话'}
            id_keywords = {'身份证', '身份证号', '公民身份号码', '居民身份证号', 'id', 'ID'}
            birth_date_keywords = {'出生日期', '出生年月', '生日', '出生时间', 'DOB', 'dob'}
            social_security_keywords = {'社保号', '社保编号', '社会保障号', '社会保险号', '医保号', '养老保险号'}
            passport_keywords = {'护照号', '护照号码', 'passport', 'Passport'}
            for index, token in enumerate(tokens):
                if token in phone_keywords:
                    span = ''.join(tokens[index + 1:index + 5]).replace(' ', '')
                    phone_matches = re.findall(self.regex_patterns['phone'], span)
                    if phone_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'phone',
                            'confidence': 0.95,
                            'match_details': phone_matches
                        }
                if token in id_keywords:
                    span = ''.join(tokens[index + 1:index + 6]).replace(' ', '')
                    id_matches = re.findall(self.regex_patterns['id_card'], span)
                    if id_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'id_card',
                            'confidence': 0.95,
                            'match_details': id_matches
                        }
                if token in birth_date_keywords:
                    span = ''.join(tokens[index + 1:index + 8]).replace(' ', '')
                    date_matches = re.findall(self.regex_patterns['birth_date'], span)
                    if date_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'birth_date',
                            'confidence': 0.9,
                            'match_details': date_matches
                        }
                if token in social_security_keywords:
                    span = ''.join(tokens[index + 1:index + 6]).replace(' ', '')
                    ssn_matches = re.findall(self.regex_patterns['social_security'], span)
                    if ssn_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'social_security',
                            'confidence': 0.9,
                            'match_details': ssn_matches
                        }
                if token in passport_keywords:
                    span = ''.join(tokens[index + 1:index + 5]).replace(' ', '')
                    passport_matches = re.findall(self.regex_patterns['passport'], span)
                    if passport_matches:
                        return {
                            'is_sensitive': True,
                            'type': 'passport',
                            'confidence': 0.9,
                            'match_details': passport_matches
                        }

        # 检测“身份证号：”后面跟数字的模式
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