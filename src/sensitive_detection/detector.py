import re
from .patterns import REGEX_PATTERNS, KEYWORD_CATEGORIES, CHINESE_NAME_REGEX
from .uie_extractor import UIEXNameExtractor

class SensitiveDetector:
    def __init__(self, use_nlp=False, enable_uie=False, uie_model='uie-x-base', uie_schema=None):
        self.regex_patterns = REGEX_PATTERNS
        self.keyword_categories = KEYWORD_CATEGORIES
        self.chinese_name_pattern = re.compile(CHINESE_NAME_REGEX)
        self.name_blacklist = {
            '联系方式', '联系人', '联系电话', '用户名', '姓名', '名字', '名称',
            '电话', '手机', '地址', '住址', '公司', '单位', '部门', '职位',
            '籍贯', '经验', '工作经验', '项目经验'
        }
        self.use_nlp = use_nlp
        self.enable_uie = enable_uie
        self.has_jieba = False
        self.has_uie = False
        self.uie_model = uie_model
        self.uie_schema = uie_schema or ['姓名', '名字', '联系人', '申请人']
        self.uie_extractor = None
        self.name_context_patterns = [
            re.compile(r'(?:我叫|我名叫|我姓|姓名(?:是|：|:)?|名字(?:是|：|:)?|名叫|称呼(?:是|：|:)?)\s*([\u4e00-\u9fa5]{2,3})'),
            re.compile(r'([\u4e00-\u9fa5]{2,3})\s*(?:是|为)\s*(?:我的)?(?:名字|姓名)'),
        ]
        if use_nlp:
            self._init_nlp()
        if enable_uie:
            self._init_uie()

    def _init_nlp(self):
        """初始化轻量级NLP（例如基于jieba分词+规则）"""
        try:
            import jieba
            self.jieba = jieba
            self.has_jieba = True
        except ImportError:
            print("Warning: jieba not installed, jieba tokenization disabled")
            self.has_jieba = False

    def _init_uie(self):
        """初始化 UIE-X 抽取器（可选，失败时自动降级）。"""
        self.uie_extractor = UIEXNameExtractor(
            model=self.uie_model,
            schema=self.uie_schema,
            allow_model_fallback=False,
        )
        self.has_uie = bool(self.uie_extractor.available)
        if not self.has_uie:
            print(f"Warning: UIE extractor unavailable, fallback to rules/NLP. Detail: {self.uie_extractor.init_error}")

    def detect(self, text: str):
        """返回单一最佳分类，兼容旧接口。"""
        results = self.detect_all(text)
        return self._summarize_results(results)

    def detect_all(self, text: str):
        """
        检测单个文本中的敏感信息，返回多种类型命中结果。
        返回: [
            {
                'is_sensitive': True,
                'type': str,
                'confidence': float,
                'match_details': [...]
            },
            ...
        ]
        """
        if not text or not text.strip():
            return []

        text_clean = text.strip()
        results = []

        # 0. 强语境优先：社保号标签下的数值优先归类为 social_security，避免被身份证正则抢先命中
        social_label_match = re.search(r'(?:社保号|社保编号|社会保障号|社会保险号|医保号|养老保险号)\s*[：:]\s*([A-Za-z0-9\-]{8,25})', text_clean)
        if social_label_match:
            value = social_label_match.group(1).strip()
            ssn_matches = re.findall(self.regex_patterns['social_security'], value)
            if ssn_matches:
                self._append_result(results, 'social_security', 0.9, ssn_matches)

        id_matches = []
        # 1. 正则匹配（基础命中）
        for sens_type, pattern in self.regex_patterns.items():
            # 出生日期属于高频普通信息，避免全局日期误报，仅在语境规则里检测
            if sens_type in ('birth_date', 'social_security', 'passport'):
                continue
            matches = self._find_regex_matches(pattern, text_clean)
            if matches:
                if sens_type == 'id_card':
                    id_matches.extend(matches)
                confidence = 0.85 if sens_type == 'address' else 1.0
                if sens_type == 'phone':
                    matches = self._filter_phone_matches(matches, id_matches)
                self._append_result(results, sens_type, confidence, matches)

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
            if not match:
                continue
            value = match.group(1).strip()
            if label == 'name':
                name_matches = self.chinese_name_pattern.findall(value)
                if name_matches:
                    self._append_result(results, 'name', 0.9, name_matches)
            elif label == 'phone':
                phone_matches = re.findall(self.regex_patterns['phone'], value)
                phone_matches = self._filter_phone_matches(phone_matches, id_matches)
                if phone_matches:
                    self._append_result(results, 'phone', 1.0, phone_matches)
            elif label == 'id':
                id_matches = re.findall(self.regex_patterns['id_card'], value)
                if id_matches:
                    self._append_result(results, 'id_card', 1.0, id_matches)
            elif label == 'address':
                self._append_result(results, 'address', 0.8, [value])
            elif label == 'birth_date':
                date_matches = re.findall(self.regex_patterns['birth_date'], value)
                if date_matches:
                    self._append_result(results, 'birth_date', 0.9, date_matches)
            elif label == 'social_security':
                ssn_matches = re.findall(self.regex_patterns['social_security'], value)
                if ssn_matches:
                    self._append_result(results, 'social_security', 0.9, ssn_matches)
            elif label == 'passport':
                passport_matches = re.findall(self.regex_patterns['passport'], value)
                if passport_matches:
                    self._append_result(results, 'passport', 0.9, passport_matches)

        # 3. UIE-X 语义抽取（姓名增强）
        if self.enable_uie and self.has_uie:
            uie_result = self._uie_detect(text_clean)
            if uie_result:
                self._append_result(results, uie_result['type'], uie_result['confidence'], uie_result.get('match_details', []))

        # 4. 轻量级NLP兜底
        if self.use_nlp:
            nlp_result = self._nlp_detect(text_clean)
            if nlp_result:
                self._append_result(results, nlp_result['type'], nlp_result['confidence'], nlp_result.get('match_details', []))

        structured_types = {'id_card', 'phone', 'email', 'bank_card', 'address'}
        has_structured_pii = any(item.get('type') in structured_types for item in results)

        if has_structured_pii:
            leading_names = self._extract_leading_names(text_clean)
            if leading_names:
                self._append_result(results, 'name', 0.88, leading_names)

        # 5. 中文姓名正则检测（纯值，仅当不含标签且文本较短时）
        if not has_structured_pii and not any(kw in text_clean for kw_list in self.keyword_categories.values() for kw in kw_list):
            if len(text_clean) <= 4 or self._is_name_context(text_clean):
                name_matches = self._filter_name_matches(self.chinese_name_pattern.findall(text_clean))
                if name_matches:
                    self._append_result(results, 'name', 0.9, name_matches)

        return results

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
        for suffix in ('公司', '电话', '地址', '方式', '信息', '号码', '部门', '职位', '姓名', '经验', '籍贯', '盖章', '印章'):
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

    def _uie_detect(self, text):
        """使用 UIE-X 抽取姓名并返回统一结构。"""
        if not self.has_uie or self.uie_extractor is None:
            return None

        name_hits = self.uie_extractor.extract_names(text)
        if not name_hits:
            return None

        filtered_names = []
        max_confidence = 0.0
        for hit in name_hits:
            candidate = str(hit.get('text', '')).strip()
            if not candidate:
                continue
            if not re.fullmatch(r'[\u4e00-\u9fa5]{2,4}|[\u4e00-\u9fa5]{1,3}[·•][\u4e00-\u9fa5]{1,3}', candidate):
                continue
            if self._is_blacklisted_name(candidate):
                continue
            filtered_names.append(candidate)
            max_confidence = max(max_confidence, float(hit.get('confidence', 0.0) or 0.0))

        if not filtered_names:
            return None

        filtered_names = list(dict.fromkeys(filtered_names))
        keyword_boost = 0.05 if re.search(r'(姓名|名字|联系人|申请人|病人姓名|客户姓名|员工姓名)\s*[：:]?', text) else 0.0
        confidence = min(0.98, max(0.82, max_confidence + keyword_boost))
        return {
            'is_sensitive': True,
            'type': 'name',
            'confidence': confidence,
            'match_details': filtered_names
        }

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
            id_matches_in_text = self._find_regex_matches(self.regex_patterns['id_card'], text)

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
                    phone_matches = self._filter_phone_matches(phone_matches, id_matches_in_text)
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

    def _summarize_results(self, results):
        if not results:
            return {'is_sensitive': False}
        priority = [
            'id_card', 'phone', 'email', 'bank_card', 'social_security',
            'passport', 'address', 'birth_date', 'name',
            'id', 'medical'
        ]
        priority_index = {name: idx for idx, name in enumerate(priority)}
        best = sorted(
            results,
            key=lambda r: (-r.get('confidence', 0.0), priority_index.get(r.get('type'), 999))
        )[0]
        summary = {
            'is_sensitive': True,
            'type': best.get('type', 'unknown'),
            'confidence': best.get('confidence', 0.0),
            'match_details': best.get('match_details', [])
        }
        return summary

    def _find_regex_matches(self, pattern, text):
        return [m.group(0) for m in re.finditer(pattern, text)]

    def _append_result(self, results, sens_type, confidence, match_details):
        if not match_details:
            return
        for existing in results:
            if existing.get('type') == sens_type:
                details = existing.setdefault('match_details', [])
                for item in match_details:
                    if item not in details:
                        details.append(item)
                existing['confidence'] = max(existing.get('confidence', 0.0), confidence)
                return
        results.append({
            'is_sensitive': True,
            'type': sens_type,
            'confidence': confidence,
            'match_details': match_details
        })

    def _extract_leading_names(self, text):
        anchor_positions = []
        for key in ('id_card', 'phone', 'email', 'bank_card'):
            pattern = self.regex_patterns.get(key)
            if not pattern:
                continue
            match = re.search(pattern, text)
            if match:
                anchor_positions.append(match.start())

        if not anchor_positions:
            digit_match = re.search(r'\d', text)
            if digit_match:
                anchor_positions.append(digit_match.start())

        if not anchor_positions:
            return []
        anchor = min(anchor_positions)

        prefix = text[:anchor]
        if not prefix:
            return []
        prefix = re.sub(r'(姓名|身份证|身份证号|手机号|手机|电话|住址|地址|邮箱|电子邮箱|银行卡号)', ' ', prefix)
        prefix = re.sub(r'[，,。;；:：|/\\\-]+', ' ', prefix)
        matches = self.chinese_name_pattern.findall(prefix)
        filtered = self._filter_name_matches(matches)
        # 保持顺序去重
        return list(dict.fromkeys(filtered))

    def _filter_phone_matches(self, phone_matches, id_matches):
        if not phone_matches or not id_matches:
            return phone_matches
        filtered = []
        for phone in phone_matches:
            if any(phone in id_value for id_value in id_matches):
                continue
            filtered.append(phone)
        return filtered

    def detect_cells(self, cells):
        """
        批量检测单元格列表。
        cells: list of dict, 每个包含 'text' 和可选的 'bbox'
        返回: 每个单元格增加 'sensitives' 字段
        """
        for cell in cells:
            results = self.detect_all(cell['text'])
            cell['sensitives'] = results
        return cells