import os
import re
from typing import Dict, List, Optional


# UIE Taskflow may fail under PIR mode in some Paddle/PaddleNLP combinations.
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"


class UIEXNameExtractor:
    """Optional UIE-X based name extractor with graceful fallback."""

    def __init__(self, model: str = "uie-x-base", schema: Optional[List[str]] = None, allow_model_fallback: bool = False):
        self.model = model
        self.schema = schema or ["姓名", "名字", "联系人", "申请人"]
        self.allow_model_fallback = allow_model_fallback
        self.available = False
        self._task = None
        self.init_error = None
        self._try_init_task()

    def _try_init_task(self) -> None:
        try:
            from paddlenlp import Taskflow
        except Exception as exc:
            self.init_error = f"paddlenlp import failed: {exc}"
            self.available = False
            return

        model_candidates = [self.model]
        if self.allow_model_fallback and self.model != "uie-base":
            model_candidates.append("uie-base")

        last_error = None
        for model_name in model_candidates:
            try:
                self._task = Taskflow(
                    "information_extraction",
                    schema=self.schema,
                    model=model_name,
                )
                self.model = model_name
                self.available = True
                self.init_error = None
                return
            except Exception as exc:
                last_error = exc

        self.init_error = f"Taskflow init failed: {last_error}"
        self.available = False

    def extract_names(self, text: str) -> List[Dict]:
        if not self.available or not text or not text.strip():
            return []

        try:
            raw_result = self._task(text)
        except Exception:
            return []

        rows = raw_result if isinstance(raw_result, list) else [raw_result]
        candidates = []
        for row in rows:
            if isinstance(row, dict):
                candidates.extend(self._collect_name_candidates(row))

        return self._deduplicate(candidates)

    def _collect_name_candidates(self, row: Dict) -> List[Dict]:
        candidates = []
        for label, spans in row.items():
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict):
                    continue
                text = str(span.get("text", "")).strip()
                if self._is_valid_name(text):
                    candidates.append(
                        {
                            "text": text,
                            "confidence": float(span.get("probability", 0.0) or 0.0),
                            "label": str(label),
                        }
                    )
                relations = span.get("relations")
                if isinstance(relations, dict):
                    candidates.extend(self._collect_name_candidates(relations))
        return candidates

    def _is_valid_name(self, candidate: str) -> bool:
        if not candidate:
            return False

        # Accept common Chinese name styles, including names with middle dot.
        if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", candidate):
            return True
        if re.fullmatch(r"[\u4e00-\u9fa5]{1,3}[·•][\u4e00-\u9fa5]{1,3}", candidate):
            return True
        return False

    def _deduplicate(self, candidates: List[Dict]) -> List[Dict]:
        by_text: Dict[str, Dict] = {}
        for item in candidates:
            name_text = item["text"]
            prev = by_text.get(name_text)
            if prev is None or item["confidence"] > prev["confidence"]:
                by_text[name_text] = item

        return sorted(by_text.values(), key=lambda x: x["confidence"], reverse=True)
