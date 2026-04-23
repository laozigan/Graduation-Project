#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AdvBox-backed ROI adversarial generator.

This module imports adversarialbox from the local AdvBox repository first,
then wraps OCR recognition as an AdvBox Model-compatible class. The attack
objective is non-targeted: maximize text perplexity so recognized text tends
to drift from the original sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class AdvBoxROIAttackParams:
    epsilon: float
    alpha: float
    steps: int
    epsilon_steps: int
    attack_name: str
    spsa_sigma: float
    spsa_samples: int
    text_change_bonus: float


def _ensure_local_advbox_path() -> None:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    local_advbox = project_root / "AdvBox"
    if local_advbox.exists():
        local_path = str(local_advbox)
        if local_path not in sys.path:
            sys.path.insert(0, local_path)


def _import_advbox_symbols() -> tuple[Any, Any, Any, Any, Any]:
    import collections
    import collections.abc

    # advbox 0.4.1 uses legacy Iterable import.
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable

    _ensure_local_advbox_path()

    from adversarialbox.adversary import Adversary
    from adversarialbox.attacks.gradient_method import BIM, FGSM, MIFGSM
    from adversarialbox.models.base import Model

    return Model, Adversary, FGSM, BIM, MIFGSM


def _iter_prediction_items(pred_output: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if any(k in obj for k in ("rec_text", "rec_score", "text", "score")):
                items.append(obj)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)

    _walk(pred_output)
    return items


def _recognize_text_and_confidence(recognizer: Any, image_bgr_uint8: np.ndarray) -> tuple[str, float]:
    try:
        pred = recognizer.predict(image_bgr_uint8)
    except Exception:
        return "", 1e-6

    items = _iter_prediction_items(pred)
    texts: list[str] = []
    scores: list[float] = []

    for item in items:
        txt = str(item.get("rec_text", item.get("text", ""))).strip()
        if txt:
            texts.append(txt)

        score = item.get("rec_score", item.get("score", None))
        if score is None:
            continue
        try:
            scores.append(float(score))
        except Exception:
            continue

    joined = " ".join(texts).strip()
    if not scores:
        # Unknown confidence is treated as very uncertain.
        return joined, 1e-6

    conf = float(np.mean(np.clip(np.asarray(scores, dtype=np.float32), 1e-6, 1.0)))
    return joined, conf


class AdvBoxPaddleModelCompat:
    """AdvBox Model-compatible wrapper around OCR recognition.

    This class behaves like an adversarialbox model and exposes:
    - predict(data)
    - gradient(data, label)
    - bounds/channel_axis/num_classes/predict_name

    The objective score is perplexity-like, based on recognition confidence:
        score = exp(-log(confidence)) = 1 / confidence
    plus optional bonus when recognized text changes from original text.
    """

    def __init__(
        self,
        model_cls: Any,
        recognizer: Any,
        original_roi: np.ndarray,
        writable_3c: np.ndarray,
        params: AdvBoxROIAttackParams,
        rng: np.random.Generator,
    ) -> None:
        self._recognizer = recognizer
        self._mask = writable_3c.astype(np.float32)
        self._params = params
        self._rng = rng

        self._original_roi = np.clip(original_roi.astype(np.float32), 0.0, 255.0)
        self._original_text, _ = _recognize_text_and_confidence(
            self._recognizer,
            self._original_roi.astype(np.uint8),
        )

        class _InnerModel(model_cls):
            def __init__(self, outer: "AdvBoxPaddleModelCompat") -> None:
                super().__init__(bounds=(0.0, 255.0), channel_axis=3, preprocess=(0.0, 1.0))
                self._outer = outer

            def predict(self, data: np.ndarray) -> np.ndarray:
                score = self._outer._objective(data)
                # Keep original label at index 0 so non-targeted attacks move away from it.
                return np.array([score, -score], dtype=np.float32)

            def num_classes(self) -> int:
                return 2

            def gradient(self, data: np.ndarray, label: int) -> np.ndarray:
                # Estimate gradient of objective via SPSA.
                grad = self._outer._spsa_gradient(data)
                # label=0 corresponds to objective class, maintain sign consistency.
                return grad if int(label) == 0 else -grad

            def predict_name(self) -> str:
                return "ocr_perplexity"

        self.model = _InnerModel(self)

    def _objective(self, data: np.ndarray) -> float:
        roi = np.clip(data.astype(np.float32), 0.0, 255.0)
        roi = self._original_roi + (roi - self._original_roi) * self._mask
        roi = np.clip(roi, 0.0, 255.0).astype(np.uint8)

        rec_text, conf = _recognize_text_and_confidence(self._recognizer, roi)
        conf = float(max(1e-6, min(1.0, conf)))

        nll = -np.log(conf)
        perplexity = float(np.exp(nll))

        bonus = 0.0
        if self._original_text and rec_text and rec_text != self._original_text:
            bonus = float(self._params.text_change_bonus)

        # Keep objective sensitive even when OCR confidence is locally flat.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture_term = float(np.mean(np.abs(lap)) / 255.0)

        return perplexity + bonus + 0.05 * texture_term

    def _spsa_gradient(self, data: np.ndarray) -> np.ndarray:
        sigma = float(max(1e-4, self._params.spsa_sigma))
        samples = int(max(1, self._params.spsa_samples))

        x = np.clip(data.astype(np.float32), 0.0, 255.0)
        grad = np.zeros_like(x, dtype=np.float32)

        for _ in range(samples):
            direction = self._rng.choice([-1.0, 1.0], size=x.shape).astype(np.float32)
            x_plus = np.clip(x + sigma * direction, 0.0, 255.0)
            x_minus = np.clip(x - sigma * direction, 0.0, 255.0)

            f_plus = self._objective(x_plus)
            f_minus = self._objective(x_minus)
            grad += ((f_plus - f_minus) / (2.0 * sigma)) * direction

        grad /= float(samples)
        return grad * self._mask


def _pgd_maximize(
    model: AdvBoxPaddleModelCompat,
    start_roi: np.ndarray,
    params: AdvBoxROIAttackParams,
) -> np.ndarray:
    eps = float(max(0.0, params.epsilon))
    alpha = float(max(1e-6, params.alpha))
    steps = int(max(1, params.steps))

    x0 = np.clip(start_roi.astype(np.float32), 0.0, 255.0)
    adv = x0.copy()

    for _ in range(steps):
        grad = model.model.gradient(adv, 0)
        adv = adv + alpha * np.sign(grad)
        adv = np.clip(adv, x0 - eps, x0 + eps)
        adv = np.clip(adv, 0.0, 255.0)

    return adv


def run_advbox_roi_attack(
    roi_delta: np.ndarray,
    original_roi: np.ndarray,
    writable_3c: np.ndarray,
    params: AdvBoxROIAttackParams,
    recognizer: Any,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Run AdvBox-backed attack for one ROI and return pixel-space delta.

    Returns None when adversarialbox import/execution fails so caller can
    fallback to internal optimizer.
    """
    if recognizer is None:
        return None
    if int(np.count_nonzero(writable_3c)) <= 0:
        return roi_delta

    try:
        model_cls, adversary_cls, fgsm_cls, bim_cls, mifgsm_cls = _import_advbox_symbols()
    except Exception:
        return None

    start_roi = np.clip(original_roi.astype(np.float32) + roi_delta.astype(np.float32), 0.0, 255.0)
    wrapped = AdvBoxPaddleModelCompat(
        model_cls=model_cls,
        recognizer=recognizer,
        original_roi=original_roi,
        writable_3c=writable_3c,
        params=params,
        rng=rng,
    )

    attack_name = (params.attack_name or "PGD").strip().upper()

    try:
        if attack_name == "PGD":
            adv_img = _pgd_maximize(wrapped, start_roi, params)
        else:
            attack_cls = bim_cls
            if attack_name == "FGSM":
                attack_cls = fgsm_cls
            elif attack_name == "MIFGSM":
                attack_cls = mifgsm_cls

            attack = attack_cls(wrapped.model)
            adversary = adversary_cls(start_roi.astype(np.float32), original_label=None)
            adversary.set_target(is_targeted_attack=False)
            adversary = attack(
                adversary,
                epsilons=float(params.alpha),
                epsilons_max=float(params.epsilon),
                steps=max(1, int(params.steps)),
                epsilon_steps=max(1, int(params.epsilon_steps)),
            )
            adv_img = adversary.adversarial_example
            if adv_img is None:
                adv_img = adversary.bad_adversarial_example
            if adv_img is None:
                return roi_delta
    except Exception:
        return None

    eps = float(params.epsilon)
    out_delta = np.clip(adv_img.astype(np.float32) - original_roi.astype(np.float32), -eps, eps)
    out_delta = out_delta * writable_3c.astype(np.float32)
    return out_delta.astype(np.float32)
