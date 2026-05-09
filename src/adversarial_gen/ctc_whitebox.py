#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""White-box CTC adversarial attack helpers for OCR recognition models.

This module keeps the core attack logic model-agnostic: it only requires a
callable that returns raw recognition logits before CTC decoding/postprocess.
When the supplied model is differentiable, the attack uses backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Tuple

import numpy as np


class CtcLogitsAdapter(Protocol):
    """Callable adapter that exposes raw logits for a single OCR crop."""

    supports_grad: bool

    def __call__(self, image_batch: Any) -> Any:
        ...


@dataclass
class CtcAttackConfig:
    epsilon: float = 24.0
    alpha: float = 6.0
    steps: int = 9
    blank_index: int = 0
    random_start: bool = True
    layout_hint: str = "auto"
    clip_min: float = 0.0
    clip_max: float = 1.0
    spsa_sigma: float = 2.0
    spsa_samples: int = 4


def load_charset(charset: Optional[str] = None, charset_path: Optional[str] = None) -> str:
    if charset:
        return charset
    if charset_path:
        path = Path(charset_path)
        if not path.exists():
            raise FileNotFoundError(f"charset file not found: {charset_path}")
        content = path.read_text(encoding="utf-8").splitlines()
        return "".join(line.strip("\ufeff") for line in content if line.strip())
    return "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def encode_text(text: str, charset: str, blank_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if not text:
        raise ValueError("target text is empty")

    char_to_id = {ch: idx + 1 for idx, ch in enumerate(charset)}
    labels = []
    for ch in text:
        if ch not in char_to_id:
            raise ValueError(f"character {ch!r} is not present in the provided charset")
        labels.append(char_to_id[ch])

    label_array = np.asarray(labels, dtype=np.int32)
    if blank_index in label_array:
        raise ValueError("blank_index collides with encoded labels")
    return label_array, np.asarray([len(label_array)], dtype=np.int64)


def _as_float32_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.float32:
        return image
    return image.astype(np.float32)


def _to_time_major_logits(logits: Any, layout_hint: str = "auto"):
    import paddle

    if not isinstance(logits, paddle.Tensor):
        logits = paddle.to_tensor(logits)

    if logits.ndim != 3:
        raise ValueError(f"expected a 3D logits tensor, got shape={list(logits.shape)}")

    hint = (layout_hint or "auto").strip().lower()
    if hint == "time_major":
        return logits
    if hint == "batch_major":
        return logits.transpose([1, 0, 2])

    if logits.shape[1] == 1 and logits.shape[0] != 1:
        return logits
    if logits.shape[0] == 1 and logits.shape[1] != 1:
        return logits.transpose([1, 0, 2])
    if logits.shape[0] >= logits.shape[1]:
        return logits
    return logits.transpose([1, 0, 2])


class PaddleLayerCtcAdapter:
    """Wrap a differentiable Paddle layer that returns raw recognition logits."""

    supports_grad = True

    def __init__(self, model: Any, layout_hint: str = "auto", logits_are_log_probs: bool = False):
        self.model = model
        self.layout_hint = layout_hint
        self.logits_are_log_probs = logits_are_log_probs

    def __call__(self, image_batch: Any) -> Any:
        logits = self.model(image_batch)
        logits = _to_time_major_logits(logits, self.layout_hint)
        if self.logits_are_log_probs:
            return logits

        import paddle

        return paddle.nn.functional.log_softmax(logits, axis=-1)


class PaddleTextRecognitionCtcAdapter:
    """Wrap a PaddleOCR text recognition predictor and expose raw logits."""

    supports_grad = False

    def __init__(self, predictor: Any, layout_hint: str = "auto", output_index: int = 0):
        self.predictor = predictor
        self.layout_hint = layout_hint
        self.output_index = int(output_index)

    @staticmethod
    def _as_image_list(image_batch: Any) -> Sequence[np.ndarray]:
        if image_batch is None:
            return []

        if hasattr(image_batch, "numpy"):
            image_batch = image_batch.numpy()

        if isinstance(image_batch, np.ndarray):
            if image_batch.ndim == 3:
                return [image_batch]
            if image_batch.ndim == 4:
                return [image_batch[idx] for idx in range(image_batch.shape[0])]

        if isinstance(image_batch, (list, tuple)):
            images = []
            for item in image_batch:
                if hasattr(item, "numpy"):
                    item = item.numpy()
                if not isinstance(item, np.ndarray):
                    item = np.asarray(item)
                images.append(item)
            return images

        return [np.asarray(image_batch)]

    def _call_predictor(self, image_batch: Any) -> Any:
        images = list(self._as_image_list(image_batch))
        if not images:
            raise ValueError("image batch is empty")

        if hasattr(self.predictor, "pre_tfs") and hasattr(self.predictor, "infer"):
            pre_tfs = self.predictor.pre_tfs
            batch_raw_imgs = pre_tfs["Read"](imgs=images) if "Read" in pre_tfs else images
            batch_imgs = (
                pre_tfs["ReisizeNorm"](imgs=batch_raw_imgs)
                if "ReisizeNorm" in pre_tfs
                else batch_raw_imgs
            )
            x = pre_tfs["ToBatch"](imgs=batch_imgs) if "ToBatch" in pre_tfs else batch_imgs
            return self.predictor.infer(x=x)

        if hasattr(self.predictor, "predict"):
            return self.predictor.predict(images)

        raise TypeError("predictor does not expose a usable inference interface")

    def __call__(self, image_batch: Any) -> Any:
        outputs = self._call_predictor(image_batch)
        if isinstance(outputs, (list, tuple)):
            if not outputs:
                raise ValueError("predictor returned no outputs")
            index = max(0, min(self.output_index, len(outputs) - 1))
            logits = outputs[index]
        else:
            logits = outputs

        return _to_time_major_logits(logits, self.layout_hint)


class CtcPgdAttack:
    """Iterative CTC loss maximization for OCR crops."""

    def __init__(
        self,
        logits_adapter: CtcLogitsAdapter,
        charset: Optional[str] = None,
        charset_path: Optional[str] = None,
        config: Optional[CtcAttackConfig] = None,
        logits_are_log_probs: bool = False,
    ) -> None:
        self.adapter = logits_adapter
        self.charset = load_charset(charset=charset, charset_path=charset_path)
        self.config = config or CtcAttackConfig()
        self.logits_are_log_probs = logits_are_log_probs

    def _prepare_log_probs(self, logits: Any):
        import paddle

        logits_t = _to_time_major_logits(logits, self.config.layout_hint)
        if self.logits_are_log_probs:
            return logits_t
        return paddle.nn.functional.log_softmax(logits_t, axis=-1)

    def _compute_loss(self, adv_image: np.ndarray, target_text: str):
        import paddle

        adv_t = paddle.to_tensor(adv_image[np.newaxis, ...], dtype="float32")
        logits = self._call_model(adv_t)
        log_probs = self._prepare_log_probs(logits)

        if log_probs.ndim != 3:
            raise ValueError(f"model must return a 3D logits tensor, got {list(log_probs.shape)}")

        labels_np, label_lengths_np = encode_text(target_text, self.charset, blank_index=self.config.blank_index)
        labels_t = paddle.to_tensor(labels_np.reshape([1, -1]), dtype="int32")
        label_lengths_t = paddle.to_tensor(label_lengths_np, dtype="int64")
        input_lengths_t = paddle.to_tensor(np.asarray([int(log_probs.shape[0])], dtype=np.int64), dtype="int64")

        loss = paddle.nn.functional.ctc_loss(
            log_probs,
            labels_t,
            input_lengths_t,
            label_lengths_t,
            blank=int(self.config.blank_index),
            reduction="mean",
            norm_by_times=False,
        )
        return float(loss.numpy())

    def _estimate_spsa_gradient(
        self,
        original: np.ndarray,
        delta: np.ndarray,
        writable: np.ndarray,
        target_text: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        cfg = self.config
        eps = float(cfg.epsilon) / 255.0
        sigma = max(float(cfg.spsa_sigma), 1e-6) / 255.0
        samples = max(1, int(cfg.spsa_samples))

        grad = np.zeros_like(delta, dtype=np.float32)
        for _ in range(samples):
            noise = rng.choice([-1.0, 1.0], size=delta.shape).astype(np.float32) * writable
            delta_plus = np.clip(delta + sigma * noise, -eps, eps)
            delta_minus = np.clip(delta - sigma * noise, -eps, eps)

            loss_plus = self._compute_loss(np.clip(original + delta_plus, cfg.clip_min, cfg.clip_max), target_text)
            loss_minus = self._compute_loss(np.clip(original + delta_minus, cfg.clip_min, cfg.clip_max), target_text)
            grad += ((loss_plus - loss_minus) / (2.0 * sigma)) * noise

        return grad / float(samples)

    def _call_model(self, image_t):
        return self.adapter(image_t)

    def attack(
        self,
        image: np.ndarray,
        target_text: str,
        writable_mask: Optional[np.ndarray] = None,
        delta_init: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        import paddle

        if image is None or image.size == 0:
            raise ValueError("input image is empty")
        if not target_text:
            raise ValueError("target_text is required for CTC attack")

        rng = rng or np.random.default_rng()
        cfg = self.config
        eps = float(cfg.epsilon) / 255.0
        alpha = float(cfg.alpha) / 255.0

        original = np.clip(_as_float32_image(image) / 255.0, cfg.clip_min, cfg.clip_max)
        if delta_init is None:
            if cfg.random_start:
                delta = rng.uniform(-eps, eps, size=original.shape).astype(np.float32)
            else:
                delta = np.zeros_like(original, dtype=np.float32)
        else:
            delta = np.clip(_as_float32_image(delta_init) / 255.0, -eps, eps).astype(np.float32)

        if writable_mask is None:
            writable = np.ones_like(original, dtype=np.float32)
        else:
            writable = writable_mask.astype(np.float32)
            if writable.ndim == 2:
                writable = writable[..., None]
            if writable.shape[-1] == 1 and original.shape[-1] == 3:
                writable = np.repeat(writable, 3, axis=2)

        labels_np, label_lengths_np = encode_text(target_text, self.charset, blank_index=cfg.blank_index)
        labels_t = paddle.to_tensor(labels_np.reshape([1, -1]), dtype="int32")
        label_lengths_t = paddle.to_tensor(label_lengths_np, dtype="int64")

        supports_grad = bool(getattr(self.adapter, "supports_grad", True))

        for _ in range(max(1, int(cfg.steps))):
            adv = np.clip(original + delta, cfg.clip_min, cfg.clip_max)
            if supports_grad:
                adv_t = paddle.to_tensor(adv[np.newaxis, ...], dtype="float32")
                adv_t.stop_gradient = False

                logits = self._call_model(adv_t)
                log_probs = self._prepare_log_probs(logits)

                if log_probs.ndim != 3:
                    raise ValueError(f"model must return a 3D logits tensor, got {list(log_probs.shape)}")

                input_lengths_t = paddle.to_tensor(np.asarray([int(log_probs.shape[0])], dtype=np.int64), dtype="int64")
                loss = paddle.nn.functional.ctc_loss(
                    log_probs,
                    labels_t,
                    input_lengths_t,
                    label_lengths_t,
                    blank=int(cfg.blank_index),
                    reduction="mean",
                    norm_by_times=False,
                )
                loss.backward()

                adv_grad = adv_t.grad
                if adv_grad is None:
                    break

                grad_np = adv_grad.numpy()[0]
            else:
                grad_np = self._estimate_spsa_gradient(
                    original=original,
                    delta=delta,
                    writable=writable,
                    target_text=target_text,
                    rng=rng,
                )

            delta = np.clip(delta + alpha * np.sign(grad_np) * writable, -eps, eps)
            delta = np.clip(original + delta, cfg.clip_min, cfg.clip_max) - original

        return (delta * 255.0).astype(np.float32)


def build_paddle_layer_ctc_attack(
    model: Any,
    *,
    charset: Optional[str] = None,
    charset_path: Optional[str] = None,
    config: Optional[CtcAttackConfig] = None,
    layout_hint: str = "auto",
    logits_are_log_probs: bool = False,
) -> CtcPgdAttack:
    resolved_model = model
    resolved_charset = charset
    resolved_charset_path = charset_path
    if isinstance(model, (str, Path)):
        from paddleocr import TextRecognition

        model_source = Path(model)
        if model_source.exists():
            try:
                resolved_model = TextRecognition(text_recognition_model_dir=str(model_source))
            except TypeError:
                resolved_model = TextRecognition(model_name=str(model_source))
        else:
            resolved_model = TextRecognition(model_name=str(model))

    if hasattr(resolved_model, "paddlex_predictor"):
        predictor = resolved_model.paddlex_predictor
        if resolved_charset is None and resolved_charset_path is None:
            try:
                model_charset = predictor.config["PostProcess"]["character_dict"]
                if isinstance(model_charset, list):
                    resolved_charset = "".join(str(ch) for ch in model_charset)
            except Exception:
                pass
        adapter: CtcLogitsAdapter = PaddleTextRecognitionCtcAdapter(
            predictor=predictor,
            layout_hint=layout_hint,
        )
    elif hasattr(resolved_model, "pre_tfs") and hasattr(resolved_model, "infer"):
        adapter = PaddleTextRecognitionCtcAdapter(
            predictor=resolved_model,
            layout_hint=layout_hint,
        )
    else:
        adapter = PaddleLayerCtcAdapter(model=resolved_model, layout_hint=layout_hint, logits_are_log_probs=logits_are_log_probs)

    return CtcPgdAttack(
        logits_adapter=adapter,
        charset=resolved_charset,
        charset_path=resolved_charset_path,
        config=config,
        logits_are_log_probs=logits_are_log_probs,
    )
