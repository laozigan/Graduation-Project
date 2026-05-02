from __future__ import annotations

import os
from typing import Optional, Tuple


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_paddle_use_gpu(prefer_gpu: Optional[bool] = None) -> Tuple[bool, str]:
    """Resolve whether Paddle-based components should run with GPU.

    Returns:
        (use_gpu, message)
    """
    should_prefer_gpu = _env_flag("PPS_PREFER_GPU", True) if prefer_gpu is None else bool(prefer_gpu)

    if not should_prefer_gpu:
        return False, "Paddle runtime: GPU preference disabled, using CPU."

    try:
        import paddle  # type: ignore
    except Exception as exc:
        return False, f"Paddle runtime: cannot import paddle ({exc}), fallback to CPU."

    try:
        cuda_compiled = bool(paddle.device.is_compiled_with_cuda())
    except Exception as exc:
        return False, f"Paddle runtime: CUDA capability check failed ({exc}), fallback to CPU."

    if not cuda_compiled:
        return False, "Paddle runtime: no CUDA support in current Paddle build, using CPU."

    try:
        current_device = str(paddle.device.get_device())
    except Exception:
        current_device = "gpu"

    return True, f"Paddle runtime: CUDA available, using GPU ({current_device})."
