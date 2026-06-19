"""
Device auto-selection: CUDA -> MPS -> CPU.

Each stage calls select_device(config_value) at init time. Logs a warning
on fallback so operator knows GPU inference is not active.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def select_device(preference: str = "auto") -> str:
    """
    Resolve a device string to a concrete torch device.

    Args:
        preference: "auto" | "cuda" | "mps" | "cpu"

    Returns:
        Resolved device string, e.g. "cuda", "mps", "cpu".
    """
    if preference not in ("auto", "cuda", "mps", "cpu"):
        logger.warning("unknown_device_preference", preference=preference, fallback="auto")
        preference = "auto"

    if preference == "cpu":
        return "cpu"

    try:
        import torch  # type: ignore
    except ImportError:
        logger.warning("torch_not_installed", fallback="cpu")
        return "cpu"

    if preference in ("auto", "cuda"):
        if torch.cuda.is_available():
            device = "cuda"
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("device_selected", device=device, name=name, vram_gb=round(vram_gb, 1))
            return device
        if preference == "cuda":
            logger.warning("cuda_unavailable", fallback="cpu")
            return "cpu"

    if preference in ("auto", "mps"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("device_selected", device="mps")
            return "mps"
        if preference == "mps":
            logger.warning("mps_unavailable", fallback="cpu")
            return "cpu"

    logger.warning("no_gpu_available", fallback="cpu")
    return "cpu"
