"""Local RynnVLA inference; model loading is lazy and needs no RPC service."""

import torch

from ..registry import INFERENCE_WRAPPER_REGISTRY
from .base import BaseInferenceWrapper, BaseVLMInferenceWrapper, BaseVLAInferenceWrapper


def build_inference_wrapper(
    model_type: str,
    model_path: str,
    dtype: torch.dtype,
    attn_implementation: str,
    device: str = "cuda:0",
    local_files_only: bool = True,
) -> BaseInferenceWrapper:
    if model_type == "rynn_brain_vla":
        from . import rynn_brain_vla  # noqa: F401 — register only the supported wrapper
    else:
        raise ValueError(f"Unsupported inference model type: {model_type!r}")
    return INFERENCE_WRAPPER_REGISTRY[model_type](
        model_path=model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device=device,
        local_files_only=local_files_only,
    )
