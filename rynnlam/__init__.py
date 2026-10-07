"""RynnLAM public API; configuration imports do not load torch or model dependencies."""

from .config import RynnLAMConfig

__all__ = ["RynnLAMConfig", "RynnLAM", "build_model"]


def __getattr__(name):
    if name == "RynnLAM":
        from .modules.lam_v5 import RynnLAM

        return RynnLAM
    if name == "build_model":
        from .model import build_model

        return build_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
