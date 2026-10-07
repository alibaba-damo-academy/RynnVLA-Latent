"""Public model construction API."""

from dataclasses import asdict, is_dataclass
from inspect import signature
from collections.abc import Mapping

from .modules.lam_v5 import RynnLAM


def build_model(config):
    """Build the complete training graph from a dataclass or flat mapping."""
    if is_dataclass(config) and not isinstance(config, type):
        values = asdict(config)
    elif isinstance(config, Mapping):
        values = dict(config)
    else:
        raise TypeError("config must be a dataclass instance or flat mapping")
    unsupported_flags = (
        "hard_bottleneck",
        "use_vae_latent",
        "use_triframe_encoder",
        "use_domain_view_cond",
        "use_feature_predictor",
        "use_recon_decoder",
        "use_flow_from_z_head",
        "use_multilayer_agg",
        "use_multiscale_feat_pred",
        "use_reversibility",
        "use_additivity",
    )
    enabled = [key for key in unsupported_flags if values.get(key, False)]
    if enabled:
        raise ValueError(f"Unsupported experimental architecture: {', '.join(enabled)}")
    for key, expected in (
        ("model_version", "v5"),
        ("latent_encoder_type", "selfattn"),
        ("flow_decoder_version", "v5"),
        ("use_k_token_recon", True),
        ("delta_refine_target_dim", 0),
    ):
        if key in values and values[key] != expected:
            raise ValueError(
                f"Unsupported {key}={values[key]!r}; expected {expected!r}"
            )
    supported = signature(RynnLAM).parameters
    return RynnLAM(**{key: value for key, value in values.items() if key in supported})


__all__ = ["RynnLAM", "build_model"]
