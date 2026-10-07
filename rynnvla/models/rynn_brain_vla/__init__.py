from .configuration_rynn_brain_vla import RynnBrainVLAConfig
from .modeling_rynn_brain_vla import RynnBrainVLAModel
from .processing_rynn_brain_vla import RynnBrainVLAProcessor

# Re-exported so rynnvla.models.build_processor can explicitly re-register
# this processor class into PROCESSOR_MAPPING after the monkey-patch chain.
PROCESSOR_CLASS = RynnBrainVLAProcessor


_REGISTERED = False


def apply_monkey_patch():
    """Register ``rynn_brain_vla`` with transformers' Auto* mappings.

    Idempotent — safe to call multiple times. Runs automatically when this
    package is imported so ``from_pretrained`` resolves the custom config.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import CONFIG_MAPPING, MODEL_MAPPING, PROCESSOR_MAPPING
    CONFIG_MAPPING.register("rynn_brain_vla", RynnBrainVLAConfig)
    MODEL_MAPPING.register(RynnBrainVLAConfig, RynnBrainVLAModel)
    PROCESSOR_MAPPING.register(RynnBrainVLAConfig, RynnBrainVLAProcessor)
    _REGISTERED = True


apply_monkey_patch()
