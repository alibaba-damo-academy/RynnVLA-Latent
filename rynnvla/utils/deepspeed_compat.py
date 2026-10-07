"""Process-local compatibility for DeepSpeed's ZeRO-2 accumulation regression."""

import inspect
import textwrap

from . import logging


logger = logging.get_logger(__name__)


def fix_zero2_gradient_accumulation() -> bool:
    """Keep all microbatch gradients in affected DeepSpeed implementations.

    DeepSpeed 0.18.2 accumulates into ``all_grad_tensors``, but tests
    ``averaged_gradients`` to decide whether to initialize that accumulator.
    The latter is only populated at the accumulation boundary, so each earlier
    microbatch overwrites the previous one. DeepSpeed 0.17.4 uses a different,
    unaffected implementation.

    Replace only the incorrect predicate in the installed method, in memory.
    A source guard leaves older and already fixed implementations unchanged and
    avoids copying version-dependent reduction / offload code into this repo.
    No installed package files or checkpoint weights are modified.
    """
    import deepspeed
    import deepspeed.runtime.zero.stage_1_and_2 as zero_module

    cls = zero_module.DeepSpeedZeroOptimizer
    method = cls.independent_gradient_partition_epilogue
    if getattr(method, "_rynn_accumulation_fixed", False):
        return False
    source = textwrap.dedent(inspect.getsource(method))
    # Older releases accumulate directly into averaged_gradients and are correct.
    if "self.all_grad_tensors" not in source:
        return False
    incorrect = "if not i in self.averaged_gradients or self.averaged_gradients[i] is None:"
    corrected = "if i not in self.all_grad_tensors or self.all_grad_tensors[i] is None:"
    if incorrect not in source:
        return False
    if source.count(incorrect) != 1 or "accumulated_grad.add_(new_avg_grad)" not in source:
        raise RuntimeError("Unrecognized DeepSpeed ZeRO-2 accumulation implementation; verify before training")

    namespace = {}
    # Compile the installed, trusted function with the single guarded correction;
    # its globals remain the matching installed DeepSpeed module's globals.
    exec(compile(source.replace(incorrect, corrected), __file__, "exec"), zero_module.__dict__, namespace)
    replacement = namespace[method.__name__]
    replacement._rynn_accumulation_fixed = True
    cls.independent_gradient_partition_epilogue = replacement
    logger.warning(
        f"Applied ZeRO-2 gradient accumulation correction for DeepSpeed {deepspeed.__version__}: "
        "preserve all microbatches before the optimizer step"
    )
    return True
