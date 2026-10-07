"""Exponential Moving Average (EMA) of trainable weights for RynnVLA training.

Keeps an fp32 shadow copy of each trainable parameter and updates it after every
optimizer step: ``shadow = decay * shadow + (1 - decay) * param``. fp32 accumulation
is required — a bf16 shadow would round the ``(1 - decay) * param`` term to zero.

At save time, ``overlay`` produces a state dict with the EMA values substituted in,
which is written as an inference-ready checkpoint. Works with DeepSpeed ZeRO-1/2
(parameters are replicated); ZeRO-3 shards parameters and is not supported.
"""
import torch


class EMA:
    def __init__(self, model, decay: float = 0.999, warmup: bool = True, device=None):
        self.decay = decay
        self.warmup = warmup
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().to(device=device or param.device, dtype=torch.float32, copy=True)

    def _current_decay(self, step):
        if self.warmup and step is not None:
            # Ramp the decay up early so the shadow tracks fast at the start.
            return min(self.decay, (1.0 + step) / (10.0 + step))
        return self.decay

    @torch.no_grad()
    def update(self, model, step=None):
        d = self._current_decay(step)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                shadow = self.shadow[name]
                shadow.mul_(d).add_(param.detach().to(shadow), alpha=1.0 - d)

    def overlay(self, state_dict):
        """Return ``state_dict`` with EMA values substituted, materialised on CPU.

        The result goes straight to ``torch.save``, so building it on the GPU only
        duplicated the whole model in device memory: at 8.9B params that is +17.7GB and it
        OOM'd the first checkpoint save of a config whose training steps fit comfortably.
        """
        out = {}
        for name, value in state_dict.items():
            if not torch.is_tensor(value):
                out[name] = value
                continue
            source = self.shadow.get(name, value)
            out[name] = source.detach().to(device="cpu", dtype=value.dtype)
        return out

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        missing = self.shadow.keys() - sd.keys()
        unexpected = sd.keys() - self.shadow.keys()
        if missing or unexpected:
            raise ValueError(f"EMA checkpoint keys differ: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        for name, value in sd.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"EMA checkpoint shape differs for {name}")
        for name, value in sd.items():
            self.shadow[name].copy_(value.to(self.shadow[name]))
