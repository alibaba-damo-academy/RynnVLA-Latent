from types import SimpleNamespace

import pytest
import torch

from rynnvla.training.ema import EMA
from rynnvla.training.trainer import Trainer


def test_ema_roundtrip_continues_same_average():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    first = EMA(model, decay=0.9, warmup=False)
    with torch.no_grad():
        model.weight.fill_(3)
    first.update(model, step=1)
    resumed = EMA(model, decay=0.9, warmup=False)
    resumed.load_state_dict(first.state_dict())
    with torch.no_grad():
        model.weight.fill_(5)
    first.update(model, step=2)
    resumed.update(model, step=2)
    assert torch.equal(first.shadow["weight"], resumed.shadow["weight"])
    assert torch.allclose(first.shadow["weight"], torch.full((2, 2), 1.58))


def test_ema_rejects_incomplete_or_misshapen_state():
    ema = EMA(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="keys differ"):
        ema.load_state_dict({"weight": torch.ones(2, 2)})
    with pytest.raises(ValueError, match="shape differs"):
        ema.load_state_dict({"weight": torch.ones(1), "bias": torch.ones(2)})


def _trainer(offload=False):
    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.ModuleDict({"trunk": torch.nn.Linear(2, 2), "action_expert": torch.nn.Linear(2, 2)})
    trainer.args = SimpleNamespace(
        learning_rate=2.5e-5, action_head_lr=1e-4,
        action_head_modules=["action_expert"], weight_decay=0.01,
        adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
        deepspeed_config={"zero_optimization": {"offload_optimizer": {"device": "cpu"}}} if offload else {},
    )
    trainer.get_decay_parameter_names = lambda model: [name for name, _ in model.named_parameters() if name.endswith("weight")]
    trainer.get_optimizer_cls_and_kwargs = lambda args, model: (torch.optim.AdamW, {"lr": args.learning_rate})
    return trainer


@pytest.mark.parametrize("offload", [False, True])
def test_learning_rate_groups_preserved_for_cpu_offload(monkeypatch, offload):
    import deepspeed.ops.adam

    monkeypatch.setattr(deepspeed.ops.adam, "DeepSpeedCPUAdam", torch.optim.AdamW)
    trainer = _trainer(offload)
    trainer.create_optimizer()
    seen = set()
    for group in trainer.optimizer.param_groups:
        expected_lr = 1e-4 if group["name"].startswith("action_head") else 2.5e-5
        assert group["lr"] == expected_lr
        for parameter in group["params"]:
            assert id(parameter) not in seen
            seen.add(id(parameter))
    assert seen == {id(parameter) for parameter in trainer.model.parameters()}


def test_unknown_action_head_prefix_is_rejected():
    trainer = _trainer()
    trainer.args.action_head_modules = ["action_decoder"]
    with pytest.raises(ValueError, match="matched no parameters"):
        trainer.create_optimizer()
