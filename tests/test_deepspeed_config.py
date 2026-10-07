"""Checkpoint metadata must size ZeRO buckets before custom model registration."""
import json
from types import SimpleNamespace

import pytest

from rynnvla.arguments import TrainingArguments


def test_unregistered_checkpoint_uses_text_width(tmp_path):
    # AutoConfig cannot instantiate this type. Reading its metadata needs no
    # custom-model import, CUDA setup or weight allocation.
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "unregistered_vla_checkpoint_test",
        "text_config": {"hidden_size": 2560},
        "vision_config": {"hidden_size": 1024},
    }))
    args = SimpleNamespace(model_path=str(tmp_path))
    result = TrainingArguments._process_deepspeed_config(args, {
        "zero_optimization": {"stage": 2, "reduce_bucket_size": "auto"},
    })
    assert result["zero_optimization"]["reduce_bucket_size"] == 6553600


def test_missing_text_width_reports_the_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unregistered_vla_checkpoint_test"}))
    args = SimpleNamespace(model_path=str(tmp_path))
    with pytest.raises(ValueError, match="no valid text hidden_size"):
        TrainingArguments._process_deepspeed_config(args, {
            "zero_optimization": {"stage": 2, "reduce_bucket_size": "auto"},
        })
