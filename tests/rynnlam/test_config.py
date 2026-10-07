import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from rynnlam.config import RynnLAMConfig

ROOT = Path(__file__).resolve().parents[2]


def test_lightweight_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from rynnlam import RynnLAMConfig; "
            "assert 'torch' not in sys.modules; assert 'yaml' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_reference_recipes():
    for name, k in (("stage1", 8), ("stage1_k2", 2), ("stage1_k4", 4), ("stage2", 8)):
        config = RynnLAMConfig.from_yaml(ROOT / "rynnlam" / "configs" / f"{name}.yaml")
        assert config.num_k_tokens == k
        assert config.latent_dim == 64 and config.motion_hint_dim == 128
        assert config.k_token_source == "features" and config.use_z_residual_loss
        assert config.lambda_recon == 5 and config.lr == 5e-5
        # No object-store support ships in this release: reads are local-path only. Pinning the
        # absence, not just a default of False, so re-adding the field is a visible change.
        # "oss" as a substring would also match "loss", hence the prefix form.
        assert not hasattr(config, "oss_mode")
        assert not [field for field in asdict(config) if field.startswith("oss")]
        assert config.output_dir == "./runs"
        assert RynnLAMConfig.from_dict(asdict(config)) == config
        if name == "stage2":
            with pytest.raises(ValueError, match="requires explicit"):
                config.validate_training()
            config.resume = "./runs/stage1/latest.pt"
            assert config.reset_optimizer and config.encoder_finetune_mode == "full"
        config.validate_training()


def test_parsing_and_typo_detection():
    config = RynnLAMConfig.from_dict(
        {"training": {"lr": "1e-5"}, "reset_optimizer": True}
    )
    assert config.lr == 1e-5 and config.reset_optimizer
    assert RynnLAMConfig.from_dict({"old_unused_field": 1}).latent_dim == 64
    with pytest.raises(ValueError, match="Unknown"):
        RynnLAMConfig.from_dict({"training": {"lern_rate": 1}}, strict=True)
    with pytest.raises(ValueError, match="boolean"):
        RynnLAMConfig.from_dict({"reset_optimizer": "false"})
