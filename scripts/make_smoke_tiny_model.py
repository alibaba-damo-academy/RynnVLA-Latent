#!/usr/bin/env python3
"""Generate a tiny, self-contained RynnVLA backbone for the offline training smoke.

Goal: let someone who cloned this repo run `rynnvla.api.train` end-to-end -- real
processor, real expert/latent head, real DeepSpeed ZeRO-1 step -- with NO corpus
download and NO 4.8 GB RynnBrain checkpoint.

What this builds:
  * a `rynn_brain_vla` model whose VLM trunk (text + vision) is shrunk to a few
    million parameters, but whose **tokenizer / chat template / processor assets are
    the real Qwen3-VL ones**, copied from `--src`. Reusing the real tokenizer is the
    point: the smoke then exercises the actual special-token, image-token and
    state-token path instead of a stand-in, so a green run means the real data pipeline
    is wired correctly.
  * RANDOM weights. This fixture proves the code *runs*; it is not a usable policy and
    must never be passed off as a recipe checkpoint.

Nothing binary is committed. The tokenizer assets are copied from a local `--src`
backbone (any Qwen3-VL-family checkpoint -- e.g. the RynnBrain base); the weights are
materialized locally into a gitignored directory (default `data/smoke_tiny_model/`).

Dimension shrink constraints (all enforced below, see the comments at each field):
  * vision `out_hidden_size` == text `hidden_size` (vision merges into the text space)
  * text rope `mrope_section` sums to `head_dim // 2`
  * `deepstack_visual_indexes` are valid layer indices (< vision `depth`)
  * the action expert derives its head layout (heads / kv-heads / head_dim / depth)
    from text_config, so only `expert_hidden_size` / `expert_intermediate_size` are free
  * `vocab_size` stays at the real value so the copied tokenizer's ids are all in range

Usage:
    python scripts/make_smoke_tiny_model.py --src <your local RynnBrain-2B or Qwen3-VL dir>

Then run the smoke (single GPU, from the repo root):
    torchrun --standalone --nproc_per_node=1 -m rynnvla.api.train \
        --config rynnvla/configs/stage1_smoke_tiny.json \
        --model_path data/smoke_tiny_model \
        --data_mixture data/sample/sample_mixture.json \
        --output_dir runs/smoke_vla
"""
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _rel(path: Path) -> str:
    """Repo-root-relative when inside the repo, so the printed commands stay portable."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


# Tokenizer / processor assets copied verbatim from the real backbone. These are the
# only files that carry the true Qwen3-VL vocabulary, merges, chat template and image
# preprocessing -- the thing that makes this a faithful smoke rather than a toy.
ASSET_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)

# ---- tiny trunk (see module docstring for the constraints these satisfy) -----------
TEXT_HIDDEN = 256
TEXT_HEAD_DIM = 64
TEXT_HEADS = 4
TEXT_KV_HEADS = 2
TEXT_LAYERS = 2
VISION_HIDDEN = 128
VISION_DEPTH = 4
VOCAB_SIZE = 151936          # keep the real vocab so copied tokenizer ids stay in range
EXPERT_HIDDEN = 128
EXPERT_INTERMEDIATE = 256


def _text_config() -> dict:
    return {
        "model_type": "qwen3_vl_text",
        "hidden_size": TEXT_HIDDEN,
        "intermediate_size": TEXT_HIDDEN * 2,
        "num_hidden_layers": TEXT_LAYERS,
        "num_attention_heads": TEXT_HEADS,
        "num_key_value_heads": TEXT_KV_HEADS,
        "head_dim": TEXT_HEAD_DIM,
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": 4096,
        # mrope_section must sum to head_dim // 2 (= 32 here).
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [TEXT_HEAD_DIM // 4, TEXT_HEAD_DIM // 8, TEXT_HEAD_DIM // 8],
            "rope_type": "default",
        },
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-06,
        "initializer_range": 0.02,
    }


def _vision_config() -> dict:
    return {
        "model_type": "qwen3_vl",
        "hidden_size": VISION_HIDDEN,
        "depth": VISION_DEPTH,
        "intermediate_size": VISION_HIDDEN * 2,
        "num_heads": 4,
        "out_hidden_size": TEXT_HIDDEN,   # must equal text hidden_size
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": [0, 1, 2],   # valid layer indices, < depth
        "hidden_act": "gelu_pytorch_tanh",
        "initializer_range": 0.02,
    }


def _copy_assets(src: Path, out: Path) -> list:
    copied = []
    for name in ASSET_FILES:
        path = src / name
        if path.is_file():
            shutil.copyfile(path, out / name)
            copied.append(name)
    if "tokenizer_config.json" not in copied:
        raise SystemExit(
            f"--src {src} has no tokenizer_config.json; point it at a Qwen3-VL-family "
            "backbone so the real tokenizer can be copied."
        )
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True,
                    help="local Qwen3-VL-family backbone dir to copy tokenizer assets from")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "smoke_tiny_model"),
                    help="fixture output dir (created fresh; must not exist unless --force)")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if os.path.lexists(out):
        if not args.force:
            raise SystemExit(f"--out already exists: {out}\n  Remove it, pass --force, or a new --out.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")
    copied = _copy_assets(src, out)
    print(f"[tiny] copied {len(copied)} tokenizer/processor asset(s): {', '.join(copied)}")

    import torch
    from rynnvla.models import _apply_monkey_patch
    from rynnvla.models.rynn_brain_vla.configuration_rynn_brain_vla import RynnBrainVLAConfig
    from rynnvla.models.rynn_brain_vla.processing_rynn_brain_vla import ACTION_DIM
    from transformers import AutoModel

    _apply_monkey_patch("rynn_brain_vla")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    config = RynnBrainVLAConfig(
        # Baked to the values the processor injects at train time (get_config_overrides),
        # so re-applying them over this config.json is a no-op and shapes always match.
        action_dim=ACTION_DIM,
        action_chunk_size=30,
        state_token_id=-1,
        action_head_type="expert",
        expert_backbone="qwen3_vl",
        expert_hidden_size=EXPERT_HIDDEN,
        expert_intermediate_size=EXPERT_INTERMEDIATE,
        expert_num_foresight_tokens=0,
        expert_per_layer_adanorm=True,
        expert_train_repeat=1,
        expert_time_concat=True,
        time_conditioning="adaln",
        use_latent_actions=True,
        use_latent_head_readout=False,
        use_view_cond_slots=True,
        use_view_role_embedding=False,
        latent_action_dim=608,
        latent_action_chunk_size=6,
        latent_action_stride=5,
        num_view_slots=6,
        use_sf_align=False,
        use_lb_align=False,
        text_config=_text_config(),
        vision_config=_vision_config(),
        tie_word_embeddings=True,
    )
    config.get_text_config().tie_word_embeddings = True

    torch.manual_seed(args.seed)
    # eager attention: the fixture must build on any machine, with or without flash-attn.
    model = AutoModel.from_config(config, dtype=dtype, attn_implementation="eager")
    # safetensors' serialize_file is rejected with EACCES by some network/FUSE mounts even
    # when the target directory is world-writable, so serialize into a local temp dir and
    # move the files into place -- the workaround build_latent_pretrain_index.py already uses.
    with tempfile.TemporaryDirectory(prefix="tiny_model_") as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        for produced in sorted(Path(tmp).iterdir()):
            if produced.is_file():
                shutil.move(str(produced), str(out / produced.name))

    # save_pretrained wrote config.json; confirm the weights landed and report size.
    weights = out / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"expected weights not written: {weights}")
    state = model.state_dict()
    total = sum(v.numel() for v in state.values())
    mb = weights.stat().st_size / 1e6
    print(f"[tiny] wrote {_rel(weights)}")
    print(f"[tiny] {len(state)} tensors, {total / 1e6:.1f}M params ({mb:.1f} MB, dtype={args.dtype})")
    print(f"[tiny] fixture: {_rel(out)}")
    print("[tiny] next -- single-GPU training smoke (from the repo root):")
    print("    torchrun --standalone --nproc_per_node=1 -m rynnvla.api.train \\")
    print("        --config rynnvla/configs/stage1_smoke_tiny.json \\")
    print(f"        --model_path {_rel(out)} \\")
    print("        --data_mixture data/sample/sample_mixture.json \\")
    print("        --output_dir runs/smoke_vla")


if __name__ == "__main__":
    main()
