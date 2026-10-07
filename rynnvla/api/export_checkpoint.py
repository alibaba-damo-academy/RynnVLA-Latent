"""Export a validated Stage1 or V5 (2B/4B) checkpoint without training state.

All tensors are loaded on CPU with weights_only=True (or safetensors). The
expected module tree is constructed on meta, never on CUDA or with real weights.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import zipfile

# stdlib-only on purpose: this exporter is meant to run on a CPU machine with no torch.
# rynnvla.constants imports nothing but os/enum, so it is safe to pull in here.
from rynnvla.constants import ALLOWED_ACTION_NORM_TYPES


METADATA_FILES = (
    "config.json", "processor_config.json", "preprocessor_config.json",
    "image_processor_config.json", "video_preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
    "spiece.model", "sentencepiece.bpe.model", "chat_template.jinja",
    "chat_template.json", "generation_config.json",
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(config, processor, stage="stage1"):
    if stage not in ("stage1", "stage1_2b", "stage2", "stage2_2b"):
        raise ValueError(f"Unknown stage: {stage}")
    backbone_width = 2048 if stage in ("stage1_2b", "stage2_2b") else 2560
    stage = "stage2" if stage == "stage2_2b" else "stage1" if stage == "stage1_2b" else stage
    expected = {
        "model_type": "rynn_brain_vla", "action_head_type": "expert",
        "expert_backbone": "qwen3_vl", "expert_hidden_size": 768,
        "expert_intermediate_size": 2752, "expert_num_foresight_tokens": 0,
        "expert_per_layer_adanorm": True, "expert_time_concat": True,
        "expert_train_repeat": 1, "time_conditioning": "adaln",
        "num_view_slots": 6, "latent_action_dim": 608,
        "latent_action_chunk_size": 6, "latent_action_stride": 5,
        "use_latent_actions": stage == "stage1",
        "use_latent_head_readout": stage == "stage2",
        "use_view_cond_slots": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{stage} config mismatch: {key}={config.get(key)!r}, expected {value!r}")
    # Stage-1 is chunk30. Stage-2's formal recipe is chunk10, but chunk size is a real lever on
    # VLABench, where the chunk30 arms materially outperform chunk10. Stage-2 chunk30 arms
    # therefore have to be exportable, or a long run finishes and cannot be evaluated.
    _ALLOWED_STAGE2_CHUNKS = (10, 30)
    chunk = config.get("action_chunk_size")
    if stage == "stage1":
        if chunk != 30:
            raise ValueError(f"stage1 config mismatch: action_chunk_size={chunk!r}, expected 30")
    elif chunk not in _ALLOWED_STAGE2_CHUNKS:
        raise ValueError(
            f"stage2 config mismatch: action_chunk_size={chunk!r} "
            f"(expected one of {', '.join(map(str, _ALLOWED_STAGE2_CHUNKS))})"
        )
    if config.get("text_config", {}).get("hidden_size") != backbone_width:
        size = "2B" if backbone_width == 2048 else "4B"
        raise ValueError(f"Expected a {size} backbone (text_config.hidden_size={backbone_width})")
    for key in ("use_sf_align", "use_lb_align", "use_depth_aux", "use_view_role_embedding"):
        if config.get(key, False):
            raise ValueError(f"Formal recipe requires {key}=false")
    if processor.get("use_state") is not (stage == "stage2"):
        raise ValueError(f"{stage} processor use_state must be {stage == 'stage2'}")
    # The formal Stage-2 recipe is min_max_sym, but a recipe may deliberately vary the
    # normalization, so this accepts exactly what the processor accepts -- see
    # constants.ALLOWED_ACTION_NORM_TYPES for why the two lists must be the same one.
    if stage == "stage2" and processor.get("action_norm_type") not in ALLOWED_ACTION_NORM_TYPES:
        raise ValueError(
            f"Stage2 processor has unsupported action_norm_type="
            f"{processor.get('action_norm_type')!r} (expected one of "
            f"{', '.join(ALLOWED_ACTION_NORM_TYPES)})"
        )


def checkpoint_metadata(source, stage="stage1"):
    source = Path(source)
    for name in ("config.json", "processor_config.json", "tokenizer_config.json"):
        if not (source / name).is_file() or (source / name).stat().st_size == 0:
            raise ValueError(f"Missing or empty checkpoint metadata: {name}")
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model")
    has_vocab = all((source / name).is_file() for name in ("vocab.json", "merges.txt"))
    if not has_vocab and not any((source / name).is_file() for name in tokenizer_files):
        raise ValueError("Checkpoint is missing tokenizer vocabulary/model files")
    config = read_json(source / "config.json")
    processor = read_json(source / "processor_config.json")
    tokenizer = read_json(source / "tokenizer_config.json")
    has_template = (
        any((source / name).is_file() for name in ("chat_template.jinja", "chat_template.json"))
        or any((source / "chat_templates").glob("*.jinja"))
        or processor.get("chat_template") or tokenizer.get("chat_template")
    )
    if not has_template:
        raise ValueError("Checkpoint is missing its chat template")
    validate_config(config, processor, stage)
    return config


def expected_state_shapes(config):
    import torch
    from ..models.rynn_brain_vla import RynnBrainVLAConfig, RynnBrainVLAModel

    model_config = RynnBrainVLAConfig.from_dict(config)
    with torch.device("meta"):
        model = RynnBrainVLAModel._from_config(model_config, attn_implementation="eager")
    kwargs = {"convert": False} if "convert" in inspect.signature(model.state_dict).parameters else {}
    return {key: tuple(value.shape) for key, value in model.state_dict(**kwargs).items()}


def _load_weight_file(path):
    import torch
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path))
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Expected a nonempty flat state_dict: {path.name}")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"Non-tensor state_dict entry in {path.name}: {key!r}")
        if value.is_meta or value.layout != torch.strided:
            raise ValueError(f"Unsupported tensor in {path.name}: {key}")
    return state


def load_weights(source, weights="ema"):
    """Select explicitly; never silently fall back from EMA to live weights."""
    source = Path(source)
    if weights == "ema":
        path = source / "ema_model.bin"
        if not path.is_file():
            raise ValueError("ema_model.bin not found; use --weights model explicitly for live weights")
        return _load_weight_file(path)
    if weights != "model":
        raise ValueError(f"Unknown weight selection: {weights}")
    for name in ("model.safetensors", "model.safetensors.index.json",
                 "pytorch_model.bin", "pytorch_model.bin.index.json", "model.bin"):
        path = source / name
        if not path.is_file():
            continue
        if not name.endswith(".index.json"):
            return _load_weight_file(path)
        weight_map = read_json(path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map in {name}")
        shards = set(weight_map.values())
        state = {}
        for shard in sorted(shards):
            # HF shard names are plain filenames, never arbitrary filesystem paths.
            if not isinstance(shard, str) or Path(shard).name != shard or "\\" in shard:
                raise ValueError(f"Invalid shard filename: {shard!r}")
            part = _load_weight_file(source / shard)
            if state.keys() & part.keys():
                raise ValueError(f"Duplicate tensor keys across shards: {shard}")
            if any(weight_map.get(key) != shard for key in part):
                raise ValueError(f"Shard contents disagree with weight_map: {shard}")
            state.update(part)
        if state.keys() != weight_map.keys():
            raise ValueError("Sharded checkpoint is missing indexed tensor keys")
        return state
    raise ValueError("No complete HF model weights or model.bin found in checkpoint directory")


def validate_state_dict(state, expected):
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    mismatched = [
        f"{key}: {tuple(state[key].shape)} != {shape}"
        for key, shape in expected.items() if key in state and tuple(state[key].shape) != shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Incomplete/incompatible checkpoint: missing={missing[:12]}, "
            f"unexpected={unexpected[:12]}, shape_mismatch={mismatched[:12]}"
        )


def export_checkpoint(checkpoint_dir, output_dir, *, weights="ema", output_format="safetensors", stage="stage1"):
    import torch

    source = Path(checkpoint_dir).resolve()
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite export directory: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Export parent directory does not exist: {output.parent}")
    if output_format not in ("safetensors", "pytorch"):
        raise ValueError(f"Unknown output format: {output_format}")
    config = checkpoint_metadata(source, stage)
    expected = expected_state_shapes(config)
    state = load_weights(source, weights)
    validate_state_dict(state, expected)

    # Exclusive mkdir protects existing directories, including a concurrent export.
    # Only files created by this invocation are removed if writing fails.
    output.mkdir(exist_ok=False)
    try:
        for name in METADATA_FILES:
            path = source / name
            if path.is_file():
                if path.stat().st_size == 0:
                    raise ValueError(f"Empty metadata file: {name}")
                if path.suffix == ".json":
                    read_json(path)
                shutil.copyfile(path, output / name)
        templates = sorted((source / "chat_templates").glob("*.jinja"))
        if templates:
            (output / "chat_templates").mkdir()
            for path in templates:
                shutil.copyfile(path, output / "chat_templates" / path.name)
        if output_format == "safetensors":
            from safetensors.torch import save_file
            # Save all names, including any tied aliases; never silently drop keys.
            seen_storage = set()
            tensors = {}
            for key, value in state.items():
                value = value.detach().cpu().contiguous()
                pointer = value.untyped_storage().data_ptr()
                if pointer in seen_storage:
                    value = value.clone()
                seen_storage.add(pointer)
                tensors[key] = value
            save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
        else:
            torch.save(state, output / "pytorch_model.bin")
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--format", dest="output_format", choices=("safetensors", "pytorch"), default="safetensors")
    parser.add_argument(
        "--stage", choices=("stage1", "stage1_2b", "stage2", "stage2_2b"), default="stage1"
    )
    args = parser.parse_args(argv)
    try:
        output = export_checkpoint(**vars(args))
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Checkpoint export failed: {exc}\n")
    print(f"Exported {args.weights} weights to {output}; usable as --model-path (not --resume).")


if __name__ == "__main__":
    main()
