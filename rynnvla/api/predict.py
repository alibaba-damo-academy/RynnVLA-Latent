"""Offline RynnVLA prediction from one LIBERO-format JSON observation."""

import argparse
import json
from pathlib import Path
import random

import numpy as np


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def add_policy_arguments(parser):
    parser.add_argument("--model-path", required=True, type=Path, help="Exported local HF model directory")
    parser.add_argument("--seed", type=nonnegative_int, default=7)
    parser.add_argument("--denoising-steps", type=positive_int, default=10)
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")


def seed_policy(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def load_policy(args, schema="libero"):
    if schema not in ("libero", "vlabench"):
        raise ValueError(f"Unknown eval schema: {schema!r}; expected 'libero' or 'vlabench'")
    if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
        raise ValueError("--model-path must be an exported HF directory containing config.json")
    import torch
    from ..inference_wrappers.rynn_brain_vla import (
        RynnBrainVLAInferenceWrapper,
        validate_libero_schema,
        validate_vlabench_schema,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu --dtype float32 for CPU inference")
    if device.type == "cpu" and args.attn_implementation == "flash_attention_2":
        raise ValueError("flash_attention_2 requires CUDA; use --attn-implementation sdpa on CPU")
    seed_policy(args.seed)
    wrapper = RynnBrainVLAInferenceWrapper(
        model_path=str(args.model_path.resolve()),
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
        device=args.device,
        local_files_only=True,
    )
    # Load the actual exported architecture. Do not apply training/config overrides.
    validate = validate_libero_schema if schema == "libero" else validate_vlabench_schema
    validate(wrapper.processor, wrapper.model.config)
    return wrapper


def load_sample(path):
    from PIL import Image
    from ..inference_wrappers.rynn_brain_vla import libero_images, libero_state_to_robot_state

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        sample = json.load(stream)
    instruction = sample.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Sample needs a nonempty 'instruction' string")
    state = np.asarray(sample["state"], dtype=np.float32)
    libero_state_to_robot_state(state)
    images = {}
    for camera in ("front", "wrist"):
        image_path = path.parent / sample[camera]
        with Image.open(image_path) as image:
            images[camera] = np.asarray(image.convert("RGB")).copy()
    images = libero_images(images, convention=sample.get("image_convention", "dataset"))
    return {"text": instruction, "state": state, "images": images}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_arguments(parser)
    parser.add_argument("--sample", required=True, type=Path, help="JSON with instruction, state, front and wrist paths")
    parser.add_argument("--output", required=True, type=Path, help="Output .npy containing raw OSC commands (T, 7)")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npy":
        parser.error("--output must end in .npy")
    if args.output.exists():
        parser.error("--output already exists; choose a new file")
    try:
        sample = load_sample(args.sample)
        policy = load_policy(args)
        actions = policy.predict_libero(**sample, denoising_steps=args.denoising_steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            np.save(stream, actions, allow_pickle=False)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"RynnVLA prediction failed: {exc}\n")
    print(f"Saved {len(actions)} raw OSC commands to {args.output}")


if __name__ == "__main__":
    main()
