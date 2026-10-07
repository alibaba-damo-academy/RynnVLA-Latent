"""Training entry point with JSON presets and explicit resume semantics."""

import argparse
from copy import deepcopy
from dataclasses import fields
from functools import partial
import json
import logging as _stdlib_logging
import os
from pathlib import Path
import re
import sys

from .launch import normalize_options


JSON_FIELDS = (
    "config_overrides", "processor_overrides", "lr_scheduler_kwargs",
    "gradient_checkpointing_kwargs",
)
MODEL_DATA_FIELDS = ("action_chunk_size", "use_latent_actions", "latent_action_dim", "num_view_slots")


def load_config(path):
    """Only the recipe's DeepSpeed path is relative to the recipe JSON."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("--config must contain a JSON object of TrainingArguments fields")
    if config.get("deepspeed") and not Path(config["deepspeed"]).is_absolute():
        config["deepspeed"] = str(path.parent / config["deepspeed"])
    return config


def _json_object(value):
    if isinstance(value, str):
        value = json.loads(value)
    if value is not None and not isinstance(value, dict):
        raise ValueError("JSON override fields must be objects (or null)")
    return value


def parse_training_options(argv=None, argument_type=None):
    """Merge JSON and CLI BEFORE constructing the CUDA-initializing dataclass.

    HfArgumentParser.parse_args is argparse's Namespace-only method, deliberately
    not parse_args_into_dataclasses/parse_json_file. Unknown fields never disappear.
    argument_type is injectable for CPU-only parser tests.
    """
    from transformers import HfArgumentParser

    if argument_type is None:
        from ..arguments import TrainingArguments
        argument_type = TrainingArguments
    argv = normalize_options(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config")
    pre.add_argument("--resume", action="store_true")
    entry, remaining = pre.parse_known_args(argv)
    config = load_config(entry.config) if entry.config else {}
    valid_fields = {field.name for field in fields(argument_type) if field.init}
    unknown = set(config) - valid_fields
    if unknown:
        raise ValueError(f"Unknown TrainingArguments config fields: {sorted(unknown)}")

    parser = HfArgumentParser(argument_type, allow_abbrev=False)
    for action in parser._actions:
        if action.dest in JSON_FIELDS:
            action.type = _json_object
    parser.set_defaults(**config)
    namespace = parser.parse_args(remaining)
    values = {name: value for name, value in vars(namespace).items() if name in valid_fields}
    explicit = {token[2:].split("=", 1)[0] for token in remaining if token.startswith("--")}
    for name in JSON_FIELDS:
        if name not in values:
            continue
        value = _json_object(values[name])
        if name in explicit and value is not None:
            value = {**(_json_object(config.get(name)) or {}), **value}
        values[name] = value

    # These fields configure both the dataset and model. A single CLI override
    # should not leave the model on the preset's old chunk size or latent mode.
    model_overrides = values.get("config_overrides") or {}
    model_cli = {}
    if "config_overrides" in explicit:
        # Namespace still holds the original CLI value before the dictionary merge.
        model_cli = _json_object(namespace.config_overrides) or {}
    for name in MODEL_DATA_FIELDS:
        if name not in values:
            continue
        if name in explicit or f"no_{name}" in explicit:
            if name in model_cli and model_cli[name] != values[name]:
                raise ValueError(f"Conflicting CLI values for {name} and config_overrides.{name}")
            model_overrides[name] = values[name]
        elif name in model_cli:
            values[name] = model_cli[name]
        elif name in model_overrides and model_overrides[name] != values[name]:
            raise ValueError(f"Config has inconsistent dataset/model field: {name}")
    if "config_overrides" in values:
        values["config_overrides"] = model_overrides
    return values, entry.resume


def check_output_dir(output_dir, resume=False):
    """Do not auto-resume, mix logs, or reuse even an empty existing directory."""
    if not output_dir or "://" in str(output_dir):
        raise ValueError("output-dir must be an explicit local/shared-filesystem path")
    path = Path(output_dir)
    if resume:
        if not path.is_dir():
            raise ValueError(f"--resume requires an existing training directory: {path}")
    elif os.path.lexists(path):
        raise FileExistsError(f"Output already exists: {path}. Use a new directory or explicit --resume.")


def validate_options(values):
    if values.get("sampler_shuffle", "auto") not in ("auto", "global"):
        raise ValueError("sampler_shuffle must be auto or global")
    for name in ("model_path", "data_mixture", "output_dir", "deepspeed"):
        if not values.get(name):
            raise ValueError(f"{name} is required; pass it explicitly or in --config")
    for name in ("micro_batch_size", "gradient_accumulation_steps", "save_steps"):
        if values.get(name, 1) <= 0:
            raise ValueError(f"{name} must be positive")
    interval = values.get("save_keep_every")
    if interval and (interval < 0 or interval % values.get("save_steps", 1000)):
        raise ValueError("save_keep_every must be a positive multiple of save_steps (or null/0)")
    if values.get("min_lr_rate") is not None and values.get("lr_scheduler_type") != "cosine_with_min_lr":
        raise ValueError("min_lr_rate requires lr_scheduler_type=cosine_with_min_lr")
    # Evaluation is not wired up: the whole branch in Trainer._maybe_log_save_evaluate is a
    # commented-out TODO and no eval_dataset is ever passed, so --eval-strategy steps would run
    # to completion having validated nothing. Fail before the run rather than after it.
    eval_strategy = getattr(values.get("eval_strategy"), "value", values.get("eval_strategy", "no"))
    if str(eval_strategy).lower() != "no":
        raise NotImplementedError(
            f"eval_strategy={eval_strategy!r} is not implemented: Trainer has no evaluation loop "
            'and no eval_dataset is passed. Use "no" and evaluate via rynnvla.api.predict / '
            "rynnvla.api.eval_libero on an exported checkpoint."
        )


# The ZeRO config is snapshotted as an absolute path, and the same shared filesystem is commonly
# mounted under a different prefix on another machine, so a migrated --resume sees a new string
# for an unchanged file. Forgiving that must not forgive a real config change, hence the
# shared-suffix and content requirements below.
_RESUME_REPATH_ENV = "RYNNVLA_RESUME_ALLOW_DEEPSPEED_REPATH"
_MIN_SHARED_SUFFIX = 3  # e.g. rynnvla/configs/zero1.json -- enough to pin the file inside the repo


def _deepspeed_repath_ok(saved, current):
    """True only when saved and current address the same ZeRO config through different mounts."""
    if not (isinstance(saved, str) and isinstance(current, str)):
        return False
    shared = 0
    for left, right in zip(reversed(Path(saved).parts), reversed(Path(current).parts)):
        if left != right:
            break
        shared += 1
    if shared < _MIN_SHARED_SUFFIX:
        return False
    if os.path.isfile(saved) and os.path.isfile(current):
        with open(saved, encoding="utf-8") as handle:
            saved_json = json.load(handle)
        with open(current, encoding="utf-8") as handle:
            current_json = json.load(handle)
        return saved_json == current_json
    # The old mount is gone so its contents cannot be re-read and there is nothing left to
    # verify against. That case needs an explicit operator assertion, not a silent pass.
    return os.environ.get(_RESUME_REPATH_ENV) == "1"


def resolve_resume(values, resume, find_checkpoint):
    """Resume is training-state restoration, never a way to change stages."""
    if not resume:
        return None
    checkpoint = find_checkpoint(values["output_dir"])
    if checkpoint is None:
        raise ValueError("--resume found no complete training checkpoint; an exported model is not resumable")
    logger = _stdlib_logging.getLogger(__name__)
    with (Path(checkpoint) / "config.json").open(encoding="utf-8") as handle:
        saved_config = json.load(handle)
    with (Path(checkpoint) / "processor_config.json").open(encoding="utf-8") as handle:
        saved_processor = json.load(handle)
    for current, saved in (
        (values.get("config_overrides") or {}, saved_config),
        (values.get("processor_overrides") or {}, saved_processor),
    ):
        for name, value in current.items():
            if name not in saved:
                # Not a silent pass: the checkpoint predates this field, so there is nothing to
                # compare it against and the override is unverifiable.
                logger.warning(
                    f"--resume cannot verify {name}: absent from the checkpoint metadata. "
                    f"Proceeding with {value!r}."
                )
            elif value != saved[name]:
                raise ValueError(
                    f"--resume cannot change checkpoint setting {name}: {saved[name]!r} -> {value!r}. "
                    "Use a new output directory and --model-path for stage initialization."
                )

    # The settings that silently reshape training. create_scheduler builds its lambda from the
    # CURRENT args before the checkpoint's scheduler state_dict is loaded over it, so a changed
    # horizon redraws the LR curve; gradient_accumulation_steps and data_mixture change the epoch
    # length the batch-skip math divides by. trainer.py persists them per checkpoint.
    # Imported from constants, not the trainer: this function is unit-tested standalone and must
    # not drag in deepspeed.
    from ..constants import RESUME_ARGS_NAME, RESUME_CRITICAL_FIELDS

    snapshot_path = Path(checkpoint) / RESUME_ARGS_NAME
    if not snapshot_path.is_file():
        if values.get("sampler_shuffle", "auto") != "auto":
            raise ValueError("Cannot resume with a new sampler_shuffle without saved sampler metadata; "
                             "use a new output directory for a sampling experiment")
        logger.warning(
            f"{checkpoint} has no {RESUME_ARGS_NAME}; it predates resume-argument validation, so "
            "max_steps / warmup_steps / lr_scheduler_type / gradient_accumulation_steps / "
            "data_mixture cannot be checked against it."
        )
        return checkpoint
    with snapshot_path.open(encoding="utf-8") as handle:
        saved_args = json.load(handle)
    # Checkpoints created before this option always used auto. Treat that as an
    # explicit historical default, not permission to change order while skipping
    # already-trained batches on resume.
    saved_args.setdefault("sampler_shuffle", "auto")
    for name in RESUME_CRITICAL_FIELDS:
        if name not in saved_args:
            continue
        current = getattr(values.get(name), "value", values.get(name))
        if name == "sampler_shuffle":
            current = values.get(name, "auto")
        saved = saved_args[name]
        if name == "data_mixture" and isinstance(saved, list):
            # DataArguments expands the JSON path before Trainer snapshots it. Compare
            # the current file's contents to that snapshot, including source order and
            # weights, so an unchanged command resumes and edits at the same path fail.
            if isinstance(current, str):
                with open(current, encoding="utf-8") as handle:
                    current = json.load(handle)
            if not isinstance(current, list) or not all(isinstance(item, dict) for item in current):
                raise ValueError("data_mixture must contain a JSON list of data source objects")
        if current is None and saved is None:
            continue
        if current != saved:
            if name == "deepspeed" and _deepspeed_repath_ok(saved, current):
                logger.warning(
                    f"--resume: {name} differs only by mount prefix: {saved!r} -> {current!r}. "
                    "Accepted as the same ZeRO config."
                )
            else:
                raise ValueError(
                    f"--resume cannot change {name}: {saved!r} -> {current!r}. It is baked into the "
                    "learning-rate schedule and the epoch length used to skip batches, so changing it "
                    "silently reshapes training. Use a new output directory instead."
                )
        if name == "data_mixture" and isinstance(saved, list):
            # Train with exactly the contents just checked instead of reopening a file
            # that could change between validation and TrainingArguments construction.
            values[name] = deepcopy(current)
    return checkpoint


def train(argv=None):
    values, resume = parse_training_options(argv)
    validate_options(values)
    # Every torchrun worker ran this before, which broke multinode: node 0's Trainer creates the
    # output directory shortly after starting, so workers on a node that came up later saw it
    # already there and aborted. Only global rank 0 owns the directory. RANK is unset for a
    # direct single-process invocation, where the check must still apply.
    if int(os.environ.get("RANK", 0)) == 0:
        check_output_dir(values["output_dir"], resume=resume)

    from transformers.trainer_utils import enable_full_determinism, set_seed
    from ..arguments import TrainingArguments
    from ..datasets import build_dataset
    from ..models import build_model, init_weights, build_processor
    from ..ops import cross_entropy_loss
    from ..training import DataCollator, Trainer
    from ..training.trainer import get_last_checkpoint
    from ..utils import logging

    logger = logging.get_logger(__name__)
    checkpoint = resolve_resume(values, resume, get_last_checkpoint)
    if checkpoint:
        # Model/processor metadata on resume must come from the training checkpoint,
        # not from a raw backbone or a Stage1 initialization directory.
        values["model_path"] = checkpoint
    args = TrainingArguments(**values)
    enable_full_determinism(args.seed) if args.full_determinism else set_seed(args.seed)

    train_dataset = build_dataset(args)
    processor_overrides = deepcopy(args.processor_overrides)
    schema = train_dataset.get_schema()
    if schema is not None:
        processor_overrides["schema"] = schema
    processor_overrides.setdefault("mm_max_length", args.mm_max_length)
    processor = build_processor(
        model_type=args.model_type,
        model_path=args.model_path,
        processor_overrides=processor_overrides,
    )
    train_dataset.processor = processor
    config_overrides = processor.get_config_overrides()
    config_overrides.update(args.config_overrides)
    model = build_model(
        model_type=args.model_type,
        model_path=args.model_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        config_overrides=config_overrides,
        vision_encoder_path=args.vision_encoder_path,
        reduced_layers_in_stage_zero=args.reduced_layers_in_stage_zero,
    )
    if checkpoint:
        logger.info("Explicit resume from %s; skipping primary initialization, trunk overlay and resets", checkpoint)
    init_weights(
        model,
        pretrained_model_name_or_path=args.model_path if not checkpoint else None,
        trunk_model_name_or_path=args.trunk_model_path if not checkpoint else None,
        reset_pretrained_modules=args.reset_pretrained_modules if not checkpoint else None,
        reset_pretrained_seed=args.reset_pretrained_seed,
    )
    model.loss_function = partial(
        cross_entropy_loss,
        loss_reduction_scope=args.loss_reduction_scope,
        loss_implementation=args.loss_implementation,
    )
    if args.frozen_parameters is not None:
        for name, param in model.named_parameters():
            if any(re.match(pattern, name) for pattern in args.frozen_parameters):
                param.requires_grad_(False)
    frozen_params = [name for name, param in model.named_parameters() if not param.requires_grad]
    logger.info(
        f"Dataset: {train_dataset}\n\nModel config: {model.config}\n\n"
        f"Processor: {processor}\n\nModel: {model}\n\nFrozen parameters: {frozen_params}\n\n"
    )
    data_collator = DataCollator(processor=processor, sequence_packing=args.sequence_packing)
    trainer = Trainer(
        model=model, args=args, data_collator=data_collator,
        train_dataset=train_dataset, processing_class=processor,
    )
    return trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    train()
