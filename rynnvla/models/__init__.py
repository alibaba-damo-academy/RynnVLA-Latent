# Portions of this file are derived from HuggingFace Transformers
# (https://github.com/huggingface/transformers), Copyright The HuggingFace Inc. team,
# licensed under the Apache License, Version 2.0. The license text is in LICENSE; the
# attribution is recorded in NOTICE.
# Upstream reference: src/transformers/modeling_utils.py (checkpoint loading helpers)

import gc
import importlib
import inspect
import json
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Any

import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    PROCESSOR_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    AutoVideoProcessor,
    PreTrainedModel,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from .. import parallel_state as mpu
from ..utils import logging

logger = logging.get_logger(__name__)


def _get_local_path(pretrained_model_name_or_path: str, filename: str) -> Optional[str]:
    """Resolve `filename` inside a local checkpoint directory; None when absent.

    Everything is read off the local filesystem: no object store and no hub fetch, so an
    offline machine and a connected one resolve exactly the same files.
    """
    local_path = os.path.join(pretrained_model_name_or_path, filename)
    if os.path.exists(local_path):
        return local_path
    return None


def _remap_checkpoint_keys(
    sharded_state_dict: Dict[str, torch.Tensor],
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if not prefix:
        return sharded_state_dict

    remapped = {}
    state_dict_keys = set(state_dict.keys())
    prefix_dot = f"{prefix}."

    for key, tensor in sharded_state_dict.items():
        if key in state_dict_keys:
            remapped[key] = tensor
        elif key.startswith(prefix_dot) and key[len(prefix_dot):] in state_dict_keys:
            remapped[key[len(prefix_dot):]] = tensor
        elif f"{prefix_dot}{key}" in state_dict_keys:
            remapped[f"{prefix_dot}{key}"] = tensor
        else:
            remapped[key] = tensor

    return remapped


def _load_checkpoint_file(
    pretrained_model_name_or_path: str,
    filename: str,
    expert_param_tags: Dict[str, int],
    state_dict: Dict[str, torch.Tensor],
    missing_keys: Set[str],
    prefix: str = "",
) -> None:
    local_checkpoint_file = _get_local_path(
        pretrained_model_name_or_path,
        filename=filename,
    )

    if mpu.get_expert_model_parallel_rank() == 0:
        if filename.endswith(".safetensors"):
            sharded_state_dict = {}
            with safe_open(local_checkpoint_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    sharded_state_dict[key] = f.get_tensor(key)
        else:
            sharded_state_dict = torch.load(local_checkpoint_file, map_location="cpu", weights_only=True, mmap=True)

        sharded_state_dict = _remap_checkpoint_keys(sharded_state_dict, state_dict, prefix)

        for key, tensor in state_dict.items():
            if key in sharded_state_dict:
                source = sharded_state_dict[key]
                if source.shape != tensor.shape:
                    raise ValueError(f"Checkpoint tensor {key} has shape {tuple(source.shape)}, expected {tuple(tensor.shape)}")
                missing_keys.discard(key)
                if key not in expert_param_tags:
                    tensor.copy_(source, non_blocking=True)



def _load_checkpoint_files(
    pretrained_model_name_or_path: str,
    expert_param_tags: Dict[str, int],
    state_dict: Dict[str, torch.Tensor],
    prefix: str = "",
) -> List[str]:
    missing_keys = set(state_dict.keys())

    checkpoint_file = _get_local_path(pretrained_model_name_or_path, filename=SAFE_WEIGHTS_NAME)

    checkpoint_name = None
    if checkpoint_file is not None:
        checkpoint_name = SAFE_WEIGHTS_NAME
    else:
        checkpoint_file = _get_local_path(pretrained_model_name_or_path, filename=WEIGHTS_NAME)
        if checkpoint_file is not None:
            checkpoint_name = WEIGHTS_NAME

    if checkpoint_name is not None:
        _load_checkpoint_file(
            pretrained_model_name_or_path,
            filename=checkpoint_name,
            expert_param_tags=expert_param_tags,
            state_dict=state_dict,
            missing_keys=missing_keys,
            prefix=prefix,
        )
        return list(missing_keys)

    index_file = _get_local_path(pretrained_model_name_or_path, filename=SAFE_WEIGHTS_INDEX_NAME)

    if index_file is None:
        index_file = _get_local_path(pretrained_model_name_or_path, filename=WEIGHTS_INDEX_NAME)

    if index_file is None:
        raise FileNotFoundError(
            f"No checkpoint or weight index found under the local path '{pretrained_model_name_or_path}'"
        )

    if SAFE_WEIGHTS_INDEX_NAME in index_file:
        with open(index_file, "r") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        raise NotImplementedError

    # Build a mapping from state_dict keys to weight_map keys (handling prefix differences)
    prefix_dot = f"{prefix}." if prefix else ""
    checkpoint_files_to_load = set()
    for key in state_dict.keys():
        if key in weight_map:
            checkpoint_files_to_load.add(weight_map[key])
        elif prefix_dot and key.startswith(prefix_dot) and key[len(prefix_dot):] in weight_map:
            # Model key has prefix, weight_map key doesn't
            checkpoint_files_to_load.add(weight_map[key[len(prefix_dot):]])
        elif prefix_dot and f"{prefix_dot}{key}" in weight_map:
            # Model key lacks prefix, weight_map key has it
            checkpoint_files_to_load.add(weight_map[f"{prefix_dot}{key}"])

    for checkpoint_file in tqdm(
        checkpoint_files_to_load,
        desc="Loading checkpoint shards",
    ):
        _load_checkpoint_file(
            pretrained_model_name_or_path,
            filename=checkpoint_file,
            expert_param_tags=expert_param_tags,
            state_dict=state_dict,
            missing_keys=missing_keys,
            prefix=prefix,
        )

    return list(missing_keys)


@contextmanager
def _init_empty_params():
    old_device = torch.get_default_device()

    def move_init_to_device(func):
        def decorator(self, *args, **kwargs):
            torch.set_default_device(old_device)
            func(self, *args, **kwargs)
            torch.set_default_device("meta")

        return decorator

    def apply_patch(cls):
        if hasattr(cls, "_orig_init"):
            return

        if "RotaryEmbedding" in cls.__name__:
            cls._orig_init = cls.__init__
            cls.__init__ = move_init_to_device(cls.__init__)

        if not hasattr(cls, "_orig_init_subclass"):
            cls._orig_init_subclass = cls.__init_subclass__

            @classmethod
            def patched_init_subclass(sub_cls, **kwargs):
                sub_cls._orig_init_subclass(**kwargs)
                apply_patch(sub_cls)

            cls.__init_subclass__ = patched_init_subclass

        for sub in cls.__subclasses__():
            if "__init__" in sub.__dict__:
                apply_patch(sub)

    def restore_patch(cls):
        if hasattr(cls, "_orig_init"):
            cls.__init__ = cls._orig_init
            del cls._orig_init

        if hasattr(cls, "_orig_init_subclass"):
            cls.__init_subclass__ = cls._orig_init_subclass
            del cls._orig_init_subclass

        for sub in cls.__subclasses__():
            restore_patch(sub)

    try:
        torch.set_default_device("meta")
        apply_patch(torch.nn.Module)
        yield
    finally:
        torch.set_default_device(old_device)
        restore_patch(torch.nn.Module)


def _load_pretrained_weights(
    model: PreTrainedModel,
    state_dict: Dict[str, torch.Tensor],
    pretrained_model_name_or_path: str,
    trunk_model_name_or_path: Optional[str] = None,
):
    if mpu.get_expert_data_parallel_rank() == 0:
        expert_param_tags = {}

        original_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        tie_word_embeddings = original_config.tie_word_embeddings

        head_key = "lm_head.weight"
        # TODO: handle embedding keys for general models
        embedding_key = "model.language_model.embed_tokens.weight"

        if mpu.get_data_parallel_rank() == 0 and tie_word_embeddings and head_key in state_dict and embedding_key not in state_dict:
            state_dict[embedding_key] = state_dict[head_key].clone()

        missing_keys = _load_checkpoint_files(
            pretrained_model_name_or_path,
            expert_param_tags=expert_param_tags,
            state_dict=state_dict,
            prefix=model.base_model_prefix,
        )

        state_dict_args = {}
        if "convert" in inspect.signature(model.state_dict).parameters:
            state_dict_args["convert"] = True
        original_state_dict = model.state_dict(**state_dict_args)

        if mpu.get_data_parallel_rank() == 0 and tie_word_embeddings and head_key in original_state_dict:
            assert embedding_key not in missing_keys
            assert head_key in missing_keys
            missing_keys.remove(head_key)
            state_dict[head_key].copy_(state_dict[embedding_key])
            if embedding_key not in original_state_dict:
                state_dict.pop(embedding_key)

        if original_config.model_type == "rynn_brain_vla":
            allowed_missing = {"latent_readout_proj.weight", "latent_readout_proj.bias"} if (
                original_config.use_latent_actions and model.config.use_latent_head_readout
            ) else set()
            unexpected_missing = set(missing_keys) - allowed_missing
            if unexpected_missing:
                raise ValueError(f"Incomplete RynnVLA checkpoint transfer: {sorted(unexpected_missing)}")

        logger.info(
            f"Loaded checkpoint from '{pretrained_model_name_or_path}', missing keys: {missing_keys}"
        )

        if trunk_model_name_or_path is not None:
            trunk_state_dict = {
                key: tensor
                for key, tensor in state_dict.items()
                if key.startswith(("visual.", "language_model."))
            }
            if not trunk_state_dict:
                raise ValueError("The model has no visual.* or language_model.* tensors to overlay")
            trunk_missing_keys = _load_checkpoint_files(
                trunk_model_name_or_path,
                expert_param_tags={},
                state_dict=trunk_state_dict,
                prefix=model.base_model_prefix,
            )
            if trunk_missing_keys:
                raise ValueError(
                    f"Trunk checkpoint '{trunk_model_name_or_path}' is missing target tensors: "
                    f"{sorted(trunk_missing_keys)}"
                )
            visual_count = sum(key.startswith("visual.") for key in trunk_state_dict)
            language_count = sum(key.startswith("language_model.") for key in trunk_state_dict)
            logger.info(
                f"Overlaid raw trunk from '{trunk_model_name_or_path}': "
                f"visual={visual_count}, language_model={language_count}"
            )

        return missing_keys
    else:
        return list(state_dict.keys())


def _init_missing_weights(
    model: PreTrainedModel,
    missing_keys: List[str],
):
    if not missing_keys:
        return

    modules_to_init = set()
    named_modules = dict(model.named_modules())

    for key in missing_keys:
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            module_name = parts[0]
        else:
            module_name = ""

        if module_name in named_modules:
            modules_to_init.add(module_name)

    # sorted() is load-bearing, not cosmetic. _init_weights draws from the global RNG, so the
    # iteration order decides which module consumes which slice of the random stream. A set of
    # str iterates in hash order, and CPython randomises str hashing per process unless
    # PYTHONHASHSEED is pinned -- so without the sort, two runs with an identical seed produce
    # DIFFERENT initial weights. Measured on RynnBrain-2B: action_in_proj.weight summed to
    # -3.783240 under PYTHONHASHSEED=1 and -2.971881 under =2. It matters here because the
    # action expert is initialised entirely from scratch (the checkpoint has nothing for it),
    # so each ablation arm was starting from a different random draw and every arm-to-arm
    # delta carried an uncontrolled confound.
    for module_name in sorted(modules_to_init):
        module = named_modules[module_name]
        model._init_weights(module)
        logger.debug(f"Initialized missing weights for module: {module_name}")

    logger.info(
        f"Initialized {sorted(modules_to_init)} with missing pretrained weights"
    )


def _reset_pretrained_modules(
    model: PreTrainedModel,
    module_prefixes: Optional[List[str]],
    missing_keys: List[str],
    seed: int,
    validate_missing_keys: bool = True,
):
    if not module_prefixes:
        return

    prefixes = [prefix.strip().rstrip(".") for prefix in module_prefixes]
    if any(not prefix for prefix in prefixes):
        raise ValueError("reset_pretrained_modules contains an empty prefix")
    for index, prefix in enumerate(prefixes):
        for other in prefixes[index + 1:]:
            if prefix == other or prefix.startswith(f"{other}.") or other.startswith(f"{prefix}."):
                raise ValueError(f"Overlapping reset module prefixes: '{prefix}' and '{other}'")

    named_parameters = dict(model.named_parameters())
    parameter_names = list(named_parameters.keys())
    selected_keys = []
    missing_set = set(missing_keys)
    for prefix in prefixes:
        matched = [
            name
            for name in parameter_names
            if name == prefix or name.startswith(f"{prefix}.")
        ]
        if not matched:
            raise ValueError(f"Reset module prefix matched no parameters: '{prefix}'")
        naturally_missing = sorted(missing_set.intersection(matched))
        if validate_missing_keys and naturally_missing:
            raise ValueError(
                f"Reset module prefix '{prefix}' includes tensors absent from the primary checkpoint: "
                f"{naturally_missing}"
            )
        selected_keys.extend(matched)

    cuda_devices = sorted({
        parameter.device.index
        for name, parameter in named_parameters.items()
        if name in selected_keys and parameter.device.type == "cuda"
    })
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        for device in cuda_devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        _init_missing_weights(model, selected_keys)

    logger.info(
        f"Reset pretrained modules {prefixes} with seed={seed}; tensors={len(selected_keys)}"
    )


def init_weights(
    model: PreTrainedModel,
    pretrained_model_name_or_path: Optional[str] = None,
    trunk_model_name_or_path: Optional[str] = None,
    reset_pretrained_modules: Optional[List[str]] = None,
    reset_pretrained_seed: int = 42,
):
    if pretrained_model_name_or_path is None and (
        trunk_model_name_or_path is not None or reset_pretrained_modules
    ):
        raise ValueError("Trunk overlay and pretrained-module reset require a primary checkpoint")

    state_dict_args = {}
    if "convert" in inspect.signature(model.state_dict).parameters:
        state_dict_args["convert"] = True
    original_state_dict = model.state_dict(**state_dict_args)

    # Ensuring continuous memory allocation
    state_dict = {
        key: torch.empty_like(tensor, memory_format=torch.contiguous_format, device="cuda")
        for key, tensor in original_state_dict.items()
    }

    missing_keys = []
    if pretrained_model_name_or_path is not None:
        missing_keys = _load_pretrained_weights(
            model,
            state_dict=state_dict,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            trunk_model_name_or_path=trunk_model_name_or_path,
        )

    state_dict_args = {"strict": True, "assign": True}
    if "convert" in inspect.signature(model.load_state_dict).parameters:
        state_dict_args["convert"] = True
    model.load_state_dict(state_dict, **state_dict_args)

    model.to("cuda")
    _init_missing_weights(model, missing_keys)
    _reset_pretrained_modules(
        model,
        module_prefixes=reset_pretrained_modules,
        missing_keys=missing_keys,
        seed=reset_pretrained_seed,
        validate_missing_keys=mpu.get_expert_data_parallel_rank() == 0,
    )
    model.tie_weights()

    torch.distributed.barrier()

    return model


def _infer_model_type(
    model_path: str,
    model_type: Optional[str] = None,
):
    if model_type is None:
        config = AutoConfig.from_pretrained(model_path)
        return config.model_type
    return model_type


def _apply_monkey_patch(model_type: str):
    module_dir = os.path.join(os.path.dirname(__file__), model_type)
    assert os.path.isdir(module_dir)
    module = importlib.import_module(f".{model_type}", package=__package__)
    assert hasattr(module, "apply_monkey_patch")
    logger.info(f"Apply monkey patch for `{model_type}` using {module.apply_monkey_patch}")
    module.apply_monkey_patch()


def build_processor(
    model_type: Optional[str],
    model_path: str,
    processor_overrides: Dict[str, Any] = {},
):
    model_type = _infer_model_type(model_path=model_path, model_type=model_type)
    _apply_monkey_patch(model_type)

    # Re-register the model-type-specific processor class AFTER the monkey patch.
    # The patch chain may import sibling model packages (e.g. qwen3_vl when building
    # rynn_brain_vla) whose apply_monkey_patch replaces the parent Qwen3VLConfig entry
    # in PROCESSOR_MAPPING with a generic processor subclass, clobbering the
    # subclass registration done at module import time.
    module = importlib.import_module(f".{model_type}", package=__package__)
    processor_class = getattr(module, "PROCESSOR_CLASS", None)
    config_class = CONFIG_MAPPING[model_type]
    if processor_class is not None:
        PROCESSOR_MAPPING.register(config_class, processor_class)
    else:
        processor_class = PROCESSOR_MAPPING[config_class]

    processor = processor_class.from_pretrained(model_path, **processor_overrides)

    return processor


def build_model(
    model_type: Optional[str],
    model_path: str,
    dtype: torch.dtype,
    attn_implementation: str,
    config_overrides: Dict[str, Any] = {},
    vision_encoder_path: Optional[str] = None,
    reduced_layers_in_stage_zero: int = 0,
):
    model_type = _infer_model_type(model_path=model_path, model_type=model_type)
    _apply_monkey_patch(model_type)

    config_class = CONFIG_MAPPING[model_type]
    config = config_class.from_pretrained(model_path, **config_overrides)

    tie_word_embeddings = config.tie_word_embeddings
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        tie_word_embeddings = False
    config.tie_word_embeddings = tie_word_embeddings
    config.get_text_config().tie_word_embeddings = tie_word_embeddings

    with _init_empty_params():
        try:
            model = AutoModelForImageTextToText.from_config(
                config=config,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
        except ValueError:
            model = AutoModel.from_config(
                config=config,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )

    pp_world_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    if pp_world_size > 1:
        assert hasattr(model, "apply_pipeline_parallel")
        model.apply_pipeline_parallel(
            num_stages=pp_world_size,
            stage_index=pp_rank,
            reduced_layers_in_stage_zero=reduced_layers_in_stage_zero,
        )

    return model
