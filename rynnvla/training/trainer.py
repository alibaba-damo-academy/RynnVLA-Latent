# Portions of this file are derived from HuggingFace Transformers
# (https://github.com/huggingface/transformers), Copyright The HuggingFace Inc. team,
# licensed under the Apache License, Version 2.0. The license text is in LICENSE; the
# attribution is recorded in NOTICE.
# Upstream reference: src/transformers/trainer.py (Trainer / TrainerState)

import contextlib
import functools
import gc
import inspect
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List, Optional, Union

import deepspeed
import numpy as np
import shutil
import tempfile
import torch
import torch.nn as nn
from deepspeed.runtime.checkpoint_engine import CheckpointCommitInfo
from packaging import version
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import Trainer as _Trainer
from transformers.trainer import (
    DEFAULT_CALLBACKS,
    DEFAULT_PROGRESS_CALLBACK,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    BaseImageProcessor,
    CallbackHandler,
    DataCollator,
    ExportableState,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PrinterCallback,
    ProcessorMixin,
    TrainerCallback,
    TrainerControl,
    TrainerMemoryTracker,
    TrainOutput,
    enable_full_determinism,
    get_model_param_count,
    get_reporting_integration_callbacks,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.trainer import (
    TrainerState as _TrainerState,
)

from ..arguments import TrainingArguments
from ..constants import RESUME_ARGS_NAME, RESUME_CRITICAL_FIELDS
from ..utils import logging
from ..utils.pipeline_parallel import ALL_PIPELINE_SCHEDULES, PipelineModule, PipelineStage, gather_pp_params
from .sampler import DistributedBatchSampler
from .ema import EMA

logger = logging.get_logger(__name__)

# ── opt-in step probe ────────────────────────────────────────────────────────────────
# Exists because the full-corpus run loses 59-71% of wall clock to 20-120s stalls whose
# layer was unknown: queue depth (prefetch_factor 2 -> 8) provably did not move them
# (permutation p=0.80 over 309 steps), so guessing further config knobs was wasting
# 64-GPU restarts. This attributes each stall to dataloader-wait vs everything-else, and
# records gen-2 GC so the "period == num_workers == GC allocation period" coincidence can
# be settled by timestamp instead of by arithmetic.
#
# Off unless RYNNVLA_STEP_PROBE=1. Cost when on: two monotonic() calls per batch plus one
# early-returning gc.callbacks entry per collection. Logs only above a threshold, so a
# multi-day run cannot flood the 30s log mirror. Pure observation -- no training semantics
# change, and nothing here appears in RESUME_CRITICAL_FIELDS.
_STEP_PROBE = os.environ.get("RYNNVLA_STEP_PROBE", "") == "1"
_STEP_PROBE_SLOW = float(os.environ.get("RYNNVLA_STEP_PROBE_SECONDS", "8"))
_STEP_PROBE_GC_SLOW = float(os.environ.get("RYNNVLA_STEP_PROBE_GC_SECONDS", "1"))
_probe = {"batches": 0, "data": 0.0, "gc_start": 0.0, "gc_total": 0.0, "gc_count": 0}


def _probe_gc(phase, info):
    # gen-0/gen-1 fire thousands of times per step; bail before touching the clock.
    if info.get("generation") != 2:
        return
    # Only the training process, not the forked dataloader workers: 64 loggers rather than
    # 64 x num_workers, and the main process is the one whose stall shows up as a step time.
    if torch.utils.data.get_worker_info() is not None:
        return
    if phase == "start":
        _probe["gc_start"] = time.monotonic()
        return
    dt = time.monotonic() - _probe["gc_start"]
    _probe["gc_total"] += dt
    _probe["gc_count"] += 1
    if dt >= _STEP_PROBE_GC_SLOW:
        logger.warning(
            f"[stepprobe] GC gen2 {dt:.2f}s (cum {_probe['gc_total']:.1f}s over "
            f"{_probe['gc_count']}) after batch {_probe['batches']}"
        )


if _STEP_PROBE:
    gc.callbacks.append(_probe_gc)


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False
    except AttributeError:
        # Ray DataSets raises an AttributeError: https://github.com/ray-project/ray/blob/master/python/ray/data/dataset.py#L5616
        return False


def _is_fuse_mount(path: str) -> bool:
    """Detect FUSE mounts that require checkpoint staging on local disk."""
    try:
        real = os.path.realpath(path)
        fuse_mounts = []
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and "fuse" in parts[2]:
                    fuse_mounts.append(os.path.realpath(parts[1]).rstrip("/"))
        for mp in fuse_mounts:
            if real == mp or real.startswith(mp + "/"):
                return True
    except Exception:
        pass
    return False


class _FuseStagedDirectory:
    """Stage checkpoint writes on local disk, then copy back to a FUSE mount.

    Each rank writes its files to a local temp dir, and ``cleanup()`` copies them
    to the shared FUSE destination via a retry-aware copy callable. This avoids
    silent corruption from writing large tensors directly to a FUSE mount.
    """

    def __init__(self, final_path: str, copy_fn):
        self.final_path = final_path
        self._copy_fn = copy_fn  # bound Trainer._retry_copy(src, dst, is_dir=...)
        self.name = tempfile.mkdtemp(prefix="rynn_ckpt_stage_")

    def _copy_back(self):
        if not os.path.isdir(self.name):
            return
        os.makedirs(self.final_path, exist_ok=True)
        # trainer_state.json goes last. get_last_checkpoint gates on it and this loop is
        # sequential on one thread, so once it lands, everything else this rank staged is
        # already whole on the shared mount -- including files resume opens bare that the
        # gate never checks (latest, scheduler.pt, rng_state_*, config.json). os.listdir
        # order is arbitrary: `latest` was measured landing 27 s after the gate. Most of
        # that stretch is already excluded by get_last_checkpoint's EMA weights/shadow
        # pairing check, leaving ~1 s per save where a preemption yields a checkpoint that
        # passes get_last_checkpoint and then dies on a bare open; this ordering closes it.
        # Non-zero ranks do not stage this file, so the key is uniformly False for them.
        items = sorted(os.listdir(self.name), key=lambda name: name == TRAINER_STATE_NAME)
        for item in items:
            src = os.path.join(self.name, item)
            dst = os.path.join(self.final_path, item)
            self._copy_fn(src, dst, is_dir=os.path.isdir(src))
        shutil.rmtree(self.name, ignore_errors=True)

    def cleanup(self):
        self._copy_back()

    def start_async_cleanup(self) -> threading.Thread:
        """Copy back off the critical path; the caller must join before relying on the files."""
        thread = threading.Thread(target=self._copy_back, name="ckpt-stage-copy")
        thread.start()
        return thread


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    pattern = re.compile("checkpoint" + r"\-(\d+)$")
    checkpoints = sorted(
        [path for path in content if pattern.search(path) is not None],
        key=lambda x: int(pattern.search(x).groups()[0]),
        reverse=True,
    )
    if len(checkpoints) == 0:
        return

    for checkpoint_name in checkpoints:
        step = int(pattern.search(checkpoint_name).groups()[0])
        checkpoint = os.path.join(folder, checkpoint_name)

        trainer_state = os.path.join(checkpoint, TRAINER_STATE_NAME)
        ds_state_path = os.path.join(checkpoint, "deepspeed_state.pt")
        model_path = os.path.join(checkpoint, "model_pp_rank_00_ep_rank_00.pt")
        if not (os.path.isfile(trainer_state) and os.path.isfile(ds_state_path) and os.path.isfile(model_path)):
            logger.warning(f"Skipping incomplete checkpoint {checkpoint}: missing trainer/deepspeed/model files")
            continue

        # _load_model reads the EMA shadow unconditionally when args.use_ema, and both shipped
        # recipes turn it on -- so a checkpoint interrupted mid-EMA-write would be reported as
        # complete here and then die with a bare FileNotFoundError during model load. _save_model
        # always writes the EMA weights and the shadow together, so the weights' presence is what
        # says "this checkpoint was written with EMA on" and therefore requires the shadow.
        ema_weights_path = os.path.join(checkpoint, "ema_model_pp_rank_00_ep_rank_00.pt")
        ema_shadow_path = os.path.join(checkpoint, "ema_shadow_pp_rank_00_ep_rank_00.pt")
        if os.path.isfile(ema_weights_path) and not os.path.isfile(ema_shadow_path):
            logger.warning(f"Skipping incomplete checkpoint {checkpoint}: EMA weights present but ema shadow missing")
            continue

        try:
            ds_state = torch.load(ds_state_path, map_location="cpu", weights_only=False)
            dp_world_size = int(ds_state.get("dp_world_size", 1))
            tag = f"global_step{step}"
            tag_dir = os.path.join(checkpoint, tag)
            missing = []
            for rank in range(dp_world_size):
                fname = f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                fpath = os.path.join(tag_dir, fname)
                if not os.path.isfile(fpath):
                    missing.append(fname)
            if missing:
                logger.warning(f"Skipping incomplete checkpoint {checkpoint}: missing optimizer states {missing[:3]}{'...' if len(missing) > 3 else ''}")
                continue
        except Exception as exc:
            logger.warning(f"Skipping checkpoint {checkpoint}: failed completeness check: {exc}")
            continue

        return checkpoint

    return None


def rotate_checkpoints(
    output_dir: str,
    save_total_limit: Optional[int] = None,
    keep_every: Optional[int] = None,
):
    if save_total_limit is None or save_total_limit <= 0:
        return

    content = os.listdir(output_dir)

    pattern = re.compile("checkpoint" + r"\-(\d+)$")
    checkpoints = sorted(
        [path for path in content if pattern.search(path) is not None],
        key=lambda x: int(pattern.search(x).groups()[0])
    )

    # Milestones are exempt from the rolling window and do not consume its depth, so the
    # window keeps its full length regardless of how many milestones have accumulated.
    if keep_every and keep_every > 0:
        milestones = {c for c in checkpoints if int(pattern.search(c).groups()[0]) % keep_every == 0}
        rotatable = [c for c in checkpoints if c not in milestones]
    else:
        milestones, rotatable = set(), checkpoints

    if len(rotatable) <= save_total_limit:
        return

    for checkpoint in rotatable[:-save_total_limit]:
        logger.info(f"Rotating out {checkpoint} (keeping {len(milestones)} milestone(s))")
        checkpoint = os.path.join(output_dir, checkpoint)
        shutil.rmtree(checkpoint, ignore_errors=True)


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


class LazyBatchLoader(object):
    _torch_dtype_map = {
        str(dtype): dtype for dtype in [
            torch.float, torch.float32, torch.float16, torch.bfloat16,
            torch.long, torch.int64, torch.int32, torch.int16, torch.int8,
            torch.uint64, torch.uint32, torch.uint16, torch.uint8, torch.bool,
        ]
    }

    def __init__(
        self,
        epoch_iterator: Iterator,
        num_batches: int,
        training_args: TrainingArguments,
    ):
        self.epoch_iterator = epoch_iterator
        self.num_batches = num_batches
        self.args = training_args

        self._batch_samples = []

    def __len__(self):
        return self.num_batches

    def _load_one_batch(self):
        assert len(self._batch_samples) < self.num_batches

        if (not self.args.cp_broadcast_data or self.args.cp_rank == 0) and \
            (not self.args.pp_broadcast_data or self.args.pp_rank == 0):
            if _STEP_PROBE:
                _t0 = time.monotonic()
                batch = next(self.epoch_iterator)
                _wait = time.monotonic() - _t0
                _probe["batches"] += 1
                _probe["data"] += _wait
                # Per-event line pinpoints WHICH batch stalled; the per-step summary emitted
                # near global_step += 1 splits the same stall across five buckets (data /
                # fwd_bwd / opt / ema / resid). A stall that produces neither is not in this
                # process at all -- with 64 ranks logging it, that means this rank was waiting
                # in a collective on some other rank, which is itself the answer.
                if _wait >= _STEP_PROBE_SLOW:
                    logger.warning(
                        f"[stepprobe] DATALOADER WAIT {_wait:.1f}s at batch {_probe['batches']} "
                        f"(gc gen2 cum {_probe['gc_total']:.1f}s / {_probe['gc_count']} collections)"
                    )
            else:
                batch = next(self.epoch_iterator)
        else:
            batch = {}

        if self.args.cp_broadcast_data and (not self.args.pp_broadcast_data or self.args.pp_rank == 0):
            if self.args.cp_rank == 0:
                meta_data = defaultdict(list)
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        meta_data[str(value.dtype)].append((key, tuple(value.shape)))
                    else:
                        meta_data["others"].append((key, value))
            else:
                meta_data = None

            meta_data = [meta_data]
            torch.distributed.broadcast_object_list(
                meta_data,
                group=self.args.cp_group,
                group_src=0,
            )
            meta_data = meta_data[0]

            others = meta_data.pop("others", [])
            if self.args.cp_rank != 0:
                for key, value in others:
                    batch[key] = value

            for dtype, items in meta_data.items():
                dtype = self._torch_dtype_map[dtype]
                sizes = [math.prod(shape) for _, shape in items]

                if self.args.cp_rank == 0:
                    flattened_tensors = []
                    for key, _ in items:
                        batch[key] = batch[key].to(self.args.device)
                        flattened_tensors.append(batch[key].flatten())
                    buffer = torch.cat(flattened_tensors, dim=0)
                else:
                    buffer = torch.empty(sum(sizes), dtype=dtype, device=self.args.device)

                torch.distributed.broadcast(
                    buffer,
                    group=self.args.cp_group,
                    group_src=0,
                )

                if self.args.cp_rank != 0:
                    buffers = buffer.split(sizes, dim=0)
                    for (key, shape), tensor in zip(items, buffers):
                        batch[key] = tensor.view(shape)

        if self.args.pp_broadcast_data:
            cu_seq_lens = torch.empty(
                (self.args.micro_batch_size * self.args.dp_world_size + 2,),
                dtype=torch.int32,
                device=self.args.device,
            )

            if self.args.pp_rank == 0:
                cu_seq_lens[-1] = len(batch["cu_seq_lens_q"])
                cu_seq_lens[:len(batch["cu_seq_lens_q"])] = batch["cu_seq_lens_q"]

            torch.distributed.broadcast(
                cu_seq_lens,
                group=self.args.pp_group,
                group_src=0,
            )
            cu_seq_lens = cu_seq_lens[:cu_seq_lens[-1]]

            if self.args.pp_rank == 0:
                batch["position_ids"] = batch["position_ids"].to(self.args.device)
                assert batch["position_ids"].size() == (3, 1, cu_seq_lens[-1])
                assert batch["position_ids"].dtype == torch.long
                position_ids = batch["position_ids"]
                batch["labels"] = batch["labels"].to(self.args.device)
                assert batch["labels"].size() == (1, cu_seq_lens[-1])
                assert batch["labels"].dtype == torch.long
                labels = batch["labels"]
            else:
                position_ids = torch.empty(
                    (3, 1, cu_seq_lens[-1]),
                    dtype=torch.long,
                    device=self.args.device,
                )
                labels = torch.empty(
                    (1, cu_seq_lens[-1]),
                    dtype=torch.long,
                    device=self.args.device,
                )

            torch.distributed.broadcast(
                position_ids,
                group=self.args.pp_group,
                group_src=0,
            )
            torch.distributed.broadcast(
                labels,
                group=self.args.pp_group,
                group_src=0,
            )

            if self.args.pp_rank != 0:
                max_length = torch.amax(cu_seq_lens[1:] - cu_seq_lens[:-1]).item()
                batch["cu_seq_lens_q"] = cu_seq_lens
                batch["cu_seq_lens_k"] = cu_seq_lens
                batch["max_length_q"] = max_length
                batch["max_length_k"] = max_length
                batch["position_ids"] = position_ids
                batch["labels"] = labels

        if self.args.synchronize_experts_before_forward:
            torch.distributed.barrier(group=self.args.ep_group)

        return batch

    def __getitem__(self, index: int):
        if index < 0 or index >= self.num_batches:
            raise IndexError(f"Index {index} is out of range")

        if index < len(self._batch_samples):
            return self._batch_samples[index]

        torch.cuda.nvtx.range_push("load_data")

        num_batches = index - len(self._batch_samples) + 1
        batch_samples = []

        for _ in range(num_batches):
            batch_samples.append(self._load_one_batch())

        num_items_in_batch = None
        count_num_items_in_batch = "labels" in batch_samples[0]

        if count_num_items_in_batch:
            if self.args.loss_reduction_scope == "batch":
                num_batches = self.num_batches - len(self._batch_samples) - len(batch_samples)
                for _ in range(num_batches):
                    batch_samples.append(self._load_one_batch())

                num_items_in_batch = sum((batch["labels"].ne(-100)).sum() for batch in batch_samples) / len(
                    batch_samples
                )
                if self.args.average_tokens_across_devices and self.args.dp_world_size > 1:
                    num_items_in_batch = num_items_in_batch.to(self.args.device)
                    torch.distributed.all_reduce(
                        num_items_in_batch,
                        op=torch.distributed.ReduceOp.SUM,
                        group=self.args.dp_group,
                    )
                    num_items_in_batch = num_items_in_batch / self.args.dp_world_size

            elif self.args.loss_reduction_scope == "sequence":
                num_items_in_batch = self.args.micro_batch_size

            else:
                raise ValueError(f"Unknown loss reduction scope: {self.args.loss_reduction_scope}")

        for batch in batch_samples:
            batch["num_items_in_batch"] = num_items_in_batch

        self._batch_samples.extend(batch_samples)

        torch.cuda.nvtx.range_pop()

        return self._batch_samples[index]


@dataclass
class TrainerState(_TrainerState):
    num_input_tokens_seen: float = 0.0
    running_time: float = 0.0


class Trainer(object):
    # Reuse some functions from huggingface transformers
    create_scheduler = _Trainer.create_scheduler
    get_optimizer_cls_and_kwargs = staticmethod(_Trainer.get_optimizer_cls_and_kwargs)
    _load_callback_state = _Trainer._load_callback_state
    _get_learning_rate = _Trainer._get_learning_rate

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: DataCollator,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        self.args = args
        # Seed must be set before instantiating the model when using model
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)

        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False
        self.model = model

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker()
        self._memory_tracker.start()

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        self.is_deepspeed_enabled = True

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model
        self.optimizer = None
        self.lr_scheduler = None
        self.ema = None

        # Check if the model has explicit setup for loss kwargs,
        # if not, check if `**kwargs` are in model.forward
        if hasattr(model, "accepts_loss_kwargs"):
            self.model_accepts_loss_kwargs = model.accepts_loss_kwargs
        else:
            forward_params = inspect.signature(model.forward).parameters
            self.model_accepts_loss_kwargs = any(
                k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values()
            )

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.callback_handler.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        if self.args.global_rank == 0:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise TypeError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.args.local_rank == 0,
            is_world_process_zero=self.args.global_rank == 0,
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # very last
        self._memory_tracker.stop_and_update_metrics()

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        logger.warning("Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.")
        return self.processing_class

    @tokenizer.setter
    def tokenizer(self, processing_class) -> None:
        logger.warning(
            "Trainer.tokenizer is now deprecated. You should use `Trainer.processing_class = processing_class` instead."
        )
        self.processing_class = processing_class

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        sampler_seed = torch.as_tensor(self.args.seed).cuda()
        torch.distributed.broadcast(sampler_seed, src=0)

        if self.args.decoder_load_balancing or self.args.dynamic_batching:
            assert hasattr(train_dataset, "get_sequence_lengths")
            sequence_lengths = train_dataset.get_sequence_lengths(
                num_workers=self.args.dataloader_num_workers,
                cache_dir=self.args.output_dir,
            )
        else:
            sequence_lengths = None

        batch_sampler = DistributedBatchSampler(
            train_dataset,
            sequence_lengths=sequence_lengths,
            num_replicas=self.args.dp_world_size,
            rank=self.args.dp_rank,
            micro_batch_size=self.args.micro_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            shuffle=True,
            seed=sampler_seed.item(),
            drop_last=self.args.dataloader_drop_last,
            decoder_load_balancing=self.args.decoder_load_balancing,
            dynamic_batching=self.args.dynamic_batching,
            dynamic_batching_window_size=self.args.dynamic_batching_window_size,
            model_max_length=self.args.model_max_length,
            # The locality window is derived inside the sampler from this, micro_batch_size,
            # gradient_accumulation_steps and dp_world_size; it must track the real worker
            # count or the window stops dividing the worker batch stride and latent LRU
            # reuse silently collapses to ~0.
            num_workers=self.args.dataloader_num_workers,
            shuffle_mode=self.args.sampler_shuffle,
        )

        def worker_init_fn(worker_id, num_workers, rank):
            seed_worker(worker_id, num_workers=num_workers, rank=rank)

        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": functools.partial(
                worker_init_fn,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.dp_rank,
            ),
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        return DataLoader(train_dataset, **dataloader_params)

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_layer_types = [nn.LayerNorm]
        forbidden_layer_names = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        forbidden_layer_patterns = (
            [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
        )

        def get_decay_parameter_names(model):
            result = []
            for name, child in model.named_children():
                child_params = get_decay_parameter_names(child)
                result += [
                    f"{name}.{n}"
                    for n in child_params
                    if not isinstance(child, tuple(forbidden_layer_types))
                    and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
                ]
            # Add model specific parameters that are not in any child
            result += [
                k
                for k in model._parameters
                if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
            ]
            return result

        return get_decay_parameter_names(model)

    # Modules built from scratch on top of the pretrained VLM: the pi0-style action expert, the
    # action/state projections, and (latent line) the MultiViewActionDecoder. These are the only
    # randomly-initialized parameters, so they are the ones that may want a higher peak LR.
    # Matching allows a leading dot because the latent line trains a composite model whose
    # parameters are prefixed "vla_model." / "action_decoder.".
    #
    # The criterion is "built from scratch", NOT "lives in the action head" -- view_role_emb
    # sits in the VLM prefix but is still a fresh, zero-initialised table, and it was missing
    # here until 2026-08-21 because this tuple predates the V2 arm. The effect was measurable
    # and one-sided: with action_head_lr=1e-4 / learning_rate=2.5e-5, V2's only new tensor
    # trained 4x slower than the new tensors of every sibling arm (V3 latent_action_head,
    # V4 slot_seed_proj, ...). For a module that is exactly zero at step 0 and has to grow
    # entirely from gradient, that turns "does the flag help?" into "does the flag help at a
    # quarter of the learning rate?". Anything added here must be a from-scratch module, or
    # it will silently get 4x the intended LR on pretrained weights.
    DEFAULT_ACTION_HEAD_MODULES = (
        "action_expert",
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_proj",
        "time_mlp",
        "adaln_in",
        "adaln_final",
        "confidence_head",
        "slot_seed_norm",
        "slot_seed_proj",
        "action_decoder",
        "view_role_emb",
    )

    def create_optimizer(self):
        opt_model = self.model

        decay_parameters = set(self.get_decay_parameter_names(opt_model))

        action_head_lr = getattr(self.args, "action_head_lr", None)
        if action_head_lr is None:
            # One rate for everything: keep the groups exactly as they have always been.
            head_parameters = set()
        else:
            requested_prefixes = getattr(self.args, "action_head_modules", None)
            prefixes = requested_prefixes or self.DEFAULT_ACTION_HEAD_MODULES
            parameter_names = [name for name, _ in opt_model.named_parameters()]

            def matches_prefix(name, prefix):
                return re.search(rf"(?:^|\.){re.escape(prefix)}(?:\.|$)", name) is not None

            if requested_prefixes:
                unmatched = [
                    prefix
                    for prefix in prefixes
                    if not any(matches_prefix(name, prefix) for name in parameter_names)
                ]
                if unmatched:
                    raise ValueError(
                        f"action_head_modules contains prefixes that matched no parameters: {unmatched}"
                    )
            head_parameters = {
                name
                for name in parameter_names
                if any(matches_prefix(name, prefix) for prefix in prefixes)
            }
            if not head_parameters:
                raise ValueError(
                    f"action_head_lr={action_head_lr} was requested but none of {list(prefixes)} "
                    "matched a parameter name. Set action_head_modules to the correct prefixes."
                )
            logger.info(
                f"action_head_lr={action_head_lr} applies to {len(head_parameters)} tensors "
                f"matching {list(prefixes)}; the remaining parameters keep lr={self.args.learning_rate}"
            )

        transfer_lr = getattr(self.args, "transfer_lr", None)
        transfer_prefixes = getattr(self.args, "transfer_modules", None)
        transfer_parameters = set()
        if transfer_lr is not None or transfer_prefixes is not None:
            if transfer_lr is None or not math.isfinite(transfer_lr) or transfer_lr <= 0 or not transfer_prefixes:
                raise ValueError("transfer_lr must be positive and finite and requires nonempty transfer_modules")
            parameter_names = [name for name, _ in opt_model.named_parameters()]

            def transfer_matches(name, prefix):
                return re.search(rf"(?:^|\.){re.escape(prefix)}(?:\.|$)", name) is not None

            unmatched = [prefix for prefix in transfer_prefixes
                         if not any(transfer_matches(name, prefix) for name in parameter_names)]
            if unmatched:
                raise ValueError(f"transfer_modules contains prefixes that matched no parameters: {unmatched}")
            transfer_parameters = {name for name in parameter_names
                                   if any(transfer_matches(name, prefix) for prefix in transfer_prefixes)}
            overlap = transfer_parameters & head_parameters
            if overlap:
                raise ValueError(f"transfer_modules overlaps action_head_modules: {sorted(overlap)[:8]}")
            logger.info(f"transfer_lr={transfer_lr} applies to {len(transfer_parameters)} tensors "
                        f"matching {list(transfer_prefixes)}")

        def collect(in_decay: bool, in_head: bool):
            return [
                p
                for n, p in opt_model.named_parameters()
                if p.requires_grad
                and n not in transfer_parameters
                and (n in decay_parameters) == in_decay
                and (n in head_parameters) == in_head
            ]

        optimizer_grouped_parameters = [
            {
                "name": "decay",
                "params": collect(in_decay=True, in_head=False),
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            {
                "name": "no_decay",
                "params": collect(in_decay=False, in_head=False),
                "lr": self.args.learning_rate,
                "weight_decay": 0.0,
            },
        ]

        if head_parameters:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "name": "action_head_decay",
                        "params": collect(in_decay=True, in_head=True),
                        "lr": action_head_lr,
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "name": "action_head_no_decay",
                        "params": collect(in_decay=False, in_head=True),
                        "lr": action_head_lr,
                        "weight_decay": 0.0,
                    },
                ]
            )
            optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

        if transfer_parameters:
            for in_decay, label in ((True, "decay"), (False, "no_decay")):
                params = [p for n, p in opt_model.named_parameters()
                          if p.requires_grad and n in transfer_parameters and (n in decay_parameters) == in_decay]
                if params:
                    optimizer_grouped_parameters.append({
                        "name": f"transfer_{label}", "params": params, "lr": transfer_lr,
                        "weight_decay": self.args.weight_decay if in_decay else 0.0,
                    })

        offload = self.args.deepspeed_config.get("zero_optimization", {}).get("offload_optimizer", {})
        if offload.get("device") == "cpu":
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            self.optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def _retry_torch_save(self, obj, path, max_retries=3, delay=10):
        """Save with retry to handle transient I/O errors on OSS-mounted filesystems."""
        import time
        for attempt in range(max_retries):
            try:
                torch.save(obj, path)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"torch.save to {path} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"torch.save to {path} failed after {max_retries} attempts: {e}")
                    raise

    def _retry_copy(self, src, dst, is_dir=False, max_retries=3, delay=10):
        """Copy file/dir with retry to handle transient I/O errors on OSS-mounted filesystems."""
        import time
        for attempt in range(max_retries):
            try:
                if is_dir:
                    shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=shutil.copy)
                else:
                    shutil.copy(src, dst)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"copy {src} -> {dst} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"copy {src} -> {dst} failed after {max_retries} attempts: {e}")
                    raise

    def _save_model(self, output_dir, full: bool = False):
        if self.args.edp_rank != 0:
            return

        if self.args.global_rank == 0:
            # Save to local tmp first to avoid safetensors mmap issues on OSS-mounted fs
            tmp_save_dir = tempfile.mkdtemp(prefix="rynn_ckpt_")
            self.model.config.save_pretrained(tmp_save_dir)
            self.processing_class.save_pretrained(tmp_save_dir)
            os.makedirs(output_dir, exist_ok=True)
            for f in os.listdir(tmp_save_dir):
                self._retry_copy(os.path.join(tmp_save_dir, f), os.path.join(output_dir, f), is_dir=False)
            shutil.rmtree(tmp_save_dir, ignore_errors=True)

        kwargs = {}
        if "convert" in inspect.signature(self.model.state_dict).parameters:
            kwargs["convert"] = False
        state_dict = self.model.state_dict(**kwargs)

        ckpt_name = f"model_pp_rank_{self.args.pp_rank:02d}_ep_rank_{self.args.ep_rank:02d}.pt"
        self._retry_torch_save(state_dict, os.path.join(output_dir, ckpt_name))

        # Also save in .bin format for from_pretrained compatibility (evaluation)
        if self.args.pp_rank == 0 and self.args.ep_rank == 0:
            bin_path = os.path.join(output_dir, "model.bin")
            self._retry_torch_save(state_dict, bin_path)
            logger.info(f"Saved model.bin for evaluation compatibility")

        # EMA weights (fp32 shadow overlaid on the live state dict) as an
        # inference-ready checkpoint.
        if self.ema is not None:
            ema_state_dict = self.ema.overlay(state_dict)
            ema_ckpt_name = f"ema_model_pp_rank_{self.args.pp_rank:02d}_ep_rank_{self.args.ep_rank:02d}.pt"
            self._retry_torch_save(ema_state_dict, os.path.join(output_dir, ema_ckpt_name))
            if self.args.pp_rank == 0 and self.args.ep_rank == 0:
                self._retry_torch_save(ema_state_dict, os.path.join(output_dir, "ema_model.bin"))
                logger.info("Saved ema_model.bin (EMA weights) for evaluation")

            # Persist the EMA's internal fp32 shadow state separately so resume can
            # restore accumulated momentum. Without this, the lazy-init in the
            # training loop rebuilds EMA from a single noisy snapshot.
            ema_shadow_name = f"ema_shadow_pp_rank_{self.args.pp_rank:02d}_ep_rank_{self.args.ep_rank:02d}.pt"
            self._retry_torch_save(self.ema.state_dict(), os.path.join(output_dir, ema_shadow_name))

    def _load_model(self, checkpoint):
        if self.args.edp_rank == 0:
            ckpt_name = f"model_pp_rank_{self.args.pp_rank:02d}_ep_rank_{self.args.ep_rank:02d}.pt"
            ckpt_path = os.path.join(checkpoint, ckpt_name)
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)

            kwargs = {"strict": True}
            if "convert" in inspect.signature(self.model.state_dict).parameters:
                kwargs["convert"] = False
            self.model.load_state_dict(state_dict, **kwargs)
            del state_dict

            if self.args.use_ema:
                shadow_name = f"ema_shadow_pp_rank_{self.args.pp_rank:02d}_ep_rank_{self.args.ep_rank:02d}.pt"
                shadow_path = os.path.join(checkpoint, shadow_name)
                shadow_sd = torch.load(shadow_path, map_location="cpu", weights_only=True, mmap=True)
                self.ema = EMA(self.model, decay=self.args.ema_decay, device=self.args.ema_device)
                self.ema.load_state_dict(shadow_sd)
                del shadow_sd
                logger.info(f"Restored EMA shadow from {shadow_path}")

        self.model_wrapped._broadcast_model()

    def _save_optimizer_and_scheduler(self, output_dir, is_tmp_dir):
        if hasattr(self.optimizer, "checkpoint_event_prologue"):
            self.optimizer.checkpoint_event_prologue()

        tag = f"global_step{self.state.global_step}"
        save_dir = os.path.join(output_dir, tag)
        if self.args.global_rank == 0 or is_tmp_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.distributed.barrier()

        commit_info = CheckpointCommitInfo(tag=tag, save_dir=output_dir, save_latest=True)
        self.model_wrapped.checkpoint_engine.create(commit_info)

        if self.model_wrapped.save_zero_checkpoint:
            self.model_wrapped._create_zero_checkpoint_files(output_dir, tag)
            # Monkey-patch torch.save with retry for DeepSpeed's internal optimizer state saves.
            # This fixes transient OSS I/O failures that caused rank 1-7 crashes.
            _original_torch_save = torch.save
            def _patched_torch_save(obj, f, *args, **kwargs):
                if isinstance(f, str):
                    import time as _time
                    for _attempt in range(3):
                        try:
                            _original_torch_save(obj, f, *args, **kwargs)
                            return
                        except Exception as _e:
                            if _attempt < 2:
                                logger.warning(f"torch.save to {f} failed (attempt {_attempt+1}/3): {_e}. Retrying in 10s...")
                                _time.sleep(10)
                            else:
                                logger.error(f"torch.save to {f} failed after 3 attempts: {_e}")
                                raise
                else:
                    _original_torch_save(obj, f, *args, **kwargs)
            torch.save = _patched_torch_save
            try:
                self.model_wrapped._save_zero_checkpoint(output_dir, tag)
            finally:
                torch.save = _original_torch_save

        if hasattr(self.optimizer, "checkpoint_event_epilogue"):
            self.optimizer.checkpoint_event_epilogue()

        if self.args.global_rank == 0:
            self._retry_torch_save(
                {
                    "skipped_steps": self.model_wrapped.skipped_steps,
                    "global_steps": self.model_wrapped.global_steps,
                    "global_samples": self.model_wrapped.global_samples,
                    "dp_world_size": self.model_wrapped.dp_world_size,
                    "mp_world_size": self.model_wrapped.mp_world_size,
                },
                os.path.join(output_dir, "deepspeed_state.pt"),
            )

        if not self.model_wrapped.checkpoint_engine.is_decoupled():
            self.model_wrapped.checkpoint_engine.commit(tag)
            if self.args.global_rank == 0:
                with open(os.path.join(output_dir, "latest"), "w") as fd:
                    fd.write(tag)

        if self.args.global_rank == 0:
            self._retry_torch_save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

        torch.distributed.barrier()

    def _load_optimizer_and_scheduler(self, checkpoint):
        if hasattr(self.optimizer, "checkpoint_event_prologue"):
            self.optimizer.checkpoint_event_prologue()

        latest_path = os.path.join(checkpoint, "latest")
        with open(latest_path, "r") as fd:
            tag = fd.read().strip()

        deepspeed_state_path = os.path.join(checkpoint, "deepspeed_state.pt")
        deepspeed_state = torch.load(deepspeed_state_path)

        self.model_wrapped.global_steps = deepspeed_state["global_steps"]
        self.model_wrapped.global_samples = deepspeed_state["global_samples"]
        self.model_wrapped.skipped_steps = deepspeed_state["skipped_steps"]
        self.model_wrapped.loaded_checkpoint_dp_world_size = deepspeed_state["dp_world_size"]
        self.model_wrapped.loaded_checkpoint_mp_world_size = deepspeed_state["mp_world_size"]

        success = self.model_wrapped._load_zero_checkpoint(checkpoint, tag, load_optimizer_states=True)
        assert success

        if hasattr(self.optimizer, "checkpoint_event_epilogue"):
            self.optimizer.checkpoint_event_epilogue()

        self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME), weights_only=True))

    def _save_rng_state(self, output_dir):
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.random.get_rng_state_all(),
        }
        self._retry_torch_save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.global_rank}.pth"))

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        rng_file = os.path.join(checkpoint, f"rng_state_{self.args.global_rank}.pth")
        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file)

        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])

    def _save_checkpoint(self):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # One checkpoint in flight at a time: the previous copy must land (and be
        # validated/rotated) before this one starts filling local disk again.
        self._finish_pending_stage()

        # Save model checkpoint
        checkpoint_folder = f"checkpoint-{self.state.global_step}"

        final_output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        staged_dir = None

        if _is_fuse_mount(final_output_dir):
            # FUSE mount: stage on local disk, copy back on cleanup. Writing
            # the multi-GB checkpoint directly to a fuse mount can silently corrupt it.
            staged_dir = _FuseStagedDirectory(final_output_dir, self._retry_copy)
            output_dir = staged_dir.name
            staged = True
        else:
            output_dir = final_output_dir
            staged = False
            if self.args.global_rank == 0:
                os.makedirs(output_dir, exist_ok=True)
        torch.distributed.barrier()

        self._save_model(output_dir, full=False)
        self._save_optimizer_and_scheduler(output_dir, is_tmp_dir=staged)
        self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.global_rank == 0:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            # Snapshot the settings that would silently reshape training if changed on resume.
            # trainer_state.json does not carry warmup_steps / lr_scheduler_type /
            # gradient_accumulation_steps / data_mixture, so without this there is nothing to
            # compare them against and a resumed run quietly redraws its LR curve and miscounts
            # the batches to skip.
            snapshot = {}
            for name in RESUME_CRITICAL_FIELDS:
                value = getattr(self.args, name, None)
                snapshot[name] = getattr(value, "value", value)
            with open(os.path.join(output_dir, RESUME_ARGS_NAME), "w", encoding="utf-8") as fd:
                json.dump(snapshot, fd, indent=2, sort_keys=True)

        if staged_dir is not None:
            # A 4B checkpoint is ~68GB on rank 0 alone, and the FUSE mount takes >10min for
            # it. Copying inline held every other rank in the barrier below until NCCL's
            # watchdog killed the job, so drain it in the background instead and validate
            # at the join point, where the files are actually there.
            self._pending_stage = {
                "thread": staged_dir.start_async_cleanup(),
                "step": self.state.global_step,
            }
        # Cheap for the staged path: every rank has only handed its copy to a thread.
        torch.distributed.barrier()
        if getattr(self, "_pending_stage", None) is None:
            self._validate_and_rotate(self.state.global_step)

    def _finish_pending_stage(self):
        """Join the in-flight checkpoint copy, then validate and rotate that checkpoint."""
        pending = getattr(self, "_pending_stage", None)
        if pending is None:
            return
        self._pending_stage = None
        started = time.monotonic()
        pending["thread"].join()
        step = pending["step"]
        logger.info(
            "Checkpoint %d finished staging to the shared mount (waited %.0fs at the join point)",
            step, time.monotonic() - started,
        )
        torch.distributed.barrier()
        self._validate_and_rotate(step)

    def _validate_and_rotate(self, step: int):
        if self.args.global_rank == 0:
            # Validate the checkpoint before rotating old ones.
            # This prevents deleting valid old checkpoints when the new save is incomplete.
            latest_ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
            latest_tag_dir = os.path.join(latest_ckpt_dir, f"global_step{step}")
            checkpoint_valid = True
            if os.path.isdir(latest_tag_dir) and self.model_wrapped.save_zero_checkpoint:
                dp_world_size = self.model_wrapped.dp_world_size if hasattr(self.model_wrapped, 'dp_world_size') else 1
                for rank in range(dp_world_size):
                    fname = f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                    if not os.path.isfile(os.path.join(latest_tag_dir, fname)):
                        logger.warning(f"Checkpoint {step} missing {fname}, skipping rotation to preserve old checkpoints")
                        checkpoint_valid = False
                        break
            if checkpoint_valid:
                rotate_checkpoints(
                    output_dir=self.args.output_dir,
                    save_total_limit=self.args.save_total_limit,
                    keep_every=self.args.save_keep_every,
                )

        torch.distributed.barrier()

    def _staged_checkpoint(self, checkpoint: str) -> str:
        # Every rank reads every optimizer shard, and on top of that _load_model has only
        # edp_rank 0 read the whole weights file. Off a fuse-mounted share that puts rank 0
        # minutes behind the other ranks, which are already blocked in _broadcast_model() on
        # NCCL's 600 s default. The launcher copies the checkpoint to node-local disk and
        # points us at it.
        staged = os.environ.get("RESUME_STAGE_DIR", "").strip()
        if not staged:
            return checkpoint
        if not os.path.isdir(staged):
            logger.warning(f"RESUME_STAGE_DIR={staged} does not exist; resuming from {checkpoint}")
            return checkpoint
        if os.path.basename(staged.rstrip("/")) != os.path.basename(checkpoint.rstrip("/")):
            logger.warning(
                f"RESUME_STAGE_DIR={staged} is a copy of a different step than {checkpoint}; "
                "resuming from the original path"
            )
            return checkpoint
        logger.info(f"Resuming from node-local copy {staged}")
        return staged

    def _load_checkpoint(self, checkpoint: Optional[str]):
        self._load_optimizer_and_scheduler(checkpoint)
        self._load_model(checkpoint)

    def _save_full_model(self):
        state_dict = self.model.state_dict()
        state_dict = gather_pp_params(state_dict)

        if self.args.global_rank == 0:
            final_output_dir = self.args.output_dir
            if _is_fuse_mount(final_output_dir):
                # FUSE-mounted filesystem: save to local tmp first, then copy
                import tempfile as _tempfile
                tmp_save_dir = _tempfile.mkdtemp(prefix="rynn_full_")
                output_dir = tmp_save_dir
            else:
                output_dir = final_output_dir

            # The EMA shadow is otherwise only ever written inside checkpoint-N, where
            # nothing that loads the run root as MODEL_PATH can reach it. `ema.shadow` is
            # keyed on self.model.named_parameters(), the same namespace as the gathered
            # dict, so the overlay applies directly.
            #
            # Build the overlay BEFORE the root save, not after. transformers>=5
            # save_pretrained pops every tensor out of the dict it is handed
            # (modeling_utils.py: "Get the tensor, and remove it from state_dict to avoid
            # keeping the ref"), so saving the root first leaves {} behind and
            # overlay({}) returns {}. That produced an ema/ holding only config+tokenizer
            # while still logging "Model weights saved" -- the only visible trace was
            # "Writing model shards: 0it" -- and pointing MODEL_PATH at it later died on a
            # bare AssertionError in models/__init__. overlay() materialises on CPU, so
            # holding both dicts costs host RAM only; device memory is unchanged.
            ema_state = (
                self.ema.overlay(state_dict)
                if (self.args.export_ema and self.ema is not None)
                else None
            )
            if ema_state is not None and not ema_state:
                raise RuntimeError(
                    "EMA overlay came back empty; refusing to write a weightless ema/ "
                    "export. The gathered state dict was consumed before the overlay was "
                    "built."
                )

            self.model.save_pretrained(output_dir, state_dict=state_dict)
            self.processing_class.save_pretrained(output_dir)

            if ema_state is not None:
                ema_dir = os.path.join(output_dir, "ema")
                self.model.save_pretrained(ema_dir, state_dict=ema_state)
                self.processing_class.save_pretrained(ema_dir)
                logger.info(f"Saved EMA export to {os.path.join(final_output_dir, 'ema')}")

            # Copy from local tmp to OSS if needed
            if output_dir != final_output_dir:
                os.makedirs(final_output_dir, exist_ok=True)
                for item in os.listdir(output_dir):
                    src = os.path.join(output_dir, item)
                    dst = os.path.join(final_output_dir, item)
                    if os.path.isdir(src):
                        self._retry_copy(src, dst, is_dir=True)
                    else:
                        self._retry_copy(src, dst, is_dir=False)
                shutil.rmtree(output_dir, ignore_errors=True)

        torch.distributed.barrier()

    def log(self, logs: dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.log_seen_tokens:
            num_tokens_tensor = torch.tensor(
                self.state.num_input_tokens_seen, dtype=torch.float32, device=self.args.device
            )
            torch.distributed.all_reduce(
                num_tokens_tensor, op=torch.distributed.ReduceOp.AVG, group=self.args.dcp_group
            )
            self.state.num_input_tokens_seen = num_tokens_tensor.item()
            logs["num_tokens_seen"] = self.state.num_input_tokens_seen * self.args.dp_world_size
            logs["throughput"] = self.state.num_input_tokens_seen / self.state.running_time
        if self.args.log_flops:
            flops_tensor = torch.tensor(self.state.total_flos, dtype=torch.float32, device=self.args.device)
            torch.distributed.all_reduce(flops_tensor, op=torch.distributed.ReduceOp.AVG, group=self.args.dcp_group)
            self.state.total_flos = flops_tensor.item()
            logs["tflops"] = self.state.total_flos / self.state.running_time

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _get_distinct_learning_rates(self) -> dict:
        """Distinct LRs across optimizer param groups, keyed by the group name.

        ``_get_learning_rate`` (inherited from HF) returns ``get_last_lr()[0]``, which is
        always the "decay" group. With ``action_head_lr`` set -- 4x the trunk rate in both
        shipped recipes -- the head's rate never appeared in the logs, so a schedule problem
        on the head was invisible. Groups sharing a rate are collapsed to keep the log small.
        """
        try:
            last_lr = self.lr_scheduler.get_last_lr()
        except AssertionError as exc:
            if "need to call step" not in str(exc):
                raise
            return {}
        distinct = {}
        for index, (group, lr) in enumerate(zip(self.optimizer.param_groups, last_lr)):
            value = lr.item() if torch.is_tensor(lr) else float(lr)
            if value in distinct:
                continue
            distinct[value] = group.get("name") or f"group_{index}"
        return distinct

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, epoch, learning_rate=None):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # get average loss over all processes
            torch.distributed.all_reduce(
                tr_loss,
                op=torch.distributed.ReduceOp.AVG,
                group=self.args.dcp_group,
            )
            tr_loss_scalar = tr_loss.item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()
            # learning_rate above is group 0 only; surface the other distinct rates so the
            # action-head schedule is actually visible.
            for rate, name in self._get_distinct_learning_rates().items():
                logs[f"lr_{name}"] = round(rate, 8)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        # No mid-training evaluation: this recipe passes no eval_dataset, so there is no
        # metric to rank checkpoints by and SaveStrategy.BEST has nothing to select on.
        if self.control.should_save:
            self._save_checkpoint()
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def compare_trainer_and_checkpoint_args(self, training_args, trainer_state):
        attributes_map = {
            "logging_steps": "logging_steps",
            "eval_steps": "eval_steps",
            "save_steps": "save_steps",
        }

        has_warning = False
        warning_str = "Warning: The following arguments do not match the ones in the `trainer_state.json` within the checkpoint directory: "
        for arg_attr, state_attr in attributes_map.items():
            arg_value = getattr(training_args, arg_attr, None)
            state_value = getattr(trainer_state, state_attr, None)

            if arg_value is not None and state_value is not None and arg_value != state_value:
                warning_str += f"\n\t{arg_attr}: {arg_value} (from args) != {state_value} (from trainer_state.json)"
                has_warning = True

        # train bs is special as we need to account for multi-GPU
        train_bs_args = training_args.micro_batch_size
        train_bs_state = trainer_state.train_batch_size // max(1, training_args.dp_world_size)

        if train_bs_args != train_bs_state:
            warning_str += (
                f"\n\tmicro_batch_size: {train_bs_args} (from args) != {train_bs_state} (from trainer_state.json)"
            )
            has_warning = True

        if has_warning:
            logger.warning_once(warning_str)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        args = self.args

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        # api.train resolves --resume to a string before calling Trainer. Apply
        # node-local staging to that path too, not only to resume=True callers.
        if isinstance(resume_from_checkpoint, str):
            resume_from_checkpoint = self._staged_checkpoint(resume_from_checkpoint)

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = self.args.micro_batch_size * args.gradient_accumulation_steps * args.dp_world_size

        # The banner below prints this number, but it sits between Num examples and Num Epochs
        # and is easy to scan past -- and nothing says the LR was NOT rescaled, which is the
        # non-obvious part. A warning rather than an error: 16 ranks with an explicit
        # gradient_accumulation_steps=2 is a legitimate way to hold the design point.
        if args.reference_global_batch and total_train_batch_size != args.reference_global_batch:
            logger.warning(
                f"Global batch is {total_train_batch_size}, but this recipe's learning rate was "
                f"calibrated for {args.reference_global_batch}. The learning rate was NOT "
                "rescaled -- if the rank count changed, set gradient_accumulation_steps "
                "explicitly to hold the design point."
            )

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.train_batch_size = self.args.micro_batch_size * args.dp_world_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
            encoder = self.model.get_encoder(modality="image")
            if args.encoder_gradient_checkpointing_interval is not None and encoder is not None and hasattr(encoder, "gradient_checkpointing_interval"):
                encoder.gradient_checkpointing_disable()
                encoder.gradient_checkpointing_interval = args.encoder_gradient_checkpointing_interval

        self.model.train()

        if self.args.pp_world_size > 1:
            module = PipelineModule(self.model)
            mpu = module.mpu()
        else:
            module = self.model
            mpu = None

        if (
            args.deepspeed_config.get("zero_optimization", {}).get("stage") == 2
            and args.gradient_accumulation_steps > 1
        ):
            from ..utils.deepspeed_compat import fix_zero2_gradient_accumulation

            fix_zero2_gradient_accumulation()

        model = deepspeed.DeepSpeedEngine(
            args=args,
            model=module,
            optimizer=self.optimizer,
            mpu=mpu,
            config=args.deepspeed_config,
            config_class=deepspeed.DeepSpeedConfig(args.deepspeed_config, mpu=mpu),
        )

        self.optimizer = model.optimizer
        assert model.lr_scheduler is None

        self.model_wrapped = model
        self.deepspeed_engine = self.model_wrapped

        pipeline_stage = PipelineStage(
            self.model,
            deepspeed_engine=self.deepspeed_engine,
            group=args.pp_group,
        )

        pipeline_schedule = ALL_PIPELINE_SCHEDULES[args.pipeline_parallel_schedule](
            stages=[pipeline_stage],
            deepspeed_engine=self.deepspeed_engine,
        )

        # ckpt loading
        if resume_from_checkpoint is not None:
            self._load_checkpoint(resume_from_checkpoint)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.micro_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None:
            trainer_state_path = os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)

            self.state = TrainerState.load_from_json(trainer_state_path)
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            # load_from_json replaced the state wholesale, which reverted the logging/save/eval
            # intervals that compute_steps had just derived from the CURRENT args. Recompute so
            # an explicit --save-steps / --logging-steps on a resumed run actually takes effect
            # instead of silently inheriting the previous cadence (max_steps is refreshed by
            # init_training_references below).
            self.state.compute_steps(args, max_steps)

            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first"
                f" {steps_trained_in_current_epoch} batches in the first epoch."
            )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, None)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._total_grad_norm_scaler = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # if args.eval_on_start:
        #     self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        start_time = time.time()
        # speed_metrics needs a snapshot taken before the whole run, and start_time is
        # reassigned at the end of every step below to feed state.running_time. Without a
        # separate clock, train_runtime came out as one step's duration and the throughput
        # metrics were inflated by (total runtime / one step).
        train_start_time = start_time
        initial_global_step = self.state.global_step
        initial_num_input_tokens_seen = float(self.state.num_input_tokens_seen)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            epoch_dataloader.batch_sampler.set_epoch(epoch)

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            update_step = -1
            rng_to_sync = False

            # Handle resumption from checkpoint
            if epoch == epochs_trained and resume_from_checkpoint is not None:
                if steps_trained_in_current_epoch > 0:
                    epoch_dataloader.batch_sampler.skip_first_batches(steps_trained_in_current_epoch)
                    step = steps_trained_in_current_epoch - 1
                    update_step = steps_trained_in_current_epoch // args.gradient_accumulation_steps - 1
                    rng_to_sync = True
                else:
                    self._load_rng_state(resume_from_checkpoint)

            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps

            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )

            for _ in range(update_step + 1, total_updates):
                update_step += 1
                if _STEP_PROBE:
                    _p_step0 = time.monotonic()
                    _p_data0 = _probe["data"]
                    _p_fb = _p_opt = _p_ema = 0.0

                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples = LazyBatchLoader(
                    epoch_iterator=epoch_iterator,
                    num_batches=num_batches,
                    training_args=args,
                )
                step += num_batches

                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                if _STEP_PROBE:
                    _p_fb0 = time.monotonic()
                losses = pipeline_schedule.step(batch_samples)
                if _STEP_PROBE:
                    # Includes the lazy _load_one_batch calls it triggers, so the pure
                    # forward+backward figure is this minus the data wait accumulated below.
                    _p_fb = time.monotonic() - _p_fb0
                tr_loss = tr_loss + losses.mean()

                if args.pp_rank == 0:
                    for inputs in batch_samples:
                        if args.log_seen_tokens:
                            main_input_name = getattr(self.model, "main_input_name", "input_ids")
                            if main_input_name not in inputs:
                                logger.warning(
                                    "Tried to track the number of tokens seen, however the current model is "
                                    "not configured properly to know what item is the input. To fix this, add "
                                    "a `main_input_name` attribute to the model class you are using."
                                )
                            else:
                                if "attention_mask" in inputs:
                                    input_tokens = inputs["attention_mask"].sum()
                                elif (
                                    self.processing_class is not None
                                    and hasattr(self.processing_class, "pad_token_id")
                                    and self.processing_class.pad_token_id is not None
                                ):
                                    input_tokens = (inputs[main_input_name] != self.processing_class.pad_token_id).sum()
                                else:
                                    input_tokens = inputs[main_input_name].numel()

                                self.state.num_input_tokens_seen += input_tokens

                        if args.log_flops:
                            self.state.total_flos += (
                                float(self.model.floating_point_ops(inputs)) / 1e12 * 3
                            )

                if self.args.cleanup_before_optimizer_step:
                    del batch_samples
                    gc.collect()
                    torch.cuda.empty_cache()

                self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                if _STEP_PROBE:
                    _p_opt0 = time.monotonic()
                with torch.cuda.nvtx.range("optimizer_step"):
                    if self.args.pp_world_size > 1:
                        self.optimizer.step()
                    else:
                        self.deepspeed_engine.step()
                if _STEP_PROBE:
                    _p_opt = time.monotonic() - _p_opt0
                self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    grad_norm = self.deepspeed_engine.get_global_grad_norm()
                    if grad_norm is None and hasattr(self.optimizer, '_global_grad_norm'):
                        grad_norm = self.optimizer._global_grad_norm
                    # In some cases the grad norm may not return a float
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()
                    self._total_grad_norm_scaler += grad_norm

                # get leaning rate before update
                learning_rate = self._get_learning_rate()

                if not getattr(self.optimizer, "overflow", False):
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()

                    # Maintain EMA of trainable weights (fp32 shadow), built lazily so it
                    # captures the post-resume / post-init parameters at the first step.
                    # Only on the rank that persists it: _save_model returns early for
                    # edp_rank != 0, so building it everywhere allocated a whole extra fp32
                    # model copy per GPU just to discard it, and on resume only edp_rank 0
                    # restores the saved shadow while the rest rebuilt from one snapshot.
                    # ZeRO-1/2 replicate parameters, so the single copy is sufficient.
                    if self.args.use_ema and self.args.edp_rank == 0:
                        if _STEP_PROBE:
                            _p_ema0 = time.monotonic()
                        if self.ema is None:
                            self.ema = EMA(self.model, decay=self.args.ema_decay, device=self.args.ema_device)
                        self.ema.update(self.model, step=self.state.global_step)
                        if _STEP_PROBE:
                            # rank0-only work inside an all-synchronized job: ema.update
                            # allocates a transient fp32 copy of every trainable parameter
                            # (ema.py:35), so on the 2B model that is ~9.4 GB and ~600
                            # allocations per step that no other rank pays. Whatever it costs
                            # is added to every step for all 64 ranks.
                            _p_ema = time.monotonic() - _p_ema0

                self.deepspeed_engine.zero_grad()

                if _STEP_PROBE:
                    _p_step = time.monotonic() - _p_step0
                    if _p_step >= _STEP_PROBE_SLOW:
                        _p_data = _probe["data"] - _p_data0
                        logger.warning(
                            f"[stepprobe] STEP {_p_step:.1f}s = data {_p_data:.1f} "
                            f"+ fwd_bwd {max(0.0, _p_fb - _p_data):.1f} + opt {_p_opt:.1f} "
                            f"+ ema {_p_ema:.1f} + resid {_p_step - _p_fb - _p_opt - _p_ema:.1f} "
                            f"| gc gen2 {_probe['gc_total']:.1f}s/{_probe['gc_count']} "
                            f"| at global_step {self.state.global_step}"
                        )

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.state.running_time += time.time() - start_time
                start_time = time.time()

                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                self._maybe_log_save_evaluate(
                    tr_loss,
                    grad_norm,
                    model,
                    epoch,
                    learning_rate=learning_rate,
                )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, epoch, learning_rate=learning_rate)

            if self.control.should_training_stop:
                break

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        # These accumulators only cover this invocation, including after resume.
        completed_steps = self.state.global_step - initial_global_step
        effective_global_step = max(completed_steps, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            train_start_time,
            num_samples=completed_steps * total_train_batch_size,
            num_steps=completed_steps,
            num_tokens=float(self.state.num_input_tokens_seen) - initial_num_input_tokens_seen,
        )
        metrics["train_loss"] = train_loss
        metrics["grad_norm"] = self._total_grad_norm_scaler / effective_global_step

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # The last periodic checkpoint may still be draining to the shared mount.
        self._finish_pending_stage()

        if args.save_full_model:
            self._save_full_model()

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def set_initial_training_values(
        self, args: TrainingArguments, dataloader: DataLoader, total_train_batch_size: int
    ):
        # Case 1: we rely on `args.max_steps` first
        max_steps = args.max_steps
        # If max_steps is negative, we use the number of epochs to determine the number of total steps later
        epoch_based = max_steps < 0
        len_dataloader = len(dataloader) if has_length(dataloader) else None

        # Case 2: We have a dataloader length and can extrapolate
        if len_dataloader is not None:
            num_update_steps_per_epoch = max(
                len_dataloader // args.gradient_accumulation_steps
                + int(len_dataloader % args.gradient_accumulation_steps > 0),
                1,
            )
            # Case 3: We have a length but are using epochs, we can extrapolate the number of steps
            if epoch_based:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

        # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
        if len_dataloader:
            num_examples = len(dataloader)
            if args.max_steps > 0:
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                    max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = max_steps * total_train_batch_size
            else:
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        )

    def is_local_process_zero(self):
        return self.args.local_rank == 0

    def is_world_process_zero(self):
        return self.args.global_rank == 0
