---
license: apache-2.0
library_name: rynnvla
pipeline_tag: robotics
language:
  - en
tags:
  - vision-language-action
  - vla
  - latent-action
  - latent-action-model
  - robot-manipulation
  - flow-matching
  - mixture-of-transformers
  - embodied-ai
  - pytorch
  - deepspeed
# base_model is omitted on purpose: the hub warns about and drops an id that does not resolve to a
# live repo, and the RynnBrain-2B/4B backbones are not published yet (see the TODO(hf) note at the
# bottom of this file). This distribution also ships no weights, so nothing here is a fine-tune of
# anything. Uncomment once the backbone lands.
# base_model: Alibaba-DAMO-Academy/RynnBrain-2B
---

# RynnVLA-Latent

**Latent Action Pretraining for Robotic Manipulation Foundation Models**

DAMO Academy, Alibaba Group · Hong Kong Embodied AI Lab · CUHK · Hupan Lab

| | |
| --- | --- |
| Homepage | https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io |
| GitHub | https://github.com/alibaba-damo-academy/RynnVLA-Latent |
| HuggingFace | https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-Latent |
| ModelScope | https://www.modelscope.cn/models/DAMO_Academy/RynnVLA-Latent |

---

## Model weights

The trained checkpoints are published **separately from this code repository**, on both
Hugging Face and ModelScope. This repository ships no weights (everything is loaded with
`local_files_only=True`); download the artifact you need from the links below and point the
training / export / eval commands at that local directory.

| Artifact | What it is | Hugging Face | ModelScope |
| --- | --- | --- | --- |
| **RynnLAM** | self-supervised latent-action model (`ktoken_zcam`, 608-dim) — the auto-labeler that produces the Stage-1 latent corpus | [Alibaba-DAMO-Academy/RynnLAM](https://huggingface.co/Alibaba-DAMO-Academy/RynnLAM) | [DAMO_Academy/RynnLAM](https://modelscope.cn/models/DAMO_Academy/RynnLAM) |
| **RynnVLA-Latent-2B** | Stage-1 latent-action pretrained VLA, 2B backbone (bfloat16, EMA @ step 236442) | [Alibaba-DAMO-Academy/RynnVLA-Latent-2B](https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-Latent-2B) | [DAMO_Academy/RynnVLA-Latent-2B](https://modelscope.cn/models/DAMO_Academy/RynnVLA-Latent-2B) |
| **RynnVLA-Latent-4B** | Stage-1 latent-action pretrained VLA, 4B backbone (bfloat16, EMA @ step 236442) | [Alibaba-DAMO-Academy/RynnVLA-Latent-4B](https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-Latent-4B) | [DAMO_Academy/RynnVLA-Latent-4B](https://modelscope.cn/models/DAMO_Academy/RynnVLA-Latent-4B) |

The 2B / 4B checkpoints are **Stage-1 latent-action pretraining** exports: they generate latent
actions and serve as the initialization for Stage-2 embodiment alignment. They are *not*
closed-loop control policies — do **not** point LIBERO / RoboTwin / VLABench evaluation at them
directly; run Stage-2 post-training first (see
[Inference and evaluation](#inference-and-evaluation)). RynnLAM is the motion checkpoint that
*produces* the 608-dim latents Stage 1 consumes (see
[Latent-action model (RynnLAM)](#latent-action-model-rynnlam)).

Not yet published: the **RynnBrain-2B/4B VLM backbones** and the **DA3-Large encoder** required
to *train* RynnLAM. Obtain those separately (see [Optional components](#optional-components)).

---

## Abstract

Vision-language-action (VLA) models are limited by the scarcity and fragmentation of
action-annotated robot data: different embodiments define incompatible action spaces, while
abundant human videos remain unusable because they lack action labels. We present a unified
latent-action framework that turns heterogeneous manipulation video into a shared,
embodiment-agnostic action space.

We first train **RynnLAM**, a Latent Action Model, in a fully self-supervised manner to
compress inter-frame dynamics into compact latent actions. Unlike prior latent actions that
primarily explain 2D appearance changes, RynnLAM grounds its representations in physical
motion through future visual-feature prediction, dynamic 3D flow, and explicit
camera-ego-motion decoupling. Using RynnLAM as a scalable auto-labeler, we densely annotate a
large multi-source corpus spanning real-robot trajectories, simulation rollouts, and human
egocentric videos, aligning all sources under one consistent action semantics.

On this aligned corpus we pretrain **RynnVLA-Latent**, a Mixture-of-Transformers
vision-language-action model that generates latent-action chunks from language instructions
and multi-view observations via flow matching. A subsequent post-training stage retargets the
same flow-matching objective from latent actions to robot-specific action spaces, requiring
action-labeled demonstrations only for final embodiment alignment. This *label-once,
align-everything* recipe enables joint pretraining on robot and human data under a unified
action representation. Experiments show that RynnVLA-Latent obtains consistent scaling gains
with pretraining data volume and achieves state-of-the-art results on RoboTwin 2.0, LIBERO and
VLABench.

Those benchmark numbers come from the paper, not from anything runnable here. This distribution
ships the training stack, the LIBERO closed-loop evaluator, and the **policy side** of the
RoboTwin and VLABench interfaces; the RoboTwin and VLABench simulators, their runners and their
corpora are not redistributed, so no command in this repository reproduces a RoboTwin or VLABench
score on its own. See [RoboTwin and VLABench tracks](#robotwin-and-vlabench-tracks) for exactly
what ships and what you have to bring.

---

## Method

### RynnLAM: latent actions grounded in physical motion

RynnLAM encodes a pair of RGB frames `(t, t+gap)` into a compact latent action. It is trained
fully self-supervised — no action labels, no robot state, no simulator. What separates it from
latent-action models that only have to explain 2D appearance change is that three separate
objectives force the latent to describe *physical motion*:

| Grounding signal | What it constrains | Where it appears in training |
| --- | --- | --- |
| Future visual-feature prediction | the latent must anticipate how features evolve, not just how pixels differ | `feat_loss` (`use_contrastive_feat_loss`) |
| Dynamic 3D flow | the latent must decode into metric 3D scene motion | `flow_loss` (`FlowDecoderV5`) |
| Camera-ego-motion decoupling | agent motion must be separable from camera motion | `pose_loss` + `adversarial_loss` (gradient-reversal on `cam_dec_adversarial`) |

A K-token reconstruction branch (`recon_loss`) additionally compresses the target-frame
features into `K` motion tokens, which is what makes the delivered latent scalable to a large
corpus. All five terms are active in the shipped recipes; you can see them reported per step
by `scripts/train_rynnlam.py`.

The delivered representation is `ktoken_zcam`, **608 floats per frame pair**:

```
[ k_tokens 512 | latent_z 64 | camera_pose 32 ]
```

The full byte layout, stride cadence, on-disk schema and normalization rules are normative in
[Latent-action protocol](#latent-action-protocol-608-dim-ktoken_zcam) below.

### Label once, align everything

Because RynnLAM only needs RGB pairs, it acts as a scalable auto-labeler over sources that
share no action space: real-robot trajectories, simulation rollouts, and human egocentric
video. Every source is annotated into the *same* 608-dim semantics, which is what makes joint
pretraining on robot and human data possible. Latents are labeled with `gap = 4` and
`pair_stride = 4` — one latent action per 4 RGB frames — and each episode is published
atomically with a protocol fingerprint, so a re-run with a different stride, checkpoint or
schema is refused rather than silently overwritten.

### RynnVLA-Latent: flow matching over latent-action chunks

RynnVLA-Latent is a Mixture-of-Transformers VLA. Given a language instruction and multi-view
observations, it generates a **chunk of latent actions** by flow matching. Stage 1 therefore
needs no robot actions at all: the flow-matching target is the latent action, so human video
contributes gradient exactly like robot video does.

Post-training keeps the *same* flow-matching objective and retargets it from latent actions to
a robot-specific action space. Action-labeled demonstrations are only needed for this final
embodiment alignment — which is the whole point of the recipe: expensive action annotation is
confined to the last stage, while pretraining scales on unlabeled video.

Concretely in this repository:

- **Stage 1** — `LatentPretrainDataset` + `rynnvla/configs/stage1_*.json`. Latent prediction
  on, latent-head readout off, robot actions unused.
- **Stage 2** — `LiberoPlusDataset` + `rynnvla/configs/stage2_v5_*.json`. Latent prediction
  off, `latent_readout_proj` new, initialized from the *complete* exported Stage-1 EMA model.
  Robot action chunk 30 → 10; this does not change any shared tensor shape.

Up to 6 camera roles are carried per sample (`head`, `left_wrist`, `right_wrist`,
`front_third`, `side_left`, `side_right`), with `slot_mask` marking the valid ones, so a source
with one camera and a source with four train in the same batch.

---

## Repository layout

```
RynnVLA-Latent/
├── rynnvla/          # VLA package: api/, models/, datasets/, training/, utils/, configs/
├── rynnlam/          # RynnLAM package: model, modules/ (encoder + flow decoder), inference, configs/
├── scripts/          # launchers, smoke.sh, sample generators, latent manifest/index/stats builders
├── configs/          # *.example.json data-mixture templates
├── data/             # bundled synthetic sample corpora (data/sample, data/sample_lam)
├── tests/            # CPU-only suite: pytest tests/ (tests/rynnlam/ covers the RynnLAM side)
├── requirements.txt  # pinned reference environment
├── NOTICE            # third-party attribution
└── README.md         # this file — all documentation lives here
```

`import rynnvla` and `import rynnlam` both resolve after an editable install.

The two sample corpora under `data/` are the only data that ships; everything else there is
downloaded or generated. See [Quickstart: the bundled sample](#quickstart-the-bundled-sample).

---

## Installation

Linux, Python ≥ 3.10, CUDA GPU. Create a dedicated environment; do not replace packages in an
existing training environment.

### Reference environment (recommended)

```bash
conda create -y -n rynnvla python=3.10 && conda activate rynnvla
pip install -r requirements.txt
```

`requirements.txt` installs a CUDA build of PyTorch, this repository with every optional
extra, and a prebuilt flash-attn wheel. The reference combination, verified end to end on
linux-x86_64 with one GPU:

```
python 3.10   torch 2.7.1+cu126   torchvision 0.22.1+cu126
transformers 5.2.0   deepspeed 0.18.2   accelerate 1.11.0   flash-attn 2.8.3.post1
numpy 2.2.6   opencv-python-headless 5.0.0   safetensors 0.8.0
```

The flash-attn wheel is ABI-locked to python 3.10 + torch 2.7 + cu12 + cxx11abi TRUE. On a
different combination, change the index URL, the `+cuXXX` local versions and the flash-attn
asset together (see the comments in `requirements.txt`).

### Manual install

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .                              # or -e '.[preprocess]'
python -m pip install flash-attn --no-build-isolation
```

**flash-attn is a hard runtime requirement, not an optimization.**
`rynnvla/utils/context_parallel.py` imports it at module load and
`rynnvla/training/data_collator.py` imports that module, so no training run can even start
without it. PyPI ships only an sdist and a source build needs `nvcc` plus roughly an hour,
which is why `requirements.txt` points at a prebuilt wheel instead.

Optional extras: `[preprocess]` (zarr/numcodecs), `[test]` (pytest, plus zarr≥3 where the
Python version allows it).

Everything runs offline against local paths: no object store, no hub download and no
experiment-tracking service is contacted at runtime. RynnLAM training metrics are appended
to `<output_dir>/<exp_name>/metrics.jsonl` when `track: true` is set in the config.

### Verify the install

A successful package installation alone does not establish binary compatibility; check real
imports before training:

```bash
python -c 'import torch, transformers, deepspeed, flash_attn; print(torch.__version__, transformers.__version__, deepspeed.__version__, flash_attn.__version__); print(torch.cuda.is_available())'
python -c 'import rynnvla; print(rynnvla.__file__)'
python -c 'import rynnlam; print(rynnlam.__file__)'
```

`transformers` must be 5.x: `rynnvla/arguments.py` imports `PreTrainedConfig`, which does not
exist in the 4.x line. The reported `rynnvla` / `rynnlam` paths must refer to *this*
installation, not an older checkout with the same package name — do not put another checkout
on `PYTHONPATH`.

### Tests

```bash
pip install -e '.[test]'
pytest                      # testpaths is already set to tests/
```

The suite is CPU-only and needs no GPU, no backbone weights, no simulator and no network: model
fixtures are tiny synthetic configs, and data fixtures are built under `tmp_path`. It runs in
under a minute — 392 passed, 13 skipped on the reference environment above.

What it covers: the sampler's locality permutation and the weighted virtual index that has to
stay compatible with it; the Stage-1 index/stats/manifest builders; the export contract and
Stage-1 → Stage-2 transfer invariants; camera-role wiring across train and eval; the LIBERO,
RoboTwin and VLABench adapters; the optimizer's parameter groups and the resume guards; and on
the RynnLAM side the config, model, trainer, dataset, labeling primitives and evaluation CLI.

What it does not cover: learned checkpoint quality, GPU kernels, rendering, multi-node
rendezvous, or any benchmark score. A green suite says the wiring is intact, not that a model
works — that is `scripts/smoke.sh` plus a real rollout.

Three tests skip unless you supply something, and each says so in its skip reason:

| Skip | Set this to run it |
| --- | --- |
| production-scale sampler periodicity | `RYNNVLA_LATENT_INDEX` → an index npz from your own corpus |
| VLABench wire interop with the real client | `VLABENCH_HARNESS_SCRIPTS` → a directory containing `policy_client.py` |
| RynnLAM historical source parity | `RYNNLAM_BASELINE_ROOT` → an upstream RynnLAM checkout |
| EgoVerse Zarr-v3 store decode | `zarr>=3`, which needs Python ≥ 3.11 (see `[test]` in pyproject) |

### Training resources

The full recipe uses a 4B vision-language backbone plus its action expert. ZeRO-1/2 shards
optimizer state across data-parallel GPUs; a single GPU does not receive this memory saving.
Gradient checkpointing and a microbatch of one are useful for smoke tests but do not remove
the optimizer or fp32 EMA costs.

For single-GPU validation, `--ema-device cpu` stores the EMA shadow in host memory while
preserving FP32 arithmetic and the trained parameter set. A DeepSpeed configuration may
instead offload the optimizer to CPU (RynnVLA uses `DeepSpeedCPUAdam` for that setting);
offload needs substantial host memory, a C++ compiler and a writable PyTorch extensions cache,
and can exceed a 100 GB container limit. Neither strategy guarantees a particular model fits.
Do not freeze the vision backbone to present an incomplete test as full-model training.

### Optional components

- **LIBERO evaluation** additionally needs LIBERO, its simulator dependencies, task assets and
  a working offscreen renderer. See [Inference and evaluation](#inference-and-evaluation).
- **Weights and data are separate resources.** Two distinct checkpoints are involved and they
  are *not* interchangeable:
  - the **DA3-Large encoder** (`checkpoints/DA3-LARGE-1.1/model.safetensors`) is required to
    *train* RynnLAM. It must be a DA3-Large build with QK-norm enabled on every block:
    `rynnlam/modules/lam.py` loads it with `strict=True` and expects 407 tensors including
    per-block `attn.q_norm` / `attn.k_norm`. A DA3 release built with QK-norm disabled
    (`qknorm_start: -1`) is missing those 65 tensors and the load fails.
  - a trained **RynnLAM motion checkpoint** is what *produces* the 608-dim latents Stage 1
    consumes (via `rynnlam.inference.RynnLAMEncoder`). It is published on Hugging Face and
    ModelScope — see [Model weights](#model-weights).
- HDF5 RGB decoding uses h5py; RoVid-X tar decoding uses PyAV plus explicit tar/index roots.

No private tracker, internal package mirror, CUTLASS submodule or DeepEP build is required for
the released dense RynnVLA recipe.

---

## Quickstart: the bundled sample

Two tiny **synthetic** corpora ship in `data/` so a fresh clone trains immediately, with no
download, no credentials and no external weights:

| Path | Format | Consumed by | Contents |
| --- | --- | --- | --- |
| `data/sample/` | Stage-1 latent npz + mp4 + index + stats | `rynnvla.api.train` | 3 episodes, 24 latents each, 608-dim |
| `data/sample_lam/` | preprocessed per-scene safetensors | `scripts/train_rynnlam.py` | 3 scenes, 12 frames, 56×56 |

Regenerate either with `python scripts/make_sample_data.py` /
`python scripts/make_sample_lam_data.py`. Every path inside them is **repo-root-relative**, so
they keep working wherever you check the repo out. The frames and latents are synthetic: the
losses these produce are meaningless, and a green run proves the *pipeline* works, not that a
model learns.

### One command

```bash
bash scripts/smoke.sh                        # dataset check + RynnLAM Stage-1, no external weights
bash scripts/smoke.sh --src /path/to/RynnBrain-2B   # also runs VLA Stage-1
bash scripts/smoke.sh --only data            # CPU-only dataset read, no GPU
```

Run it from the repository root. Outputs land in `runs/` (gitignored).

The `vla` stage needs a local Qwen3-VL-family checkpoint (RynnBrain-2B or
Qwen3-VL-2B-Instruct) purely for its **tokenizer / chat-template / processor assets** — those
are not redistributed here. `scripts/make_smoke_tiny_model.py` copies them and materializes a
44M random-weight backbone under `data/smoke_tiny_model/`, so the smoke exercises the real
special-token, image-token and DeepSpeed ZeRO-1 path without the multi-GB download. Without
`--src` that stage is skipped, not failed.

### The same steps by hand

```bash
# 1. CPU-only: read data/sample through the real Stage-1 dataset (no GPU, no backbone)
python scripts/verify_sample_data.py --mixture data/sample/sample_mixture.json

# 2. RynnLAM Stage-1: 2 optimizer steps, all five losses, checkpoint + internal eval
python scripts/make_sample_lam_data.py                     # if data/sample_lam is absent
python scripts/train_rynnlam.py --config rynnlam/configs/smoke.yaml

# 3. VLA Stage-1: 2 optimizer steps against the bundled latent corpus
python scripts/make_smoke_tiny_model.py --src /path/to/RynnBrain-2B
torchrun --standalone --nproc_per_node=1 -m rynnvla.api.train \
    --config rynnvla/configs/stage1_smoke_tiny.json \
    --model_path data/smoke_tiny_model \
    --data_mixture data/sample/sample_mixture.json \
    --output_dir runs/smoke_vla
```

`rynnlam/configs/smoke.yaml` and `rynnvla/configs/stage1_smoke_tiny.json` differ from the real
recipes only mechanically (small strides, `batch_size 1`, `max_steps 2`, every loss warmup at 0
so all branches fire). The model *architecture* is unchanged: the RynnLAM smoke still builds the
full 608-dim `ktoken_zcam` encoder and 397M-parameter model.

### What the smoke does not cover

`scripts/evaluate_rynnlam.py regress` (LARYBench checkout), `rynnvla.api.eval_libero`
(LIBERO + simulator) and `scripts/export_checkpoint.py` / `rynnvla.api.predict` (which validate
against the real 2B/4B Stage-2 architecture and correctly refuse the tiny fixture) all need
resources that cannot be bundled. See [Inference and evaluation](#inference-and-evaluation).

**Stage-2 post-training has no offline smoke, and that is a real gap rather than an omission to
paper over.** The smoke therefore proves capabilities 1–3 (RynnLAM training → latent labeling →
Stage-1 latent-action pretraining) end to end, and stops there. Two independent reasons it cannot
currently go further:

- The bundled `data/sample/` corpus carries latent actions, not robot actions, so there is no
  offline Stage-2 data source. `LiberoPlusDataset` needs a real LeRobot v2.1 suite and
  `RoboTwinDataset` needs a real HDF5 corpus.
- The Stage-1 → Stage-2 handoff goes through `export_checkpoint`, which enforces the formal
  recipe's architecture (expert hidden 768 / intermediate 2752). The smoke's 44M random-weight
  backbone is 128-wide, so the exporter refuses it — correctly. There is no tiny fixture that is
  both cheap enough to bundle and wide enough to export.

What covers Stage-2 instead is the unit suite, which runs on tiny synthetic models and needs no
data: `tests/test_stage2_transfer.py` (weight transfer under
`bypass_latent_output_projection`, config surviving a `from_pretrained` roundtrip, RNG
restoration on module reset, one-LR-group-per-parameter, the resume lock on the LR recipe,
direct-vs-cached forward agreement and backward reach), `tests/test_robotwin_stage2.py`
(RoboTwin action semantics, `transfer_lr` isolation, resume rejection, index/schema pinning),
`tests/test_checkpoint_transfer.py` and `tests/test_export.py`. Those verify the wiring; they do
**not** verify that a Stage-2 run trains. A green suite plus a green smoke is still not a green
post-training run — that needs a real backbone and a real robot corpus.

---

## Conventions

- **No hard-coded machine paths.** Data/model roots come from the environment or explicit CLI
  arguments; required-but-unset variables fail loudly rather than defaulting to a private path.
- **No credentials anywhere.** Nothing in this repository talks to a remote service, so there
  are no keys, tokens or endpoints to configure. `*.local.json` and `config.json` are
  git-ignored.
- **Keep caches and artifacts outside the source tree.** `RYNNVLA_CACHE_DIR` controls dataset
  schema caches; `LATENT_PRETRAIN_PATH_MAP` remaps video-path prefixes embedded in existing
  latent indices without modifying the index or the video files.
- **Relative paths resolve against the working directory**, not against the config file. Run
  from the repository root. The single exception is a recipe JSON's own `deepspeed` path,
  which resolves relative to that JSON.

---

## Data: latent index, stats and mixtures

Model weights, datasets and offline RynnLAM latents are **not** bundled. This section covers
the formats the training entry points consume and the three builders that produce them.

### Stage 1: index + stats + mixture

Stage-1 latent-action pretraining (`rynnvla.api.train`, `LatentPretrainDataset`) reads three
artifacts, assembled from per-episode latent `.npz` files. The whole chain runs on a single
node from your own local video. Run every command from the repository root: relative paths
resolve against the working directory (see [Conventions](#conventions)).

```bash
# 0) local video -> 608-dim ktoken_zcam latent .npz (needs a trained RynnLAM checkpoint;
#    see "Latent-action model (RynnLAM)" below). Writes the npz schema + meta.protocol of
#    "Latent-action protocol" (schema_version 2, gap 4, pair_stride 4).
python scripts/label_latent.py --checkpoint runs/lam/stage1/best.pt \
    --video demo.mp4 --output-dir data/labeled          # or --metadata episodes.json
#    -> data/labeled/latents/<dataset>/<episode_id>/<view>/latent.npz
#    The bundled sample is itself a valid --metadata input:
#      python scripts/label_latent.py --checkpoint <ckpt> \
#          --metadata data/sample/source/SampleToy.json --output-dir data/labeled

# 1) latent .npz parts -> a manifest (jsonl). --data-root is the directory holding the
#    <Dataset>.json episode files; --latent-dir is absolutized into the manifest, so a
#    manifest built on one host is host-specific (see "Path remapping" below).
python scripts/rebuild_latent_manifest.py --data-root data/labeled/source \
    --latent-dir data/labeled/latents --out-dir data/labeled/build \
    --name mycorpus --combine

# 2) manifest -> the Stage-1 training index (npz, version 4)
python scripts/build_latent_pretrain_index.py \
    --manifest data/labeled/build/manifests/mycorpus.jsonl \
    --out data/labeled/build/index_c6.npz \
    --latent-chunk 6 --latent-step-seconds 0.25 --default-fps 30

# 3) manifest -> normalization stats (json); latents are stored raw and normalized at load
python scripts/build_latent_stats.py \
    --manifest data/labeled/build/manifests/mycorpus.jsonl \
    --out data/labeled/build/latent_stats.json --latent-dim 608
```

Each script prints the exact next command with your own paths filled in, so the chain can be
followed without referring back here. `data/sample/build/` is a worked example of all three
outputs for the bundled corpus.

`build_latent_pretrain_index.py` delegates to
`rynnvla.datasets.vla_datasets.latent_pretrain.build_index`, reads the whole supplied
manifest, rejects an existing output file, nonpositive sampling parameters and output under
the manifest directory, and creates its temporary ZIP beside the requested output.

A Stage-1 **data mixture** (see `configs/data_latent_pretrain.example.json`) points at those
artifacts. This is the bundled sample's own mixture, `data/sample/sample_mixture.json`:

```json
[
  {
    "data_type": "LatentPretrainDataset",
    "data_path": "data/sample/build/index_c6.npz",
    "latent_stats_path": "data/sample/build/latent_stats.json",
    "latent_chunk": 6,
    "latent_step_seconds": 0.25,
    "dataset_weights": {"SampleToy": 1.0}
  }
]
```

The pretraining index must use the same role axis as the library (`head`, `left_wrist`,
`right_wrist`, `front_third`, `side_left`, `side_right`, IDs 0–5), and its video and latent
paths must be accessible on **all** workers. `latent_step_seconds=0.25` determines the
per-episode sampling stride from FPS; it is distinct from the model's `latent_action_stride=5`.

### Balancing the mixture: `dataset_weights` and alpha

Sampling uniformly over the merged index gives each dataset a probability proportional to its
window count, so a corpus dominated by one large source spends most of its steps there.
`dataset_weights` in the mixture entry overrides that with an explicit per-dataset weight.
`scripts/build_dataset_weights.py` derives them by temperature:

```bash
python scripts/build_dataset_weights.py \
    --mixture configs/data_latent_pretrain.local.json \
    --out     configs/data_latent_pretrain_a07.json \
    --alpha   0.7
```

| alpha | effect |
| --- | --- |
| `1.0` | natural proportions exactly — **no** `dataset_weights` is written |
| `< 1.0` | flattens: small sources up, large sources down |
| `> 1.0` | sharpens |

Weights are `windows ** alpha`, where `windows` is the per-dataset count of chunk starts read
straight out of the index with the same formula `LatentPretrainDataset` uses to enumerate
samples (`tests/test_build_dataset_weights.py` pins the two together). The script prints a
`nat% -> eff%` table and flags any source upsampled more than 8×, which is where overfitting
shows up first.

Three properties worth knowing before you use it:

- **`len(dataset)` does not move.** The per-dataset targets are a largest-remainder rounding of
  `w / sum(w) * total`, which sums back to `total`. So `max_steps` derived from the length is
  the same as the unweighted run, and an alpha arm is comparable to its baseline at equal steps
  and equal wall clock. `max_steps` is also resume-critical, so a weighting that moved the total
  would reject every existing checkpoint of that arm. The one exception is the third bullet
  below: a source whose exact target rounds to 0 is bumped to one slot, so the total grows by
  one per such source.
- **Coverage does move.** A downsampled source no longer visits every window in one epoch; an
  upsampled one revisits. That is inherent to weighting at a fixed total, not a defect.
- **A weight of 0 does not exclude a dataset.** It still buys one slot. Weighting is a
  rebalancing knob, not a filter — drop the entry from the mixture to remove a corpus.

The output must be a new file: `data_mixture` is resume-critical, so the script refuses to
overwrite an existing mixture that a queued or running job may already have read. There is
deliberately no `--force`. It also rejects an index built for a different `latent_chunk` /
`latent_step_seconds` than the mixture asks for, because those weights would describe a sample
space training never walks.

What this does not establish: that a given alpha is *better*. It only makes the realised
mixture ratio the requested one.

### Path remapping

A real index embeds the **absolute** video/latent paths of the host that built it. To read it
on a different mount without rebuilding, set `LATENT_PRETRAIN_PATH_MAP=src=dst` (repeatable,
longest-prefix-first, matched on a path-component boundary). This remaps prefixes at load time
only; it never rewrites the index.

### Stage 2: LIBERO mixture

Stage-2 robot fine-tuning reads `LiberoPlusDataset` entries, one per suite (see
`configs/data_libero_joint40.example.json`):

```json
[
  {"data_type": "LiberoPlusDataset", "data_path": "data/libero/libero_spatial_no_noops_lerobot_v21"},
  {"data_type": "LiberoPlusDataset", "data_path": "data/libero/libero_object_no_noops_lerobot_v21"},
  {"data_type": "LiberoPlusDataset", "data_path": "data/libero/libero_goal_no_noops_lerobot_v21"},
  {"data_type": "LiberoPlusDataset", "data_path": "data/libero/libero_10_no_noops_lerobot_v21"}
]
```

`LiberoPlusDataset` derives its schema from the suite metadata (`get_schema()`); it does not
require precomputed stats. Point `RYNNVLA_CACHE_DIR` at scratch space for the schema cache.

### Stage 2: RoboTwin and VLABench mixtures

Two further Stage-2 sources ship, each with a template in `configs/`:

```json
[
  {
    "data_type": "RoboTwinDataset",
    "data_path": "data/robotwin/robotwin_raw_hdf5",
    "index_cache": "data/robotwin/robotwin_index.json",
    "index_sha256": "<sha256 printed by build_robotwin_index.py>",
    "variants": ["aloha-agilex_clean_50", "aloha-agilex_randomized_500"],
    "action_space": "ee"
  }
]
```

```json
[
  {"data_type": "VLABenchDataset", "data_path": "data/vlabench",
   "task_category": "primitive", "video_backend": "pyav", "strict": true}
]
```

`RoboTwinDataset` reads a **prebuilt** JSON index rather than walking the corpus, so every rank
does not repeat the same walk over shared storage. Build it with
`scripts/build_robotwin_index.py` (see
[RoboTwin and VLABench tracks](#robotwin-and-vlabench-tracks)); `index_sha256` is optional but
worth pinning, since it makes a resumed run refuse an index that was rebuilt underneath it.
`task_category` is `primitive`, `composite` or `all`; `strict: true` drops an episode whose
cameras or action tables are incomplete instead of training on a partially-decoded sample.

Both derive their schema from the data, so neither needs precomputed stats. Note that the
RoboTwin recipes train at chunk 30 with `mean_std` normalization while the LIBERO V5 recipe
trains at chunk 10 with `min_max_sym`; an export from one is not loadable by the other's
evaluation adapter.

### Validating the data path

`scripts/verify_sample_data.py` exercises the CPU-side data path with no GPU, no backbone and
no training dependencies: it builds a dataset through the public
`rynnvla.datasets.build_dataset` entry point, reads samples, and asserts the expected cameras
come back with finite 608-dim latent targets. Point `--mixture` at your own mixture to validate
your corpus the same way:

```bash
python scripts/verify_sample_data.py --mixture data/sample/sample_mixture.json
```

Import `rynnvla` from **this** checkout. GPU model/processor forwards, optimizer steps and
resume are separate validations that a CPU loader read does not establish — `scripts/smoke.sh`
covers those.

---

## Latent-action protocol (608-dim `ktoken_zcam`)

This is the normative description of the latent-action corpus format. It exists so the producer
(the `rynnlam` encoder), the consumers (`rynnvla.datasets.vla_datasets.*`) and any third-party
re-labeling job agree on one format. **Where code and this document disagree, the code wins and
this document is a bug.** The authoritative byte layout is
[`rynnlam/inference.py`](rynnlam/inference.py) (`RynnLAMEncoder.__call__`).

Stage 1 does not consume robot actions; it consumes a self-supervised latent action computed by
the RynnLAM encoder over pairs of RGB frames.

### 1. The 608-dimensional latent

The deliverable representation is `ktoken_zcam`, formed by a single concatenation in
`rynnlam/inference.py`:

```python
tokens = torch.cat([k_tokens.flatten(1), z, camera], dim=-1)
```

For the b512 checkpoint this is 608 floats per frame pair:

| Slice | Dim | Field | Meaning |
| --- | --- | --- | --- |
| `[0:512]` | 512 | `k_tokens` | K=8 motion tokens × d=64, flattened. Produced by `hint_compressor`; the compression input is `k_token_source` — `"hints"` (per-patch `motion_hints`) or `"features"` (target-frame features). **The shipped recipes use `k_token_source: features`** (`rynnlam/config.py`, `rynnlam/configs/stage1.yaml`). |
| `[512:576]` | 64 | `z` (`latent_action`) | the global latent action; **the only slice the flow decoder consumes** |
| `[576:608]` | 32 | `camera_pose_latent` | relative camera ego-motion between the pair (auxiliary) |

Do not hard-code 512/64/32. Derive the offsets from the checkpoint:
`k = total - model.latent_dim - model.camera_pose_latent_dim`, then slice from the end
(`z = vec[-(z_dim+cam_dim):-cam_dim]`, `cam = vec[-cam_dim:]`) so a different K still decodes.

> **Only `z` drives 3D flow — 64 of 608 dims (10.5%).** `FlowDecoderV5.forward(patches,
> latent_z, motion_hints, H, W)` conditions on `latent_z`; `patches` and `motion_hints` come
> from the frozen encoder on the *real* frames. `k_tokens` and `camera_pose` never enter the
> flow decode, so a flow visualization hides 544 of the 608 dims — and in particular hides
> `camera_pose`, the weakest-predicted slice. Read k-token and camera-pose quality from their
> own heads/metrics, not from flow. Treat flow agreement as evidence about `z` only.

### 2. Frame indexing and cadence (stride-4)

The latent corpus is labeled per episode with:

- `gap = 4` — RGB frame interval *within* a pair: row *r* encodes `(t, t+gap)`.
- `pair_stride = 4` — start-frame advance *between* rows: row *r* starts at
  `start_frame + r*pair_stride`, i.e. pairs `(0,4), (4,8), (8,12), …`.

So there is exactly **one latent action per 4 RGB frames**, matching the world model's Wan-VAE
4× temporal cadence (4 RGB frames → 1 video latent → 1 action).

`pair_indices[row] = (t, t+gap)` is stored in **stream-relative** frames. The absolute source
frame is `view_start_frame + pair_indices[row]` (see `latent_pretrain.py::_read_view_frame`);
each view carries its own labeling-window offset, so the per-view `view_start_frame` — not a
single per-episode scalar — is what converts a latent index back to a frame.

Flow over a longer horizon (8/12 frames) is obtained by chaining per-row flows with a
warp-and-add composition; it is **not** stored. Labeling is resume-safe: each episode is
published atomically with a protocol fingerprint, and a stride/checkpoint/protocol mismatch on
re-run is refused rather than silently overwritten.

### 3. On-disk npz schema

Each labeled `(episode, view)` is one uncompressed `.npz` (zlib saves nothing on dense
float16), written in this order:

| Key | Dtype / shape | Notes |
| --- | --- | --- |
| `latent_action` | **float16** `[N, 608]` | raw encoder output, one row per pair; **not** normalized on disk (see §5) |
| `pair_indices` | **int32** `[N, 2]` | the `(t, t+gap)` stream-relative frame indices per row |
| `meta` | JSON string | saved via `json.dumps`, which `np.savez` stores as a 0-d array; read back with `json.loads(str(npz["meta"].item()))` |

`meta` decodes to (this is the full key set `scripts/label_latent.py` writes — the bundled
`data/sample/` npz files carry exactly these keys, so they can be diffed against real output):

```json
{
  "protocol": {
    "schema_version": 2,
    "checkpoint_sha256": "<sha256 of the RynnLAM checkpoint>",
    "implementation_sha256": "<sha256 of the rynnlam encoder implementation>",
    "labeler_sha256": "<sha256 of scripts/label_latent.py itself>",
    "source": "<video path the labeler opened>",
    "source_size": 27386,
    "source_mtime_ns": 1790084433000000000,
    "gap": 4, "pair_stride": 4, "start_frame": 0, "end_frame": null,
    "bucket": "auto",
    "normalization": "imagenet",
    "representation": "ktoken_zcam",
    "precision": "bf16",
    "device": "cuda",
    "device_name": "<torch.cuda.get_device_name(), or \"cpu\">"
  },
  "view": "head", "num_latents": 24, "num_frames": 100,
  "unpaired_tail_frames": 3, "code_shape": [608],
  "bucket": "square", "target_hw": [280, 280], "source_hw": [64, 64], "fps": 30.0
}
```

Two `bucket` fields, deliberately different: **`protocol.bucket` is the bucket that was
*requested*** (the `--bucket` flag, `"auto"` by default) while **the top-level `bucket` is the
one that *resolved*** for this video's aspect ratio, with `target_hw` = `BUCKETS[bucket]` from
`rynnlam/video.py` (`square` 280×280, `4x3` 238×322, `16x9` 210×364) and `source_hw` the
native frame size. Read the resolved pair, not the requested one, if you need to know the
geometry the bytes were produced at.

`unpaired_tail_frames = num_frames - ((num_latents - 1) * pair_stride + gap + 1)`: the trailing
frames too close to the end to form another pair.

The `protocol` block is the fingerprint. The three `*_sha256` fields pin *what* produced the
bytes — the checkpoint, the encoder implementation and the labeler script — so a corpus and a
decoder with mismatched pins are detectably incomparable rather than silently reinterpreted.
`source_size` / `source_mtime_ns` pin the input video. A loader must refuse a corpus whose
`schema_version`, `gap`, `pair_stride` or `representation` differ from what training expects.
Bump `schema_version` whenever the geometry or meaning of the bytes changes.

`device` / `device_name` are provenance, not a compatibility gate — deliberately absent from the
refuse list above. bf16 matmul is hardware-dependent, so the same checkpoint at the same
`precision` yields different bits on different accelerators; recording the silicon explains that
divergence instead of leaving it to be rediscovered as a mystery diff. Adding them changes
neither the geometry nor the meaning of the bytes, which is why `schema_version` stays at 2 and
why a corpus labeled on mixed hardware remains one corpus. The logical `device` alone
(`"cuda:0"`) does not identify the accelerator, hence `device_name` beside it.

In `data/sample/` all three hashes read `"synthetic-sample"`: no encoder ran, so there is
nothing to fingerprint. `device` / `device_name` read `"cpu"` there for the same reason — those
arrays come from numpy, not a forward pass. Everything else in those files is real, including
`source_size` and `source_mtime_ns` for the bundled mp4.

### 4. Views and roles

View roles are fixed in [`rynnvla/constants.py`](rynnvla/constants.py):

```
0 head   1 left_wrist   2 right_wrist   3 front_third   4 side_left   5 side_right
```

`NUM_VIEW_SLOTS = 6`; a sample carries up to 6 slots, `slot_mask` marks the valid ones, and
**slot id == role id**. For HDF5/Zarr containers that hold several cameras behind one path, the
labeler's source view is recovered through the `VIEW_NAME_TO_ROLE` / `_VIEW_ROLE` tables (e.g.
EgoVerse stores its single camera as `images.front_1` while the manifest view is `head`).
Reordering those tables silently pairs latents with the wrong camera — treat them as part of
the protocol, not as display metadata.

### 5. Normalization

The npz stores **raw** float16 latents. Stage 1 normalizes at *load* time, per dimension, as
`(x - mean) / std` using `latent_stats.json`, which `latent_loader.load_latent_stats` reads
(expecting `mean`/`std` arrays whose length equals the recipe's `latent_action_dim`). The
Stage-1 producer is **`scripts/build_latent_stats.py`**. The same statistics file must be used
at training and at inference.

Because the per-dimension std spans roughly 67× across the 608 dims, **raw-space cosine
similarity is inflated and must not be used as a quality metric**. Evaluate in normalized space
(MSE / R²) or in flow space instead.

### 6. EgoVerse (Zarr v3) decode convention

EgoVerse episodes are Zarr v3 stores: `sharding_indexed` containers with inner chunk shape
`[1]`, `vlen-bytes` + `zstd(level 0)` codecs and a `crc32c` checksum appended to the shard
index; each element carries an 8-byte `[u32 count][u32 nbytes]` header before the encoded
image. Frame numbering follows the labeler's `range(array.shape[0])` with **no clamp** to the
`total_frames` attribute. The two RynnLAM decode paths historically disagreed here; the
**labeling path is authoritative** because it produced `num_frames`.

### 7. Producers and consumers

- **Producer:** the `rynnlam` encoder (`RynnLAMEncoder`, `rynnlam/inference.py`), driven over
  local video by `scripts/label_latent.py` (writes the npz schema and `meta.protocol` below).
  The encoder checkpoint itself is trained by `scripts/train_rynnlam.py`
  (`rynnlam/configs/*.yaml`); `checkpoint_sha256` in `meta.protocol` pins exactly that
  checkpoint.
- **Manifests / index:** `scripts/rebuild_latent_manifest.py`,
  `scripts/build_latent_pretrain_index.py`.
- **Stats:** `scripts/build_latent_stats.py`.
- **Consumer (training):** `rynnvla/datasets/vla_datasets/latent_*.py`.

Changing the layout, the stride or the schema version is a coordinated edit across all of the
above.

---

## Training

`scripts/train.sh` is a thin wrapper over `python -m rynnvla.api.launch`, which builds the
torchrun command for a single node. Presets: `stage1` (4B), `stage1_2b`, `stage2` (4B),
`stage2_2b`. The portable entry points do not install packages, download datasets, select
network interfaces or invoke any external launcher. There is no implicit environment-variable
expansion in JSON: supply model, data-mixture and output paths yourself.

`rynnvla/configs/` holds three further recipes that are **not** launch presets, because they are
not LIBERO V5: `stage2_robotwin_2b.json` and `stage2_robotwin_4b_2node16gpu.json` (see
[RoboTwin and VLABench tracks](#robotwin-and-vlabench-tracks)) and `stage1_smoke_tiny.json`
(used only by `scripts/smoke.sh`). Pass them to torchrun with `--config`, as shown in
[Overrides and the direct training entry point](#overrides-and-the-direct-training-entry-point).

### Presets

| Setting | `stage1_4b.json` | `stage2_v5_4b.json` |
| --- | --- | --- |
| Initialization | user-supplied RynnBrain-4B backbone | complete exported Stage-1 EMA model |
| Expert | Qwen3-VL, hidden 768 / intermediate 2752 | same |
| Time conditioning | AdaLN, per-layer AdaRMSNorm, time concat on | same |
| Foresight / training repeats | 0 / 1 | 0 / 1 |
| Latent prediction / latent-head readout | true / false | false / true |
| Camera context tokens | true | true |
| Latent geometry | 6 roles, dim 608, chunk 6, stride 5 | retained from Stage 1 (no latent prediction) |
| Robot action chunk | 30 | 10 |
| State input | false | true |
| Action normalization | not used for latent targets | `min_max_sym` |
| Frozen parameters | none, including vision | none |
| Image token budget (`mm_max_length`) | 64 | 2048 (256×256 LIBERO views normally yield 64) |
| Micro batch × accumulation × ranks | 4 × 4 × 8 = 128 | 4 × 4 × 8 = 128 |
| Steps / warmup | 60000 / 5000 | 30000 / 3000 |
| Backbone / head LR | 1e-5 / 1e-4 | 2.5e-5 / 1e-4 |
| Scheduler | `cosine_with_min_lr`, floor 0.05 of each group's peak | `cosine` |
| EMA decay | 0.999 | 0.99 |
| Visual augmentation / chunk overlap | true / 0 | false / 0.5 |
| DeepSpeed | ZeRO-2 | ZeRO-1 |
| Save interval / rolling limit | 2000 / 3, plus permanent 6000-step milestones | 2000 / 3 |

Both stages use bf16, gradient checkpointing, Adam beta2 0.95, weight decay 0, max gradient
norm 1, raw (not state-delta) action targets, `rot_6d`, sequence packing off, and no SF/LB
teacher. Stage-1 latent normalization comes from the data stats, not from robot-action
normalization. Stage 2 uses the existing interleaved LIBERO rot6d layout; do not change
`LIBERO_ROT6D_LAYOUT` unless the data was converted accordingly.

### Parameter groups: `action_head_lr`, `transfer_lr`, `frozen_parameters`

Three independent knobs decide which parameters train, and at what rate. All three are
resume-critical: changing one on `--resume` is rejected rather than silently applied, because
each changes what the restored optimizer state means.

**Prefix matching is component-wise, not substring.** A prefix `p` matches a parameter name
when `p` is a whole dotted component of it (`(?:^|\.)p(?:\.|$)`). So `action_expert` matches
`action_expert.layers.0.mlp.weight` but not `my_action_expert_extra.bias`. A prefix that matches
nothing **raises** rather than being ignored — a typo would otherwise cost you a full run at the
wrong learning rate.

| Knob | Applies to | Effect |
| --- | --- | --- |
| `learning_rate` | everything not claimed below | base peak LR |
| `action_head_lr` + `action_head_modules` | the newly-initialized head | its own peak LR |
| `transfer_lr` + `transfer_modules` | inherited modules you want to move slower or faster than the trunk | a third peak LR |
| `frozen_parameters` | regex list matched with `re.match` on parameter names | `requires_grad_(False)`; excluded from every optimizer group |

`action_head_lr` alone falls back to `DEFAULT_ACTION_HEAD_MODULES`; passing
`action_head_modules` explicitly is what makes an unmatched prefix an error. `transfer_lr` must
be positive and finite and requires a non-empty `transfer_modules`, and the two sets must not
overlap — overlapping prefixes would leave a parameter in two groups, and which one wins depends
on group order rather than on anything you wrote.

`frozen_parameters` is applied after model construction, trunk overlay and module resets, and is
re-applied on resume. Because it uses `re.match`, patterns are anchored at the **start** of the
name: `visual` freezes `visual.*`, and `.*visual` freezes nothing. Freezing the vision backbone
to make a test fit in memory produces a result that is not full-model training — say so if you
do it.

Each rate gets its own scheduler, so `cosine_with_min_lr`'s floor is a fraction of *that group's*
peak, not of `learning_rate`.

Notes that matter when editing the prefix lists:

- `state_proj` is still constructed in Stage 1 even when the processor suppresses state input.
- Shared-path modules `action_time_proj`, `time_mlp`, `adaln_in`, `adaln_final`, the action
  in/out projections and `confidence_head` exist even where a given forward path does not use
  them; they remain in the LR list.
- `latent_in_proj` is Stage-1-only; Stage 2 instead builds `latent_readout_proj`.
- `action_expert` includes its time-concat MLPs.

The RoboTwin Stage-2 recipes are worked examples: both set
`transfer_modules: [action_expert, latent_action_head, slot_seed_norm, slot_seed_proj]` at
`transfer_lr: 1e-4` while the trunk runs at `1e-5` and the fresh head at `1e-4`.

### Sampler shuffle order

`sampler_shuffle` selects between two orderings over the same sample set:

| Value | Order | Use it when |
| --- | --- | --- |
| `auto` (default) | episode-locality permutation: samples from one episode are kept near each other | large latent corpora — this is what makes the latent LRU effective |
| `global` | uniform `randperm` over every sample | small corpora, or an experiment that needs sample order to carry no episode structure |

This is a throughput knob before it is a statistics one. Under `auto`, consecutive samples
usually come from the same episode, so the per-worker latent cache serves them without reopening
the `.npz`; under `global` nearly every sample re-reads a whole file to take a few rows from it.
On a 121M-sample latent corpus at alpha=0.7 that difference measured **5.2×** wall clock
(0.841 → 4.13 s/step). In a synchronous run one rank's cold read stalls every rank at the
gradient all-reduce, and the step probe books that as `fwd_bwd` rather than `data` — so the run
*looks* compute-bound while it is waiting. If a Stage-1 run is unexpectedly slow and the GPU
utilization is low, check this before profiling the model.

`global` is still the right choice where locality buys nothing: `stage2_robotwin_4b_2node16gpu.json`
uses it, because that corpus is small enough to stay cached either way and a uniform order is the
cleaner control.

The value is resume-critical and recorded in the checkpoint, so an existing run cannot be
switched mid-flight. A checkpoint written before this option existed is treated as `auto`, which
is the order it was trained with.

### Prepare local data

Copy and edit the appropriate template:

- `configs/data_latent_pretrain.example.json` — prebuilt latent index and latent stats; the
  optional `dataset_weights` map sets per-source mixture weights (omit it for uniform).
- `configs/data_latent_pretrain_rynnvla_base.example.json` — RynnVLA-Base derived indices.
- `configs/data_libero_joint40.example.json` — four standard LeRobot v2.1 LIBERO suites.
- `configs/data_robotwin_mixed.example.json` — RoboTwin HDF5 corpus plus its episode index.
- `configs/data_vlabench.example.json` — a converted VLABench tree.

All `data/...` paths are placeholders relative to the **process working directory**, not the
JSON directory. Populate the files before launching: the launcher does not build an index,
extract latents or fetch datasets.

For a Stage-1 latent mixture, `dataset_weights` can be derived by temperature rather than set by
hand — see [Balancing the mixture](#balancing-the-mixture-dataset_weights-and-alpha).

### Launch Stage 1

The output directory must **not exist**, even as an empty folder; create only its parent.

```bash
python -m rynnvla.api.launch \
  --stage stage1 \
  --model-path models/RynnBrain-4B \
  --data-mixture configs/data_latent_pretrain.local.json \
  --output-dir runs/stage1 \
  --nproc-per-node 8
```

`bash scripts/train.sh` accepts the same options and uses `${PYTHON_BIN:-python}`.
`--dry-run` prints the torchrun command without checking data/model existence, importing torch,
creating files or starting workers — it is **not** a training smoke test. For a user-run
one-GPU, two-step smoke:

```bash
python -m rynnvla.api.launch \
  --stage stage1 --model-path models/RynnBrain-4B \
  --data-mixture configs/data_latent_pretrain.local.json \
  --output-dir runs/stage1-smoke --nproc-per-node 1 \
  --max-steps 2 --micro-batch-size 1 --gradient-accumulation-steps 1 \
  --warmup-steps 0 --dataloader-num-workers 0 --dry-run
```

Remove `--dry-run` only when ready to allocate a GPU; a one-GPU 4B job can still exceed GPU
memory. Changing rank count, micro batch or accumulation does **not** automatically scale batch
size or LR — to keep global batch 128 on one rank with micro batch 4, use accumulation 32.

For two nodes, invoke on **each** node with the same paths, rendezvous address/port and
options, changing only `--node-rank`:

```bash
python -m rynnvla.api.launch \
  --stage stage1 --model-path models/RynnBrain-4B \
  --data-mixture configs/data_latent_pretrain.local.json \
  --output-dir runs/stage1-two-nodes \
  --nnodes 2 --nproc-per-node 8 --node-rank 0 \
  --master-addr node0.example --master-port 29512 \
  --gradient-accumulation-steps 2
```

Multinode requires an explicit reachable master address and shared output storage. Single-node
defaults to torchrun `--standalone`; to use a fixed port pass both `--master-addr 127.0.0.1`
and `--master-port`. Node/rank topology is never inferred from cluster-specific variables. The
implementation supports data-parallel single/multinode torchrun; the formal recipes do not
enable pipeline or expert parallelism.

### Export the Stage-1 EMA initialization

Choose any complete Stage-1 checkpoint, not necessarily step 60000. Create the export parent
yourself; the destination must not exist.

```bash
python -m rynnvla.api.export_checkpoint \
  --checkpoint-dir runs/stage1/checkpoint-60000 \
  --output-dir models/stage1-ema \
  --weights ema --format safetensors
```

Equivalent installed-package wrapper: `python scripts/export_checkpoint.py`. `--weights ema`
requires exactly that checkpoint's `ema_model.bin` and **never** falls back to live weights.
`--weights model` explicitly selects live HF safetensors / `pytorch_model.bin` (including
indexed shards) or the Trainer's `model.bin`. Use `--format pytorch` to emit
`pytorch_model.bin` instead of `model.safetensors`. No optimizer, RNG state, EMA shadow,
trainer state, history logs or code files are copied.

The exporter checks the formal config and processor contract, then constructs the real model on
**meta** and validates the entire weight key set and every shape. Tensor files load only on
CPU; pickle loading uses `weights_only=True`. It retains the config, processor/image/video
processor, tokenizer and chat-template sidecars. Missing tokenizer/template metadata, partial
weights, wrong widths or shapes, and existing destinations all fail explicitly. No byte-count
or step-number assertion is used. The CPU machine needs sufficient RAM and disk for the
complete weights; meta construction does not allocate model-sized parameter storage.

### Launch Stage 2

```bash
python -m rynnvla.api.launch \
  --stage stage2 --model-path models/stage1-ema \
  --data-mixture configs/data_libero_joint40.local.json \
  --output-dir runs/stage2-v5 --nproc-per-node 8
```

This is a **new optimizer run** initialized from the whole exported Stage-1 model, not a resume
of Stage 1. The launcher validates Stage-1 metadata before starting workers. VLM, vision,
expert, latent head and camera projection transfer together; `latent_readout_proj` is new, and
the Stage-1-only `latent_in_proj` is not used. Strict transfer rules are enforced by the model
loader, independently of export validation. Chunk 30 → 10 does not change shared tensor shapes.
For Stage-2 EMA inference exports use the same exporter with `--stage stage2`; the Stage-2
preset retains per-checkpoint EMA but does not enable automatic final-root EMA export.

### Resume versus initialization

A fresh launch rejects any existing output directory. To resume **the same stage**, repeat the
original command with `--resume` and the same `--output-dir`:

```bash
python -m rynnvla.api.launch \
  --stage stage2 --model-path models/stage1-ema \
  --data-mixture configs/data_libero_joint40.local.json \
  --output-dir runs/stage2-v5 --nproc-per-node 8 --resume
```

Resume selects the latest **complete** training checkpoint according to the Trainer
completeness check, reads its model/processor metadata and restores training state. It bypasses
primary initialization, trunk overlay and module resets. The required CLI model path is ignored
for metadata/weights on resume. An exported model directory is **not** a resumable training
checkpoint. Do not change architecture, frozen modules, stage or optimizer partitioning when
resuming; those changes require a new output directory and an appropriate initialization. Do not
run two concurrent jobs against one output directory.

### Overrides and the direct training entry point

Precedence is preset JSON → explicit CLI. Kebab-case and the original underscore
`TrainingArguments` options are both accepted. JSON object overrides merge by key (one level);
lists and scalars replace the preset value. Unknown top-level JSON fields and CLI options fail.
The model/data fields `action_chunk_size`, `use_latent_actions`, `latent_action_dim` and
`num_view_slots` are kept in sync when explicitly overridden; conflicting simultaneous values
fail. Adjust the data manifest separately if you change its latent geometry.

```bash
python -m rynnvla.api.launch \
  --stage stage2 --model-path models/stage1-ema \
  --data-mixture configs/data_libero_joint40.local.json \
  --output-dir runs/stage2-custom \
  --max-steps 100 --learning-rate 0.00002 \
  --processor-overrides '{"use_state": true}'
```

For a completely custom JSON, use torchrun directly rather than combining `--config` with
launch's `--stage`:

```bash
python -m torch.distributed.run --standalone --nproc_per_node 8 \
  --module rynnvla.api.train --config rynnvla/configs/stage1_4b.json \
  --model-path models/RynnBrain-4B \
  --data-mixture configs/data_latent_pretrain.local.json \
  --output-dir runs/stage1-custom --max-steps 100
```

In a recipe JSON only its `deepspeed` path is resolved relative to that JSON. Explicit CLI
paths, model/data/output paths, and paths within the data mixture remain relative to the
working directory. API training merges and parses all values **before** constructing
`TrainingArguments` (which initializes CUDA and distributed training); it does not first
instantiate defaults and then overwrite them.

---

## Latent-action model (RynnLAM)

The `rynnlam` package is the self-supervised latent-action model that *produces* the 608-dim
latents consumed at Stage 1. Its training and evaluation entry points are included so the
architecture is reproducible end to end.

Training RynnLAM needs the DA3-Large encoder weights described under
[Optional components](#optional-components):

```bash
# Train RynnLAM (single-node DDP via torchrun); recipes are YAML configs.
NPROC_PER_NODE=8 bash scripts/train_rynnlam_stage1.sh --output-dir ./runs/stage1
NPROC_PER_NODE=8 bash scripts/train_rynnlam_stage2.sh --resume ./runs/stage1/latest.pt
# equivalently:
python scripts/train_rynnlam.py --config rynnlam/configs/stage1.yaml \
    --encoder-checkpoint checkpoints/DA3-LARGE-1.1/model.safetensors
```

To check the training path without any external weights, use the bundled synthetic corpus and
the smoke recipe — the encoder is randomly initialized, so nothing is learned, but the dataset,
model, all five losses, checkpointing and the internal eval all run:

```bash
python scripts/make_sample_lam_data.py                      # -> data/sample_lam/ (3 scenes)
python scripts/train_rynnlam.py --config rynnlam/configs/smoke.yaml
```

`scripts/train_rynnlam.py` accepts `--config`, `--output-dir`, `--resume`,
`--encoder-checkpoint`, `--max-steps`, `--device {auto,cpu,cuda}` and
`--reset-optimizer/--no-reset-optimizer`. `batch_size` is per rank; `global_step`, checkpoint
names, save intervals and loss warmups count per-rank batches. With
`target_effective_batch > 0`, `max_steps` counts successful optimizer updates; otherwise it
counts per-rank batches. LR warmup always counts optimizer updates. Resetting the optimizer
loads weights strictly and restarts all counters; a normal resume restores them.

RynnLAM training reads preprocessed per-scene safetensors (see `data.manifest_path` /
`data.safetensors_root` in the recipes). `rynnlam/configs/*.yaml` are the reference recipes;
`rynnlam/config.py:validate_training()` enforces their invariants — in particular Stage-2 full
finetuning requires an explicit `--resume`, and `data.manifest_path` /
`data.safetensors_root` must be set explicitly rather than defaulting.

**Not every recipe emits the 608-dim protocol.** The delivered latent width is
`num_k_tokens * k_token_out_dim + latent_dim + camera_pose_latent_dim`, where `k_token_out_dim`
is `k_token_bottleneck_dim` when set and otherwise the full token width (`embed_dim * 2 = 2048`
for `k_token_source: features`):

| Recipe | K | `k_token_bottleneck_dim` | Delivered latent |
| --- | --- | --- | --- |
| `stage1.yaml`, `stage2.yaml`, `smoke.yaml` | 8 | 64 | **608** — the `ktoken_zcam` protocol |
| `stage1_k2.yaml` | 2 | absent → 0 | 4192 |
| `stage1_k4.yaml` | 4 | absent → 0 | 8288 |

The two `k*` files are ablation arms that deliberately drop the bottleneck, so they run the
full-width-token branch rather than a narrow-K version of the delivered recipe. Stage-1 training
is configured for `latent_action_dim: 608` and will reject a corpus labeled from either. To
ablate K at a comparable width, add `k_token_bottleneck_dim: 64` (K=2 → 224, K=4 → 352). Both
files carry the same warning in their header.

This is a **different format from the Stage-1 latent npz** above: Stage 1 consumes latents, RynnLAM
training consumes the raw geometry those latents are learned from. `data/sample_lam/` is a
worked example of both files.

`data.manifest_path` is a directory scanned for `manifest_*.json` (or `manifest.json`, or one
level down). Each manifest is a JSON list of scene entries:

| Key | Type | Notes |
| --- | --- | --- |
| `scene_id` | str | unique; `<dataset>/<episode>/<view>`. The view suffix sets the role used for role-weighted sampling |
| `file` | str | scene safetensors path, resolved **relative to the manifest's own directory** — the one exception to the working-directory rule, and what keeps a corpus portable |
| `num_frames` | int | `T`; must match the stored tensors or the scene is rejected |
| `num_flows` | int | `T-1` |
| `height`, `width` | int | validated against the actual tensor shape; a mismatch raises rather than silently resampling |
| `bucket` | str | resolution bucket; batches are built per bucket so a batch stays single-resolution |
| `depth_stride` | int | depth decimation relative to RGB |

Inside the safetensors, every tensor key is prefixed with the file's own stem (`scene_id` with
`/` replaced by `__`):

| Tensor | Dtype / shape | Notes |
| --- | --- | --- |
| `<stem>__rgb` | uint8 `[T, H, W, 3]` | |
| `<stem>__depth` | float16 `[T, H, W]` | metres; `> 0` and finite marks a valid pixel, and the flow mask is derived from it |
| `<stem>__flow` | float16 `[T-1, H, W, 3]` | precomputed **3D** flow; a last dim of 3 selects the precomputed branch, 2 selects 2D optical flow (which the loader lifts to 3D using the stored cameras) |
| `<stem>__extrinsics` | float32 `[T, 4, 4]` | world-to-camera |
| `<stem>__intrinsics` | float32 `[T, 3, 3]` | |
| `<stem>__mask` | uint8 `[T, H, W]` | 255 = valid |

A sample is a frame pair `(t, t+stride)` with `stride` drawn from
`[data.min_frame_stride, data.max_sample_stride]`, clamped so the pair stays inside the scene;
scenes are chunked non-overlappingly at `data.max_frame_stride`, so a scene contributes
`(num_frames - 1) // max_frame_stride` samples. A load that still fails (a truncated flow
interval, a resolution that disagrees with the manifest) is retried on a same-resolution
substitute scene rather than dropped, which keeps every batch single-resolution.
`scripts/make_sample_lam_data.py` generates a minimal valid corpus and is the shortest readable
specification of this format.

```bash
# Evaluate: validate/merge an extraction, or run the external LARYBench regression probe.
bash scripts/evaluate_rynnlam.sh extract --help
bash scripts/evaluate_rynnlam.sh regress --lary-root <your LARYBench checkout> --help
```

The `regress` subcommand shells out to a **user-supplied** LARYBench checkout
(`--lary-root/regression/main.py`); that external benchmark is not bundled (see NOTICE).
The child process is launched with `WANDB_MODE=disabled` / `WANDB_DISABLED=true`, so no
external service is contacted from this repository or from the probe it runs.

---

## Inference and evaluation

The supported method is **Stage-1 RynnLAM latent pretraining → full-transfer Stage-2 V5 LIBERO
joint40**. The public model name is RynnVLA; Python imports use `rynnvla`. These entry points
run locally, without a neighboring repository or an external launcher.

```bash
python -m rynnvla.api.export_checkpoint --help   # Trainer ckpt -> HF directory (Stage-1 -> Stage-2 handoff)
python -m rynnvla.api.eval_libero       --help   # LIBERO closed-loop evaluation
python -m rynnvla.api.predict           --help   # single-observation action prediction
python scripts/vlabench_policy_server.py --help  # serve a Stage-2 checkpoint to a VLABench runner
```

LIBERO is the only benchmark whose simulator orchestration ships here. RoboTwin and VLABench
are supported as *policy* sides — see
[RoboTwin and VLABench tracks](#robotwin-and-vlabench-tracks) for exactly what that means and
what you have to bring.

### Checkpoint and observation contract

Use an **exported Stage-2 Hugging Face directory**, not a Trainer checkpoint and not a
Stage-1-only latent model. It must contain `config.json`, model weights (and their index when
sharded), tokenizer/image-processor assets, and the saved `RynnBrainVLAProcessor` configuration
including the Franka action/state schema and normalization statistics. Loading is
`local_files_only=True`: missing assets are an error, not a download. Architecture, chunk
length, camera-conditioning flags and action normalization come from the export; the CLI does
not replace trained flags. `--attn-implementation` selects the runtime attention backend.

The narrow adapter matches `rynnvla/datasets/vla_datasets/libero_plus.py`:

- `front` → `VIEW_ROLE_TO_ID['front_third']` (3), `wrist` → `left_wrist` (1). Camera order and
  labels are handled by the saved processor.
- State is **world-frame EEF xyz + axis-angle + raw gripper qpos**. The JSON accepts seven
  values (first qpos only) or eight (both qpos); only `qpos[0]` reaches the model. No
  base-offset subtraction, joint-state substitution, gripper normalization or missing-state
  zero fallback is applied.
- Simulator observations supply an **xyzw** quaternion, converted with the LIBERO client's
  `quat2axisangle` convention.
- Actions are raw OSC inputs `[dx, dy, dz, dax, day, daz, gripper]`. The internal
  81-dimensional RobotAction uses **interleaved** rot6d (the first two matrix columns flattened
  row-major). Processor statistics are reversed, then rot6d is converted to axis-angle.
  Commands are neither absolute EEF targets nor state-relative training targets: do **not**
  subtract the current pose or scale OSC commands again. No clipping, gripper inversion or
  binarization is added.
- Legacy non-interleaved rot6d and state-relative action checkpoints are outside this
  adapter's contract. Explicit legacy metadata and relative schemas are rejected; historical
  server heuristics that guessed layout or inverted grippers from statistics are intentionally
  not reproduced. Exports without layout metadata must be known V5/interleaved exports, not
  arbitrary old models.

### Exact image orientation

Raw LIBERO simulator renders must be rotated with `image[::-1, ::-1]` — a **180° rotation**, not
just a vertical flip — on **both** cameras, exactly once. `libero_images()` in
[`rynnvla/inference_wrappers/rynn_brain_vla.py`](rynnvla/inference_wrappers/rynn_brain_vla.py)
implements this behind a `convention` argument: `libero_raw` applies the rotation, `dataset`
does not.

Dataset-decoded and replay images already carry that orientation, so running them through the
`libero_raw` path rotates them a second time and silently evaluates the model on upside-down
input. Images are RGB, uint8, HWC; no BGR swap, manual resize, crop or tensor normalization is
introduced. The saved processor owns image preprocessing.

### Offline prediction / replay (not a closed-loop benchmark)

Create a JSON observation, for example `samples/observation.json`:

```json
{
  "instruction": "pick up the black bowl and place it on the plate",
  "state": [0.45, 0.02, 0.85, 0.0, 0.0, 0.0, 0.025, -0.025],
  "front": "front.png",
  "wrist": "wrist.png",
  "image_convention": "dataset"
}
```

Image paths are relative to the JSON file (absolute paths also work). `image_convention`
defaults to `dataset`, meaning already model-oriented RGB. Use `libero_raw` **only** for
unmodified raw `OffScreenRenderEnv` renders. The state values above are illustrative; real
replay must use the recorded state.

```bash
python -m rynnvla.api.predict \
  --model-path exports/stage2_v5 \
  --sample samples/observation.json --output results/actions.npy \
  --seed 7 --denoising-steps 10
```

The `.npy` result is float32 `(checkpoint_action_chunk_size, 7)` in raw OSC units; read it with
`numpy.load(..., allow_pickle=False)`. Existing outputs are not overwritten. For a sufficiently
provisioned CPU host, add `--device cpu --dtype float32 --attn-implementation sdpa`; this still
loads the real model and may be slow and memory-intensive — it is not a lightweight model test.

Repeated offline observations can check action consistency, but **offline replay never measures
task success**: its recorded observations do not respond to the predicted actions.

### Local LIBERO closed loop

Install and configure **upstream LIBERO** separately, including its compatible
robosuite/MuJoCo dependencies, benchmark BDDL files, task initial-state assets and an
offscreen-rendering backend. No simulator or rendering environment is installed or changed by
these CLIs. The LIBERO-Plus fork's perturbed task registry is not the standard joint40
benchmark. Missing LIBERO produces an actionable error and does not affect offline prediction
or pure adapter imports.

```bash
python -m rynnvla.api.eval_libero \
  --model-path exports/stage2_v5 \
  --suite libero_spatial --task-id 0 --episodes 1 \
  --seed 7 --output-dir results/libero_spatial_task0 \
  --replan-steps 10 --denoising-steps 10 --max-steps 220 \
  --save-frames
```

- One `OffScreenRenderEnv` is owned per episode. Episode `k` uses task initial state `k`, with
  simulator and policy seed `seed + k`. Asking for more episodes than available initial states
  is an error; states are not silently recycled.
- Ten warmup steps use `[0, 0, 0, 0, 0, 0, -1]` by default (`--warmup-steps` changes this).
  Termination or success during warmup is respected.
- Every replan uses the **current** observation and a fresh prefill, predicts a chunk, then
  executes at most `--replan-steps` commands (default 10). Success, termination or truncation
  stops execution immediately, even mid-chunk.
- Policy steps are capped at `--max-steps` and the suite budget, whichever is smaller:
  spatial 220, object 280, goal 300, libero_10 520, libero_90 400. Without `--max-steps`, the
  suite budget applies. Warmup is additional and separately recorded, so total steps are
  bounded by policy cap + warmup cap.
- `check_success()` determines success, not arbitrary `done`. Environment close runs in
  `finally`, including failures during reset, warmup, prediction or frame saving. Errors are
  recorded as unknown outcomes (`success: null`), not measured task failures, and abort the run.
- The output directory must be empty. `run.json` records effective arguments; `results.jsonl`
  records real per-episode outcomes, seeds, step counts, stop reason and duration;
  `episode_XXXX_actions.npy` records executed policy commands (excluding warmup). Optional PNGs
  record both model-oriented camera views at reset and every step. `summary.json` is written
  only after all episodes complete.

### RoboTwin and VLABench tracks

Both benchmarks are supported on the **policy side only**. Read this before expecting a score.

| | ships here | you must bring |
| --- | --- | --- |
| RoboTwin | `RoboTwinDataset` (training), `scripts/robotwin_policy.py` (eval adapter), `scripts/build_robotwin_index.py`, `rynnvla/configs/stage2_robotwin_{2b,4b_2node16gpu}.json`, `configs/data_robotwin_mixed.example.json` | the RoboTwin simulator and its own eval entry point, and the HDF5 episode corpus |
| VLABench | `VLABenchDataset` (training), `scripts/vlabench_policy_server.py` (eval server), `configs/data_vlabench.example.json` | the VLABench simulator tree and whatever runner drives the policy server |

Neither simulator, neither runner and neither corpus is redistributed here, so **no command in
this repository produces a RoboTwin or VLABench number on its own.** What the shipped pieces do
guarantee is that the model side of the interface — observation decoding, camera→role mapping,
action decoding and normalization — is the one the training dataset uses, which is the half
where silent train/eval skew happens. `tests/test_vlabench_inference_adapter.py`,
`tests/test_vlabench_policy_server.py` and `tests/test_robotwin_stage2.py` pin that half on CPU.

**RoboTwin training.** Build the index the dataset requires, then point a mixture at the pair:

```bash
python scripts/build_robotwin_index.py \
    --data-root data/robotwin/robotwin_raw_hdf5 \
    --instructions data/robotwin/robotwin_instructions.json \
    --out data/robotwin/robotwin_index.json
```

`--instructions` is a JSON object mapping task name → list of phrasings; it is the one field
that cannot be derived from the corpus, because RoboTwin's HDF5 files carry endpose, joint
action and pixels but no language. Tasks with no entry are **refused** rather than written with
an empty instruction list, since an empty prompt trains a different model and nothing complains
about it later. The script prints the index's sha256 — record it as `index_sha256` in the
mixture so a resumed run refuses an index that was rebuilt underneath it.

**RoboTwin eval.** `scripts/robotwin_policy.py` is a module, not a CLI: RoboTwin's own eval
entry point imports it and calls `get_model(usr_args)` / `reset_model(model)` /
`eval(TASK_ENV, model, observation)`. Put it on the path RoboTwin loads policies from, and
export the checkpoint first with `python -m rynnvla.api.export_checkpoint`. Two environment
variables are read: `ROBOTWIN_ACTION_SPACE` must be `ee` (anything else raises), and
`RYNNVLA_ROBOTWIN_INFER_HORIZON` (default 30) caps how many commands of a predicted chunk are
executed before replanning.

The adapter's `validate_checkpoint` requires `action_dim=81`, `action_chunk_size=30`,
`latent_action_dim=608`, `use_latent_head_readout=true`, `use_view_cond_slots=true`,
`bypass_latent_output_projection=false`, `action_norm_type="mean_std"`, `use_state=true`, and a
dual-arm EEF schema with canonical interleaved rot6d and absolute grippers. **A LIBERO V5
Stage-2 export does not satisfy this** — it is chunk 10 with `min_max_sym` — and will be
rejected rather than silently evaluated with the wrong action semantics. Train with the RoboTwin
recipe instead, which is not a launch preset and so goes to torchrun directly:

```bash
# 2B, single node
python -m torch.distributed.run --standalone --nproc_per_node 8 \
  --module rynnvla.api.train --config rynnvla/configs/stage2_robotwin_2b.json \
  --model-path exports/stage1-ema \
  --data-mixture configs/data_robotwin_mixed.local.json \
  --output-dir runs/stage2-robotwin-2b

# 4B, two nodes x 8 GPUs: run on EACH node, changing only --node-rank
python -m torch.distributed.run \
  --nnodes 2 --nproc_per_node 8 --node-rank 0 \
  --master_addr node0.example --master_port 29512 \
  --module rynnvla.api.train --config rynnvla/configs/stage2_robotwin_4b_2node16gpu.json \
  --model-path exports/stage1-ema \
  --data-mixture configs/data_robotwin_mixed.local.json \
  --output-dir runs/stage2-robotwin-4b
```

Both recipes set `transfer_lr: 1e-4` over
`[action_expert, latent_action_head, slot_seed_norm, slot_seed_proj]` while the trunk stays at
`1e-5`; the 4B two-node one also sets `sampler_shuffle: global`. See
[Parameter groups](#parameter-groups-action_head_lr-transfer_lr-frozen_parameters) and
[Sampler shuffle order](#sampler-shuffle-order).

**VLABench eval.** Serve an exported Stage-2 checkpoint over a socket:

```bash
python scripts/vlabench_policy_server.py \
    --model-path exports/stage2_vlabench --direct --weights ema \
    --device cuda:0 --port 0 --ready-file /tmp/vlabench_ready.json
```

`--port 0` binds an ephemeral port and the actual one is written to `--ready-file` as JSON
(`host`, `port`, `pid`, `code_root`), which is how a runner discovers it. The wire protocol is
specified in full in that file's docstring: a 4-byte big-endian length covering a one-byte type
plus a **pickle** body, in both directions; the request carries
`{"mode": "sync", "policy_seed": int, "obs": {"images": {image, second_image, wrist_image},
"state": (7,), "prompt": str}}` and the reply is a `(T, 7)` float32 chunk of **absolute**
targets that the client converts back by adding `base_pos`, running IK and binarizing the
gripper. State is `[ee_pos - base_pos, ee_euler_xyz_world, raw_gripper_flag]`, and the raw flag
is inverted inside the adapter to match training.

Unpickling executes arbitrary code, so the server binds **loopback only** by default and
refuses a reachable address unless you pass `--allow-remote-bind`. Keep it that way and keep the
port off any network you do not control; the server is meant to be driven by a runner on the
same node. `tests/test_vlabench_policy_server.py` pins both the framing and that refusal, and
will additionally round-trip through the real harness client if you set
`VLABENCH_HARNESS_SCRIPTS` to a directory containing `policy_client.py`.

Camera→role mapping is `image`→`front_third`, `second_image`→`side_left`,
`wrist_image`→`left_wrist`, identical to `VLABenchDataset._CAMERA_SLOTS` on the training side.

### Validation boundary

```bash
python -B -m rynnvla.api.predict --help
python -B -m rynnvla.api.eval_libero --help
```

These import the policy and evaluation entry points and build their argument parsers, so they
catch import-time and wiring breakage without a GPU or a simulator. They do **not** validate
learned checkpoint quality, GPU inference, rendering or benchmark success — a green `--help` is
not a green rollout.

A single task/episode is a smoke test, **not** a suite score or a zero-shot claim. Any published
metric must state the checkpoint, suite/task coverage, initial states, seeds, step/replan
budgets and simulator version, and must come from actual completed rollouts.

---

## Third-party dependencies and attribution

RynnVLA uses Qwen3-VL model and processor interfaces from Hugging Face Transformers. The
positional indexing and attention helpers in the RynnVLA implementation follow the
corresponding Qwen/Transformers implementations. Transformers is distributed under Apache-2.0;
its applicable license and notices apply to derived code. Every derived file carries a header
naming its upstream reference, and the full list is in [NOTICE](NOTICE).

The training runtime uses PyTorch, DeepSpeed, Accelerate, FlashAttention, NumPy, SciPy, PyAV
and PyArrow. These are installed dependencies, not vendored copies in this repository; their
upstream licenses continue to apply.

The action expert includes implementations of, and architectural references to, conditional
normalization, two-stream action attention and optional auxiliary heads, documented in the
source.

Not included in this distribution — obtain each separately and follow its own license and usage
conditions: RynnBrain pretrained weights, the DA3-Large encoder weights referenced at
`checkpoints/DA3-LARGE-1.1/model.safetensors`, offline RynnLAM latent files, LIBERO datasets
and LIBERO simulator assets, the RoboTwin HDF5 corpus and simulator, and the VLABench simulator
tree plus whatever runner drives `scripts/vlabench_policy_server.py`. No robot mesh assets,
private cluster submission configuration or benchmark datasets were copied into this release.

This distribution bundles two first-party packages — `rynnvla` (the VLA training/inference
stack) and `rynnlam` (the latent-action model that produces the 608-dim latents consumed at
Stage 1). `rynnlam` embeds a DA3/DINOv2-derived backbone under `modules/dinov2/` and a logger
derived from ByteDance source; both carry Apache-2.0 headers. The full attribution is in
[NOTICE](NOTICE).

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights and datasets
are separate resources and are not granted by this source license. Obtaining a checkpoint does
not grant permission to redistribute its training data or third-party weights.

<!-- TODO(hf): RynnLAM and the Stage-1 RynnVLA-Latent-2B/4B checkpoints are now published — see
     the "Model weights" section above. Still outstanding: once the RynnBrain-2B/4B VLM backbones
     and the DA3-Large encoder are published, add their hub links to that section with the exact
     revision each recipe was trained against, and replace the "Not included in this distribution"
     list with per-artifact license notes. Everything published so far is a separate download
     loaded with local_files_only=True, so no link here may imply a download this repository
     performs. -->
