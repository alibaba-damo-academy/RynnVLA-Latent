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
---

<div align="center">

## RynnVLA-Latent: Scaling Robotic Manipulation Foundation Models with 3D-Grounded Latent Actions

</div>

<p align="center">
  💫 <a href="https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io"><b>Project Page</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
  🤗 <a href="https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-Latent"><b>Hugging Face</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
  🤖 <a href="https://www.modelscope.cn/models/DAMO_Academy/RynnVLA-Latent"><b>ModelScope</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
  📄 <a href=""><b>arXiv</b></a>
</p>

---

## 🌟 Abstract

RynnVLA-Latent turns heterogeneous interaction videos into scalable supervision for robot policy pretraining. Its core, **RynnLAM**, learns continuous 3D-grounded latent actions from visual transitions without native robot controls; trained on 30.1K hours of human, robot and simulation videos, it produces 745M latent-action descriptors. A flow-matching VLA then predicts latent-action chunks from observations and language (Stage 1), before the pretrained backbone and action expert are adapted to embodiment-specific robot actions (Stage 2). The resulting 4B policy achieves **98.8% on LIBERO**, **84.7% on LIBERO-Plus** without benchmark-specific adaptation, **51.9%** average on VLABench and **88.6%** average on RoboTwin 2.0, plus top success rates among compared methods on six real-world tasks.

## 📖 Model weights

The trained checkpoints are published **separately from this code repository**, on both Hugging Face and ModelScope. This repository ships no weights (everything is loaded with `local_files_only=True`).

| Artifact | What it is | Hugging Face | ModelScope |
| --- | --- | --- | --- |
| **RynnLAM** | self-supervised latent-action model (`ktoken_zcam`, 608-dim) — the auto-labeler that produces the Stage-1 latent corpus | [Alibaba-DAMO-Academy/RynnLAM](https://huggingface.co/Alibaba-DAMO-Academy/RynnLAM) | [DAMO_Academy/RynnLAM](https://modelscope.cn/models/DAMO_Academy/RynnLAM) |
| **RynnVLA-Latent-2B** | Stage-1 latent-action pretrained VLA, 2B backbone (bfloat16, EMA @ step 236442) | [Alibaba-DAMO-Academy/RynnVLA-Latent-2B](https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-Latent-2B) | [DAMO_Academy/RynnVLA-Latent-2B](https://modelscope.cn/models/DAMO_Academy/RynnVLA-Latent-2B) |
| **RynnVLA-Latent-4B** | Stage-1 latent-action pretrained VLA, 4B backbone (bfloat16, EMA @ step 236442) | [Alibaba-DAMO-Academy/RynnVLA-Latent-4B](https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-Latent-4B) | [DAMO_Academy/RynnVLA-Latent-4B](https://modelscope.cn/models/DAMO_Academy/RynnVLA-Latent-4B) |

The 2B / 4B checkpoints are **Stage-1 latent-action pretraining** exports: they generate latent actions and serve as the initialization for Stage-2 embodiment alignment. They are *not* closed-loop control policies — do **not** point LIBERO / RoboTwin / VLABench evaluation at them directly; run Stage-2 post-training first (see [Inference and evaluation](#-inference-and-evaluation)).

## 🌳 Repository layout

```
RynnVLA-Latent/
├── rynnvla/          # VLA package: api/, models/, datasets/, training/, utils/, configs/
├── rynnlam/          # RynnLAM package: model, modules/ (encoder + flow decoder), inference, configs/
├── scripts/          # launchers, smoke.sh, sample generators, latent manifest/index/stats builders
├── configs/          # *.example.json data-mixture templates
├── docs/             # detailed data pipeline & advanced training guides
├── data/             # bundled synthetic sample corpora (data/sample, data/sample_lam)
├── tests/            # CPU-only suite: pytest tests/ (tests/rynnlam/ covers the RynnLAM side)
├── requirements.txt  # pinned reference environment
├── NOTICE            # third-party attribution
└── README.md         # this file
```

`import rynnvla` and `import rynnlam` both resolve after an editable install. The two sample corpora under `data/` are the only data that ships; everything else there is downloaded or generated. See [Quickstart: the bundled sample](#-quickstart).

## 📦 Installation

Linux, Python ≥ 3.10, CUDA GPU. Create a dedicated environment; do not replace packages in an existing training environment.

```bash
conda create -y -n rynnvla python=3.10 && conda activate rynnvla
pip install -r requirements.txt
```

`requirements.txt` installs a CUDA build of PyTorch, this repository with every optional extra, and a prebuilt flash-attn wheel.

## 🚀 Quickstart

Two tiny **synthetic** corpora ship in `data/` so a fresh clone trains immediately, with no download, no credentials and no external weights:

| Path | Format | Consumed by | Contents |
| --- | --- | --- | --- |
| `data/sample/` | Stage-1 latent npz + mp4 + index + stats | `rynnvla.api.train` | 3 episodes, 24 latents each, 608-dim |
| `data/sample_lam/` | preprocessed per-scene safetensors | `scripts/train_rynnlam.py` | 3 scenes, 12 frames, 56×56 |

```bash
bash scripts/smoke.sh                        # dataset check + RynnLAM Stage-1, no external weights
bash scripts/smoke.sh --src /path/to/RynnBrain-2B   # also runs VLA Stage-1
bash scripts/smoke.sh --only data            # CPU-only dataset read, no GPU
```

## ⚙️ Training

`scripts/train.sh` is a thin wrapper over `python -m rynnvla.api.launch`, which builds the torchrun command for single- or multi-node data-parallel runs. Presets: `stage1` (4B), `stage1_2b`, `stage2` (4B), `stage2_2b`. There is no implicit environment-variable expansion in JSON: supply model, data-mixture and output paths yourself.

Data mixtures are JSON templates in `configs/` (`data_latent_pretrain`, `data_libero_joint40`, `data_robotwin_mixed`, `data_vlabench` — each `.example.json`); copy, edit and populate the referenced data yourself. The bundled `data/sample/sample_mixture.json` is a ready-to-run example exercised by the [Quickstart](#-quickstart); the launcher does not build an index, extract latents or fetch datasets. Formats and build tooling are documented in [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md).

The `stage1` / `stage2` presets differ mainly in initialization source, action chunk (30 vs 10), state input, normalization and LR; the full recipes are `rynnvla/configs/stage1_4b.json` and `stage2_v5_4b.json` and a side-by-side table is in [`docs/ADVANCED_TRAINING.md`](docs/ADVANCED_TRAINING.md).

### Launch Commands

Stage 1 (the output directory must **not** exist; initialize from a RynnBrain backbone):

```bash
python -m rynnvla.api.launch \
  --stage stage1 \
  --model-path models/RynnBrain-4B \
  --data-mixture configs/data_latent_pretrain.local.json \
  --output-dir runs/stage1 \
  --nproc-per-node 8
```

Export the Stage-1 EMA initialization for Stage 2 (create the export parent yourself; the destination must not exist):

```bash
python -m rynnvla.api.export_checkpoint \
  --checkpoint-dir runs/stage1/checkpoint-60000 \
  --output-dir models/stage1-ema \
  --weights ema --format safetensors
```

`--weights ema` requires that checkpoint's `ema_model.bin` and **never** falls back to live weights; `--weights model` explicitly selects the live weights.

Stage 2 — a **new optimizer run** initialized from the exported Stage-1 model, not a resume of Stage 1:

```bash
python -m rynnvla.api.launch \
  --stage stage2 --model-path models/stage1-ema \
  --data-mixture configs/data_libero_joint40.local.json \
  --output-dir runs/stage2-v5 --nproc-per-node 8
```

### Resuming & Overrides

To resume **the same stage**, repeat the original command with `--resume` and the same `--output-dir`; an exported model directory is **not** a resumable checkpoint. Otherwise a fresh launch rejects an existing output directory. Precedence is preset JSON → explicit CLI (`--help` for the flags); fully custom recipes go to torchrun directly. See [`docs/ADVANCED_TRAINING.md`](docs/ADVANCED_TRAINING.md) for resume-critical fields, learning-rate groups and multinode launches.

## 🧠 Latent-Action Model (RynnLAM)

The `rynnlam` package is the self-supervised latent-action model that *produces* the 608-dim `ktoken_zcam` latents consumed at Stage 1. Its training and evaluation entry points are included so the architecture is reproducible end to end. Training needs the DA3-Large encoder weights:

```bash
NPROC_PER_NODE=8 bash scripts/train_rynnlam_stage1.sh --output-dir ./runs/lam
NPROC_PER_NODE=8 bash scripts/train_rynnlam_stage2.sh --resume ./runs/lam/latest.pt
# equivalently:
python scripts/train_rynnlam.py --config rynnlam/configs/stage1.yaml \
    --encoder-checkpoint checkpoints/DA3-LARGE-1.1/model.safetensors
```

To check the whole training path without any external weights (encoder randomly initialized, all five losses, checkpointing and internal eval run):

```bash
python scripts/make_sample_lam_data.py                      # -> data/sample_lam/ (3 scenes)
python scripts/train_rynnlam.py --config rynnlam/configs/smoke.yaml
```

RynnLAM training reads preprocessed per-scene safetensors (RGB, depth, 3D flow, cameras); `scripts/make_sample_lam_data.py` generates a minimal valid corpus and is the shortest specification of that format. Evaluation: `bash scripts/evaluate_rynnlam.sh` provides `extract`, `merge` and `regress` subcommands — `regress --lary-root <your LARYBench checkout>` runs the external LARYBench regression probe (that benchmark is not bundled).

## 🚀 Inference and Evaluation

The supported method is **Stage-1 RynnLAM latent pretraining → full-transfer Stage-2 V5 LIBERO joint40**. Install and configure **upstream LIBERO** (robosuite/MuJoCo, BDDL tasks, offscreen rendering) separately; no simulator is installed by this repository.

```bash
python -m rynnvla.api.export_checkpoint --help   # Trainer ckpt -> HF directory
python -m rynnvla.api.eval_libero       --help   # LIBERO closed-loop evaluation
python -m rynnvla.api.predict           --help   # single-observation offline prediction

python -m rynnvla.api.eval_libero \
  --model-path exports/stage2_v5 \
  --suite libero_spatial --task-id 0 --episodes 1 \
  --seed 7 --output-dir results/libero_spatial_task0 \
  --replan-steps 10 --denoising-steps 10 --save-frames
```

- Point evaluation at an **exported Stage-2 HF directory** — never a Trainer checkpoint and never a Stage-1-only model. Loading is `local_files_only=True`; missing assets are an error, not a download.
- Do not pre-transform observations or rescale actions: the adapter in [`rynnvla/inference_wrappers/rynn_brain_vla.py`](rynnvla/inference_wrappers/rynn_brain_vla.py) owns image orientation, state and rot6d → OSC decoding.
- A single task/episode is a smoke test, **not** a suite score; report checkpoints, suites, seeds and step/replan budgets with any number.
- **RoboTwin and VLABench are supported on the policy side only** — neither simulator, runner nor corpus is redistributed, so no command here produces their numbers on its own.

Closed-loop mechanics, offline replay, and the RoboTwin / VLABench adapters are detailed in [`docs/ADVANCED_TRAINING.md`](docs/ADVANCED_TRAINING.md).

## ⚖️ License & Attribution

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). This project uses Qwen3-VL model and processor interfaces from Hugging Face Transformers; the RynnLAM backbone is DA3/DINOv2-derived and the action expert implements conditional normalization, two-stream action attention and optional auxiliary heads — the full third-party attribution list is in [NOTICE](NOTICE).

Model weights and datasets are separate resources and are not granted by this source license. Not included in this distribution: RynnBrain pretrained weights, the DA3-Large encoder weights, offline RynnLAM latent files, LIBERO datasets and simulator assets, the RoboTwin HDF5 corpus and simulator, and the VLABench simulator tree. Obtain each separately and follow its own license and usage conditions.
