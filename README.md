# Canonical Observation Pretraining for Visuomotor Control

[Website](https://agenticlearning.ai/canonical/) | [Model](https://huggingface.co/agentic-learning-ai-lab/canonical) | [Dataset](https://huggingface.co/datasets/agentic-learning-ai-lab/canonical)

Implementation of ***Canonical***, a self-supervised method that makes visuomotor policies robust to
camera-viewpoint changes by treating viewpoint as a learned rotation in latent space
rather than suppressing it.

## Overview

A viewpoint-equivariant encoder preserves viewpoint as a recoverable latent factor and warps
observations from any camera into a shared canonical feature space.

At test time a lightweight angle-prediction head estimates the camera viewpoint from features
and canonicalizes them before the policy, with no ground-truth camera pose required.

## Repository structure

```
canonical/
├── models/
│   ├── ssl.py                 # CANON_RotationSO3AnglePred_6D_Spatial (MetaWorld), CANON_RotationSO2AnglePred (LIBERO)
│   ├── encoder.py             # resnet18_spatial (MetaWorld), resnet18 (LIBERO)
│   ├── projector.py           # MLP rotation projector
│   └── policy.py              # DiTFlowSpatial (MetaWorld), DiTFlowPolicy (LIBERO)
├── metaworld_bench/            # MetaWorld (name avoids clashing with the pip `metaworld`)
│   ├── configs/               # encoder.yaml, policy.yaml
│   ├── datasets/              # multi_view.py (encoder), single_view.py (policy)
│   ├── envs/                  # thin wrapper over the pip `metaworld` package
│   ├── eval_configs_ablation/ # harder-split eval camera configs (ablations)
│   ├── train_encoder.py
│   └── train_policy.py
├── libero_goal/                # LIBERO-Goal
│   ├── configs/               # encoder.yaml, policy.yaml, policy_vqbet.yaml
│   ├── datasets/              # multi_view.py (encoder), single_view.py (policy)
│   ├── envs/                  # vendored simulator (Canonical-modified spherical multi-view camera)
│   ├── train_encoder.py
│   └── train_policy.py
├── data_generation/            # scripts that render the released datasets
└── utils/                      # trainer, checkpoint loaders (load_encoder / load_policy)
```

## Installation

The two benchmarks use different simulator stacks; install into separate environments.

```bash
# MetaWorld (metaworld is a second --no-deps step so its git gymnasium requirement does not
# override the pinned gymnasium / mujoco 2.3.7):
pip install -r requirements_metaworld.txt
pip install --no-deps git+https://github.com/Farama-Foundation/Metaworld.git@83ac03ca3207c0060112bfc101393ca794ebf1bd

# LIBERO-Goal:
pip install -r requirements_libero.txt
```

LIBERO-Goal also needs its vendored simulator's 3D assets (410MB, not stored in this
repo — see [`libero_goal/envs/NOTICE`](libero_goal/envs/NOTICE) for provenance):

```bash
huggingface-cli download agentic-learning-ai-lab/canonical \
    --repo-type dataset --include "libero_sim_assets/*" \
    --local-dir /tmp/libero_sim_assets
mv /tmp/libero_sim_assets/libero_sim_assets libero_goal/envs/assets
```

## Usage

All commands assume the repository root as the working directory and the data location exported:

```bash
export CANON_DATA_ROOT=/path/to/downloaded/canonical
```

Download the dataset (linked above) and point `CANON_DATA_ROOT` at it.

Every entry point runs a smoke test with `+debug=true` (limits data/epochs); drop it for a full run.

### MetaWorld

```bash
# Encoder pretraining (SO(3)-6D rotation prediction + canonical-warp SpatialTransformer)
python -m metaworld_bench.train_encoder +debug=true

# Policy training (canonical-warp DiTFlowSpatial, self-calibrating angle prediction);
# one job per task: dooropenv2, drawerclosev2, windowopenv2, assemblyv2
python -m metaworld_bench.train_policy --config-name policy \
    dataset.task_name=dooropenv2 +encoder_path=/path/to/snapshot_epoch399.pt +debug=true
```

The encoder checkpoint (`snapshot_epoch399.pt`) holds `encoder` (frozen `resnet18_spatial`),
`view_ssl` (`CANON_RotationSO3AnglePred_6D_Spatial`, angle head + SpatialTransformer), and
`view_projector` (`MLP(6→64)`).

### LIBERO-Goal

```bash
# Encoder pretraining (SO(2) rotation prediction + canonical-warp forward-dynamics transformer)
python -m libero_goal.train_encoder +debug=true

# Policy training (DiTFlow)
python -m libero_goal.train_policy --config-name policy \
    +encoder_path=/path/to/snapshot_epoch39.pt +debug=true

# Policy training (VQ-BeT)
python -m libero_goal.train_policy --config-name policy_vqbet \
    +encoder_path=/path/to/snapshot_epoch39.pt +debug=true
```

Policies train on the single 0° canonical view and are evaluated unchanged at perturbed cameras.

## Pretrained checkpoints

Converted, ready-to-load Canonical weights (both benchmarks, encoder + BC policies) plus DynaMo
reference encoders are available as a model bundle (linked above). Load them with the helpers in
`utils/checkpoint.py`:

```python
from utils.checkpoint import load_encoder, load_policy
enc = load_encoder("metaworld/encoder.pt")        # {encoder, view_ssl, view_projector, epoch}
pol = load_policy("metaworld/policy/dooropenv2.pt")
```

The encoder path is the same one `train_policy` reads via `+encoder_path=`. See the bundle's own
README for the file map and provenance.

## Citation

```bibtex
@inproceedings{canon2026,
  title     = {Canonical Observation Pretraining for Visuomotor Control},
  author    = {Yeung, Yuen-Hei and Hoang, Christopher and Zhao, Zifan and Ren, Mengye},
  year      = {2026},
  note      = {Under review},
}
```

This code builds on [DynaMo](https://github.com/jeffacce/dynamo_ssl).
