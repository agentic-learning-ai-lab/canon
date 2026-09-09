#!/bin/bash
#SBATCH --job-name=collect-encoder
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=12:00:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-16         # one job per environment (17 total) → 17 GPUs in parallel
#SBATCH --output=slurm_%A_%a.out

# ---------------------------------------------------------------------------
# Multi-view SSL encoder data collection (collect_encoder.py).
#
# 17 array tasks × 15 trajs/task = 255 trajectories (one env per array task).
# Camera pool: 20 cameras, elevation [-60,-20]°, distance [1.1,1.5] m, cam 6 canonical
# (az=136°), loaded from this repo's bundled camera_configs.json (deterministic, seed=42).
# Resume-safe: re-submitting skips envs already complete (idempotent per env).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Environment list — must match sorted(set(ALL_ENVIRONMENTS)) in common/utils.py
# ---------------------------------------------------------------------------
ENV_LIST=(
    "assembly-v2"           # 0
    "basketball-v2"         # 1
    "button-press-v2"       # 2
    "coffee-button-v2"      # 3
    "coffee-push-v2"        # 4
    "dial-turn-v2"          # 5
    "door-open-v2"          # 6
    "door-unlock-v2"        # 7
    "drawer-open-v2"        # 8
    "faucet-open-v2"        # 9
    "handle-pull-side-v2"   # 10
    "pick-place-v2"         # 11
    "push-v2"               # 12
    "reach-v2"              # 13
    "soccer-v2"             # 14
    "window-close-v2"       # 15
    "window-open-v2"        # 16
)

ENV_NAME=${ENV_LIST[$SLURM_ARRAY_TASK_ID]}
echo "Array task $SLURM_ARRAY_TASK_ID → collecting env: $ENV_NAME (15 trajs)"

nvidia-smi -L;
nvidia-smi;

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

. ~/.bashrc
cd "$SCRIPT_DIR"   # run from this metaworld/ dir (finds vendored utils/ + common/)

conda activate reviwo;
export CANON_DATA_ROOT=${CANON_DATA_ROOT:-/path/to/canon_public_data}

export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline;


# ---------------------------------------------------------------------------
# SO(3)-6D MetaWorld SSL pretraining — 15 trajs per env, parallel across envs
# ---------------------------------------------------------------------------

python collect_encoder.py \
    --gaussian_expert_epoch 15 \
    --img_size 224 \
    --save_trajectory_path data/metaworld_encoder \
    --env_names "$ENV_NAME"

# Each array task writes one HDF5 shard for its env (trajectories_<env>.hdf5).
# After all array tasks complete, merge shards into the single trajectories.hdf5
# that MetaworldGoalMultiViewDataset reads:
# python merge_trajectories_metaworld.py --data_dir data/metaworld_encoder


# ---------------------------------------------------------------------------
# trajectories_<env>.hdf5  (shard, one per array task; 15 trajs each)
# trajectories.hdf5        (merged output — 17 × 15 = 255 trajs total)
# └── assembly-v2/
#     ├── traj_0/  …  traj_14/
#     │   ├── image_0    (T, 224, 224, 3)  uint8, lzf   az=8°,   el=[-60,-20]°, d=[1.1,1.5]m
#     │   ├── image_1    (T, 224, 224, 3)  uint8, lzf   az=11°
#     │   ├── ...
#     │   ├── image_19   (T, 224, 224, 3)  uint8, lzf   az=351°
#     │   ├── state, action, reward, mujoco_state
#     │   └── attrs: is_success, rand_vec
#     └── ...
#
# camera_configs.json (shared across all tasks; pre-generated):
#   {"pool": [{"id": 0, "azimuth": 8.0, "elevation": ..., "distance": ..., "lookat": [...]},
#             ..., {"id": 19, ...}]}
# Cam 6 entry has the canonical override: az=136°, el=-25°, d=1.3 m.
# ---------------------------------------------------------------------------

# sbatch collect_encoder.sbatch
