#!/bin/bash
#SBATCH --job-name=collect-policy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=8:00:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-10           # one job per BC task (11 total)
#SBATCH --output=slurm_%A_%a.out

# ---------------------------------------------------------------------------
# BC policy data collection (collect_policy.py) — 50 trajs/task across 11 tasks.
#
# The released policy uses the single canonical view (cam 6, az=136°); set
# CANONICAL_ONLY=False in the script to collect the full camera pool + a 90° OOD view.
# One env per array task, 4 parallel workers. Resume-safe (dedup on resume).
# ---------------------------------------------------------------------------

ENV_LIST=(
    "assembly-v2"           # 0
    "basketball-v2"         # 1
    "button-press-v2"       # 2
    "door-open-v2"          # 3
    "drawer-close-v2"       # 4
    "drawer-open-v2"        # 5
    "lever-pull-v2"         # 6
    "pick-place-v2"         # 7
    "push-v2"               # 8
    "reach-v2"              # 9
    "window-open-v2"        # 10
)

ENV_NAME=${ENV_LIST[$SLURM_ARRAY_TASK_ID]}
echo "Array task $SLURM_ARRAY_TASK_ID → collecting env: $ENV_NAME (50 trajs)"

SEED=$((42 + SLURM_ARRAY_TASK_ID))

nvidia-smi -L
nvidia-smi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

. ~/.bashrc
cd "$SCRIPT_DIR"   # run from this metaworld/ dir (finds vendored utils/ + common/)

conda activate reviwo
export CANON_DATA_ROOT=${CANON_DATA_ROOT:-/path/to/canon_public_data}

export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

python collect_policy.py \
    --expert_epoch 50 \
    --img_size 224 \
    --save_trajectory_path "data/metaworld_policy" \
    --env_names "$ENV_NAME" \
    --seed $SEED \
    --egl_refresh_interval 3 \
    --num_workers 4
