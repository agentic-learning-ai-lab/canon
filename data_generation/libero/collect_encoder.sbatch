#!/bin/bash
#SBATCH --job-name=collect-encoder
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=8:00:00
# (cluster-specific SLURM settings removed for release; set --qos/--partition/etc. for your scheduler)
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=slurm_%A_%a.out

# ---------------------------------------------------------------------------
# Task subsets — 10 LIBERO-Goal tasks split across 4 array jobs (~2-3 each)
# ---------------------------------------------------------------------------
TASK_SUBSETS=(
    "open_the_middle_drawer_of_the_cabinet,open_the_top_drawer_and_put_the_bowl_inside,push_the_plate_to_the_front_of_the_stove"
    "put_the_bowl_on_the_plate,put_the_bowl_on_the_stove,put_the_bowl_on_top_of_the_cabinet"
    "put_the_cream_cheese_in_the_bowl,put_the_wine_bottle_on_the_rack"
    "put_the_wine_bottle_on_top_of_the_cabinet,turn_on_the_stove"
)

TASK_SUBSET=${TASK_SUBSETS[$SLURM_ARRAY_TASK_ID]}
echo "Array task $SLURM_ARRAY_TASK_ID → collecting tasks: $TASK_SUBSET"

nvidia-smi -L;
nvidia-smi;

. ~/.bashrc;
cd "${CANON_REPO:-/path/to/canonical}"   # repo root; the collector is invoked below by its full path from here

conda activate dynamo-repro;

export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export ASSET_PATH=$(pwd)
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

: "${LIBERO_STATES_DIR:?Set LIBERO_STATES_DIR to your LIBERO demo states directory}"
: "${LIBERO_ACTIONS_DIR:?Set LIBERO_ACTIONS_DIR to your LIBERO demo actions directory}"

# ---------------------------------------------------------------------------
# Collect multi-view encoder data. Each demo renders one view pair (view_i, view_j)
# from the 0-based azimuth pool, assigned by exhaustive round-robin so every pair is
# covered across demos. Each array job writes trajectories_<SLURM_ARRAY_TASK_ID>.hdf5;
# combine the shards with merge_trajectories_libero.py afterwards.
# ---------------------------------------------------------------------------

python data_generation/libero/collect_encoder.py \
    --states_dir "$LIBERO_STATES_DIR" --actions_dir "$LIBERO_ACTIONS_DIR" \
    --task_subset "$TASK_SUBSET" \
    --view_num 20 --azimuth_range=-20,20 --canonical_proportion 0 \
    --save_path "canon_public_data/generated/libero_encoder/trajectories_${SLURM_ARRAY_TASK_ID}.hdf5"

# After all array tasks complete, merge shards:
# python merge_trajectories_libero.py --data_dir /path/to/libero_demo_actions --delete_shards