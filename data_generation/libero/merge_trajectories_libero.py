"""Merge per-job HDF5 trajectory shards into a single trajectories.hdf5,
and merge per-job pair_assignments_*.pkl shards into a single pair_assignments.pkl.

Run this after all SLURM array jobs finish:
    python merge_trajectories_libero.py --data_dir data/libero_multiview

Each array job writes to:
    trajectories_<N>.hdf5: trajectory data
    pair_assignments_<N>.pkl: (task, demo_idx) → selected camera pair

Both are merged into their respective single-file counterparts.
Use --delete_shards to remove the per-job files after a successful merge.
"""
import argparse
import glob
import os
import pickle
import h5py


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/libero_multiview",
                        help="directory containing the per-job shard files")
    parser.add_argument("--output", type=str, default=None,
                        help="output HDF5 path (default: <data_dir>/trajectories.hdf5)")
    parser.add_argument("--delete_shards", action="store_true",
                        help="delete per-job shard files after successful merge")
    return parser.parse_args()


def merge_hdf5(data_dir, output_path, delete_shards):
    """Merge trajectories_*.hdf5 shards → trajectories.hdf5."""
    shard_pattern = os.path.join(data_dir, "trajectories_*.hdf5")
    shards = sorted(glob.glob(shard_pattern))

    if not shards:
        print(f"No HDF5 shards found matching: {shard_pattern}")
        raise SystemExit(1)

    print(f"Found {len(shards)} HDF5 shard(s):")
    for s in shards:
        print(f"  {s}")

    with h5py.File(output_path, "a") as dst:
        for shard_path in shards:
            with h5py.File(shard_path, "r") as src:
                for task_name in src:
                    n_src = len(src[task_name])
                    if task_name in dst:
                        n_dst = len(dst[task_name])
                        if n_dst >= n_src and n_dst > 0:
                            # Already have at least as many trajs: skip
                            print(f"  SKIP {task_name} (already has {n_dst} trajs)")
                            continue
                        # Destination group exists but is empty or shorter: replace it
                        print(f"  REPLACE {task_name} (dst={n_dst}, src={n_src})")
                        del dst[task_name]
                    src.copy(task_name, dst)
                    n_trajs = len(dst[task_name])
                    print(f"  Merged {task_name}: {n_trajs} trajectories")

    print(f"\nAll tasks merged into: {output_path}")

    if delete_shards:
        for shard_path in shards:
            os.remove(shard_path)
            print(f"  Deleted: {shard_path}")


def merge_pair_assignments(data_dir, delete_shards):
    """Merge pair_assignments_*.pkl shards → pair_assignments.pkl.

    Each shard is a dict {(task_name, demo_idx): selected_pair}.
    Shards are combined with dict.update(); later shards win on key collision,
    but shards are designed to be disjoint so this is safe.
    """
    shard_pattern = os.path.join(data_dir, "pair_assignments_*.pkl")
    shards = sorted(glob.glob(shard_pattern))

    if not shards:
        # pair_assignments are optional (v1 datasets don't have them)
        print("No pair_assignments shards found, skipping pkl merge.")
        return

    print(f"\nFound {len(shards)} pair_assignments shard(s):")
    merged = {}
    for shard_path in shards:
        with open(shard_path, "rb") as f:
            data = pickle.load(f)
        merged.update(data)
        print(f"  {os.path.basename(shard_path)}: {len(data)} entries")

    out_path = os.path.join(data_dir, "pair_assignments.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"Merged {len(merged)} entries → {out_path}")

    if delete_shards:
        for shard_path in shards:
            os.remove(shard_path)
            print(f"  Deleted: {os.path.basename(shard_path)}")


if __name__ == "__main__":
    args = get_args()
    output_path = args.output or os.path.join(args.data_dir, "trajectories.hdf5")

    merge_hdf5(args.data_dir, output_path, args.delete_shards)
    merge_pair_assignments(args.data_dir, args.delete_shards)

'''
python merge_trajectories_libero.py \
    --data_dir /path/to/libero_shards_dir \
    --delete_shards
'''
