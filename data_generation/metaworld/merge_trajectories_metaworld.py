"""Merge per-env HDF5 trajectory shards into a single trajectories.hdf5.

Run this after all SLURM array jobs finish:
    python merge_trajectories_metaworld.py --data_dir data/metaworld_encoder

Each array task (one env per task, via --env_names) writes:
    trajectories_<env>.hdf5: that env's trajectory data

All shards are merged into a single trajectories.hdf5, the file
MetaworldGoalMultiViewDataset actually reads.

Use --delete_shards to remove the per-env files after a successful merge.
"""
import argparse
import glob
import os
import h5py


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/metaworld_encoder",
                        help="directory containing the per-env shard files")
    parser.add_argument("--output", type=str, default=None,
                        help="output HDF5 path (default: <data_dir>/trajectories.hdf5)")
    parser.add_argument("--delete_shards", action="store_true",
                        help="delete per-env shard files after successful merge")
    return parser.parse_args()


def merge_hdf5(data_dir, output_path, delete_shards):
    """Merge trajectories_*.hdf5 shards -> trajectories.hdf5."""
    shard_pattern = os.path.join(data_dir, "trajectories_*.hdf5")
    shards = sorted(glob.glob(shard_pattern))
    # The merge output itself matches the shard glob once created; never re-merge it.
    shards = [s for s in shards if os.path.basename(s) != "trajectories.hdf5"]

    if not shards:
        print(f"No HDF5 shards found matching: {shard_pattern}")
        raise SystemExit(1)

    print(f"Found {len(shards)} HDF5 shard(s):")
    for s in shards:
        print(f"  {s}")

    with h5py.File(output_path, "a") as dst:
        for shard_path in shards:
            with h5py.File(shard_path, "r") as src:
                for env_name in src:
                    n_src = len(src[env_name])
                    if env_name in dst:
                        n_dst = len(dst[env_name])
                        if n_dst >= n_src and n_dst > 0:
                            print(f"  SKIP {env_name} (already has {n_dst} trajs)")
                            continue
                        print(f"  REPLACE {env_name} (dst={n_dst}, src={n_src})")
                        del dst[env_name]
                    src.copy(env_name, dst)
                    n_trajs = len(dst[env_name])
                    print(f"  Merged {env_name}: {n_trajs} trajectories")

    print(f"\nAll envs merged into: {output_path}")

    if delete_shards:
        for shard_path in shards:
            os.remove(shard_path)
            print(f"  Deleted: {shard_path}")


if __name__ == "__main__":
    args = get_args()
    output_path = args.output or os.path.join(args.data_dir, "trajectories.hdf5")
    merge_hdf5(args.data_dir, output_path, args.delete_shards)
