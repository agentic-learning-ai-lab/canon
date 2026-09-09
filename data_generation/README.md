# Canonical dataset generation

These data-generation scripts are provided as reference and are not currently a supported turnkey
reproduction pipeline. Per-script arguments and the exact camera / HDF5 layout are documented in
each script's own docstring.

```bash
export CANON_DATA_ROOT=/path/to/downloaded/canonical  # wherever you extracted the HF dataset
```

## MetaWorld
The required ReViWo helper modules are vendored under [`metaworld/utils/`](metaworld/utils/) and
[`metaworld/common/`](metaworld/common/). Needs the `metaworld` package + MuJoCo (`MUJOCO_GL=egl`).
Camera configs are bundled in-repo ([`metaworld/camera_configs.json`](metaworld/camera_configs.json));
override with `CANON_MW_CAM_CFG` if you need a different pool.

| dataset | collector | array launcher |
|---|---|---|
| encoder: 255 trajs, 20-view pool | [`metaworld/collect_encoder.py`](metaworld/collect_encoder.py) | [`collect_encoder.sbatch`](metaworld/collect_encoder.sbatch) |
| BC policy: 550 trajs, canonical 136° | [`metaworld/collect_policy.py`](metaworld/collect_policy.py) | [`collect_policy.sbatch`](metaworld/collect_policy.sbatch) |

Array jobs write per-env HDF5 shards; after all encoder tasks finish, merge them with
[`merge_trajectories_metaworld.py`](metaworld/merge_trajectories_metaworld.py) (the policy loader
globs its shards directly, no merge needed).

## LIBERO-Goal
Replays the original LIBERO demonstrations and renders multi-view: pass `--states_dir` /
`--actions_dir` pointing at your own copy of the LIBERO benchmark's demo data, obtained separately
(this repo does not include it). Array jobs write per-shard HDF5s; combine them with
[`merge_trajectories_libero.py`](libero/merge_trajectories_libero.py).

| dataset | collector | array launcher |
|---|---|---|
| encoder: 500 trajs, 2-view pairs | [`libero/collect_encoder.py`](libero/collect_encoder.py) | [`collect_encoder.sbatch`](libero/collect_encoder.sbatch) |
| BC policy: 500 trajs, 0° canonical | [`libero/collect_policy.py`](libero/collect_policy.py) | [`collect_policy.sbatch`](libero/collect_policy.sbatch) |
