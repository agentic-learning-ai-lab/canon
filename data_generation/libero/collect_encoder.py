#!/usr/bin/env python3
"""
Multi-view data collection for the CANON LIBERO-Goal encoder dataset (0-based camera pool).

Configuration (pure 0-based camera pool):
  - No separate front/canonical camera. The pool is `--view_num` azimuths sampled over
    `--azimuth_range`, with 0-based camera ids 0..N-1 (cam_id i renders pool_azimuths[i]).
  - The near-0° pool view is the canonical anchor: it is RENDERED at its sampled ~0.44°,
    and its recorded azimuth is RELABELED to exactly 0.0° in the camera metadata only.
  - Each trajectory renders exactly two pool views: a general pair (view_i, view_j), i < j.
  - Pair assignment uses exhaustive round-robin over all general pairs (shuffled once with
    the global seed), so every azimuth pair is covered across demos.

Output HDF5 structure:
  trajectories.hdf5
  └── open_the_middle_drawer_of_the_cabinet/
      ├── traj_0/
      │   ├── image_<i>          (T, H, W, 3)  uint8, lzf  [pool cam, 0-based id]
      │   ├── image_<j>          (T, H, W, 3)  uint8, lzf  [pool cam, 0-based id]
      │   ├── state              (T, 79)        float32
      │   ├── action             (T, 7)         float32
      │   └── attrs: pair_cam_ids, is_canonical_pair (always False in the release path)
      └── traj_1/ ...

Usage (single task subset, e.g. from SLURM array):
    MUJOCO_GL=egl ASSET_PATH=$(pwd) python collect_encoder.py \\
        --task_subset "open_the_middle_drawer_of_the_cabinet,put_the_bowl_on_the_plate" \\
        --view_num 20 --azimuth_range=-20,20 --canonical_proportion 0
"""
import os
import sys
import json
import h5py
import pickle
import random
import argparse
import numpy as np
import imageio
from pathlib import Path

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)
# LIBERO sim lives at <repo root>/libero_goal/envs; add repo root so it's importable
# as libero_goal.envs regardless of the collector's own working directory.
repo_root = os.path.dirname(os.path.dirname(project_dir))
sys.path.insert(0, repo_root)

from libero_goal.envs import benchmark, get_libero_path
from libero_goal.envs.envs.env_wrapper import OffScreenRenderEnv

# ── Camera parameters ──────────────────────────────────────────────────────────
# Valid paired-view azimuth range for LIBERO (centred on frontal view at 0°).
# [20, 40] is reserved for BC policy training / evaluation.
AZIMUTH_RANGE = [-20.0, 20.0]   # released encoder pool spans ±20° around the 0° canonical

# Canonical view: azimuth=0 is the canonical front (robot facing +X).
# Camera id 0 is reserved for the canonical view across all pair assignments.
FRONT_AZIMUTH = 0
# Elevation and distance matching libero env's _setup_camera_specific_spherical
ELEVATION = 30   # degrees above horizontal
DISTANCE  = 1.3  # metres


def stratified_sample_azimuths(az_range, n, rng):
    """Sample n azimuths via stratified (equal-width bins) sampling within az_range.

    No snap-to-zero here: the near-0° bin comes out at its sampled value (~0.44° for the
    released seed). The camera is RENDERED at that sampled value; the canonical 0° anchor
    is applied later as a METADATA-ONLY relabel of that view's recorded azimuth to 0.0°
    (see the caller's `_zero_idx` handling). This matches how the released data was
    produced (rendered ~0.44°, recorded 0.0°).

    Args:
        az_range: [lo, hi] in degrees.
        n:        number of azimuths to sample.
        rng:      random.Random instance for reproducibility.

    Returns:
        Sorted list of n float azimuths.
    """
    lo, hi = az_range
    bin_width = (hi - lo) / n
    azimuths = []
    for i in range(n):
        azimuths.append(round(rng.uniform(lo + i * bin_width, lo + (i + 1) * bin_width), 2))
    return sorted(azimuths)


TASK_NAMES = [
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
]


# ── Pair assignment ────────────────────────────────────────────────────────────

def generate_pair_assignments_roundrobin(
    task_names, num_demos, pool_ids, canonical_proportion, seed
):
    """Assign one view pair per (task, demo) using exhaustive round-robin lists.

    Canonical pair : (v_n, 0)   for each v_n in pool_ids  [v1=pool → v2=FRONT_AZIMUTH]
    General  pair  : (v_i, v_j) for all i < j in pool_ids

    For each task, round(num_demos * canonical_proportion) demos are randomly
    designated as canonical.  Both pair lists are shuffled once and then assigned
    round-robin (counters shared across tasks for maximal variety).

    Returns:
        dict {(task_name, demo_idx): (cam_id_a, cam_id_b)}
    """
    rng = random.Random(seed)

    canonical_pairs = [(v, 0) for v in sorted(pool_ids)]
    general_pairs = [
        (pool_ids[i], pool_ids[j])
        for i in range(len(pool_ids))
        for j in range(i + 1, len(pool_ids))
    ]

    rng.shuffle(canonical_pairs)
    rng.shuffle(general_pairs)

    # If no general pairs exist (single pool cam), force all demos to canonical.
    if not general_pairs:
        canonical_proportion = 1.0
    # If no canonical pairs exist (shouldn't happen with cam_id=0 always present),
    # force all demos to general.
    if not canonical_pairs:
        canonical_proportion = 0.0

    n_canonical = round(num_demos * canonical_proportion)

    assignments = {}
    c_rr = 0  # canonical round-robin index, shared across tasks
    g_rr = 0  # general  round-robin index, shared across tasks

    for task_name in task_names:
        # Randomly pick which demo indices in this task are canonical
        demo_indices = list(range(num_demos))
        rng.shuffle(demo_indices)
        canonical_set = set(demo_indices[:n_canonical])

        for demo_idx in range(num_demos):
            if demo_idx in canonical_set:
                assignments[(task_name, demo_idx)] = canonical_pairs[c_rr % len(canonical_pairs)]
                c_rr += 1
            else:
                assignments[(task_name, demo_idx)] = general_pairs[g_rr % len(general_pairs)]
                g_rr += 1

    return assignments


# ── HDF5 I/O ──────────────────────────────────────────────────────────────────

def write_trajectory(env_group, traj_idx, trajectory):
    """Write one trajectory into an open HDF5 env group, images compressed with lzf."""
    grp = env_group.create_group(f"traj_{traj_idx}")

    if "image_wrist" in trajectory:
        grp.create_dataset("image_wrist", data=trajectory["image_wrist"], compression="lzf")

    for cam_id, frames in trajectory["image"].items():
        grp.create_dataset(f"image_{cam_id}", data=frames, compression="lzf")

    grp.create_dataset("state",  data=np.array(trajectory["state"],  dtype=np.float32))
    grp.create_dataset("action", data=np.array(trajectory["action"], dtype=np.float32))

    if "pair_cam_ids" in trajectory:
        grp.attrs["pair_cam_ids"] = trajectory["pair_cam_ids"]
    if "is_canonical_pair" in trajectory:
        grp.attrs["is_canonical_pair"] = trajectory["is_canonical_pair"]


def save_debug_video(trajectory, video_path, fps=10):
    """Save a 2-wide (or 2x2 with wrist) debug MP4 of the two rendered views."""
    T = len(trajectory["state"])
    cam_ids = sorted(trajectory["image"].keys())
    views = [trajectory["image"][cid] for cid in cam_ids]
    H, W = views[0][0].shape[:2]
    black = np.zeros((H, W, 3), dtype=np.uint8)

    while len(views) < 2:
        views.append([black] * T)

    has_wrist = "image_wrist" in trajectory
    grid_frames = []
    for t in range(T):
        top = np.concatenate([views[0][t], views[1][t]], axis=1)
        if has_wrist:
            bot = np.concatenate([trajectory["image_wrist"][t], black], axis=1)
            frame = np.concatenate([top, bot], axis=0)
        else:
            frame = top
        grid_frames.append(frame)

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    try:
        imageio.mimwrite(video_path, grid_frames, fps=fps)
        print(f"  Debug video saved: {video_path}")
    except Exception as e:
        print(f"  WARNING: could not save debug video: {e}")


# ── Environment helpers ────────────────────────────────────────────────────────

def get_bddl_file(task_suite, task_name):
    task_id = task_suite.get_task_names().index(task_name)
    task = task_suite.get_task(task_id)
    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)


def collect_trajectory(
    bddl_file, states, actions, selected_pair,
    img_size, use_wrist_camera, pool_azimuths,
):
    """
    Replay a trajectory and render the assigned view pair.

    selected_pair: (cam_id_a, cam_id_b)
      cam_id i → pool camera at pool_azimuths[i]   (0-based). This uses a pure
      pool (canonical_proportion=0), so there is no separate front/canonical camera;
      the ~0° pool view is the canonical anchor and is recorded as 0.0° in the metadata.

    All cameras are registered via multi_view_camera_configs; agentview is not used.
    Returns trajectory dict, or None on failure.
    """
    def cam_azimuth(cid):
        return pool_azimuths[cid]

    multi_view_configs = [
        {
            "name":      f"view_{cid}",
            "azimuth":   cam_azimuth(cid),
            "elevation": ELEVATION,
            "distance":  DISTANCE,
        }
        for cid in selected_pair
    ]

    camera_names = [f"view_{cid}" for cid in selected_pair]
    if use_wrist_camera:
        camera_names.append("robot0_eye_in_hand")

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_names=camera_names,
            camera_heights=img_size,
            camera_widths=img_size,
            multi_view_camera_configs=multi_view_configs,
        )
        env.seed(0)
        env.reset()
    except Exception as e:
        print(f"  WARNING: env init failed: {e}")
        return None

    image_by_cam = {cid: [] for cid in selected_pair}
    image_wrist  = []

    for mujoco_state in states:
        try:
            obs = env.regenerate_obs_from_state(mujoco_state)
        except Exception as e:
            print(f"  WARNING: regenerate_obs_from_state failed: {e}")
            break

        for cid in selected_pair:
            image_by_cam[cid].append(obs[f"view_{cid}_image"][::-1].copy())
        if use_wrist_camera:
            image_wrist.append(obs["robot0_eye_in_hand_image"][::-1].copy())

    env.close()

    first_cid = selected_pair[0]
    if len(image_by_cam[first_cid]) != len(states):
        print(f"  WARNING: only got {len(image_by_cam[first_cid])}/{len(states)} frames")
        return None

    trajectory = {
        "image":  {cid: np.array(frames, dtype=np.uint8) for cid, frames in image_by_cam.items()},
        "state":  states.astype(np.float32),
        "action": actions.astype(np.float32),
    }
    if use_wrist_camera and image_wrist:
        trajectory["image_wrist"] = np.array(image_wrist, dtype=np.uint8)

    return trajectory


# ── Main ──────────────────────────────────────────────────────────────────────


import multiprocessing
from multiprocessing import Pool


# ── Entrypoint: multiprocessing collection (self-contained) ──────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Collect multi-view LIBERO demonstrations into HDF5 (v2, multiprocessing)."
    )
    parser.add_argument(
        "--states_dir", type=str,
        default="/path/to/libero_demo_states",
    )
    parser.add_argument(
        "--actions_dir", type=str,
        default="/path/to/libero_demo_actions",
    )
    parser.add_argument(
        "--save_path", type=str,
        default="canon_public_data/generated/libero_encoder/trajectories.hdf5",
    )
    parser.add_argument("--img_size", type=int, default=224)   # render at 224 (no post-resize)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--view_num", type=int, default=20,
        help="Number of pool azimuths to stratified-sample from AZIMUTH_RANGE (20 in the released dataset)",
    )
    parser.add_argument(
        "--fixed_azimuths", type=str, default=None,
        help="Comma-separated exact azimuth values to use as the pool. Overrides --view_num.",
    )
    parser.add_argument(
        "--canonical_proportion", type=float, default=0.0,
        help="Fraction of demos assigned a canonical pair; 0.0 in the released dataset (pure pool).",
    )
    parser.add_argument("--use_wrist_camera", action="store_true")
    parser.add_argument(
        "--task_subset", type=str, default=None,
        help="Comma-separated task names to collect (default: all 10 tasks)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--azimuth_range", type=str, default=None,
        help="Comma-separated lo,hi in degrees (e.g. '-30,30'). Overrides the AZIMUTH_RANGE constant.",
    )
    return parser.parse_args()


# ── Worker (must be top-level for spawn pickling) ─────────────────────────────

def _worker_init(asset_path, proj_dir):
    """Pool initializer: propagate ASSET_PATH and sys.path into spawned workers."""
    os.environ["ASSET_PATH"] = asset_path
    if proj_dir not in sys.path:
        sys.path.insert(0, proj_dir)


def collect_demo_worker(work_item):
    """
    Top-level worker: load states/actions from disk, render the assigned pair, return trajectory.
    Returns (task_name, demo_idx, demo_name, trajectory_or_None).
    """
    task_name        = work_item["task_name"]
    demo_idx         = work_item["demo_idx"]
    demo_name        = work_item["demo_name"]
    bddl_file        = work_item["bddl_file"]
    states_path      = work_item["states_path"]
    actions_path     = work_item["actions_path"]
    selected_pair    = work_item["selected_pair"]
    img_size         = work_item["img_size"]
    use_wrist_camera = work_item["use_wrist_camera"]
    pool_azimuths    = work_item["pool_azimuths"]

    if not os.path.exists(states_path):
        print(f"  [{task_name}/{demo_name}] WARNING: missing states, skipping")
        return task_name, demo_idx, demo_name, None
    if not os.path.exists(actions_path):
        print(f"  [{task_name}/{demo_name}] WARNING: missing actions, skipping")
        return task_name, demo_idx, demo_name, None

    states  = np.load(states_path)
    actions = np.load(actions_path)

    trajectory = collect_trajectory(
        bddl_file, states, actions,
        selected_pair, img_size,
        use_wrist_camera, pool_azimuths,
    )
    return task_name, demo_idx, demo_name, trajectory


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle
    import random

    multiprocessing.set_start_method("spawn")

    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    states_dir  = Path(args.states_dir)
    actions_dir = Path(args.actions_dir)
    save_path   = Path(args.save_path)
    # Safety: this collector deletes/overwrites its save_path; never let it touch the
    # protected original datasets. Only write inside the release tree (canon_public_data).
    assert "canon_public" in str(save_path.resolve()), (
        f"save_path must live under canon_public_data/ (got {save_path}); refusing to "
        f"write outside the release tree to protect the original datasets."
    )
    # path is a pure 0-based camera pool (ids 0..N-1, one of which is the ~0°
    # canonical anchor). There is no separate front/canonical camera, so canonical pairs
    # are not supported here: the renderer maps cam_id i -> pool_azimuths[i].
    assert args.canonical_proportion == 0, (
        "release collector uses a pure 0-based pool; run with --canonical_proportion 0"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    asset_path = os.environ.get("ASSET_PATH", project_dir)

    # ── Camera pool ────────────────────────────────────────────────────────────
    az_range = [float(x) for x in args.azimuth_range.split(",")] if args.azimuth_range else AZIMUTH_RANGE
    _zero_idx = None   # pool view whose recorded azimuth is snapped to 0° (canonical anchor)
    if args.fixed_azimuths is not None:
        pool_azimuths = [float(a) for a in args.fixed_azimuths.split(",")]
        print(f"Using fixed azimuths: {pool_azimuths}")
    else:
        assert args.view_num >= 2, "--view_num must be >= 2"
        _az_rng       = random.Random(args.seed)
        pool_azimuths = stratified_sample_azimuths(az_range, args.view_num, _az_rng)
        print(f"Pool azimuths ({args.view_num}) from {az_range}: {pool_azimuths}")
        # ── Explicit 0° canonical anchor: GENUINELY NECESSARY, not cosmetic. ─────────────
        # CANON's SSL angle supervision measures each view's azimuth RELATIVE to a 0°
        # canonical (theta = azimuth - 0; a view is canonical iff theta == 0). The seeded
        # stratified sample NEVER lands exactly on 0° (the bin around 0° comes out ~0.44°)
        # so without this step there is no exact-0° reference and the canonicalization has no
        # anchor. We therefore RELABEL the single near-0° pool view's recorded azimuth to
        # exactly 0.0° (done in the camera_configs write below). The camera is still RENDERED
        # at its sampled ~0.44° (visually indistinguishable from 0°); only the recorded
        # azimuth is snapped: reproducing exactly how the released dataset was generated.
        _zero_idx = min(range(len(pool_azimuths)), key=lambda i: abs(pool_azimuths[i]))
        print(f"  canonical anchor: pool view {_zero_idx} rendered at "
              f"{pool_azimuths[_zero_idx]:.2f}°, recorded as 0.0°")

    pool_ids = list(range(len(pool_azimuths)))   # 0-based camera ids (0..N-1)

    n_canonical_pairs = len(pool_ids)
    n_general_pairs   = len(pool_ids) * (len(pool_ids) - 1) // 2
    print(f"Canonical pairs: {n_canonical_pairs}  |  General pairs: {n_general_pairs}")

    # ── Task selection ─────────────────────────────────────────────────────────
    task_names = TASK_NAMES
    if args.task_subset:
        task_names = [t.strip() for t in args.task_subset.split(",")]

    num_demos = len(sorted((states_dir / task_names[0]).glob("demo_*")))

    # ── Early exit if shard already complete ───────────────────────────────────
    if save_path.exists():
        with h5py.File(str(save_path), "r") as f:
            incomplete = [
                t for t in task_names
                if t not in f or len(f[t]) < num_demos
            ]
        if not incomplete:
            print(f"Shard already complete ({num_demos} demos × {len(task_names)} tasks). Exiting: {save_path}")
            sys.exit(0)
        else:
            print(f"Shard incomplete, missing/short tasks: {incomplete}. Deleting and recollecting.")
            save_path.unlink()
    n_canon_per_task   = round(num_demos * args.canonical_proportion)
    n_general_per_task = num_demos - n_canon_per_task
    print(
        f"{len(task_names)} tasks, {num_demos} demos/task, {args.num_workers} workers  "
        f"({n_canon_per_task} canonical, {n_general_per_task} general)"
    )

    # ── Camera configs ──────────────────────────────────────────────────────
    # camera_configs.json is what libero_goal/datasets/multi_view.py actually reads;
    # camera_configs.pkl is kept alongside for backward-compatible reference.
    camera_config_dict = {
        "canonical":           {"azimuth": FRONT_AZIMUTH, "elevation": ELEVATION, "distance": DISTANCE},
        "azimuth_range":       az_range,  # = args.azimuth_range if provided, else AZIMUTH_RANGE constant
        # "azimuth_range":       AZIMUTH_RANGE,
        "canonical_proportion": args.canonical_proportion,
        "seed":                args.seed,
        "pool": [
            {"id": i, "azimuth": (0.0 if i == _zero_idx else az),
             "elevation": ELEVATION, "distance": DISTANCE}   # near-0° view relabeled 0.0° (canonical)
            for i, az in enumerate(pool_azimuths)
        ],
    }
    camera_configs_json_path = save_path.parent / "camera_configs.json"
    if not camera_configs_json_path.exists():
        with open(camera_configs_json_path, "w") as f:
            json.dump(camera_config_dict, f, indent=2, default=float)
        print(f"Saved camera configs → {camera_configs_json_path}")
    camera_configs_path = save_path.parent / "camera_configs.pkl"
    if not camera_configs_path.exists():
        with open(camera_configs_path, "wb") as f:
            pickle.dump(camera_config_dict, f)
        print(f"Saved camera configs → {camera_configs_path}")

    # ── Pair assignments ───────────────────────────────────────────────────────
    # Shard-local pkl: named after the HDF5 shard so array jobs don't collide.
    # e.g. trajectories_0.hdf5 → pair_assignments_0.pkl
    assignments_path = save_path.parent / f"pair_assignments_{save_path.stem.split('_')[-1]}.pkl"
    if assignments_path.exists():
        with open(assignments_path, "rb") as f:
            pair_assignments = pickle.load(f)
        print(f"Loaded pair assignments from {assignments_path}")
    else:
        pair_assignments = generate_pair_assignments_roundrobin(
            task_names, num_demos, pool_ids, args.canonical_proportion, args.seed
        )
        with open(assignments_path, "wb") as f:
            pickle.dump(pair_assignments, f)
        print(f"Generated and saved pair assignments → {assignments_path}")

    # ── BDDL task suite ────────────────────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict["libero_goal"]()

    # ── Collection ────────────────────────────────────────────────────────────
    with Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(asset_path, project_dir),
    ) as pool:
        with h5py.File(str(save_path), "a") as hdf5_file:
            for task_name in task_names:
                print(f"\n[Task] {task_name}")

                if task_name in hdf5_file:
                    n_existing = len(hdf5_file[task_name])
                    if n_existing >= num_demos:
                        print(f"  Already complete ({n_existing} trajs). Skipping.")
                        continue
                    print(f"  Partial ({n_existing}/{num_demos}). Delete group to recollect.")
                    continue

                bddl_file = get_bddl_file(task_suite, task_name)
                demos = sorted(
                    (states_dir / task_name).glob("demo_*"),
                    key=lambda p: int(p.name.split("_")[1]),
                )

                # Build work items: paths only, workers load arrays themselves
                work_items = []
                for demo_idx, demo_path in enumerate(demos):
                    work_items.append({
                        "task_name":        task_name,
                        "demo_idx":         demo_idx,
                        "demo_name":        demo_path.name,
                        "bddl_file":        bddl_file,
                        "states_path":      str(demo_path / "states.npy"),
                        "actions_path":     str(actions_dir / task_name / demo_path.name / "actions.npy"),
                        "selected_pair":    pair_assignments[(task_name, demo_idx)],
                        "img_size":         args.img_size,
                        "use_wrist_camera": args.use_wrist_camera,
                        "pool_azimuths":    pool_azimuths,
                    })

                env_group   = hdf5_file.create_group(task_name)
                saved_debug = False
                n_saved     = 0

                # imap preserves order → traj indices match demo indices
                for task_name_r, demo_idx_r, demo_name, trajectory in pool.imap(
                    collect_demo_worker, work_items, chunksize=1
                ):
                    if trajectory is None:
                        print(f"  WARNING: failed for {demo_name}, skipping")
                        continue

                    selected_pair    = pair_assignments[(task_name_r, demo_idx_r)]
                    # Pure 0-based pool (canonical_proportion=0): every id is a real pool
                    # view, so there is no "canonical pair". id 0 is no longer special.
                    is_canonical     = False
                    trajectory["pair_cam_ids"]     = list(selected_pair)
                    trajectory["is_canonical_pair"] = is_canonical

                    write_trajectory(env_group, n_saved, trajectory)
                    hdf5_file.flush()

                    if not saved_debug:
                        save_debug_video(
                            trajectory,
                            str(save_path.parent / "debug_videos" / f"{task_name}.mp4"),
                        )
                        saved_debug = True

                    n_saved += 1
                    T    = len(trajectory["state"])
                    kind = "canonical" if is_canonical else "general"
                    print(f"  [{n_saved}/{len(demos)}] {demo_name}  T={T}  "
                          f"pair={selected_pair}  ({kind})")

                print(f"  Done: {n_saved}/{len(demos)} trajs written for {task_name}")
