#!/usr/bin/env python3
"""
Data collection for the CANON LIBERO-Goal BC-policy dataset.

Replays existing demonstrations (states.npy + actions.npy) and renders them from a
spherical camera. The released policy uses the single 0° canonical (front) view
(CANONICAL_ONLY = True); set CANONICAL_ONLY = False to also render an N-azimuth
perturbed pool.

Output HDF5 structure:
  trajectories.hdf5
  └── <task>/
      ├── traj_0/
      │   ├── image_0       (T, H, W, 3)  uint8, lzf   0° canonical view
      │   ├── state         (T, 79)       float32
      │   ├── action        (T, 7)        float32
      │   └── attrs: selected_camera_ids
      └── traj_1/ ...
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
AZIMUTH_RANGE = [-20.0, 20.0]

# LIBERO frontal view: azimuth=0 is the canonical front (robot facing +X)
FRONT_AZIMUTH = 0
# Elevation and distance matching libero env's _setup_camera_specific_spherical
ELEVATION = 30   # degrees above horizontal
DISTANCE  = 1.3  # metres


def stratified_sample_azimuths(az_range, n, rng):
    """Sample n azimuths via stratified (equal-width bins) sampling within az_range.

    Args:
        az_range: [lo, hi] in degrees; negative values are valid (e.g. [-20, 20]).
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


# ── View assignment helpers ────────────────────────────────────────────────────

def generate_view_assignments(task_names, num_demos, non_front_ids, num_render_views, seed):
    """Pre-generate per-trajectory random paired-view assignments for reproducibility."""
    rng = random.Random(seed)
    assignments = {}
    for task_name in task_names:
        for demo_idx in range(num_demos):
            selected = rng.sample(non_front_ids, num_render_views)
            selected.sort()
            assignments[(task_name, demo_idx)] = selected
    return assignments


# ── HDF5 I/O ──────────────────────────────────────────────────────────────────

def write_trajectory(env_group, traj_idx, trajectory):
    """Write one trajectory into an open HDF5 env group, images compressed with lzf."""
    grp = env_group.create_group(f"traj_{traj_idx}")

    # HDF5 key is image_0 (what libero_goal/datasets/single_view.py reads); the
    # in-memory trajectory dict keeps the render-time key "image_front".
    grp.create_dataset("image_0", data=trajectory["image_front"], compression="lzf")
    if "image_wrist" in trajectory:
        grp.create_dataset("image_wrist", data=trajectory["image_wrist"], compression="lzf")

    for cam_id, frames in trajectory["image"].items():
        grp.create_dataset(f"image_{cam_id}", data=frames, compression="lzf")

    grp.create_dataset("state",  data=np.array(trajectory["state"],  dtype=np.float32))
    grp.create_dataset("action", data=np.array(trajectory["action"], dtype=np.float32))

    if "selected_camera_ids" in trajectory:
        grp.attrs["selected_camera_ids"] = trajectory["selected_camera_ids"]


def save_debug_video(trajectory, video_path, fps=10):
    """Save a 2×2 grid debug video: [front, wrist | view_a, view_b]."""
    T = len(trajectory["state"])
    H, W = trajectory["image_front"][0].shape[:2]
    black = [np.zeros((H, W, 3), dtype=np.uint8)] * T

    paired = [frames for _, frames in sorted(trajectory["image"].items())]
    while len(paired) < 2:
        paired.append(black)

    wrist = trajectory.get("image_wrist", black)

    grid_frames = []
    for t in range(T):
        top = np.concatenate([trajectory["image_front"][t], wrist[t]],    axis=1)
        bot = np.concatenate([paired[0][t],                 paired[1][t]], axis=1)
        grid_frames.append(np.concatenate([top, bot], axis=0))

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
    bddl_file, states, actions, selected_cam_ids,
    img_size, use_wrist_camera, pool_azimuths,
):
    """
    Replay a trajectory and render all requested views simultaneously.

    Uses:
      - 'agentview'            → image_front  (azimuth=FRONT_AZIMUTH via view_angle=)
      - 'view_{cam_id}'        → image[cam_id] (extra cameras via multi_view_camera_configs)
      - 'robot0_eye_in_hand'   → image_wrist   (built-in wrist camera)

    Returns trajectory dict, or None on failure.
    """
    # Build multi_view_camera_configs for extra paired cameras
    multi_view_configs = [
        {
            "name":      f"view_{cid}",
            "azimuth":   pool_azimuths[cid - 1],  # cam_id 1-based
            "elevation": ELEVATION,
            "distance":  DISTANCE,
        }
        for cid in selected_cam_ids
    ]

    camera_names = ["agentview"] + [f"view_{cid}" for cid in selected_cam_ids]
    if use_wrist_camera:
        camera_names.append("robot0_eye_in_hand")

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_names=camera_names,
            camera_heights=img_size,
            camera_widths=img_size,
            view_angle=FRONT_AZIMUTH,
            camera_distance=DISTANCE,
            multi_view_camera_configs=multi_view_configs,
        )
        env.seed(0)
        env.reset()
    except Exception as e:
        print(f"  WARNING: env init failed: {e}")
        return None

    image_front  = []
    image_by_cam = {cid: [] for cid in selected_cam_ids}
    image_wrist  = []

    for mujoco_state in states:
        try:
            obs = env.regenerate_obs_from_state(mujoco_state)
        except Exception as e:
            print(f"  WARNING: regenerate_obs_from_state failed: {e}")
            break

        # Images come out flipped (MuJoCo renders upside-down), flip back
        image_front.append(obs["agentview_image"][::-1].copy())
        for cid in selected_cam_ids:
            image_by_cam[cid].append(obs[f"view_{cid}_image"][::-1].copy())
        if use_wrist_camera:
            image_wrist.append(obs["robot0_eye_in_hand_image"][::-1].copy())

    env.close()

    if len(image_front) != len(states):
        print(f"  WARNING: only got {len(image_front)}/{len(states)} frames")
        return None

    trajectory = {
        "image_front": np.array(image_front,  dtype=np.uint8),
        "image":       {cid: np.array(frames, dtype=np.uint8)
                        for cid, frames in image_by_cam.items()},
        "state":       states.astype(np.float32),
        "action":      actions.astype(np.float32),
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
        description="Collect multi-view LIBERO demonstrations into HDF5 (multiprocessing)."
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
        default="canon_public_data/generated/libero_policy/trajectories.hdf5",
    )
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--view_num", type=int, default=8,
        help="Number of paired-view azimuths to stratified-sample from AZIMUTH_RANGE",
    )
    parser.add_argument(
        "--fixed_azimuths", type=str, default=None,
        help="Comma-separated exact azimuth values (e.g. '25.0') to use as the camera "
             "pool instead of stratified sampling. Overrides --view_num.",
    )
    parser.add_argument(
        "--num_render_views", type=int, default=0,
        help="Paired views per trajectory sampled from pool (0 = all)",
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
    return parser.parse_args()


# ── Worker (must be top-level for spawn pickling) ─────────────────────────────

def _worker_init(asset_path, proj_dir):
    """Pool initializer: propagate ASSET_PATH and sys.path into spawned workers."""
    os.environ["ASSET_PATH"] = asset_path
    if proj_dir not in sys.path:
        sys.path.insert(0, proj_dir)


def collect_demo_worker(work_item):
    """
    Top-level worker: load states/actions from disk, render all views, return trajectory.
    Returns (task_name, demo_idx, demo_name, trajectory_or_None).
    """
    task_name        = work_item["task_name"]
    demo_idx         = work_item["demo_idx"]
    demo_name        = work_item["demo_name"]
    bddl_file        = work_item["bddl_file"]
    states_path      = work_item["states_path"]
    actions_path     = work_item["actions_path"]
    selected_cam_ids = work_item["selected_cam_ids"]
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
        selected_cam_ids, img_size,
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
    save_path.parent.mkdir(parents=True, exist_ok=True)

    asset_path = os.environ.get("ASSET_PATH", project_dir)

    # ── Camera pool ────────────────────────────────────────────────────────────
    # the BC policy uses only the 0° canonical (front) view, which is always
    # rendered separately. So the perturbed-view pool is empty; no useless views are
    # rendered or saved. Set CANONICAL_ONLY=False to restore the perturbed pool.
    CANONICAL_ONLY = True
    if CANONICAL_ONLY:
        pool_azimuths = []
        print("canonical-only: rendering the 0° front view only (empty pool)")
    elif args.fixed_azimuths is not None:
        pool_azimuths = [float(a) for a in args.fixed_azimuths.split(",")]
        print(f"Using fixed azimuths: {pool_azimuths}")
    else:
        assert args.view_num >= 1
        _az_rng       = random.Random(args.seed)
        pool_azimuths = stratified_sample_azimuths(AZIMUTH_RANGE, args.view_num, _az_rng)
        print(f"Pool azimuths ({args.view_num}) from {AZIMUTH_RANGE}: {pool_azimuths}")

    camera_ids    = list(range(len(pool_azimuths) + 1))
    non_front_ids = camera_ids[1:]

    if args.num_render_views > len(non_front_ids):
        raise ValueError(
            f"--num_render_views ({args.num_render_views}) exceeds pool size "
            f"({len(non_front_ids)})"
        )
    use_view_subsampling = 0 < args.num_render_views < len(non_front_ids)

    # ── Task selection ─────────────────────────────────────────────────────────
    task_names = TASK_NAMES
    if args.task_subset:
        task_names = [t.strip() for t in args.task_subset.split(",")]

    num_demos = len(sorted((states_dir / task_names[0]).glob("demo_*")))
    print(f"{len(task_names)} tasks, {num_demos} demos/task, {args.num_workers} workers")

    # ── Early exit if shard already complete ──────────────────────────────────
    if save_path.exists():
        with h5py.File(str(save_path), "r") as f:
            incomplete = [
                t for t in task_names
                if t not in f or len(f[t]) < num_demos
            ]
        if not incomplete:
            print(f"Shard already complete ({num_demos} demos × {len(task_names)} tasks). "
                  f"Exiting: {save_path}")
            sys.exit(0)
        else:
            print(f"Shard incomplete, missing/short tasks: {incomplete}. "
                  f"Deleting and recollecting.")
            save_path.unlink()

    # ── Camera configs ──────────────────────────────────────────────────────
    # camera_configs.json is what libero_goal/datasets/single_view.py actually reads
    # (it looks for "canonical" and "pool" keys); camera_configs.pkl is kept alongside
    # for backward-compatible reference.
    camera_config_dict = {
        "canonical":     {"azimuth": FRONT_AZIMUTH, "elevation": ELEVATION, "distance": DISTANCE},
        "azimuth_range": AZIMUTH_RANGE,
        "seed":          args.seed,
        "pool": [
            {"id": i + 1, "azimuth": az, "elevation": ELEVATION, "distance": DISTANCE}
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

    # ── View assignments ───────────────────────────────────────────────────────
    view_assignments = None
    assignments_path = save_path.parent / "view_assignments.pkl"
    if use_view_subsampling:
        if assignments_path.exists():
            with open(assignments_path, "rb") as f:
                view_assignments = pickle.load(f)
        else:
            view_assignments = generate_view_assignments(
                task_names, num_demos, non_front_ids, args.num_render_views, args.seed
            )
            with open(assignments_path, "wb") as f:
                pickle.dump(view_assignments, f)
            print(f"Generated view assignments: {args.num_render_views} views/traj")

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
                    selected_cam_ids = (
                        view_assignments[(task_name, demo_idx)]
                        if use_view_subsampling else non_front_ids
                    )
                    work_items.append({
                        "task_name":        task_name,
                        "demo_idx":         demo_idx,
                        "demo_name":        demo_path.name,
                        "bddl_file":        bddl_file,
                        "states_path":      str(demo_path / "states.npy"),
                        "actions_path":     str(actions_dir / task_name / demo_path.name / "actions.npy"),
                        "selected_cam_ids": selected_cam_ids,
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

                    if use_view_subsampling:
                        trajectory["selected_camera_ids"] = view_assignments[
                            (task_name_r, demo_idx_r)
                        ]

                    write_trajectory(env_group, n_saved, trajectory)
                    hdf5_file.flush()

                    if not saved_debug:
                        save_debug_video(
                            trajectory,
                            str(save_path.parent / "debug_videos" / f"{task_name}.mp4"),
                        )
                        saved_debug = True

                    n_saved += 1
                    T = len(trajectory["state"])
                    print(f"  [{n_saved}/{len(demos)}] {demo_name}  T={T}  "
                          f"cams={work_items[demo_idx_r]['selected_cam_ids']}")

                print(f"  Done: {n_saved}/{len(demos)} trajs written for {task_name}")
