"""Multi-view BC policy data collection for the CANON MetaWorld policy dataset.

The released policy trains on the single canonical view (cam 6, azimuth 136°), so only that
view is rendered and stored (CANONICAL_ONLY = True). Set CANONICAL_ONLY = False to instead
collect the full 21-camera pool plus a 90° OOD side view.

21-camera setup (used when CANONICAL_ONLY = False):

  cam_id  azimuth    condition
  ------  -------    ---------
  0..19   ReViWo 20-view encoder pool   policy-multiview (random view per step)
  6       136°                          policy-136 canonical (in encoder pool)
  20      90°                           policy-90 OOD (excluded band [45,134])
  wrist   gripperPOV                    always included

Camera configs are read from this repo's bundled camera_configs.json (override with
CANON_MW_CAM_CFG; accepts the {canonical, pool} dict or a plain list). args.seed only
affects rand_vecs (env placement), not camera geometry.

EGL refresh: env is destroyed and recreated every --egl_refresh_interval episode
attempts (including failed ones, default=3) to prevent MuJoCo EGL framebuffer exhaustion.

HDF5 layout:
  trajectories_<env>.hdf5  (shard per array task)
  trajectories.hdf5         (merged)
  └── push-v2/
      ├── traj_0/
      │   ├── image_0..image_19   (T, 256, 256, 3) uint8 lzf  ← encoder pool
      │   ├── image_20            (T, 256, 256, 3) uint8 lzf  ← 90° OOD
      │   ├── image_wrist         (T, 256, 256, 3) uint8 lzf
      │   ├── state               (T, 39)          float32
      │   ├── action              (T, 4)            float32
      │   ├── reward              (T,)              float32
      │   ├── mujoco_state        (T, nq+nv)        float32
      │   └── attrs: is_success, rand_vec
      └── traj_1/ ...

  camera_configs.json: [{id, azimuth, elevation, distance, lookat}, ...]
                       ids 0-19 = encoder pool, id 20 = 90° OOD
"""
import numpy as np
import argparse
import os
import pickle
import json
import gymnasium as gym
import sys
import imageio
import h5py
import random
import math
import multiprocessing

project_dir = str(os.path.dirname(__file__))  # this metaworld/ dir -> vendored utils/ + common/
sys.path.insert(0, project_dir)
from utils.collect_expert_dataset import collect_multi_view_episode_v2 as collect_episode
from gymnasium.envs import register
from metaworld import Task, policies
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from utils.evaluate import EvalMetricTracker
from utils.metaworld_env import CameraConfig
from common.utils import ALL_ENVIRONMENTS

for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
    register(id=f"mw_{env_name}",
             entry_point="utils.metaworld_env:SawyerEnv4MultiCameraConfig",
             kwargs={"env_name": env_name})


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

REVIWO_AZIMUTHS = [8, 11, 17, 20, 28, 35, 136, 144, 148, 152, 159, 180,
                   194, 195, 198, 219, 324, 333, 339, 351]
OOD_AZIMUTH  = 90       # cam_id=20; excluded from encoder training ([45,134] band)
CAM_136_ID   = 6        # REVIWO_AZIMUTHS.index(136) → policy-136 condition
CAM_OOD_ID   = 20       # 90° OOD camera id → policy-90 condition

# Camera layout = the released dataset's own encoder camera_configs.json (single source;
# BC renders only the canonical cam 6 = 136°), shipped in this repo so collection works
# without an existing dataset download. Override with CANON_MW_CAM_CFG for a different layout.
POLICY_CAM_CFG_PATH = os.environ.get(
    "CANON_MW_CAM_CFG",
    os.path.join(project_dir, "camera_configs.json"),
)

# Four cameras shown in debug video: spread across encoder pool + OOD
DEBUG_CAM_INDICES = [0, 6, 10, 20]  # az ≈ 8°, 136°, 198°, 90°


# the BC policy trains on the single canonical view (cam 6, az=136°),
# so only cam 6 is rendered + stored. (The original run collected the full 21-cam pool;
# set CANONICAL_ONLY=False to restore that behaviour.)
CANONICAL_ONLY = True
CANONICAL_CAM_ID = 6

def make_camera_configs():
    """Load the camera configs from POLICY_CAM_CFG_PATH (plain list JSON).

    Keeps only the canonical cam 6 (136°) when CANONICAL_ONLY; the BC policy uses just this view.
    """
    with open(POLICY_CAM_CFG_PATH) as f:
        raw = json.load(f)
    cameras = raw if isinstance(raw, list) else raw["pool"]   # accept {canonical, pool} or a flat list
    canonical = raw.get("canonical") if isinstance(raw, dict) else None

    cam_ids, cam_cfgs = [], []
    for entry in sorted(cameras, key=lambda x: x["id"]):
        if CANONICAL_ONLY and int(entry["id"]) != CANONICAL_CAM_ID:
            continue                          # canonical view only
        cfg           = CameraConfig()
        cfg.azimuth   = entry["azimuth"]
        cfg.elevation = entry["elevation"]
        cfg.distance  = entry["distance"]
        cfg.lookat    = np.array(entry["lookat"])
        cam_ids.append(int(entry["id"]))
        cam_cfgs.append(cfg)
    return cam_ids, cam_cfgs, canonical


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_trajectory(env_group, traj_idx, trajectory, rand_vec, target_pos=None):
    """Write one trajectory. Stores image_{cam_id} per configured camera
    (only the canonical cam 6, image_6, when CANONICAL_ONLY); wrist separate."""
    grp = env_group.create_group(f'traj_{traj_idx}')

    for cam_id, frames in trajectory['image'].items():
        grp.create_dataset(f'image_{cam_id}', data=frames, compression='lzf')
    if 'image_wrist' in trajectory:
        grp.create_dataset('image_wrist', data=trajectory['image_wrist'], compression='lzf')

    grp.create_dataset('state',        data=np.array(trajectory['state'],        dtype=np.float32))
    grp.create_dataset('action',       data=np.array(trajectory['action'],       dtype=np.float32))
    grp.create_dataset('reward',       data=np.array(trajectory['reward'],       dtype=np.float32))
    grp.create_dataset('mujoco_state', data=np.array(trajectory['mujoco_state'], dtype=np.float32))

    grp.attrs['is_success'] = bool(trajectory['is_success'])
    grp.attrs['rand_vec']   = np.array(rand_vec, dtype=np.float32)
    if target_pos is not None:
        grp.attrs['target_pos'] = np.array(target_pos, dtype=np.float32)


# ---------------------------------------------------------------------------
# Debug video: 2×2 grid: cam 0 (8°), cam 6 (136°), cam 20 (90° OOD), wrist
# ---------------------------------------------------------------------------

def save_debug_video(trajectory, cam_ids, video_path, fps=10):
    if len(cam_ids) <= max(DEBUG_CAM_INDICES[:3]):
        return                                # canonical-only: too few cams for the grid
    debug_ids = [cam_ids[i] for i in DEBUG_CAM_INDICES[:3]]
    imgs = [trajectory['image'][cid] for cid in debug_ids]
    T    = len(imgs[0])
    H, W = imgs[0][0].shape[:2]
    black = np.zeros((H, W, 3), dtype=np.uint8)

    has_wrist = 'image_wrist' in trajectory
    grid_frames = []
    for t in range(T):
        top    = np.concatenate([imgs[0][t], imgs[1][t]], axis=1)
        bot_r  = trajectory['image_wrist'][t] if has_wrist else black
        bottom = np.concatenate([imgs[2][t], bot_r], axis=1)
        grid_frames.append(np.concatenate([top, bottom], axis=0))

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    try:
        imageio.mimwrite(video_path, grid_frames, fps=fps)
        azimuths = [REVIWO_AZIMUTHS[i] if i < 20 else OOD_AZIMUTH for i in DEBUG_CAM_INDICES[:3]]
        print(f"  Debug video saved (az={azimuths}° + wrist): {video_path}")
    except Exception as e:
        print(f"  WARNING: Could not save debug video: {e}")


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(env_name, cam_ids, cam_cfgs, args):
    """Create a fresh gym env with a new EGL context."""
    env = gym.make(
        f"mw_{env_name}",
        camera_ids=cam_ids,
        camera_configs=cam_cfgs,
        collect_data=True,
        use_camera=True,
        max_eps_step=args.max_eps_step,
        img_size=args.img_size,
        use_wrist_camera=args.use_wrist_camera,
        wrist_camera_name=args.wrist_camera_name,
        render_interval=args.render_interval,
    )
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)
    return env


# ---------------------------------------------------------------------------
# Parallel collection worker
# ---------------------------------------------------------------------------

def collect_shard_worker(worker_id, shard_path, env_name, n_target, args, cam_ids, cam_cfgs):
    """Collect n_target trajs into shard_path. Runs in a child process."""
    worker_seed = args.seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    if os.path.exists(shard_path):
        try:
            with h5py.File(shard_path, 'r') as f:
                existing = len(f.get(env_name, {}))
            if existing >= n_target:
                print(f"[W{worker_id}] shard complete ({existing}/{n_target}), skipping", flush=True)
                return
        except Exception as e:
            print(f"[W{worker_id}] WARNING: corrupt shard, deleting: {e}", flush=True)
            os.remove(shard_path)

    observable_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-goal-observable"]()
    observable_env._freeze_rand_vec = False
    observable_env.action_space.seed(worker_seed)

    policy_name = "".join([s.capitalize() for s in env_name.split("-")])
    policy_name = policy_name.replace("PegInsert", "PegInsertion")
    policy_name = "Sawyer" + policy_name + "Policy"
    policy = vars(policies)[policy_name]()
    metric_tracker = EvalMetricTracker()

    worker_args = argparse.Namespace(**vars(args))
    worker_args.seed = worker_seed

    with h5py.File(shard_path, 'a') as hdf5_file:
        env_group = hdf5_file.require_group(env_name)
        traj_count = len(env_group)
        n_to_collect = n_target - traj_count
        attempt_count = 0
        saved_debug = (traj_count > 0) or (worker_id > 0)

        # Resume-safety: env.reset(seed=...) gives a deterministic rand_vec
        # sequence, so a worker restart would re-roll the same first N
        # rand_vecs and silently produce duplicates. Track which rand_vecs are
        # already in the shard (and which we collect in this session) and
        # re-roll if env.reset() hands us one we've seen.
        seen_rand_vecs = set()
        for k in env_group.keys():
            seen_rand_vecs.add(tuple(env_group[k].attrs['rand_vec'].tolist()))
        print(f"[W{worker_id}] start: {traj_count}/{n_target} existing trajs, "
              f"{len(seen_rand_vecs)} known rand_vecs", flush=True)

        env = make_env(env_name, cam_ids, cam_cfgs, worker_args)

        for _ in range(n_to_collect):
            success = False
            while not success:
                # Roll env.reset() until we land on a rand_vec we haven't seen.
                while True:
                    obs, _ = env.reset()
                    last_rand_vec = env.unwrapped._last_rand_vec.copy()
                    rv_key = tuple(last_rand_vec.tolist())
                    if rv_key not in seen_rand_vecs:
                        break
                    attempt_count += 1
                    print(f"[W{worker_id}] [DUP-SKIP] attempt={attempt_count}, re-rolling", flush=True)

                data = dict(rand_vec=last_rand_vec, partially_observable=False,
                            env_cls=type(env.unwrapped._env))
                task = Task(env_name=env_name, data=pickle.dumps(data))
                observable_env.set_task(task)

                trajectory = collect_episode(
                    env, observable_env, policy, metric_tracker,
                    epsilon=worker_args.epsilon, noise_type='gaussian',
                    init_obs=obs, must_success=False,
                )
                attempt_count += 1
                success = trajectory['is_success']

                if not success:
                    print(f"[W{worker_id}] [FAIL] attempt={attempt_count}, retrying...", flush=True)

                if (worker_args.egl_refresh_interval > 0
                        and attempt_count % worker_args.egl_refresh_interval == 0):
                    print(f"[W{worker_id}] [EGL REFRESH] attempt={attempt_count}", flush=True)
                    env.close()
                    env = make_env(env_name, cam_ids, cam_cfgs, worker_args)

            seen_rand_vecs.add(rv_key)

            # observable_env was just reset with last_rand_vec via set_task(),
            # so its _target_pos is the correct goal for this trajectory.
            target_pos = observable_env._target_pos.copy()
            write_trajectory(env_group, traj_count, trajectory,
                             rand_vec=last_rand_vec, target_pos=target_pos)
            hdf5_file.flush()

            if not saved_debug:
                save_debug_video(
                    trajectory, cam_ids,
                    os.path.join(args.save_trajectory_path, "debug_videos", f"{env_name}.mp4"),
                )
                saved_debug = True

            traj_count += 1
            print(f"[W{worker_id}] [{traj_count}/{n_target}] "
                  f"T={len(trajectory['state'])}  success=True", flush=True)

        env.close()
        print(f"[W{worker_id}] Done: {traj_count} trajs → {os.path.basename(shard_path)}", flush=True)


def run_parallel_collection(hdf5_path, env_name, total_trajs, num_workers, args, cam_ids, cam_cfgs):
    """Distribute collection across num_workers child processes, then merge shards."""
    existing = 0
    if os.path.exists(hdf5_path):
        try:
            with h5py.File(hdf5_path, 'r') as f:
                existing = len(f.get(env_name, {}))
            if existing >= total_trajs:
                print(f"  Already complete ({existing} trajs). Skipping.")
                return
        except Exception as e:
            print(f"  WARNING: corrupt main file, deleting: {e}")
            os.remove(hdf5_path)
            existing = 0

    remaining = total_trajs - existing
    per_worker = math.ceil(remaining / num_workers)
    actual_workers = math.ceil(remaining / per_worker)
    print(f"  Parallel: {remaining} trajs across {actual_workers} workers (~{per_worker} each)")

    # Use 'spawn' to give each worker a clean interpreter with no inherited
    # EGL/OpenGL state from the parent process (fork inherits partial GL context
    # state from gymnasium imports, causing 'GLContext has no attribute _context').
    ctx = multiprocessing.get_context('spawn')

    shard_infos = []
    processes = []
    for w in range(actual_workers):
        n_w = min(per_worker, remaining - w * per_worker)
        shard_path = hdf5_path.replace('.hdf5', f'_shard{w}.hdf5')
        shard_infos.append((w, shard_path, n_w))
        p = ctx.Process(
            target=collect_shard_worker,
            args=(w, shard_path, env_name, n_w, args, cam_ids, cam_cfgs),
        )
        processes.append(p)
        p.start()
        print(f"  Launched worker {w}: n={n_w}  shard={os.path.basename(shard_path)}")

    failed = []
    for p, (w, shard_path, _) in zip(processes, shard_infos):
        p.join()
        if p.exitcode != 0:
            failed.append((w, p.exitcode))
            print(f"  WARNING: worker {w} exited with code {p.exitcode}")
    if failed:
        raise RuntimeError(f"Workers failed: {failed}")

    print(f"  Merging {len(shard_infos)} shards into {os.path.basename(hdf5_path)}...")
    with h5py.File(hdf5_path, 'a') as main_f:
        env_group = main_f.require_group(env_name)
        traj_idx = existing
        for w, shard_path, _ in shard_infos:
            if not os.path.exists(shard_path):
                print(f"  WARNING: shard missing: {shard_path}")
                continue
            with h5py.File(shard_path, 'r') as shard_f:
                shard_grp = shard_f[env_name]
                for k in sorted(shard_grp.keys(), key=lambda x: int(x.split('_')[1])):
                    main_f.copy(shard_grp[k], env_group, name=f'traj_{traj_idx}')
                    traj_idx += 1
        print(f"  Merged: {traj_idx} total trajs for {env_name}")

    for _, shard_path, _ in shard_infos:
        if os.path.exists(shard_path):
            os.remove(shard_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_epoch", type=int, default=50,
                        help="Must-succeed expert demos per task (50/task in the released dataset).")
    parser.add_argument("--epsilon",              type=float, default=0.1)
    parser.add_argument("--save_trajectory_path", type=str,
                        default=os.path.join(project_dir, "data", "metaworld_policy"))
    parser.add_argument("--seed",                 type=int, default=42)
    parser.add_argument("--img_size",             type=int, default=224)
    parser.add_argument("--max_eps_step",         type=int, default=128)
    parser.add_argument("--render_interval",      type=int, default=1)
    parser.add_argument("--use_wrist_camera",     action="store_true", default=False)  # wrist unused by CANON
    parser.add_argument("--wrist_camera_name",    type=str, default="gripperPOV")
    parser.add_argument("--egl_refresh_interval", type=int, default=3,
                        help="Recreate the env every N successful trajectories. "
                             "Default=3. Set 0 to disable.")
    parser.add_argument("--env_names", type=str, nargs='+', default=None,
                        help="Environments to collect. Defaults to ALL_ENVIRONMENTS.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel collection workers. Each gets its own "
                             "env + EGL context. Default=1 (single-process, original behavior).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_trajectory_path, exist_ok=True)

    cam_ids, cam_cfgs, canonical = make_camera_configs()
    print(f"Loaded {len(cam_ids)} camera configs from {POLICY_CAM_CFG_PATH}")
    print(f"Encoder pool azimuths (ids 0-19): {[c.azimuth for c in cam_cfgs[:20]]}")
    print(f"OOD cam_id={CAM_OOD_ID}: az={OOD_AZIMUTH}°  el={cam_cfgs[-1].elevation}°  d={cam_cfgs[-1].distance}m")

    # MetaworldSingleViewProprioFullDataset requires camera_configs.json with a
    # "canonical" entry and a "pool" listing only cameras actually stored on disk
    # (image_<id> present) -- in CANONICAL_ONLY mode that's just cam 6.
    policy_cam_cfg_path = os.path.join(args.save_trajectory_path, "camera_configs.json")
    if not os.path.exists(policy_cam_cfg_path):
        if canonical is None:
            # CANON_MW_CAM_CFG override without an explicit "canonical" entry: derive
            # it from the actual canonical camera (id 6) rather than writing
            # loader-invalid output. Cam 6 is always present when CANONICAL_ONLY,
            # and still loaded (just not selected) otherwise.
            canon_matches = [c for cid, c in zip(cam_ids, cam_cfgs) if cid == CANONICAL_CAM_ID]
            assert canon_matches, (
                f"{POLICY_CAM_CFG_PATH} has no camera id {CANONICAL_CAM_ID} and no "
                "top-level 'canonical' entry; MetaworldSingleViewProprioFullDataset "
                "requires one or the other."
            )
            c6 = canon_matches[0]
            canonical = {"azimuth": c6.azimuth, "elevation": c6.elevation,
                         "distance": c6.distance, "lookat": c6.lookat.tolist()}
        out_cfg = {
            "pool": [
                {"id": cid, "azimuth": c.azimuth, "elevation": c.elevation,
                 "distance": c.distance, "lookat": c.lookat.tolist()}
                for cid, c in zip(cam_ids, cam_cfgs)
            ],
            "canonical": canonical,
        }
        with open(policy_cam_cfg_path, "w") as f:
            json.dump(out_cfg, f, indent=2)
        print(f"Saved camera_configs.json → {policy_cam_cfg_path}")

    total_trajs = args.expert_epoch

    if args.env_names is not None:
        invalid = [e for e in args.env_names if e not in ALL_V2_ENVIRONMENTS]
        if invalid:
            raise ValueError(f"Unknown MetaWorld v2 environment name(s): {invalid}")
        active_envs = args.env_names
        suffix    = "_".join(e.replace("-", "") for e in active_envs)
        hdf5_path = os.path.join(args.save_trajectory_path, f"trajectories_{suffix}.hdf5")
    else:
        active_envs = ALL_ENVIRONMENTS
        hdf5_path   = os.path.join(args.save_trajectory_path, "trajectories.hdf5")

    if args.num_workers > 1:
        # Parallel path: each env is collected by num_workers child processes.
        # No main-process HDF5 handle held during collection; shards are written
        # by workers, then merged here after all workers complete.
        for env_name in active_envs:
            print(f"\nCollecting (parallel, {args.num_workers} workers): {env_name}")
            run_parallel_collection(
                hdf5_path, env_name, total_trajs, args.num_workers, args, cam_ids, cam_cfgs,
            )
    else:
        # Single-process path (original behaviour).
        if os.path.exists(hdf5_path):
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    incomplete = [e for e in active_envs
                                  if e not in f or len(f[e]) < total_trajs]
            except Exception as e:
                print(f"WARNING: Could not open existing shard (truncated/corrupt): {e}")
                print(f"  Deleting and starting fresh: {hdf5_path}")
                os.remove(hdf5_path)
                incomplete = active_envs
            if not incomplete:
                print(f"Shard already complete ({total_trajs} trajs × "
                      f"{len(active_envs)} envs). Exiting: {hdf5_path}")
                sys.exit(0)
            elif os.path.exists(hdf5_path):
                print(f"Shard incomplete, missing/short envs: {incomplete}. "
                      f"Deleting and recollecting.")
                os.remove(hdf5_path)

        with h5py.File(hdf5_path, 'a') as hdf5_file:
            for env_name in active_envs:
                print(f"\nCollecting: {env_name}")

                if env_name in hdf5_file:
                    n_existing = len(hdf5_file[env_name])
                    if n_existing >= total_trajs:
                        print(f"  Already complete ({n_existing} trajs). Skipping.")
                        continue
                    print(f"  Resuming: {n_existing}/{total_trajs} done. "
                          f"Collecting {total_trajs - n_existing} more.")
                    env_group = hdf5_file[env_name]
                else:
                    env_group = hdf5_file.create_group(env_name)

                observable_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
                    env_name + "-goal-observable"]()
                observable_env._freeze_rand_vec = False
                observable_env.action_space.seed(args.seed)

                policy_name = "".join([s.capitalize() for s in env_name.split("-")])
                policy_name = policy_name.replace("PegInsert", "PegInsertion")
                policy_name = "Sawyer" + policy_name + "Policy"
                policy       = vars(policies)[policy_name]()
                metric_tracker = EvalMetricTracker()
                traj_count   = len(env_group)
                n_to_collect = total_trajs - traj_count
                saved_debug  = (traj_count > 0)

                attempt_count = 0

                # Resume-safety: see comment in collect_shard_worker.
                seen_rand_vecs = set()
                for k in env_group.keys():
                    seen_rand_vecs.add(tuple(env_group[k].attrs['rand_vec'].tolist()))
                print(f"  start: {traj_count}/{total_trajs} existing, "
                      f"{len(seen_rand_vecs)} known rand_vecs")

                env = make_env(env_name, cam_ids, cam_cfgs, args)

                for i in range(n_to_collect):
                    success = False
                    while not success:
                        while True:
                            obs, _ = env.reset()
                            last_rand_vec = env.unwrapped._last_rand_vec.copy()
                            rv_key = tuple(last_rand_vec.tolist())
                            if rv_key not in seen_rand_vecs:
                                break
                            attempt_count += 1
                            print(f"  [DUP-SKIP] attempt_count={attempt_count}, re-rolling")

                        data = dict(rand_vec=last_rand_vec, partially_observable=False,
                                    env_cls=type(env.unwrapped._env))
                        task = Task(env_name=env_name, data=pickle.dumps(data))
                        observable_env.set_task(task)

                        trajectory = collect_episode(
                            env, observable_env, policy, metric_tracker,
                            epsilon=args.epsilon, noise_type='gaussian',
                            init_obs=obs, must_success=False,
                        )
                        attempt_count += 1
                        success = trajectory['is_success']

                        if not success:
                            print(f"  [FAIL] attempt_count={attempt_count}, retrying...")

                        if (args.egl_refresh_interval > 0
                                and attempt_count % args.egl_refresh_interval == 0):
                            print(f"  [EGL REFRESH] closing and recreating env "
                                  f"(attempt_count={attempt_count})")
                            env.close()
                            env = make_env(env_name, cam_ids, cam_cfgs, args)

                    seen_rand_vecs.add(rv_key)
                    # observable_env was just reset with last_rand_vec via set_task(),
                    # so its _target_pos is the correct goal for this trajectory.
                    target_pos = observable_env._target_pos.copy()
                    write_trajectory(env_group, traj_count, trajectory,
                                     rand_vec=last_rand_vec, target_pos=target_pos)
                    hdf5_file.flush()

                    if not saved_debug:
                        save_debug_video(
                            trajectory, cam_ids,
                            os.path.join(args.save_trajectory_path, "debug_videos",
                                         f"{env_name}.mp4"),
                        )
                        saved_debug = True

                    T = len(trajectory['state'])
                    traj_count += 1
                    print(f"  [{traj_count}/{total_trajs}] "
                          f"T={T}  success={trajectory['is_success']}")

                env.close()
                print(f"  Done: {traj_count} trajectories for {env_name}")

    print(f"\nCollection complete → {hdf5_path}")
