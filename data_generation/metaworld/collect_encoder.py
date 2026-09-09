"""Multi-view SSL pretraining data collection for the CANON MetaWorld encoder dataset.

Each expert trajectory is rendered from a pool of 20 cameras (ReViWo azimuths with random
SO(3) perturbations) so the SSL objective can learn view-invariance.

Camera setup:
  - 20 cameras (ids 0-19): ReViWo azimuths with random SO(3) perturbations
  - Elevations: uniform in [-60, -20]° per camera
  - Distances:  uniform in [1.1, 1.5] m per camera
  - Cam 6 canonical: az=136°, el=-25°, d=1.3 m (from camera_configs.json)
  - Lookat: [0, 0.5, 0] (fixed)

Collection:
  - 17 tasks × 15 gaussian-noise expert trajectories = 255 trajectories (released dataset)
  - All 20 cameras rendered every step

HDF5 layout:
  trajectories.hdf5 / <task> / traj_N /
    image_0 .. image_19   (T, H, W, 3)  uint8, lzf
    action                (T, 4)        float32: gives prev_action for 11-D policy
    state                 (T, 39)       float32: [0:4] = (hand_xyz, gripper)
    reward                (T,)          float32
    mujoco_state          (T, nq+nv)    float32
    attrs:
      is_success  bool
      rand_vec    float32[...]
      target_pos  float32[3]: random-goal target (constant for fixed-goal envs)

camera_configs.json:
  {"pool": [{"id": 0, "azimuth": 8.0, "elevation": ..., "distance": ..., "lookat": [...]},
            ..., {"id": 19, ...}]}
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
# ReViWo camera setup: SO(3) extension
# ---------------------------------------------------------------------------

NUM_CAMERAS = 20

# Canonical camera configs (20 cameras, ids 0-19) shared across all encoder collection batches.
# Generated once with seed=42 (SO(3) perturbations, tight distance range) and saved;
# loaded here so --seed only affects env placement (rand-vec), not camera layout.
# Camera layout = the released dataset's own encoder camera_configs.json (single source),
# shipped in this repo so collection works without an existing dataset download.
# Override with CANON_MW_CAM_CFG to use a different layout.
ENCODER_CAM_CFG_PATH = os.environ.get(
    "CANON_MW_CAM_CFG",
    os.path.join(project_dir, "camera_configs.json"),
)

# Camera indices used for the 2×2 debug video grid (spread across azimuth range)
DEBUG_CAM_INDICES = [0, 5, 10, 15]   # az ≈ 8°, 35°, 195°, 333°


def make_camera_configs(seed=42):
    """Load 20 fixed camera configs plus the canonical entry from ENCODER_CAM_CFG_PATH.

    Camera layout is shared across all encoder collection batches for consistency.
    The seed argument is accepted for API compatibility but not used here;
    seed only affects env placement via env.reset(seed=...).
    """
    with open(ENCODER_CAM_CFG_PATH) as f:
        d = json.load(f)
    pool = d.get('pool', d)
    canonical = d.get('canonical') if isinstance(d, dict) else None
    cam_ids, cam_cfgs = [], []
    for entry in sorted(pool, key=lambda x: x['id']):
        cfg           = CameraConfig()
        cfg.azimuth   = entry['azimuth']
        cfg.elevation = entry['elevation']
        cfg.distance  = entry['distance']
        cfg.lookat    = np.array(entry['lookat'])
        cam_ids.append(entry['id'])
        cam_cfgs.append(cfg)
    return cam_ids, cam_cfgs, canonical


def make_camera_configs_list(cam_ids, cam_cfgs):
    return [
        {
            "id":        cid,
            "azimuth":   float(cam_cfgs[i].azimuth),
            "elevation": float(cam_cfgs[i].elevation),
            "distance":  float(cam_cfgs[i].distance),
            "lookat":    cam_cfgs[i].lookat.tolist(),
        }
        for i, cid in enumerate(cam_ids)
    ]


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_trajectory(env_group, traj_idx, trajectory, rand_vec, target_pos=None):
    """Write one trajectory. All 20 cameras stored as image_{cam_id}.

    The stored ``state`` field (39-D MetaWorld obs) already includes the
    4-D proprioceptive prefix (hand_xyz + gripper). Together with the
    ``action`` field, this covers all inputs the 11-D BC policy needs
    (prev_action and finite-diff velocity are derived at training time;
    see MetaWorldBCEnvProprioFull in the policy code).
    """
    grp = env_group.create_group(f'traj_{traj_idx}')

    for cam_id, frames in trajectory['image'].items():
        grp.create_dataset(f'image_{cam_id}', data=frames, compression='lzf')

    grp.create_dataset('state',        data=np.array(trajectory['state'],        dtype=np.float32))
    grp.create_dataset('action',       data=np.array(trajectory['action'],       dtype=np.float32))
    grp.create_dataset('reward',       data=np.array(trajectory['reward'],       dtype=np.float32))
    grp.create_dataset('mujoco_state', data=np.array(trajectory['mujoco_state'], dtype=np.float32))

    grp.attrs['is_success'] = bool(trajectory['is_success'])
    grp.attrs['rand_vec']   = np.array(rand_vec, dtype=np.float32)
    if target_pos is not None:
        grp.attrs['target_pos'] = np.array(target_pos, dtype=np.float32)


# ---------------------------------------------------------------------------
# Debug video: 2×2 grid of 4 cameras spread across azimuth range
# ---------------------------------------------------------------------------

def save_debug_video(trajectory, cam_ids, cam_cfgs, video_path, fps=10):
    debug_ids = [cam_ids[i] for i in DEBUG_CAM_INDICES]
    imgs      = [trajectory['image'][cid] for cid in debug_ids]
    T         = len(imgs[0])

    grid_frames = [
        np.concatenate([
            np.concatenate([imgs[0][t], imgs[1][t]], axis=1),
            np.concatenate([imgs[2][t], imgs[3][t]], axis=1),
        ], axis=0)
        for t in range(T)
    ]
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    try:
        imageio.mimwrite(video_path, grid_frames, fps=fps)
        azimuths = [cam_cfgs[i].azimuth for i in DEBUG_CAM_INDICES]
        print(f"Debug video saved (cam_ids={debug_ids}, az={azimuths}°): {video_path}")
    except Exception as e:
        print(f"WARNING: Could not save debug video: {e}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def make_env(env_name, cam_ids, cam_cfgs, args, seed):
    """Create a fresh gym env with a new EGL context."""
    env = gym.make(
        f"mw_{env_name}",
        camera_ids=cam_ids,
        camera_configs=cam_cfgs,
        collect_data=True,
        use_camera=True,
        max_eps_step=args.max_eps_step,
        img_size=args.img_size,
        use_wrist_camera=False,
        render_interval=args.render_interval,
    )
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian_expert_epoch", type=int, default=15,
                        help="Gaussian-noise expert trajectories per task "
                             "(15/task × 17 tasks = 255 trajectories in the released dataset).")
    parser.add_argument("--epsilon",              type=float, default=0.1,
                        help="Gaussian noise std on actions.")
    parser.add_argument("--save_trajectory_path", type=str,
                        default=os.path.join(project_dir, "data", "metaworld_encoder"))
    parser.add_argument("--seed",                 type=int, default=42)
    parser.add_argument("--img_size",             type=int, default=224)
    parser.add_argument("--max_eps_step",         type=int, default=128)
    parser.add_argument("--render_interval",      type=int, default=1)
    parser.add_argument("--egl_refresh_interval", type=int, default=1,
                        help="Recreate the env (fresh EGL context) every N successful "
                             "trajectories. Default=1 (refresh after every traj). "
                             "Set 0 to disable.")
    parser.add_argument("--env_names", type=str, nargs='+', default=None,
                        help="Subset of environments to collect; defaults to ALL_ENVIRONMENTS. "
                             "Use with SLURM array jobs: each job writes trajectories_<env>.hdf5.")
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

    # --- Camera configs (shared across all tasks, deterministic from seed) ---
    cam_ids, cam_cfgs, canonical = make_camera_configs(seed=args.seed)

    cam_cfg_path = os.path.join(args.save_trajectory_path, "camera_configs.json")
    if not os.path.exists(cam_cfg_path):
        # canonical is required by MetaworldGoalMultiViewDataset, not just the pool list.
        # A CANON_MW_CAM_CFG override without one would silently write loader-invalid
        # output, so fail loudly here instead.
        assert canonical is not None, (
            f"{ENCODER_CAM_CFG_PATH} has no top-level 'canonical' entry; "
            "MetaworldGoalMultiViewDataset requires one. If overriding via "
            "CANON_MW_CAM_CFG, that file must be a {'canonical': {...}, 'pool': [...]} "
            "dict, not a plain camera list."
        )
        out_cfg = {"pool": make_camera_configs_list(cam_ids, cam_cfgs), "canonical": canonical}
        with open(cam_cfg_path, 'w') as f:
            json.dump(out_cfg, f, indent=2)
        print(f"Saved camera_configs.json → {cam_cfg_path}")

    print(f"Azimuths  : {[c.azimuth for c in cam_cfgs]}")
    print(f"Elevations: {[c.elevation for c in cam_cfgs]}")
    print(f"Distances : {[c.distance for c in cam_cfgs]}")

    total_trajs_per_env = args.gaussian_expert_epoch

    # --- Active env list and HDF5 path ---
    if args.env_names is not None:
        invalid = [e for e in args.env_names if e not in ALL_ENVIRONMENTS]
        if invalid:
            raise ValueError(f"Unknown environment name(s): {invalid}")
        active_envs = args.env_names
        suffix    = "_".join(e.replace("-", "") for e in active_envs)
        hdf5_path = os.path.join(args.save_trajectory_path, f"trajectories_{suffix}.hdf5")
    else:
        active_envs = ALL_ENVIRONMENTS
        hdf5_path   = os.path.join(args.save_trajectory_path, "trajectories.hdf5")

    with h5py.File(hdf5_path, 'a') as hdf5_file:
        for env_name in active_envs:
            print(f"\nCollecting: {env_name}")

            if env_name in hdf5_file:
                n_existing = len(hdf5_file[env_name])
                if n_existing >= total_trajs_per_env:
                    print(f"  Already complete ({n_existing} trajs). Skipping.")
                    continue
                print(f"  Resuming: {n_existing}/{total_trajs_per_env} done. "
                      f"Collecting {total_trajs_per_env - n_existing} more.")
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
            policy      = vars(policies)[policy_name]()
            metric_tracker = EvalMetricTracker()

            traj_count           = len(env_group)
            consecutive_failures = 0
            seed_offset          = 0
            current_seed         = args.seed

            env = make_env(env_name, cam_ids, cam_cfgs, args, current_seed)

            while traj_count < total_trajs_per_env:
                obs, _ = env.reset()

                _last_rand_vec = env.unwrapped._last_rand_vec
                data = dict(rand_vec=_last_rand_vec, partially_observable=False,
                            env_cls=type(env.unwrapped._env))
                task = Task(env_name=env_name, data=pickle.dumps(data))
                observable_env.set_task(task)

                trajectory = collect_episode(
                    env, observable_env, policy, metric_tracker,
                    epsilon=args.epsilon, noise_type='gaussian',
                    init_obs=obs, must_success=False,
                )

                if not trajectory['is_success']:
                    consecutive_failures += 1
                    print(f"  [FAIL] rand_vec={np.round(_last_rand_vec, 4)}"
                          f"  consecutive={consecutive_failures}")
                    if consecutive_failures >= 3:
                        seed_offset  += 1
                        current_seed  = args.seed + seed_offset * 1000
                        print(f"  [SEED ROTATE+EGL] → seed={current_seed}"
                              f" after {consecutive_failures} consecutive failures")
                        env.close()
                        env = make_env(env_name, cam_ids, cam_cfgs, args, current_seed)
                        consecutive_failures = 0
                    continue

                consecutive_failures = 0
                # observable_env was just reset with _last_rand_vec via set_task(),
                # so its _target_pos is the correct goal for this trajectory.
                target_pos = observable_env._target_pos.copy()
                write_trajectory(env_group, traj_count, trajectory,
                                 rand_vec=_last_rand_vec, target_pos=target_pos)
                hdf5_file.flush()

                if traj_count == 0:
                    save_debug_video(
                        trajectory, cam_ids, cam_cfgs,
                        os.path.join(args.save_trajectory_path, "debug_videos",
                                     f"{env_name}.mp4"),
                    )

                T = len(trajectory['state'])
                print(f"  [{traj_count + 1}/{total_trajs_per_env}] "
                      f"T={T}  success=True  seed={current_seed}")
                traj_count += 1

                # Refresh EGL context by recreating the env after every N successes.
                if (args.egl_refresh_interval > 0
                        and traj_count % args.egl_refresh_interval == 0
                        and traj_count < total_trajs_per_env):
                    print(f"  [EGL REFRESH] closing and recreating env (traj_count={traj_count})")
                    env.close()
                    env = make_env(env_name, cam_ids, cam_cfgs, args, current_seed)

            env.close()
            print(f"  Done: {traj_count} trajectories for {env_name}")

    print(f"\nCollection complete → {hdf5_path}")
