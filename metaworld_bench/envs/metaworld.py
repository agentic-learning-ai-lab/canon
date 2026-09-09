"""MetaWorld BC policy evaluation environment.

Wraps a MetaWorld v2 task with configurable camera views for BC policy rollout.
API mirrors LiberoEnvPerturb so it plugs into PolicyTrainer.eval_on_env() unchanged.

reset(goal_idx) → obs [1, C, H, W] float32 in [0, 1]
step(action)    → (obs, reward, done, info)
  reward  = 1.0 on success, 0.0 otherwise  (binary)
  info['image']              = [H, W, 3] uint8  for VideoRecorder
  info['all_completions_ids'] = int(success)     for completion tracking
  info['is_success']         = bool
"""

import json
import os
import numpy as np
from PIL import Image


class MetaWorldBCEnvProprioFull:
    """
    BC policy rollout environment for MetaWorld v2 tasks with proprioception and
    extended state (previous action + EE velocity).

    Mirrors LiberoEnvPerturb's interface so PolicyTrainer.eval_on_env() works
    without modification:
      - env.views       list[float]  azimuths (degrees) of active cameras
      - env.view_idx    int          current camera index
      - env.incr_view() advances view_idx for mixed-view evaluation
      - env.mixed_view  bool         set True to enable incr_view cycling

    Exposes proprioceptive state via:
      - ``info['state']`` on every ``step`` (float32 ndarray)
      - ``self.last_state`` attribute (latest state for any caller)
      - ``self.get_state()`` accessor

    The state vector = base proprio (state_indices) + optional prev_action
    (action_dim) + optional EE velocity (3-D). Mirrors
    MetaworldSingleViewProprioFullDataset on the rollout side.

    Args:
        id:                  Env ID string used by PolicyTrainer as self.env_name.
                             Use "metaworld_bc" so save_best_model() tracks it.
        task_name:           MetaWorld v2 task, e.g. "assembly-v2".
        camera_configs_path: Path to camera_configs.json ({"canonical":..., "pool":[...]}
                             or a plain JSON array: both accepted).
        camera_ids:          Camera ID(s) to cycle through. Default [6] = canonical
                             (azimuth ≈ 136°). Pass multiple for mixed-view eval.
        image_size:          Output image resolution in pixels. Default 128.
        max_episode_steps:   Episode time limit. Default 200.
        mixed_view:          If True, incr_view() cycles cameras across evals.
        single_view:         Unused; kept for API compat with LiberoEnvPerturb.
        view_interval:       Unused; kept for API compat.
        state_indices:       Indices into the 39-D MetaWorld obs to use as base
                             proprio. Default [0, 1, 2, 3].
        include_prev_action: Append last_action to state vector.
        include_velocity:    Append (current_hand − previous_hand) to state.
        action_dim:          Action dim for prev_action padding (default 4).
    """

    def __init__(
        self,
        id: str = "metaworld_bc",
        task_name: str = "assembly-v2",
        camera_configs_path: str = None,
        camera_ids: list = None,
        image_size: int = 128,
        max_episode_steps: int = 200,
        mixed_view: bool = False,
        single_view: bool = True,
        view_interval=None,
        view_indices=None,   # inherited from LiberoEnvPerturb base config; unused here
        camera_distance=None,
        state_indices=None,
        include_prev_action: bool = True,
        include_velocity: bool = True,
        action_dim: int = 4,
    ):
        # Normalize compact config names (e.g. "assemblyv2") to MetaWorld v2 names
        # ("assembly-v2"). If already hyphenated, the dict is a no-op.
        _NAME_MAP = {
            'assemblyv2':    'assembly-v2',
            'basketballv2':  'basketball-v2',
            'buttonpressv2': 'button-press-v2',
            'dooropenv2':    'door-open-v2',
            'drawerclosev2': 'drawer-close-v2',
            'draweropenv2':  'drawer-open-v2',
            'leverpullv2':   'lever-pull-v2',
            'pickplacev2':   'pick-place-v2',
            'pushv2':        'push-v2',
            'reachv2':       'reach-v2',
            'windowopenv2':  'window-open-v2',
        }
        task_name = _NAME_MAP.get(task_name, task_name)

        os.environ.setdefault("MUJOCO_GL", "egl")
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        self.id = id
        self.task_name = task_name
        self.image_size = image_size
        self._max_episode_steps = max_episode_steps
        self.mixed_view = mixed_view
        self.single_view = single_view

        # ── Camera configs ────────────────────────────────────────────────────
        assert camera_configs_path is not None, (
            "MetaWorldBCEnv: camera_configs_path is required"
        )
        with open(camera_configs_path) as f:
            cam_cfg = json.load(f)
        cameras = cam_cfg if isinstance(cam_cfg, list) else cam_cfg["pool"]
        self._cam_id_to_config = {int(c["id"]): c for c in cameras}

        # camera_ids = [6] if camera_ids is None else [int(c) for c in camera_ids]
        if camera_ids is None: # use all cameras from the config
            camera_ids = sorted(self._cam_id_to_config.keys())
        else: # validate specified IDs
            camera_ids = [int(c) for c in camera_ids]
            for cid in camera_ids:
                if cid not in self._cam_id_to_config:
                    raise ValueError(
                        f"MetaWorldBCEnv: camera_id {cid} not found in config. "
                        f"Available IDs: {sorted(self._cam_id_to_config.keys())}"
                    )

        self.camera_ids = camera_ids
        # views: list of azimuths in degrees: used by eval_on_env as view angles
        self.views = [float(self._cam_id_to_config[cid]["azimuth"]) for cid in camera_ids]
        self.view_idx = 0

        # ── MetaWorld env ─────────────────────────────────────────────────────
        assert task_name in ALL_V2_ENVIRONMENTS, (
            f"MetaWorldBCEnv: unknown task '{task_name}'. "
            f"Available: {sorted(ALL_V2_ENVIRONMENTS.keys())}"
        )
        self._env = ALL_V2_ENVIRONMENTS[task_name](render_mode="rgb_array")
        self._env._freeze_rand_vec = False
        self._env._set_task_called = True

        self._episode_steps = 0
        self._base_seed = 42
        self._renderer_cam_id = None  # tracks last configured camera_id

        # ── Proprio state config (from MetaWorldBCEnvProprio) ─────────────────
        self.state_indices = state_indices if state_indices is not None else [0, 1, 2, 3]
        self.state_dim = len(self.state_indices)
        print(
            f"MetaWorldBCEnvProprio: state_indices={self.state_indices} → state_dim={self.state_dim}"
        )

        # ── Extended state config (from MetaWorldBCEnvProprioFull) ────────────
        self.include_prev_action = include_prev_action
        self.include_velocity    = include_velocity
        self.action_dim          = action_dim
        self.last_action         = np.zeros(action_dim, dtype=np.float32)
        self.prev_hand_pos       = None
        extra = (action_dim if include_prev_action else 0) + (3 if include_velocity else 0)
        self.state_dim = self.state_dim + extra
        self.last_state = np.zeros(self.state_dim, dtype=np.float32)
        print(
            f"MetaWorldBCEnvProprioFull: prev_action={include_prev_action}, "
            f"velocity={include_velocity} → state_dim={self.state_dim}"
        )

    # ── Seed / view helpers ───────────────────────────────────────────────────

    def seed(self, s):
        self._base_seed = int(s)

    def incr_view(self):
        """Advance to next camera (for mixed_view evaluation)."""
        self.view_idx = (self.view_idx + 1) % len(self.views)

    # ── Camera / rendering ────────────────────────────────────────────────────

    def _configure_renderer(self, camera_id: int):
        """Lazy-configure MujocoRenderer; no-op if already set for this camera."""
        if self._renderer_cam_id == camera_id:
            return
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        cfg = self._cam_id_to_config[camera_id]
        lookat = cfg.get("lookat", [0.0, 0.5, 0.0])
        cam_dict = {
            "distance": float(cfg["distance"]),
            "azimuth":  float(cfg["azimuth"]),
            "elevation": float(cfg["elevation"]),
            "lookat":   np.array(lookat, dtype=np.float64),
        }
        # Render at ≥480 then resize; MuJoCo needs minimum resolution for quality
        render_res = max(480, self.image_size)
        self._env.mujoco_renderer = MujocoRenderer(
            self._env.model, self._env.data, cam_dict,
            width=render_res, height=render_res,
        )
        self._renderer_cam_id = camera_id

    def _render(self):
        """Render current camera. Returns (obs float32 [1,C,H,W], frame uint8 [H,W,3])."""
        cam_id = self.camera_ids[self.view_idx]
        self._configure_renderer(cam_id)
        frame = self._env.render()  # [H, W, 3] uint8
        if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
            frame = np.array(
                Image.fromarray(frame).resize((self.image_size, self.image_size)),
                dtype=np.uint8,
            )
        obs = np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]  # [1,C,H,W]
        return obs, frame

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, goal_idx: int = 0):
        """Reset; goal_idx seeds the episode for reproducible variety across evals."""
        # Reset action history + velocity tracker before env reset so that
        # _read_state (called below) sees zeroed-out history.
        self.last_action   = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_hand_pos = None

        self._env.seed(self._base_seed + goal_idx)
        self._env.reset()
        self._episode_steps = 0
        self._renderer_cam_id = None  # invalidate after reset (model/data may have changed)
        obs, _ = self._render()
        self.last_state = self._read_state()
        return obs  # [1, C, H, W] float32 in [0, 1]

    def step(self, action):
        """Step env. Returns (obs, reward, done, info); non-gymnasium signature."""
        # Update last_action BEFORE _read_state so the next-state proprio
        # sees the action that just executed (matches training-time convention).
        self.last_action = np.asarray(action, dtype=np.float32)

        self._episode_steps += 1
        _, _, terminated, truncated, info = self._env.step(action)
        success = bool(info.get("success", False))
        done = terminated or truncated or success or (self._episode_steps >= self._max_episode_steps)

        # Binary reward: 1.0 on success so _get_episode_score returns success rate
        # via the is_mimicgen (float(max_reward)) branch in online_eval_oop.py.
        reward = 1.0 if success else 0.0

        obs, frame = self._render()
        info["is_success"] = success
        info["image"] = frame                            # [H, W, 3] uint8 for VideoRecorder
        info["all_completions_ids"] = int(success)       # completion tracking

        self.last_state = self._read_state()
        info['state'] = self.last_state
        return obs, reward, done, info

    def _read_state(self) -> np.ndarray:
        """Read and assemble the extended proprio state vector."""
        full = self._env._get_obs()                              # [39] np.float32
        base = np.asarray(full[self.state_indices], dtype=np.float32)
        parts = [base]

        if self.include_prev_action:
            parts.append(self.last_action.astype(np.float32))

        if self.include_velocity:
            current_hand = np.asarray(full[0:3], dtype=np.float32)
            if self.prev_hand_pos is None:
                vel = np.zeros(3, dtype=np.float32)
            else:
                vel = current_hand - self.prev_hand_pos
            self.prev_hand_pos = current_hand
            parts.append(vel)

        return np.concatenate(parts).astype(np.float32)

    def get_state(self) -> np.ndarray:
        return self.last_state

    def close(self):
        pass
