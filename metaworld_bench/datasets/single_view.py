"""MetaWorld single-view dataset for CANON policy (BC) training.

Standalone file: no imports from the private dynamo_ssl package.

HDF5 layout (one file per task):
    <data_directory>/trajectories_<taskname>.hdf5 / <task_name> / traj_N /
        image_{i}    (T, H, W, 3)  uint8   i in 0..N_cams-1  (pool cameras)
        action       (T, 4)        float32
        state        (T, 39)       float32
    <data_directory>/camera_configs.json  ({"canonical":..., "pool":[{id, azimuth}, ...]}
                                            or a plain JSON array: both accepted)

Dataset class:
    MetaworldSingleViewProprioFullDataset  – single randomly-sampled view + rich
                                            proprioceptive state vector, for BC policy
                                            training after CANON encoder pretraining.

Returns [obs, act, goal, state, aux_views]:
    obs:       [T, 1, C, H, W]  one randomly-sampled view, float32 in [0, 1]
    act:       [T, 4]           float32  (optionally z-score normalized)
    goal:      [T, 1, C, H, W]  last frame of full trajectory, same camera as obs
    state:     [T, state_dim]   float32  proprioceptive state (default 11-D, see below)
    aux_views: [1]              absolute azimuth of sampled view in radians
                                (only present when aux_view is not None)

Default state composition (11-D):
    hand_pos(3) + gripper_width(1)  [state_indices, default [0,1,2,3]]
    + previous action (4-D, zero-padded at t=0)
    + EE velocity (3-D = curr_hand_pos − prev_hand_pos, using MetaWorld v2 39-D layout
      where prev_hand_pos is at state indices 18-20)

Key difference from MetaworldGoalMultiViewDataset (encoder SSL):
  - Samples 1 camera instead of a pair (size=1 in RNG choice)
  - obs/goal shape: [T, 1, C, H, W] instead of [T, 2, C, H, W]
  - Goal = last frame of the full trajectory from the same camera
    (true task-completion state, not the window-end frame as in the SSL version)
"""

import json
import math
import h5py
import numpy as np
import torch
import einops
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset


class MetaworldSingleViewProprioFullDataset(Dataset):
    """Single-view MetaWorld dataset with rich proprioceptive state for BC training.

    Loads per-task HDF5 files (trajectories_<taskname>.hdf5).  At training time,
    one camera is sampled on the fly from the pool using a seeded RNG.

    State composition per timestep (default 11-D):
        - base: state_indices (default [0,1,2,3]) = hand_pos_xyz + gripper_width
        - previous action (4-D, zero-padded at t=0): improves action coherence
          across re-plan boundaries (community standard: ACT, BC-Z, Octo, OpenVLA)
        - EE velocity (3-D) = curr_hand_pos − prev_hand_pos
          (MetaWorld v2 stores prev_hand_pos at state indices 18-20)

    Args:
        data_directory:  Path to the policy data dir containing trajectories_*.hdf5 and
                         camera_configs.json.
        task_name:       Optional substring filter on HDF5 filenames
                         (e.g. "assemblyv2" selects trajectories_assemblyv2.hdf5).
                         None = load all tasks found in data_directory.
        aux_view:        Any non-None string (e.g. "azimuth_rad") to include azimuth_rad
                         as a [1] float32 tensor.  None → return only [obs, act, goal, state].
        seed:            Master random seed for view sampling RNG.  In multi-worker
                         DataLoaders, pass a worker_init_fn that offsets per worker:
                         ``dataset._rng = np.random.default_rng(dataset.seed + worker_id)``.
        img_size:        If set, resize images to (img_size × img_size) via bilinear.
        camera_ids:      Subset of camera IDs available for sampling.  None = all.
        canonical_view:  Dict {azimuth, elevation, distance} for canonical view bookkeeping.
                         Default: {'azimuth': 136, 'elevation': -25, 'distance': 1.3}.
        task_subset:     Keep only the first N task HDF5 files (alphabetical order).
        subset_fraction: Keep only this fraction of trajectories per task (≥1 always kept).
        max_trajs:       Hard cap on trajectories per task (applied after subset_fraction).
        random_crop:     If True, apply pad-then-crop augmentation for temporal consistency.
        crop_pad:        Padding size for random crop augmentation.
        prop_goal:       If True, use a 3-D proprioceptive target position as goal instead of
                         the last image frame (requires 'target_pos' HDF5 attribute per traj).
        normalize_actions: If True, z-score normalize actions using dataset statistics.
        state_indices:   Indices into the 39-D HDF5 'state' field for the base proprio slice.
                         Default: [0, 1, 2, 3] (hand_pos_xyz + gripper_width).
        include_prev_action: If True, concatenate 4-D previous-action vector to state.
        include_velocity:    If True, concatenate 3-D EE velocity to state.
        action_dim:          Size of the action vector (MetaWorld = 4).
        action_noise_std:    Std of temporal jittering noise on prev_action. 0.0 = no augmentation.
    """

    DEFAULT_CANONICAL_VIEW: dict = {'azimuth': 136.0, 'elevation': -25.0, 'distance': 1.3}

    def __init__(
        self,
        data_directory: str,
        task_name: Optional[str] = None,
        aux_view: Optional[str] = "azimuth_rad",
        seed: int = 42,
        img_size: Optional[int] = None,
        camera_ids: Optional[List[int]] = None,
        canonical_view: Optional[dict] = None,
        task_subset: Optional[int] = None,
        subset_fraction: Optional[float] = None,
        max_trajs: Optional[int] = None,
        random_crop: bool = False,
        crop_pad: int = 8,
        prop_goal: bool = False,
        normalize_actions: bool = False,
        state_indices: Optional[List[int]] = None,
        include_prev_action: bool = True,
        include_velocity: bool = True,
        action_dim: int = 4,
        action_noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        data_dir = Path(data_directory)
        assert data_dir.exists(), f"MetaworldSingleViewProprioFullDataset: {data_dir} does not exist"
        print(f"MetaworldSingleViewProprioFullDataset: using data dir {data_dir}")

        self.aux_view            = aux_view
        self.seed                = seed
        self.img_size            = img_size
        self._rng                = np.random.default_rng(seed)
        self.random_crop         = random_crop
        self._crop_pad           = crop_pad
        self.prop_goal           = prop_goal
        self.normalize_actions   = normalize_actions
        self.include_prev_action = include_prev_action
        self.include_velocity    = include_velocity
        self.action_dim          = action_dim
        self.action_noise_std    = action_noise_std

        # ── Camera configs ─────────────────────────────────────────────────────
        # camera_configs.json is either {"canonical":..., "pool":[...]} or a plain array.
        cfg_path = data_dir / "camera_configs.json"
        assert cfg_path.exists(), f"camera_configs.json not found in {data_dir}"
        with open(cfg_path) as fh:
            cam_cfg = json.load(fh)
        cameras = cam_cfg if isinstance(cam_cfg, list) else cam_cfg['pool']
        self.id_to_azimuth: Dict[int, float] = {c['id']: float(c['azimuth']) for c in cameras}

        all_cam_ids = sorted(self.id_to_azimuth.keys())
        if camera_ids is not None:
            invalid = [c for c in camera_ids if c not in self.id_to_azimuth]
            if invalid:
                raise ValueError(
                    f"MetaworldSingleViewProprioFullDataset: camera_ids contains unknown IDs: {invalid}. "
                    f"Valid IDs: {all_cam_ids}"
                )
            self.pool_cam_ids: List[int] = sorted(int(c) for c in camera_ids)
        else:
            self.pool_cam_ids = all_cam_ids

        # ── Canonical view ─────────────────────────────────────────────────────
        if canonical_view is None:
            self.canonical_view: dict = dict(self.DEFAULT_CANONICAL_VIEW)
        else:
            self.canonical_view = dict(self.DEFAULT_CANONICAL_VIEW)
            self.canonical_view.update(dict(canonical_view))

        canonical_az = float(self.canonical_view['azimuth'])
        self.canonical_cam_id: int = min(
            self.pool_cam_ids,
            key=lambda cid: abs(self.id_to_azimuth[cid] - canonical_az)
        )

        # ── Trajectory index from per-task HDF5 files ─────────────────────────
        # Each trajectories_<task>.hdf5 has one top-level task group.
        hdf5_files = sorted(data_dir.glob("trajectories_*.hdf5"))
        assert hdf5_files, f"No trajectories_*.hdf5 files found in {data_dir}"

        if task_name is not None:
            expected_stem = "trajectories_" + task_name.replace("-", "")
            hdf5_files = [p for p in hdf5_files if p.stem == expected_stem]
            assert hdf5_files, (
                f"No trajectories_*.hdf5 matching task_name={task_name!r} "
                f"(expected stem '{expected_stem}') in {data_dir}"
            )

        if task_subset is not None:
            hdf5_files = hdf5_files[:task_subset]

        self._index: List[Tuple[str, str, str]] = []
        for hdf5_path in hdf5_files:
            with h5py.File(str(hdf5_path), 'r') as fh:
                for task in sorted(fh.keys()):
                    traj_keys = sorted(
                        fh[task].keys(), key=lambda k: int(k.split('_')[1])
                    )
                    if subset_fraction is not None and subset_fraction < 1.0:
                        traj_keys = traj_keys[:max(1, int(len(traj_keys) * subset_fraction))]
                    if max_trajs is not None:
                        traj_keys = traj_keys[:max_trajs]
                    for tk in traj_keys:
                        self._index.append((str(hdf5_path), task, tk))

        # Validate that every requested pool camera has a stored image. The released
        # policy data keeps only the canonical view (image_6), so camera_ids must match
        # what is on disk: otherwise sampling would hit a missing image at getitem time.
        if self._index:
            _hp, _task0, _tk0 = self._index[0]
            with h5py.File(_hp, 'r') as _fh:
                _present = {k for k in _fh[_task0][_tk0] if k.startswith('image_')}
            _missing = [c for c in self.pool_cam_ids if f'image_{c}' not in _present]
            if _missing:
                raise ValueError(
                    f"MetaworldSingleViewProprioFullDataset: camera_ids {_missing} have no "
                    f"image_<id> in the data (present: {sorted(_present)}). The released "
                    f"policy data stores only the canonical view; set camera_ids accordingly."
                )

        self.task_names: List[str] = list(dict.fromkeys(item[1] for item in self._index))
        print(
            f"MetaworldSingleViewProprioFullDataset: {len(self._index)} trajectories "
            f"from {len(self.task_names)} task(s) | "
            f"pool cameras ({len(self.pool_cam_ids)}): {self.pool_cam_ids}"
        )

        # ── Lazy HDF5 handles (one per file) ──────────────────────────────────
        self._hdf5_handles: Dict[str, Optional[h5py.File]] = {
            str(p): None for p in hdf5_files
        }

        # ── State dimensionality ───────────────────────────────────────────────
        self.state_indices = state_indices if state_indices is not None else [0, 1, 2, 3]
        base_dim = len(self.state_indices)
        extra = (action_dim if include_prev_action else 0) + (3 if include_velocity else 0)
        self.state_dim = base_dim + extra
        print(
            f"MetaworldSingleViewProprioFullDataset: state_indices={self.state_indices} "
            f"→ base_dim={base_dim}, prev_action={include_prev_action}, "
            f"velocity={include_velocity}, action_noise_std={action_noise_std} "
            f"→ state_dim={self.state_dim}"
        )

        # ── Action normalization stats ─────────────────────────────────────────
        if normalize_actions:
            self._compute_action_stats()
        else:
            self.act_mean: Optional[torch.Tensor] = None
            self.act_std:  Optional[torch.Tensor] = None

    # ── HDF5 lazy access ──────────────────────────────────────────────────────

    def _get_hdf5(self, hdf5_path: str) -> h5py.File:
        if self._hdf5_handles[hdf5_path] is None:
            self._hdf5_handles[hdf5_path] = h5py.File(hdf5_path, 'r')
        return self._hdf5_handles[hdf5_path]

    def __getstate__(self):
        # Close all open HDF5 handles before pickling (DataLoader worker fork).
        state = self.__dict__.copy()
        for path, fh in state.get('_hdf5_handles', {}).items():
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        state['_hdf5_handles'] = {path: None for path in state.get('_hdf5_handles', {})}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        for fh in getattr(self, '_hdf5_handles', {}).values():
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass

    # ── Action normalization ───────────────────────────────────────────────────

    def _compute_action_stats(self) -> None:
        """Compute per-dimension action mean and std from all indexed trajectories.

        Opens each HDF5 file directly (without going through the lazy handle cache)
        so this can be called at __init__ time before any worker processes are forked.
        Stats are stored as float32 tensors on CPU.
        """
        all_actions = []
        for hdf5_path, task, tk in self._index:
            with h5py.File(hdf5_path, 'r') as fh:
                all_actions.append(fh[task][tk]['action'][:])
        arr = np.concatenate(all_actions, axis=0).astype(np.float32)   # [N_total, A]
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0).clip(1e-6)
        self.act_mean = torch.from_numpy(mean)
        self.act_std  = torch.from_numpy(std)
        print(
            f"MetaworldSingleViewProprioFullDataset: action norm stats: "
            f"mean={mean.round(3).tolist()}  std={std.round(3).tolist()}"
        )

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _apply_random_crop(self, obs: torch.Tensor) -> torch.Tensor:
        """Pad-then-crop; same (i,j) applied to all T frames for temporal consistency.

        RandomCrop.get_params uses PyTorch RNG → safe under DataLoader workers.
        TF.crop slices [..., i:i+h, j:j+w] so it works on the full [T*V, C, H, W] batch.
        """
        T, V, C, H, W = obs.shape
        flat = TF.pad(obs.reshape(T * V, C, H, W), padding=self._crop_pad, padding_mode='reflect')
        i, j, h, w = RandomCrop.get_params(flat[0], output_size=(H, W))
        return TF.crop(flat, i, j, h, w).reshape(T, V, C, H, W)

    # ── Dataset interface ──────────────────────────────────────────────────────

    @property
    def demos(self):
        """Compatibility alias used by some trainers for goal-caching logic."""
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def get_seq_length(self, idx: int) -> int:
        hdf5_path, task, tk = self._index[idx]
        return self._get_hdf5(hdf5_path)[task][tk]['action'].shape[0]

    def get_frames(self, idx: int, frames, view=None) -> list:
        """Return [obs, act, goal, state] or [obs, act, goal, state, aux_views].

        State is the full proprioceptive vector (base slice + prev_action + velocity).
        The HDF5 'state' field (39-D) is loaded once and sliced for all components.
        """
        hdf5_path, task, tk = self._index[idx]
        frames_list = list(frames)
        grp = self._get_hdf5(hdf5_path)[task][tk]

        # ── Sample 1 camera ────────────────────────────────────────────────────
        cam_idx = self._rng.choice(len(self.pool_cam_ids), size=1)
        cam_id = self.pool_cam_ids[int(cam_idx[0])]

        # ── Observation ────────────────────────────────────────────────────────
        obs = torch.from_numpy(
            grp[f'image_{cam_id}'][frames_list].astype(np.float32)
        ) / 255.0                                    # [T, H, W, C]
        obs = obs.unsqueeze(1)                        # [T, 1, H, W, C]
        obs = einops.rearrange(obs, 'T V H W C -> T V C H W')

        if self.img_size is not None:
            T, V, C, H, W = obs.shape
            if H != self.img_size or W != self.img_size:
                obs = TF.resize(
                    obs.reshape(T * V, C, H, W),
                    [self.img_size, self.img_size],
                    antialias=True,
                )
                obs = obs.reshape(T, V, C, self.img_size, self.img_size)

        if self.random_crop:
            obs = self._apply_random_crop(obs)

        # ── Action ─────────────────────────────────────────────────────────────
        act = torch.from_numpy(grp['action'][frames_list])
        if self.normalize_actions and self.act_mean is not None:
            act = (act - self.act_mean) / self.act_std

        # ── Goal ───────────────────────────────────────────────────────────────
        if self.prop_goal:
            # Proprioceptive goal: 3-D target position stored as traj attr.
            # Shape [T, 1, 3]: V=1 dummy view dim keeps shapes compatible with
            # the image-goal path through TrajectoryEmbeddingDataset/forward_policy.
            target_pos = grp.attrs.get('target_pos', None)
            if target_pos is None:
                raise ValueError(
                    f"No 'target_pos' attr in {tk}. "
                    "Run collect_data/add_target_pos.py first."
                )
            tp = torch.from_numpy(np.array(target_pos, dtype=np.float32))  # [3]
            goal = tp.unsqueeze(0).unsqueeze(0).repeat(len(frames_list), 1, 1)  # [T, 1, 3]
        else:
            # Image goal: last frame of full trajectory, same camera as obs.
            # Index [-1] always addresses the true trajectory end regardless of what
            # `frames` slice was passed by TrajectorySlicerDataset.
            goal_raw = torch.from_numpy(
                grp[f'image_{cam_id}'][[-1]].astype(np.float32)
            ) / 255.0                                    # [1, H, W, C]
            goal_raw = goal_raw.unsqueeze(1)              # [1, 1, H, W, C]
            goal_raw = einops.rearrange(goal_raw, 'T V H W C -> T V C H W')
            if self.img_size is not None:
                _, V, C, H, W = goal_raw.shape
                if H != self.img_size or W != self.img_size:
                    goal_raw = TF.resize(
                        goal_raw.reshape(V, C, H, W),
                        [self.img_size, self.img_size],
                        antialias=True,
                    )
                    goal_raw = goal_raw.reshape(1, V, C, self.img_size, self.img_size)
            goal = goal_raw.repeat(len(frames_list), 1, 1, 1, 1)   # [T, 1, C, H, W]

        # ── Proprioceptive state (all components built from a single HDF5 load) ─
        T = len(frames_list)
        # Load 39-D state once; reuse for all components.
        state_full = grp['state'][frames_list]  # [T, 39] np.float32

        parts: List[torch.Tensor] = [
            torch.from_numpy(state_full[:, self.state_indices])  # [T, base_dim]
        ]

        if self.include_prev_action:
            actions_full = grp['action']  # (T_traj, action_dim)
            prev_act = np.zeros((T, self.action_dim), dtype=np.float32)
            for i, t in enumerate(frames_list):
                if t > 0:
                    prev_act[i] = actions_full[t - 1]
            # Temporal jittering: add small noise to prev_action for augmentation
            if self.action_noise_std > 0.0:
                prev_act += self._rng.normal(
                    0, self.action_noise_std, prev_act.shape
                ).astype(np.float32)
            parts.append(torch.from_numpy(prev_act))

        if self.include_velocity:
            # MetaWorld v2 39-D state: prev_hand_pos at [18:21], curr at [0:3]
            velocity = state_full[:, 0:3] - state_full[:, 18:21]
            parts.append(torch.from_numpy(velocity.astype(np.float32)))

        full_state = torch.cat(parts, dim=-1)  # [T, state_dim]

        data = [obs, act, goal, full_state]
        if self.aux_view is not None:
            az_rad = self.id_to_azimuth[cam_id] * (math.pi / 180.0)
            data.append(torch.tensor([az_rad], dtype=torch.float32))  # [1]

        return data

    def __getitem__(self, idx: int):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
