"""
CANON policy training dataset for LIBERO-Goal single-view data.

This module provides ``LiberoGoalSingleViewDataset``, the dataset used to train
both the DiTFlow and VQ-BeT policies in the CANON system.  It reads from the
same multi-view HDF5 files produced by the data collection pipeline but returns
only one camera view per trajectory: no pair logic is needed for BC training.

Collection format:
  The released policy data stores one azimuth view per trajectory, ``image_0``
: the 0° canonical (front) camera.  (The 25° perturbed view captured during
  data collection was stripped for the public release.)

HDF5 layout (released policy data):
    trajectories.hdf5
    └── <task_name>/
        ├── traj_0/
        │   ├── image_0         (T, H, W, 3) uint8   0° canonical view
        │   ├── state           (T, S)        float32
        │   └── action          (T, 7)        float32
        └── traj_1/ ...

Interface: returns ``[obs, act, goal]`` or ``[obs, act, goal, azimuth_rad]``
  obs:          [T, 1, C, H, W]  float32 in [0, 1]
  act:          [T, action_dim]  float32
  goal:         [T, 1, C, H, W]  last frame of the task, repeated across T
  azimuth_rad:  Tensor[1]        absolute azimuth in radians (only when
                                 ``aux_view`` is not None)

This file is fully self-contained: it does not import from any private
``datasets.*`` modules.
"""

import abc
import json
import math
import h5py
import numpy as np
import torch
import einops
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Optional, List, Tuple
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Minimal TrajectoryDataset ABC (inlined from datasets/core.py)
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset, abc.ABC):
    """Base class for trajectory datasets.

    Subclasses must implement ``get_seq_length`` and ``get_frames``.
    """

    @abc.abstractmethod
    def get_seq_length(self, idx: int) -> int:
        """Return the number of timesteps in trajectory ``idx``."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_frames(self, idx: int, frames, view=None):
        """Return data for a subset of frames from trajectory ``idx``."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LiberoGoalSingleViewDataset(TrajectoryDataset):
    """CANON policy training dataset: single camera view per trajectory.

    Loads a single camera view from a multi-view LIBERO-Goal HDF5 dataset.
    Used for training both the DiTFlow and VQ-BeT policies.

    Args:
        data_directory:   Root dataset directory (e.g. ``${env_vars.datasets.libero}``).
        task_subset:      Only use the first N tasks (``None`` = all).
        subset_fraction:  Use first fraction of demos per task (for debugging).
        aux_view:         If not ``None``, append the absolute azimuth of the
                          selected view in radians as a ``Tensor[1]`` (4th return
                          element of ``get_frames``).  Requires
                          ``camera_configs.json`` in the dataset directory.
                          Pass any non-``None`` string (e.g. ``"azimuth_rad"``) to
                          enable.
        img_size:         If set, resize images to ``(img_size, img_size)``.
    """

    def __init__(
        self,
        data_directory: str,
        task_subset: Optional[int] = None,
        subset_fraction: Optional[float] = None,
        aux_view: Optional[str] = "azimuth_rad",
        img_size: Optional[int] = None,
    ):
        self._data_directory = data_directory
        self.aux_view = aux_view
        self.img_size = img_size
        dataset_dir = Path(data_directory)   # metaworld-style: read data dir directly
        assert dataset_dir.exists(), (
            f"LiberoGoalSingleViewDataset: {dataset_dir} does not exist"
        )
        print(f"LiberoGoalSingleViewDataset: using data dir {dataset_dir}")

        self.hdf5_path = str(dataset_dir / "trajectories.hdf5")
        self._hdf5_file = None  # opened lazily per-worker

        # Camera configs: cam_id → azimuth (degrees). Accepts a flat list of view dicts
        # [{id, azimuth, ...}] (same format as metaworld/policy/camera_configs.json) or a
        # {"pool": [...]} object. Loaded only when aux_view is requested.
        self._id_to_azimuth: dict = {}
        self._canonical_az_deg: float = 0.0  # released policy data is the canonical view
        if aux_view is not None:
            cam_cfg_path = dataset_dir / "camera_configs.json"
            with open(cam_cfg_path) as f:
                cam_cfg = json.load(f)
            cams = cam_cfg if isinstance(cam_cfg, list) else cam_cfg["pool"]
            self._id_to_azimuth = {int(c["id"]): float(c["azimuth"]) for c in cams}
            self._id_to_azimuth.setdefault(0, 0.0)  # cam_id 0 is the canonical 0° camera
            # Stored view is the canonical view; take its azimuth from `canonical` so the
            # `pool` can hold the rollout-eval sweep without corrupting the training azimuth.
            if isinstance(cam_cfg, dict) and "canonical" in cam_cfg:
                self._canonical_az_deg = float(cam_cfg["canonical"]["azimuth"])

        # Index trajectories.
        # _index: (task_name, traj_key, image_key) per trajectory.
        # _azimuth: absolute azimuth in radians, parallel to _index.
        with h5py.File(self.hdf5_path, "r") as f:
            task_names = sorted(f.keys())
            if task_subset is not None:
                task_names = task_names[:task_subset]

            self._index: List[Tuple[str, str, str]] = []
            self._azimuth: List[float] = []
            for task in task_names:
                traj_keys = sorted(
                    f[task].keys(), key=lambda k: int(k.split("_")[1])
                )
                n = len(traj_keys)
                if subset_fraction is not None and subset_fraction < 1.0:
                    n = max(1, int(n * subset_fraction))
                    traj_keys = traj_keys[:n]

                for tk in traj_keys:
                    grp = f[task][tk]
                    image_key = self._resolve_image_key(grp)
                    if image_key is None:
                        print(
                            f"  WARNING: no suitable view in {task}/{tk}, skipping"
                        )
                        continue
                    self._index.append((task, tk, image_key))
                    self._azimuth.append(self._resolve_azimuth_rad(grp))

        self.task_names = list(dict.fromkeys(item[0] for item in self._index))
        print(
            f"LiberoGoalSingleViewDataset: serving the 0° canonical view (image_0); "
            f"{len(self._index)} trajectories from {len(self.task_names)} tasks"
        )

        # Pre-compute per-task goal trajectories (last frame of first traj in
        # each task) so get_frames can serve goals without an extra HDF5 scan.
        self._task_goal_traj: dict = {}  # task_name → (traj_key, image_key)
        seen: set = set()
        for task, tk, ik in self._index:
            if task not in seen:
                self._task_goal_traj[task] = (tk, ik)
                seen.add(task)

    # -- HDF5 lazy handle -----------------------------------------------------

    @property
    def hdf5(self) -> h5py.File:
        """Open HDF5 file lazily; each DataLoader worker gets its own handle."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("_hdf5_file") is not None:
            try:
                state["_hdf5_file"].close()
            except Exception:
                pass
            state["_hdf5_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass

    # -- View resolution ------------------------------------------------------

    def _resolve_image_key(self, grp: h5py.Group) -> str:
        """Return the HDF5 key for the 0° canonical view.

        The released policy data stores only the 0° canonical view as ``image_0``
        (the 25° view was stripped). Both the DiTFlow and VQ-BeT policies train on
        this single view.
        """
        if "image_0" not in grp:
            raise KeyError(f"expected the 0° canonical view 'image_0' in {grp.name}")
        return "image_0"

    def _resolve_azimuth_rad(self, grp: h5py.Group) -> float:
        """Azimuth (radians) of the training view.

        The release data is the 0° canonical view (cam_id 0), so this is 0.
        """
        return self._canonical_az_deg * (math.pi / 180.0)

    # -- Dataset interface ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def get_seq_length(self, idx: int) -> int:
        task, tk, _ = self._index[idx]
        return self.hdf5[task][tk]["state"].shape[0]

    def _load_view(self, grp: h5py.Group, image_key: str, frames) -> torch.Tensor:
        """Load image frames as float32 ``[T, H, W, C]`` in ``[0, 1]``."""
        imgs = grp[image_key][frames]  # (T, H, W, 3) uint8
        return torch.from_numpy(imgs.astype(np.float32)) / 255.0

    @property
    def demos(self):
        """Compatibility alias for eval_on_env's get_goals_cache."""
        return self._index

    def get_frames(self, idx: int, frames, view=None) -> list:
        """Return ``[obs, act, goal]`` or ``[obs, act, goal, azimuth_rad]``.

        Args:
            idx:    Trajectory index.
            frames: Frame indices to load (iterable of ints).
            view:   Unused; kept for interface compatibility.

        Returns:
            ``[obs, act, goal]`` when ``aux_view`` is ``None``, or
            ``[obs, act, goal, azimuth_rad]`` otherwise:

            obs:         Tensor[T, 1, C, H, W]
            act:         Tensor[T, action_dim]
            goal:        Tensor[T, 1, C, H, W]  last-frame goal, repeated
            azimuth_rad: Tensor[1]               absolute azimuth in radians
        """
        task, tk, image_key = self._index[idx]
        frames = list(frames)
        grp = self.hdf5[task][tk]

        obs = self._load_view(grp, image_key, frames)      # [T, H, W, C]
        obs = obs.unsqueeze(1)                              # [T, 1, H, W, C]
        obs = einops.rearrange(obs, "T V H W C -> T V C H W")
        if self.img_size is not None:
            T, V, C, H, W = obs.shape
            if H != self.img_size or W != self.img_size:
                obs = TF.resize(
                    obs.reshape(T * V, C, H, W),
                    [self.img_size, self.img_size],
                    antialias=True,
                )
                obs = obs.reshape(T, V, C, self.img_size, self.img_size)

        act = torch.from_numpy(grp["action"][frames])       # [T, action_dim]

        # Goal: last frame of the first trajectory in this task
        goal_tk, goal_ik = self._task_goal_traj[task]
        goal_grp = self.hdf5[task][goal_tk]
        goal_img = self._load_view(goal_grp, goal_ik, [-1])  # [1, H, W, C]
        goal_img = goal_img.unsqueeze(1)                      # [1, 1, H, W, C]
        goal_img = einops.rearrange(goal_img, "T V H W C -> T V C H W")
        if self.img_size is not None:
            _, V, C, H, W = goal_img.shape
            if H != self.img_size or W != self.img_size:
                goal_img = TF.resize(
                    goal_img.reshape(V, C, H, W),
                    [self.img_size, self.img_size],
                    antialias=True,
                )
                goal_img = goal_img.reshape(1, V, C, self.img_size, self.img_size)
        goal = goal_img.repeat(len(frames), 1, 1, 1, 1)     # [T, 1, C, H, W]

        data = [obs, act, goal]
        if self.aux_view is not None:
            data.append(
                torch.tensor([self._azimuth[idx]], dtype=torch.float32)
            )  # [1], not scalar
        return data

    def get_states(self, idx: int, frames):
        """Load MuJoCo sim states from the HDF5 ``state`` field."""
        task, tk, _ = self._index[idx]
        grp = self.hdf5[task][tk]
        if "state" in grp:
            return grp["state"][frames]
        state_dir_suffix = "_farther_states"
        demo_dir = tk.replace("traj_", "demo_")
        states_path = (
            Path(self._data_directory)
            / ("libero_goal" + state_dir_suffix)
            / task
            / demo_dir
            / "states.npy"
        )
        return np.load(states_path)[frames]

    def __getitem__(self, idx: int):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
