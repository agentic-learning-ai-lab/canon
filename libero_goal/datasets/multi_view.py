"""CANON encoder training dataset for LIBERO-Goal multi-view data.

Loads a pre-assigned pair of pool cameras per trajectory and returns per-view
angle supervision. Camera azimuths come from ``camera_configs.json`` (the single
source of view definitions). A view's supervision angle ``theta`` is its azimuth
relative to the canonical azimuth (in radians): because changing azimuth rotates
the camera frame rigidly about world-up, the relative camera rotation is a pure
yaw equal to the azimuth difference. A view is canonical iff ``theta == 0``.

Dataset directory (``data_directory``):
    trajectories.hdf5 / <task> / traj_N /
        image_{cam_id}  (T, H, W, 3) uint8      # cam_id is a 0-based pool id
        action          (T, 7)        float32   # retained; the SSL trainer discards it
        state           (T, S)        float32   # retained; not used by the encoder
        attrs: pair_cam_ids  [v1_id, v2_id]
    camera_configs.json:
        { "canonical": {"azimuth", ...}, "pool": [{"id", "azimuth", ...}, ...] }

Returns per sample ``[obs, act, goal, aux]``:
    obs   [T, 2, C, H, W]  float32 in [0, 1]
    act   [T, action_dim]  float32  (loaded; the SSL trainer discards it)
    goal  [T, 2, C, H, W]  last-frame goal repeated across T
    aux   {"theta_v1", "theta_v2"}  per-view yaw relative to canonical (radians)
"""

import abc
import math
import h5py
import json
import numpy as np
import torch
import einops
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Optional, List, Tuple
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Minimal TrajectoryDataset ABC
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset, abc.ABC):
    """Base class for trajectory datasets."""

    @abc.abstractmethod
    def get_seq_length(self, idx: int) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_frames(self, idx: int, frames, view=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LiberoGoalMultiViewDatasetAngle(TrajectoryDataset):
    """CANON encoder training dataset for LIBERO-Goal multi-view HDF5 data.

    Each trajectory stores a pre-assigned pair of pool cameras (``pair_cam_ids``).
    Per-view supervision ``theta`` is the azimuth relative to the JSON ``canonical``
    azimuth; a view is canonical iff ``theta == 0``.

    Args:
        data_directory:  Dataset dir with ``trajectories.hdf5`` + ``camera_configs.json``.
        task_subset:     Only use the first N tasks (``None`` = all).
        subset_fraction: Use the first fraction of demos per task (debugging).
        img_size:        If set, resize images to ``(img_size, img_size)``.
    """

    def __init__(
        self,
        data_directory: str,
        task_subset: Optional[int] = None,
        subset_fraction: Optional[float] = None,
        img_size: Optional[int] = None,
    ):
        dataset_dir = Path(data_directory)
        assert dataset_dir.exists(), (
            f"LiberoGoalMultiViewDatasetAngle: {dataset_dir} does not exist"
        )
        print(f"LiberoGoalMultiViewDatasetAngle: using data dir {dataset_dir}")
        self.img_size = img_size
        self.hdf5_path = str(dataset_dir / "trajectories.hdf5")
        self._hdf5_file = None

        # ── Camera azimuths from JSON (single source of view definitions) ─────
        with open(str(dataset_dir / "camera_configs.json")) as f:
            cam_cfg = json.load(f)
        self.id_to_azimuth = {int(c["id"]): float(c["azimuth"]) for c in cam_cfg["pool"]}
        self.canonical_azimuth = float(cam_cfg["canonical"]["azimuth"])

        # ── Trajectory index: (task, traj_key, v1_id, v2_id) ──────────────────
        self._index: List[Tuple[str, str, int, int]] = []
        with h5py.File(self.hdf5_path, "r") as f:
            task_names = sorted(f.keys())
            if task_subset is not None:
                task_names = task_names[:task_subset]
            for task in task_names:
                traj_keys = sorted(f[task].keys(), key=lambda k: int(k.split("_")[1]))
                if subset_fraction is not None and subset_fraction < 1.0:
                    traj_keys = traj_keys[: max(1, int(len(traj_keys) * subset_fraction))]
                for tk in traj_keys:
                    v1_id, v2_id = f[task][tk].attrs["pair_cam_ids"]
                    self._index.append((task, tk, int(v1_id), int(v2_id)))

        self.task_names = list(dict.fromkeys(item[0] for item in self._index))
        print(
            f"LiberoGoalMultiViewDatasetAngle: {len(self._index)} trajectories "
            f"from {len(self.task_names)} tasks"
        )

    # -- HDF5 lazy handle (one per DataLoader worker) -------------------------

    @property
    def hdf5(self) -> h5py.File:
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

    # -- Dataset interface ----------------------------------------------------

    @property
    def demos(self):
        """Compatibility alias for goal-caching logic."""
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def get_seq_length(self, idx: int) -> int:
        task, tk, _, _ = self._index[idx]
        return self.hdf5[task][tk]["action"].shape[0]

    def get_frames(self, idx: int, frames, view=None) -> list:
        """Return ``[obs, act, goal, aux]`` for the given frame indices.

        aux (dict, per-trajectory):
            theta_v1, theta_v2  float32  view azimuth minus canonical azimuth (radians)
        """
        task, tk, v1_id, v2_id = self._index[idx]
        frames = list(frames)
        grp = self.hdf5[task][tk]

        obs_v1 = torch.from_numpy(grp[f"image_{v1_id}"][frames].astype(np.float32)) / 255.0
        obs_v2 = torch.from_numpy(grp[f"image_{v2_id}"][frames].astype(np.float32)) / 255.0
        obs = torch.stack([obs_v1, obs_v2], dim=1)              # [T, 2, H, W, C]
        obs = einops.rearrange(obs, "T V H W C -> T V C H W")
        if self.img_size is not None:
            T, V, C, H, W = obs.shape
            if H != self.img_size or W != self.img_size:
                obs = TF.resize(
                    obs.reshape(T * V, C, H, W),
                    [self.img_size, self.img_size],
                    antialias=True,
                ).reshape(T, V, C, self.img_size, self.img_size)

        act = torch.from_numpy(grp["action"][frames])   # loaded; the SSL trainer discards it
        goal = obs[[-1]].repeat(len(frames), 1, 1, 1, 1)

        # theta = yaw of the canonical→view rotation = azimuth difference (radians).
        theta_v1 = math.radians(self.id_to_azimuth[v1_id] - self.canonical_azimuth)
        theta_v2 = math.radians(self.id_to_azimuth[v2_id] - self.canonical_azimuth)
        aux = {
            "theta_v1": torch.tensor(theta_v1, dtype=torch.float32),
            "theta_v2": torch.tensor(theta_v2, dtype=torch.float32),
        }
        return [obs, act, goal, aux]

    def __getitem__(self, idx: int):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
