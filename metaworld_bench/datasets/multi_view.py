"""MetaWorld multi-view dataset for CANON encoder (SSL) pretraining.

Standalone file: no imports from the private dynamo_ssl package.

HDF5 layout:
    trajectories.hdf5 / <task_name> / traj_N /
        image_{i}    (T, H, W, 3)  uint8   i in 0..N_cams-1  (pool cameras)
        action       (T, 4)        float32   # retained; the SSL trainer discards it
        state        (T, 39)       float32   # retained; not used by the encoder
    (raw-sim 'mujoco_state'/'reward' are not included in the release.)

camera_configs.json: {"pool": [{id, azimuth, elevation, distance, lookat?}, ...]}
  or a plain list (new format): [{id, azimuth, elevation, distance}, ...]

Dataset class:
    MetaworldGoalMultiViewDataset  – on-the-fly 2-view sampling, returns
                                     [obs, act, goal, aux] for SO(3) CANON encoder pretraining.

aux (dict), per view pair:
    rot6d_v1v2: pairwise rotation v1→v2   (6D rep, Zhou 2019)
    rot6d_v1_can: v1→canonical rotation     (angle_head GT for v1)
    rot6d_v2_can: v2→canonical rotation     (angle_head GT for v2)
    is_canonical: True if either view is the canonical camera
    is_can_v1: True if v1 is the canonical camera
    is_can_v2: True if v2 is the canonical camera
"""

import json
import h5py
import numpy as np
import torch
import einops
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Camera utility functions (compute camera rotation from JSON azimuth/elevation/distance)
# ---------------------------------------------------------------------------

def compute_camera_xyaxes(target_coord, distance, azimuth_deg, elevation_deg):
    """Compute camera position and xyaxes for MuJoCo camera specification.

    MuJoCo conventions:
    - Right-handed coordinate system with Z-axis vertical
    - xyaxes: 6D vector [x_right, y_right, z_right, x_down, y_down, z_down]
      where first 3 elements define camera X-axis (right)
      and last 3 elements define camera Y-axis (down)
    - Camera Z-axis (forward/optical axis) is implicit: Z = X × Y

    Args:
        target_coord: (3,) array of the target's XYZ position (m).
        distance: Radial distance from target to camera (m).
        azimuth_deg: Angle around the Z-axis (yaw/lateral) (degrees, starts from X towards Y).
        elevation_deg: Angle above the XY plane (pitch/vertical) (degrees).

    Returns:
        Tuple: (camera_position as list, xyaxes_vector as list)
    """
    target_coord = np.asarray(target_coord)

    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)

    # Compute camera position relative to target
    x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = distance * np.sin(elevation_rad)

    camera_position = target_coord + np.array([x, y, z])

    # Camera Z-axis points from camera to target (forward direction)
    forward = camera_position - target_coord
    forward = forward / np.linalg.norm(forward)

    # Define world up vector (Z-axis in MuJoCo)
    world_up = np.array([0, 0, 1])

    # Compute camera right vector (X-axis)
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)

    # Handle gimbal lock when camera is directly above/below target
    if right_norm < 1e-6:
        # Use alternative reference when looking straight up/down
        world_ref = np.array([1, 0, 0])
        right = np.cross(world_ref, forward)

    right = right / np.linalg.norm(right)

    # Compute camera down vector (Y-axis)
    # Z = X × Y, so Y = Z × X
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # Construct xyaxes: [x_right, y_right, z_right, x_down, y_down, z_down]
    xyaxes_vector = np.concatenate([right, down])

    return camera_position.tolist(), xyaxes_vector.tolist()


def xyaxes_to_rotation(xyaxes_vector: np.ndarray) -> R:
    """Convert a MuJoCo 6D xyaxes vector [Xx, Xy, Xz, Yx, Yy, Yz] into a
    scipy.spatial.transform.Rotation object.

    Args:
        xyaxes_vector: (6,) array representing the X and Y axes of the frame.

    Returns:
        scipy.spatial.transform.Rotation: The rotation object.
    """
    if len(xyaxes_vector) != 6:
        raise ValueError("Input xyaxes_vector must have shape (6,)")

    X_axis = xyaxes_vector[:3]
    Y_axis = xyaxes_vector[3:]

    # Z = X cross Y  (right-hand rule)
    Z_axis = np.cross(X_axis, Y_axis)

    # Columns of the rotation matrix are the basis vectors of the local frame
    rotation_matrix = np.column_stack([X_axis, Y_axis, Z_axis])

    # from_matrix handles normalization/orthogonalization internally
    return R.from_matrix(rotation_matrix)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MetaworldGoalMultiViewDataset(Dataset):
    """SO(3)+distance dataset for MetaWorld CANON encoder pretraining.

    Loads from MetaWorld HDF5 data (all tasks in a single trajectories.hdf5,
    with variable-elevation and variable-distance camera pool).  At training time,
    two cameras are sampled on the fly using a seeded RNG (self.seed → self._rng),
    giving diverse view pairs across iterations without pre-assigning pairs at
    collection time.

    camera_configs.json must include per-camera 'azimuth', 'elevation', and
    'distance' entries.  Images are stored at 224×224 natively; pass
    img_size=None (default) to skip resize.

    Returns [obs, act, goal, aux]:
        obs       [T, 2, C, H, W]  float32  in [0, 1]
        act       [T, action_dim]  float32  (loaded; the SSL trainer discards it)
        goal      [T, 2, C, H, W]  float32  (last frame repeated)
        aux (dict):  see module docstring

    Args:
        data_directory:  Path containing trajectories.hdf5 and camera_configs.json.
        seed:            Master random seed.  All stochastic ops use
                         self._rng = numpy.random.default_rng(seed).  In multi-worker
                         DataLoaders, pass a worker_init_fn that offsets per worker:
                         ``dataset._rng = np.random.default_rng(dataset.seed + worker_id)``.
        img_size:        If set, resize images to (img_size × img_size) via bilinear.
        camera_ids:      Subset of camera IDs (from camera_configs.json) available for
                         sampling.  None = all cameras in camera_configs.json.
        canonical_view:  Dict with keys {azimuth, elevation, distance} describing the
                         canonical viewpoint.
                         Default: {'azimuth': 136, 'elevation': -25, 'distance': 1.3}
                         (136° ≈ camera ID 6, exactly in the training pool: the
                         "seen" canonical).
        task_subset:     Keep only the first N tasks (alphabetical order).
        subset_fraction: Keep only this fraction of trajectories per task (≥1 always kept).

    Note on canonical_cam_id: uses azimuth-only proximity (argmin |az_cid - az_can|).
    For full SO(3) correctness one could replace with geodesic argmin, but azimuth-only
    works well in practice because pool cameras are separated primarily by azimuth.
    """

    def __init__(
        self,
        data_directory: str,
        seed: int = 42,
        img_size: Optional[int] = None,
        camera_ids: Optional[List[int]] = None,
        task_subset: Optional[int] = None,
        subset_fraction: Optional[float] = None,
    ):
        super().__init__()
        data_dir = Path(data_directory)
        assert data_dir.exists(), f"MetaworldGoalMultiViewDataset: {data_dir} does not exist"
        print(f"MetaworldGoalMultiViewDataset: using data dir {data_dir}")

        self.seed = seed
        self.img_size = img_size
        self._rng = np.random.default_rng(seed)

        # ── Camera configs ─────────────────────────────────────────────────────
        # cam_cfg may be {"pool": [...]} (old format) or a plain list (new format).
        cfg_path = data_dir / "camera_configs.json"
        assert cfg_path.exists(), f"camera_configs.json not found in {data_dir}"
        with open(cfg_path) as f:
            cam_cfg_raw = json.load(f)
        cam_list = cam_cfg_raw['pool'] if isinstance(cam_cfg_raw, dict) else cam_cfg_raw

        self.id_to_azimuth:   Dict[int, float] = {c['id']: float(c['azimuth'])   for c in cam_list}
        self.id_to_elevation: Dict[int, float] = {c['id']: float(c['elevation']) for c in cam_list}
        self.id_to_distance:  Dict[int, float] = {c['id']: float(c['distance'])  for c in cam_list}

        # lookat is fixed across all cameras in the pool
        lookat = np.array(cam_list[0].get('lookat', [0.0, 0.5, 0.0]))

        all_cam_ids = sorted(self.id_to_azimuth.keys())
        if camera_ids is not None:
            invalid = [c for c in camera_ids if c not in self.id_to_azimuth]
            if invalid:
                raise ValueError(
                    f"MetaworldGoalMultiViewDataset: camera_ids contains unknown IDs: {invalid}. "
                    f"Valid IDs from camera_configs.json: {all_cam_ids}"
                )
            self.pool_cam_ids: List[int] = sorted(int(c) for c in camera_ids)
        else:
            self.pool_cam_ids = all_cam_ids

        assert len(self.pool_cam_ids) >= 2, (
            f"Need at least 2 cameras to sample a pair, got {self.pool_cam_ids}"
        )

        # ── Canonical view (from JSON: the single source of view definitions) ──
        assert isinstance(cam_cfg_raw, dict) and 'canonical' in cam_cfg_raw, (
            "camera_configs.json must contain a 'canonical' entry (azimuth/elevation/distance)"
        )
        self.canonical_view: dict = dict(cam_cfg_raw['canonical'])

        # Canonical camera ID: azimuth-nearest pool camera. is_canonical=1 when either
        # sampled view is this camera; used to upweight canonical pairs (oversample weight).
        canonical_az = float(self.canonical_view['azimuth'])
        self.canonical_cam_id: int = min(
            self.pool_cam_ids,
            key=lambda cid: abs(self.id_to_azimuth[cid] - canonical_az)
        )

        # ── Precompute per-camera SO(3) rotation matrices ──────────────────────
        self._cam_R: Dict[int, np.ndarray] = {}
        for c in cam_list:
            cid = int(c['id'])
            if cid in self.pool_cam_ids:
                self._cam_R[cid] = self._compute_cam_R(
                    c['azimuth'], c['elevation'], c['distance'], lookat
                )

        # Canonical rotation matrix
        self._canonical_R: np.ndarray = self._compute_cam_R(
            self.canonical_view['azimuth'],
            self.canonical_view['elevation'],
            self.canonical_view['distance'],
            lookat,
        )

        # ── Trajectory index: (hdf5_path_str, task_name, traj_key) ────────────
        hdf5_path = data_dir / "trajectories.hdf5"
        assert hdf5_path.exists(), f"trajectories.hdf5 not found in {data_dir}"

        with h5py.File(str(hdf5_path), 'r') as f:
            all_tasks = sorted(f.keys())
            task_name_set: Optional[set] = (
                set(all_tasks[:task_subset]) if task_subset is not None else None
            )
            self._index: List[Tuple[str, str, str]] = []
            for task in all_tasks:
                if task_name_set is not None and task not in task_name_set:
                    continue
                traj_keys = sorted(
                    f[task].keys(), key=lambda k: int(k.split('_')[1])
                )
                n = len(traj_keys)
                if subset_fraction is not None and subset_fraction < 1.0:
                    n = max(1, int(n * subset_fraction))
                    traj_keys = traj_keys[:n]
                for tk in traj_keys:
                    self._index.append((str(hdf5_path), task, tk))

        self.task_names: List[str] = list(dict.fromkeys(item[1] for item in self._index))
        print(
            f"MetaworldGoalMultiViewDataset: {len(self._index)} trajectories "
            f"from {len(self.task_names)} tasks | "
            f"pool cameras ({len(self.pool_cam_ids)}): {self.pool_cam_ids}"
        )

        # ── Lazy HDF5 handle ──────────────────────────────────────────────────
        self._hdf5_handles: Dict[str, Optional[h5py.File]] = {
            str(hdf5_path): None
        }

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

    # ── Camera rotation helpers ────────────────────────────────────────────────

    @staticmethod
    def _compute_cam_R(az: float, el: float, dist: float, lookat: np.ndarray) -> np.ndarray:
        """Compute (3,3) float32 SO(3) rotation matrix from camera intrinsics."""
        _, xyaxes = compute_camera_xyaxes(np.array(lookat), dist, az, el)
        return xyaxes_to_rotation(np.array(xyaxes)).as_matrix().astype(np.float32)

    @staticmethod
    def _mat_to_rot6d(R_mat: np.ndarray) -> np.ndarray:
        """6D rotation representation (Zhou et al. 2019): first two columns of R."""
        return np.concatenate([R_mat[:, 0], R_mat[:, 1]]).astype(np.float32)  # (6,)

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

    def get_frames(self, idx: int, frames, view=None):
        """Return [obs, act, goal, aux] with SO(3)+dist targets.

        View pair is drawn from self.pool_cam_ids using self._rng, which is
        seeded by self.seed.  Each call advances the RNG state, giving varied
        pairs across training iterations.

        aux (dict): see module docstring.
        """
        hdf5_path, task, tk = self._index[idx]
        frames_list = list(frames)
        grp = self._get_hdf5(hdf5_path)[task][tk]

        # View sampling (seeded RNG, advances state each call)
        sampled = self._rng.choice(len(self.pool_cam_ids), size=2, replace=False)
        v1_id = self.pool_cam_ids[int(sampled[0])]
        v2_id = self.pool_cam_ids[int(sampled[1])]

        obs_v1 = torch.from_numpy(
            grp[f'image_{v1_id}'][frames_list].astype(np.float32)
        ) / 255.0  # [T, H, W, C]
        obs_v2 = torch.from_numpy(
            grp[f'image_{v2_id}'][frames_list].astype(np.float32)
        ) / 255.0

        obs = torch.stack([obs_v1, obs_v2], dim=1)           # [T, 2, H, W, C]
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

        act  = torch.from_numpy(grp['action'][frames_list])   # loaded; the SSL trainer discards it
        goal = obs[[-1]].repeat(len(frames_list), 1, 1, 1, 1)

        # ── SO(3)+dist targets ─────────────────────────────────────────────────
        R_v1  = self._cam_R[v1_id]
        R_v2  = self._cam_R[v2_id]
        R_can = self._canonical_R
        # δR = R_src^T @ R_tgt (rotation src→tgt frame), as 6D rep (Zhou 2019)
        rot6d_v1v2   = self._mat_to_rot6d(R_v1.T @ R_v2)    # [6] pairwise: v1→v2
        rot6d_v1_can = self._mat_to_rot6d(R_v1.T @ R_can)   # [6] v1→canonical
        rot6d_v2_can = self._mat_to_rot6d(R_v2.T @ R_can)   # [6] v2→canonical

        is_can_v1 = v1_id == self.canonical_cam_id
        is_can_v2 = v2_id == self.canonical_cam_id
        aux = {
            "rot6d_v1v2":   torch.tensor(rot6d_v1v2,   dtype=torch.float32),
            "rot6d_v1_can": torch.tensor(rot6d_v1_can, dtype=torch.float32),
            "rot6d_v2_can": torch.tensor(rot6d_v2_can, dtype=torch.float32),
            "is_can_v1":    torch.tensor(is_can_v1),
            "is_can_v2":    torch.tensor(is_can_v2),
            "is_canonical": torch.tensor(is_can_v1 or is_can_v2),
        }
        return [obs, act, goal, aux]

    def __getitem__(self, idx: int):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
