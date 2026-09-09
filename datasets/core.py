import abc
from . import embedding
import torch
import numpy as np
from torch import default_generator, randperm
from torch.utils.data import Dataset, Subset
from typing import Callable, Optional, Sequence, List, Any
from torch.nn.utils.rnn import pad_sequence


# Taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: False: invalid; True: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_frames(self, idx, frames):
        """
        Returns the frames from the idx-th trajectory at the specified frames.
        Used to speed up slicing.
        """
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.
    For debug purposes with faster iterations.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def get_frames(self, idx, frames):
        return self.dataset.get_frames(self.indices[idx], frames)


class TrajectorySlicerDataset:
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
        num_extra_predicted_actions: Optional[int] = None,
        frame_step: int = 1,
        repeat_first_frame: bool = False,
    ):
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.num_extra_predicted_actions = num_extra_predicted_actions or 0
        self.slices = []
        self.frame_step = frame_step
        min_seq_length = np.inf
        if num_extra_predicted_actions:
            window = window + num_extra_predicted_actions
        for i in range(len(self.dataset)):
            T = self.dataset.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)
            if T - window < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={window}")
            else:
                if repeat_first_frame:
                    self.slices += [(i, 0, end + 1) for end in range(window - 1)]
                window_len_with_step = (window - 1) * frame_step + 1
                last_start = T - window_len_with_step
                self.slices += [
                    (i, start, start + window_len_with_step)
                    for start in range(last_start + 1)
                ]

        if min_seq_length < window:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        T = self.dataset.get_seq_length(i)
        if (
            self.num_extra_predicted_actions is not None
            and self.num_extra_predicted_actions != 0
        ):
            assert self.frame_step == 1, "NOT TESTED"
            if self.future_conditional:
                raise NotImplementedError(
                    "num_extra_predicted_actions with future_conditional not implemented"
                )
            assert end <= T, f"end={end} > T={T}"
            observations, actions, mask = self.dataset.get_frames(i, range(start, end))
            observations = observations[: self.window]
            values = [observations, actions, mask.bool()]
        else:
            if self.future_conditional:
                assert self.frame_step == 1, "NOT TESTED"
                valid_start_range = (
                    end + self.min_future_sep,
                    self.dataset.get_seq_length(i) - self.future_seq_len,
                )
                if valid_start_range[0] < valid_start_range[1]:
                    if self.only_sample_tail:
                        future_obs_range = range(T - self.future_seq_len, T)
                    else:
                        future_start = np.random.randint(*valid_start_range)
                        future_end = future_start + self.future_seq_len
                        future_obs_range = range(future_start, future_end)
                    obs, actions, mask = self.dataset.get_frames(
                        i, list(range(start, end)) + list(future_obs_range)
                    )
                    future_obs = obs[end - start :]
                    obs = obs[: end - start]
                    actions = actions[: end - start]
                    mask = mask[: end - start]
                else:
                    obs, actions, mask = self.dataset.get_frames(i, range(start, end))
                    obs_dims = obs.shape[1:]
                    future_obs = torch.zeros((self.future_seq_len, *obs_dims))
                values = [obs, actions, mask.bool(), future_obs]
            else:
                observations, actions, mask, *rest = self.dataset.get_frames(
                    i, range(start, end, self.frame_step)
                )
                values = [observations, actions, mask, *rest]

        if end - start < self.window + self.num_extra_predicted_actions:
            values = [
                embedding.repeat_start_to_length(
                    x, self.window + self.num_extra_predicted_actions, dim=0
                )
                for x in values
            ]
            values[0] = values[0][: self.window]

        if self.transform is not None:
            values = self.transform(values)
        return tuple(values)


class TrajectoryEmbeddingDataset(TrajectoryDataset):
    def __init__(
        self,
        model,
        dataset: TrajectoryDataset,
        device="cpu",
        embed_goal=False,
    ):
        self.data = embedding.embed_trajectory_dataset(
            model,
            dataset,
            obs_only=False,
            device=device,
            embed_goal=embed_goal,
        )
        assert len(self.data) == len(dataset)

        self.seq_lengths = [len(x[0]) for x in self.data]
        n_tensors = len(self.data[0])
        # Detect time dimension before padding: a tensor slot has a time dim iff
        # its first-trajectory shape[0] matches that trajectory's sequence length.
        self.has_time_dim = [
            self.data[0][k].shape[0] == self.seq_lengths[0]
            for k in range(n_tensors)
        ]
        self.on_device_data = []
        for i in range(n_tensors):
            self.on_device_data.append(
                pad_sequence([x[i] for x in self.data], batch_first=True).to(device)
            )
        self.data = self.on_device_data

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_frames(self, idx, frames):
        return [
            x[idx, frames] if has_t else x[idx]
            for x, has_t in zip(self.data, self.has_time_dim)
        ]

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


class TrajectoryEmbeddingDatasetSO3Proprio(TrajectoryEmbeddingDataset):
    """SO(3)+proprio pre-embedding: stores gap, spatial, and robot state per traj.

    Data format: (gap [T,V,E], act [T,A], goal [T,V,E], spatial [T,V,512,H,W],
                  state [T, state_dim], *extra)
    Both spatial (index 3) and state (index 4) have time dims and will be windowed
    by PolicySlicerDatasetSO3Proprio.
    """

    def __init__(self, model, dataset, device="cpu", embed_goal=False):
        from datasets.embedding import embed_trajectory_dataset_so3_proprio
        self.data = embed_trajectory_dataset_so3_proprio(
            model, dataset, device=device, embed_goal=embed_goal
        )
        assert len(self.data) == len(dataset)

        self.seq_lengths = [len(x[0]) for x in self.data]
        n_tensors = len(self.data[0])
        self.has_time_dim = [
            self.data[0][k].shape[0] == self.seq_lengths[0]
            for k in range(n_tensors)
        ]
        self.on_device_data = []
        for i in range(n_tensors):
            self.on_device_data.append(
                pad_sequence([x[i] for x in self.data], batch_first=True).to(device)
            )
        self.data = self.on_device_data


class PolicySlicerDataset(Dataset):
    """
    Slices trajectories into (obs_window, act_chunk, goal, *rest) samples for policy learning.

    Designed for flow-matching / diffusion policies that predict a future action chunk
    conditioned on a window of past observations and a task goal.

    Observation window ends at timestep t (inclusive); action chunk starts at t.
    The start of the observation window is padded with the first frame when t < obs_window.

    Compatible with datasets returning (obs, act, goal, *rest) from get_frames(), including
    TrajectoryEmbeddingDataset wrapping LiberoGoalMultiViewDataset* variants.

    Returns per-sample:
        obs  : [obs_window, ...]  – observation window, front-padded with first frame if needed
        act  : [action_chunk, A] – action chunk starting at t
        goal : [...]             – task goal (first timestep of goal tensor, same for all t)
        rest : passed through as-is (e.g., aux_views [D])
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        obs_window: int,
        action_chunk: int,
    ):
        self.dataset = dataset
        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.slices: List[tuple] = []

        for i in range(len(dataset)):
            T = dataset.get_seq_length(i)
            # t is the last observed timestep; act = traj[t : t+action_chunk]
            # valid range: t in [0, T - action_chunk] (inclusive)
            for t in range(T - action_chunk + 1):
                self.slices.append((i, t))

        print(
            f"PolicySlicerDataset: {len(self.slices)} samples from {len(dataset)} trajectories "
            f"(obs_window={obs_window}, action_chunk={action_chunk})"
        )

    def __len__(self) -> int:
        return len(self.slices)

    def get_seq_length(self, idx: int) -> int:
        return self.obs_window

    def __getitem__(self, idx: int) -> tuple:
        i, t = self.slices[idx]

        # Load only the contiguous range that covers both the obs window and action chunk,
        # avoiding a full-trajectory load on every call.
        obs_start = max(0, t - self.obs_window + 1)
        load_end  = t + self.action_chunk           # exclusive
        items = self.dataset.get_frames(i, range(obs_start, load_end))
        obs_raw, act_raw, goal_raw, *rest_full = items

        # --- Observation window ------------------------------------------------
        # Within the loaded range, t is at relative index t_rel = t - obs_start.
        t_rel = t - obs_start
        obs = obs_raw[: t_rel + 1]                  # [<=obs_window, ...]
        if obs.shape[0] < self.obs_window:
            pad_len = self.obs_window - obs.shape[0]
            pad = obs[:1].expand(pad_len, *obs.shape[1:]).clone()
            obs = torch.cat([pad, obs], dim=0)      # [obs_window, ...]

        # --- Action chunk ------------------------------------------------------
        act = act_raw[t_rel : t_rel + self.action_chunk]   # [action_chunk, A]

        # --- Goal --------------------------------------------------------------
        # All timesteps share the same goal embedding; take the first loaded frame.
        goal = goal_raw[0]

        return (obs, act, goal, *rest_full)


class PolicySlicerDatasetSO3Proprio(PolicySlicerDataset):
    """PolicySlicerDataset for SO(3) + proprio BC: windows gap, spatial, and state.

    Expects ``TrajectoryEmbeddingDatasetSO3Proprio`` format:
        (gap, act, goal, spatial, state, *extra_rest)

    Returns: ``(obs_gap, act, goal, spatial_windowed, state_windowed, *extra_rest)``
        obs_gap          : [obs_window, V, E]
        spatial_windowed : [obs_window, V, 512, H, W]  (same front-pad as obs_gap)
        state_windowed   : [obs_window, state_dim]     (same front-pad as obs_gap)
    """

    def __getitem__(self, idx: int) -> tuple:
        i, t = self.slices[idx]
        obs_start = max(0, t - self.obs_window + 1)
        load_end  = t + self.action_chunk

        items = self.dataset.get_frames(i, range(obs_start, load_end))
        obs_raw, act_raw, goal_raw, spatial_raw, state_raw, *extra_rest = items

        # Observation window (gap)
        t_rel = t - obs_start
        obs = obs_raw[: t_rel + 1]
        if obs.shape[0] < self.obs_window:
            pad_len = self.obs_window - obs.shape[0]
            pad = obs[:1].expand(pad_len, *obs.shape[1:]).clone()
            obs = torch.cat([pad, obs], dim=0)

        # Action chunk
        act = act_raw[t_rel : t_rel + self.action_chunk]

        # Goal (single frame)
        goal = goal_raw[0]

        # Spatial window
        spatial = spatial_raw[: t_rel + 1]
        if spatial.shape[0] < self.obs_window:
            pad_len = self.obs_window - spatial.shape[0]
            pad = spatial[:1].expand(pad_len, *spatial.shape[1:]).clone()
            spatial = torch.cat([pad, spatial], dim=0)

        # State window
        state = state_raw[: t_rel + 1]
        if state.shape[0] < self.obs_window:
            pad_len = self.obs_window - state.shape[0]
            pad = state[:1].expand(pad_len, *state.shape[1:]).clone()
            state = torch.cat([pad, state], dim=0)

        return (obs, act, goal, spatial, state, *extra_rest)


def get_train_val_sliced(
    traj_dataset: TrajectoryDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    window_size: int = 10,
    future_conditional: bool = False,
    min_future_sep: int = 0,
    future_seq_len: Optional[int] = None,
    only_sample_tail: bool = False,
    transform: Optional[Callable[[Any], Any]] = None,
    num_extra_predicted_actions: Optional[int] = None,
    frame_step: int = 1,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    traj_slicer_kwargs = {
        "window": window_size,
        "future_conditional": future_conditional,
        "min_future_sep": min_future_sep,
        "future_seq_len": future_seq_len,
        "only_sample_tail": only_sample_tail,
        "transform": transform,
        "num_extra_predicted_actions": num_extra_predicted_actions,
        "frame_step": frame_step,
    }
    train_slices = TrajectorySlicerDataset(train, **traj_slicer_kwargs)
    val_slices = TrajectorySlicerDataset(val, **traj_slicer_kwargs)
    return train_slices, val_slices


def random_split_traj(
    dataset: TrajectoryDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    """
    (Modified from torch.utils.data.dataset.random_split)

    Randomly split a trajectory dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split_traj(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (TrajectoryDataset): TrajectoryDataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset,
        lengths,
        #generator=torch.Generator().manual_seed(random_seed) # follow global set_seed_everywhere()
    )
    return train_set, val_set
