# Based on DynaMo (datasets/vqbet_repro.py): https://github.com/jeffacce/dynamo_ssl/blob/main/datasets/vqbet_repro.py
import abc
from . import embedding
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError


class TrajectorySlicerDataset(TrajectoryDataset):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        action_window: int,
        vqbet_get_future_action_chunk: bool = True,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
        use_libero_goal: bool = False,
    ):
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.action_window = action_window
        self.vqbet_get_future_action_chunk = vqbet_get_future_action_chunk
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.slices = []
        self.use_libero_goal = use_libero_goal
        min_seq_length = np.inf
        if vqbet_get_future_action_chunk:
            min_window_required = window + action_window
        else:
            min_window_required = max(window, action_window)
        for i in range(len(self.dataset)):  # type: ignore
            T = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - min_window_required < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={min_window_required}"
                )
            else:
                self.slices += [
                    (i, 0, end + 1) for end in range(window - 1)
                ]  # slice indices follow convention [start, end)
                self.slices += [
                    (i, start, start + window)
                    for start in range(T - min_window_required)
                ]  # slice indices follow convention [start, end)

        if min_seq_length < min_window_required:
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
        obs, act, goal, *others = self.dataset[i] # for aux_views
        obs_len = obs.shape[0]
        if end - start < self.window:
            obs = embedding.repeat_start_to_length(
                obs[start:end], self.window, dim=0
            )
            act = embedding.repeat_start_to_length(
                act[start : end - 1 + self.action_window],
                self.window + self.action_window - 1,
                dim=0,
            )
            values = [obs, act]
        else:
            values = [
                obs[start:end],
                act[start : end - 1 + self.action_window],
            ]

        if self.use_libero_goal:
            goals = goal[start:end] # assume (obs, act, goal)
            if end - start < self.window:
                goals = embedding.repeat_start_to_length(
                    goals, self.window, dim=0
                )
        else:
            goals = torch.zeros([1, 1, 1])  # placeholder goal
        values.append(goals)
        for val in others:
            if isinstance(val, torch.Tensor) and val.ndim > 0 and val.shape[0] == obs_len: # assume same length as full trajectory
                # need to slice this too
                if end - start < self.window:
                    val_sliced = embedding.repeat_start_to_length(
                        val[start:end], self.window, dim=0
                    )
                else:
                    val_sliced = val[start:end]
                values.append(val_sliced)
            else:
                values.append(val)
        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        if len(values) == 2:  # placeholder goal
            values.append(torch.ones([1, 1, 1]))
        return tuple(values)
