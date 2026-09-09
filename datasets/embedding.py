"""Pre-embedding + sequence-padding helpers for trajectory datasets.

Datasets pre-embed each trajectory through the frozen encoder once (so training reads cached
features instead of re-running the backbone every step) and pad variable-length sequences to a
common window. These live next to the datasets that call them.

(Formerly in utils/inference.py; moved here since every caller is a dataset. `eval_mode` stayed in
utils since the trainer uses it too.)
"""
import torch
import torch.nn as nn
from accelerate import Accelerator

from utils import eval_mode


def embed_trajectory_dataset(
    model,
    dataset,
    obs_only=True,
    device=None,
    embed_goal=False,
):
    if type(model) is nn.parallel.DistributedDataParallel:
        return embed_trajectory_dataset_ddp(
            model,
            dataset,
            obs_only=obs_only,
            device=device,
            embed_goal=embed_goal,
        )
    else:
        result = []
        accelerator = Accelerator()
        device = device or accelerator.device  # result device
        with eval_mode(model, no_grad=True):
            for i in range(len(dataset)):
                # obs, *rest = dataset[i]
                obs, act, goal, *rest = dataset[i]
                obs = obs.to(accelerator.device)
                obs_enc = model(obs).to(device)
                if obs_only:
                    result.append(obs_enc)
                else:
                    goal = goal.to(accelerator.device)
                    if embed_goal:
                        # goal = rest[-1] # assuming goal comes last
                        # rest = rest[:-1]
                        goal_enc = model(goal).to(device)
                        # rest.append(goal_enc)
                    else:
                        goal_enc = goal
                    rest = [x.to(device) if hasattr(x, 'to') else x for x in rest]
                    result.append((obs_enc, act, goal_enc, *rest))
        return result


def embed_trajectory_dataset_so3_proprio(model, dataset, device=None, embed_goal=False):
    """SO(3)+proprio pre-embedding: stores gap, spatial, and robot state per traj.

    Source dataset returns (obs, act, goal, state, *rest) where state has shape
    [T, state_dim] (e.g. [T, 4] for 4-D MetaWorld proprio).

    Returns list of (gap, act, goal_enc, spatial, state, *rest) per trajectory:
        gap     : [T, V, E]
        spatial : [T, V, 512, H, W]   (for angle prediction in canonical warp)
        state   : [T, state_dim]      (passes through unchanged)
    """
    result = []
    accelerator = Accelerator()
    device = device or accelerator.device
    with eval_mode(model, no_grad=True):
        for i in range(len(dataset)):
            obs, act, goal, state, *rest = dataset[i]
            obs = obs.to(accelerator.device)
            gap, spatial = model(obs)
            gap = gap.to(device)
            spatial = spatial.to(device)
            if embed_goal:
                goal = goal.to(accelerator.device)
                goal_enc, _ = model(goal)
                goal_enc = goal_enc.to(device)
            else:
                goal_enc = goal
            state = state.to(device) if hasattr(state, 'to') else state
            rest = [x.to(device) if hasattr(x, 'to') else x for x in rest]
            result.append((gap, act, goal_enc, spatial, state, *rest))
    return result


def embed_trajectory_dataset_ddp(
    model: nn.Module,
    dataset,
    obs_only=True,
    device=None,
    embed_goal=False,
):
    assert type(model) is nn.parallel.DistributedDataParallel, "Model must be DDP"
    embeddings = []
    accelerator = Accelerator()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)
    # get the max trajectory length, so that we can pad tensors for DDP gather
    max_T = max(dataset.get_seq_length(i) for i in range(len(dataset)))
    with eval_mode(model, no_grad=True):
        for obs, *rest in dataloader:
            obs = obs.to(accelerator.device)  # obs shape 1 T V C H W
            obs_enc = model(obs)
            obs_enc = pad_to_length(obs_enc, max_T, dim=1)
            obs_enc = accelerator.gather_for_metrics(obs_enc)
            if obs_only:
                embeddings.append(obs_enc)
            else:
                if embed_goal:
                    # assuming goal comes last
                    goal = rest[-1]
                    rest = rest[:-1]
                    goal = goal.to(accelerator.device)
                    goal_enc = model(goal)
                    rest.append(goal_enc)
                rest = [x.to(accelerator.device) for x in rest]
                rest = [pad_to_length(x, max_T, dim=1) for x in rest]
                rest = [accelerator.gather_for_metrics(x) for x in rest]
                embeddings.append((obs_enc, *rest))

    device = device or accelerator.device
    # unpad the tensors
    result = []
    if obs_only:
        embeddings = torch.cat(embeddings, dim=0)
        assert len(embeddings) == len(dataset)
    else:
        embeddings = [torch.cat(x, dim=0) for x in zip(*embeddings)]
        assert len(embeddings[0]) == len(dataset)
    for i in range(len(dataset)):
        T = dataset.get_seq_length(i)
        if obs_only:
            result.append(embeddings[i, :T].to(device))
        else:
            result.append([x[i, :T].to(device) for x in embeddings])
    return result


def pad_to_length(x: torch.Tensor, length: int, dim: int = 0):
    """
    Pad tensor x to length along dim, adding zeros at the end.
    """
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    pad = torch.zeros(
        *x.shape[:dim],
        pad_size,
        *x.shape[dim + 1 :],
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([x, pad], dim=dim)


def repeat_start_to_length(x: torch.Tensor, length: int, dim: int = 0):
    """
    Pad tensor x to length along dim, repeating the first value at the start.
    """
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    first_frame = x.index_select(dim, torch.tensor(0, device=x.device))
    repeat_shape = [1] * len(x.shape)
    repeat_shape[dim] = pad_size
    pad = first_frame.repeat(*repeat_shape)
    return torch.cat([pad, x], dim=dim)
