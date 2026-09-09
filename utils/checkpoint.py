"""Checkpoint loading helpers.

Checkpoints in this release store whole `nn.Module` objects (not state_dicts), so loading them
needs `weights_only=False` and, for snapshots written by multi-GPU runs, a process group so the
pickled DDP wrappers can be reconstructed. That plumbing lives here rather than being duplicated in
each benchmark's train_policy.py.
"""
import os
from pathlib import Path

import torch


def load_model(save_path: Path):
    print('loading', save_path)
    with (save_path).open("rb") as f:
        return torch.load(f, weights_only=False)


def load_encoder(path):
    """Load a frozen encoder snapshot for policy training.

    Supports two formats and returns the same ``{encoder, view_ssl, view_projector, epoch}``
    dict of bare ``nn.Module`` objects in both cases:

      * **state_dict bundle** (``format='canon-encoder-statedict-v1'``): the released format.
        Rebuilds ``encoder`` / ``view_projector`` / ``view_ssl`` from the config embedded in the
        bundle and loads the weights (``strict=True``). Self-contained: no external YAML needed,
        no dependence on the class names used when the checkpoint was originally trained.
      * **legacy whole-module snapshot**: a pickled dict of ``nn.Module`` objects, delegated to
        :func:`load_snapshot_safe` (requires the module class paths to still exist in the code).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("format") == "canon-encoder-statedict-v1":
        import hydra.utils as hu
        from omegaconf import OmegaConf
        enc = hu.instantiate(OmegaConf.create(payload["encoder_cfg"]))
        enc.load_state_dict(payload["encoder_state_dict"], strict=True)
        vp = hu.instantiate(OmegaConf.create(payload["view_projector_cfg"]), _recursive_=False)
        vp.load_state_dict(payload["view_projector_state_dict"], strict=True)
        vs = hu.instantiate(OmegaConf.create(payload["view_ssl_cfg"]), encoder=enc, projector=vp)
        vs.load_state_dict(payload["view_ssl_state_dict"], strict=True)
        return {"encoder": enc, "view_ssl": vs, "view_projector": vp,
                "epoch": payload.get("epoch", -1)}
    return load_snapshot_safe(path)


def load_policy(path):
    """Load a trained BC policy checkpoint, returning the bare policy ``nn.Module``.

    Supported formats:
      * ``canon-policy-statedict-v1``: DiTFlow policies. Rebuilt from the model config embedded
        in the bundle and loaded (``strict=True``). Robust to the class move that renamed the DiT
        module path during the public cleanup.
      * ``canon-policy-wholemodule-v1``: VQ-BeT. Its vendored module path is unchanged, so the
        pickled module is stable and returned directly.
      * legacy trainer dict ``{'policy': <module>, ...}``: returns ``payload['policy']``.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    fmt = payload.get("format") if isinstance(payload, dict) else None
    if fmt == "canon-policy-statedict-v1":
        import hydra.utils as hu
        from omegaconf import OmegaConf
        pol = hu.instantiate(OmegaConf.create(payload["model_cfg"]))
        pol.load_state_dict(payload["policy_state_dict"], strict=True)
        return pol
    if fmt == "canon-policy-wholemodule-v1":
        return payload["policy"]
    if isinstance(payload, dict) and "policy" in payload:
        return payload["policy"]
    return payload


def load_snapshot_safe(path):
    """Load a snapshot that may contain DDP-wrapped objects from multi-GPU training.

    Unpickling a `DistributedDataParallel` requires an initialized process group, so if none exists
    we stand up a temporary single-rank gloo group backed by a scratch FileStore, load, then tear it
    down again. Any DDP wrappers found in the payload are unwrapped to the bare module.
    """
    _pg_initialized_here = False
    if not torch.distributed.is_initialized():
        import tempfile
        store_file = tempfile.mktemp(suffix='.store')
        store = torch.distributed.FileStore(store_file, 1)
        torch.distributed.init_process_group(
            backend='gloo', store=store, rank=0, world_size=1,
        )
        _pg_initialized_here = True

    snapshot = torch.load(path, weights_only=False)

    if _pg_initialized_here:
        torch.distributed.destroy_process_group()
        if os.path.exists(store_file):
            os.remove(store_file)

    for key, val in snapshot.items():
        while isinstance(val, torch.nn.parallel.DistributedDataParallel):
            val = val.module
        snapshot[key] = val

    return snapshot
