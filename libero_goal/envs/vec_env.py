"""
SubprocVecLiberoEnv: N parallel LiberoEnv instances, each in its own subprocess.

Design
------
* Workers are pure CPU (no CUDA). The parent process retains the GPU for
  batched encoder + policy inference, which is fast enough that the GPU is
  rarely the bottleneck.
* Workers are spawned with the `spawn` start method so no CUDA context is
  inherited, making this safe to call from inside a training loop.
* Communication uses multiprocessing.Pipe (one duplex pipe per worker).

Usage
-----
    from libero_goal.envs.vec_env import SubprocVecLiberoEnv

    # Build from Hydra config
    env_spec = SubprocVecLiberoEnv.spec_from_cfg(cfg.env.gym)
    vec = SubprocVecLiberoEnv(n_envs=20, **env_spec)

    obs_list = vec.reset([goal_idx] * 20)     # list of 20 obs arrays
    obs_list, rews, dones, infos = vec.step(actions)
    vec.close()
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Worker function (runs inside each subprocess)
# ---------------------------------------------------------------------------

def _env_worker(pipe: mp.connection.Connection, env_target: str, env_kwargs: dict, asset_path: str):
    """
    Subprocess entry point.

    Creates a LiberoEnv (or subclass) from ``env_target`` / ``env_kwargs`` and
    loops waiting for commands over ``pipe``:

        ('seed',      seed: int)          → None
        ('reset',     goal_idx: int)      → obs  [V, C, H, W]
        ('step',      action: np.ndarray) → (obs, reward: float, done: bool, info: dict)
        ('incr_view', None)               → current_view_idx: int  (or -1 if unsupported)
        ('get_views', None)               → views list
        ('close',     None)               → break
    """
    # Set env vars before any MuJoCo import
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['ASSET_PATH'] = asset_path

    # Dynamic import to avoid CUDA init in the subprocess at import time
    from importlib import import_module
    module_path, cls_name = env_target.rsplit('.', 1)
    cls = getattr(import_module(module_path), cls_name)
    env = cls(**env_kwargs)

    try:
        while True:
            cmd, data = pipe.recv()

            if cmd == 'seed':
                env.seed(data)
                pipe.send(None)

            elif cmd == 'reset':
                obs = env.reset(goal_idx=data)
                pipe.send(obs)

            elif cmd == 'step':
                obs, reward, done, info = env.step(data)
                # Trim info to only lightweight fields that survive pickling
                safe_info = {
                    k: v for k, v in info.items()
                    if isinstance(v, (int, float, bool, str, np.ndarray))
                }
                pipe.send((obs, float(reward), bool(done), safe_info))

            elif cmd == 'incr_view':
                if hasattr(env, 'incr_view'):
                    env.incr_view()
                    pipe.send(env.view_idx if hasattr(env, 'view_idx') else -1)
                else:
                    pipe.send(-1)

            elif cmd == 'set_view':
                # Set the view index explicitly (reset() renders views[view_idx]). Used by the
                # parallel eval to assign a precomputed, goal-balanced view per episode.
                if hasattr(env, 'view_idx'):
                    n_views = len(getattr(env, 'views', []))
                    assert 0 <= data < n_views, f'set_view index {data} out of range [0,{n_views})'
                    env.view_idx = data
                    pipe.send(env.view_idx)
                else:
                    pipe.send(-1)

            elif cmd == 'get_views':
                pipe.send(getattr(env, 'views', []))

            elif cmd == 'close':
                break

            else:
                pipe.send(RuntimeError(f'unknown command: {cmd}'))

    finally:
        pipe.close()


# ---------------------------------------------------------------------------
# Vectorised env
# ---------------------------------------------------------------------------

class SubprocVecLiberoEnv:
    """
    Manages ``n_envs`` LiberoEnv worker subprocesses.

    All communication is synchronous: the caller sends commands to the first
    ``n`` workers (n ≤ n_envs) and blocks until all responses arrive.  The
    remaining workers (if any) stay idle in their ``pipe.recv()`` loop.

    Parameters
    ----------
    n_envs      : Number of parallel env instances to spawn.
    env_target  : Dotted class path, e.g. 'libero_goal.envs.libero_env_perturb.LiberoEnvPerturb'.
    env_kwargs  : Plain-dict kwargs forwarded to the env constructor.
    asset_path  : Value for ASSET_PATH env var (pass ``str(Path.cwd())``).
    """

    def __init__(self, n_envs: int, env_target: str, env_kwargs: dict, asset_path: str):
        self.n_envs = n_envs

        ctx = mp.get_context('spawn')
        self._pipes: List[mp.connection.Connection] = []
        self._procs: List[mp.Process] = []

        for _ in range(n_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_env_worker,
                args=(child_conn, env_target, env_kwargs, asset_path),
                daemon=True,
            )
            p.start()
            child_conn.close()   # parent doesn't need the child end
            self._pipes.append(parent_conn)
            self._procs.append(p)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def seed(self, seeds: List[int]) -> None:
        """Send a seed to each of the first len(seeds) workers."""
        n = len(seeds)
        for pipe, s in zip(self._pipes[:n], seeds):
            pipe.send(('seed', s))
        for pipe in self._pipes[:n]:
            pipe.recv()

    def reset(self, goal_indices: List[int]) -> List[np.ndarray]:
        """
        Reset the first ``len(goal_indices)`` workers.

        Returns
        -------
        obs_list : list of np.ndarray, each [V, C, H, W]
        """
        n = len(goal_indices)
        for pipe, g in zip(self._pipes[:n], goal_indices):
            pipe.send(('reset', g))
        return [self._pipes[i].recv() for i in range(n)]

    def step(self, actions: List[np.ndarray]) -> Tuple[List, List, List, List]:
        """
        Step the first ``len(actions)`` workers.

        Parameters
        ----------
        actions : list of np.ndarray, one per active worker

        Returns
        -------
        (obs_list, rewards, dones, infos): each a list of length n
        """
        n = len(actions)
        for pipe, a in zip(self._pipes[:n], actions):
            pipe.send(('step', a))
        results = [self._pipes[i].recv() for i in range(n)]
        obs, rewards, dones, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(infos)

    def incr_view(self, n: Optional[int] = None) -> List[int]:
        """
        Call ``incr_view()`` on the first ``n`` workers (default: all).
        Returns the new view_idx for each worker (-1 if unsupported).
        """
        n = n if n is not None else self.n_envs
        for pipe in self._pipes[:n]:
            pipe.send(('incr_view', None))
        return [self._pipes[i].recv() for i in range(n)]

    def set_view(self, view_indices: List[int]) -> List[int]:
        """Set ``view_idx`` on the first ``len(view_indices)`` workers, one per entry.
        Returns the applied view_idx per worker (-1 if unsupported)."""
        for pipe, v in zip(self._pipes, view_indices):
            pipe.send(('set_view', int(v)))
        return [self._pipes[i].recv() for i in range(len(view_indices))]

    def get_views(self, n: Optional[int] = None) -> List[list]:
        """Return the ``views`` list from each worker."""
        n = n if n is not None else self.n_envs
        for pipe in self._pipes[:n]:
            pipe.send(('get_views', None))
        return [self._pipes[i].recv() for i in range(n)]

    def close(self) -> None:
        """Gracefully terminate all workers."""
        for pipe in self._pipes:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def spec_from_cfg(gym_cfg) -> dict:
        """
        Convert a Hydra ``env.gym`` OmegaConf node to plain-dict kwargs
        suitable for SubprocVecLiberoEnv.__init__.

        Example
        -------
            spec = SubprocVecLiberoEnv.spec_from_cfg(cfg.env.gym)
            vec  = SubprocVecLiberoEnv(n_envs=20, **spec, asset_path=str(Path.cwd()))
        """
        from omegaconf import OmegaConf
        d = OmegaConf.to_container(gym_cfg, resolve=True)
        env_target = d.pop('_target_')
        return {'env_target': env_target, 'env_kwargs': d}
