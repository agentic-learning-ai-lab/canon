import os
import torch
import random
import numpy as np
from pathlib import Path
from hydra.types import RunMode
from typing import Callable, Dict
from hydra.core.hydra_config import HydraConfig


def set_full_precision():
    print('full precision on')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

def set_seed_everywhere(seed):
    print(f'Setting random seed to {seed}')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def reduce_dict(f: Callable, d: Dict):
    return {k: reduce_dict(f, v) if isinstance(v, dict) else f(v) for k, v in d.items()}


def get_hydra_jobnum_workdir():
    if HydraConfig.get().mode == RunMode.MULTIRUN:
        job_num = HydraConfig.get().job.num
        work_dir = Path(HydraConfig.get().sweep.dir) / HydraConfig.get().sweep.subdir
    else:
        job_num = 0
        work_dir = HydraConfig.get().run.dir
    return job_num, work_dir


def set_env_vars():
    """Set the process-level env vars the entry points rely on, without clobbering the caller's."""
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    if 'HYDRA_FULL_ERROR' not in os.environ:
        os.environ['HYDRA_FULL_ERROR'] = '1'
    if 'ASSET_PATH' not in os.environ:
        os.environ['ASSET_PATH'] = os.getcwd()


class eval_mode:
    """Context manager: put models in eval() (optionally under no_grad), restore state on exit.

    Used by the trainer for evaluation and by the dataset pre-embedding helpers
    (datasets/embedding.py). None models are skipped (e.g. view-only trainers set projector=None).
    """
    def __init__(self, *models, no_grad=False):
        self.models = models
        self.no_grad = no_grad
        self.no_grad_context = torch.no_grad()

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            if model is None:
                self.prev_states.append(None)
                continue
            self.prev_states.append(model.training)
            model.train(False)
        if self.no_grad:
            self.no_grad_context.__enter__()

    def __exit__(self, *args):
        if self.no_grad:
            self.no_grad_context.__exit__(*args)
        for model, state in zip(self.models, self.prev_states):
            if model is None:
                continue
            model.train(state)
        return False
