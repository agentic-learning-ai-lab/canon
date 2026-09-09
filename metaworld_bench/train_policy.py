# Policy training and rollout evaluation for MetaWorld (CANON: canonical-warp DiTFlowSpatial).
# Usage: python train_policy.py --config-path configs --config-name policy \
#        dataset.task_name=dooropenv2 +encoder_path=/path/to/snapshot_epoch399.pt +debug=true

import os
import pickle
from collections import deque
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import tqdm
from omegaconf import OmegaConf, open_dict

from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from accelerate.logging import get_logger

from utils.video import VideoRecorder
from utils.trainer import Trainer, check_wandb, apply_debug_config_policy
from utils import set_env_vars, set_seed_everywhere
from utils.metrics import _get_episode_score, compute_metrics
from utils.checkpoint import load_model, load_snapshot_safe, load_encoder

from datasets.core import (
    split_traj_datasets,
    TrajectoryEmbeddingDatasetSO3Proprio,
    PolicySlicerDatasetSO3Proprio,
)


# ---------------------------------------------------------------------------
# _ensure_spatial_encoder
# ---------------------------------------------------------------------------

def _ensure_spatial_encoder(encoder):
    """Re-cast a `resnet18` instance to `resnet18_spatial` in-place so
    encoder(x) returns (gap, spatial). No-op if already spatial. Drills
    through accelerate / DDP wrappers (`.module` chain).
    """
    from models.encoder import resnet18 as _resnet18, resnet18_spatial as _resnet18_spatial

    target = encoder
    while not isinstance(target, (_resnet18, _resnet18_spatial)) and hasattr(target, 'module'):
        nested = target.module
        if nested is target:
            break
        target = nested

    if isinstance(target, _resnet18_spatial):
        return encoder
    if isinstance(target, _resnet18):
        target.__class__ = _resnet18_spatial
        return encoder
    return encoder


# ---------------------------------------------------------------------------
# PolicyTrainer: MetaWorld BC policy trainer (DiTFlowSpatial + canonical warp)
# ---------------------------------------------------------------------------

class PolicyTrainer(Trainer):
    """CANON MetaWorld BC policy trainer (DiTFlowSpatial policy, canonical-view warp).

    For the release this was flattened from the development codebase's deep view-aware trainer
    inheritance chain into this single class (the deepest class was the only one instantiated).

    Note it subclasses `Trainer` for method reuse only; see `__init__`, which deliberately does
    NOT call `super().__init__()` (that base initializer is for SSL pretraining and expects a
    different config).
    """

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def __init__(self, cfg):
        # NOTE: do NOT call super().__init__(); Trainer.__init__ is for SSL
        # pretraining and has incompatible config requirements.
        self.logger = get_logger(__name__, log_level="DEBUG" if cfg.get('debug', False) else "INFO")
        self.cfg = cfg
        self.mode = cfg.get('mode', 'train')
        self.epoch = 0
        self.epochs = cfg.epochs
        self.use_libero_goal = cfg.data.get("use_libero_goal", False)
        self.use_prop_goal   = cfg.data.get("use_prop_goal", False)

        self.accelerator = None
        self._init_accelerator()

        self.encoder = None
        self._init_encoder()

        self.policy = None
        if self.mode in ['eval']:
            self.setup_eval()
        self._init_policy()

        self.dataset = hydra.utils.instantiate(cfg.dataset)
        ds = self.dataset
        if getattr(ds, 'normalize_actions', False) and getattr(ds, 'act_mean', None) is not None:
            self._act_mean = ds.act_mean.numpy()
            self._act_std  = ds.act_std.numpy()
        else:
            self._act_mean = None
            self._act_std  = None
        if self.mode in ['train', 'tune']:
            self._setup_loaders()

        if self.accelerator.is_main_process:
            self.env_name = self.cfg.env.gym.id
            self.env = hydra.utils.instantiate(cfg.env.gym)

            self.wandb_run = None
            self._init_tracker()

            run_name = self.wandb_run.name or "Offline"
            if not self.cfg.get('debug', False):
                self.save_path = Path(cfg.save_path) / run_name
            else:
                self.save_path = Path(cfg.save_path).parent.parent.parent / 'test'
                print(f'debug mode: saving to {self.save_path}')

            self.save_path.mkdir(parents=True, exist_ok=False if not self.cfg.get('debug', False) else True)

            if self.mode in ['train', 'eval', 'tune'] and not self.cfg.get('debug', False):
                self.videorecorder = VideoRecorder(dir_name=self.save_path)
            else:
                self.videorecorder = None

            if hasattr(self.env, 'mixed_view') and self.env.mixed_view:
                self.reward_dicts = []

            self.metrics_history = []
            self.reward_history  = []

        self.count_params()

    def _init_accelerator(self):
        if self.accelerator is None:
            process_group_kwargs = InitProcessGroupKwargs(
                timeout=timedelta(seconds=self.cfg.get('timeout_seconds', 18000))
            )
            dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(
                mixed_precision=self._get_precision(),
                log_with="wandb",
                kwargs_handlers=[process_group_kwargs, dist_kwargs]
            )
            self.logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
            self.accelerator.wait_for_everyone()

    def _get_precision(self):
        amp = self.cfg.get('amp', False)
        if isinstance(amp, bool):
            return 'fp16' if amp is True else 'no'
        if isinstance(amp, str) and amp in ['no', 'fp16', 'bf16', 'fp8']:
            return amp
        raise NotImplementedError(f'amp: {amp}')

    def _init_encoder(self):
        """Load frozen encoder + view_ssl + view_projector from the snapshot, then re-cast to spatial.

        Loads the snapshot, sets encoder/view_projector/view_ssl, unwraps view_ssl_module, and calls
        _ensure_spatial_encoder to re-cast the backbone to its spatial variant.
        """
        assert 'encoder_path' in self.cfg, 'encoder_path must be set in config'
        self.cfg.encoder_path = self.cfg.encoder_path.replace(
            'encoder_epoch', 'snapshot_epoch'
        )
        snapshot = load_encoder(self.cfg.encoder_path)

        self.encoder = snapshot['encoder']
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.encoder = self.accelerator.prepare(self.encoder)

        _ddp = torch.nn.parallel.DistributedDataParallel
        for attr in ('view_projector', 'view_ssl'):
            m = snapshot[attr]
            for sub_name in list(m._modules.keys()):
                sub = m._modules[sub_name]
                while isinstance(sub, _ddp):
                    sub = sub.module
                m._modules[sub_name] = sub
            for sub_name in list(m.__dict__.keys()):
                sub = m.__dict__[sub_name]
                if isinstance(sub, _ddp):
                    m.__dict__[sub_name] = sub.module
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
            setattr(self, attr, self.accelerator.prepare(m))

        print(f'PolicyTrainer: loaded encoder + view branch from {self.cfg.encoder_path}')

        # Unwrap view_ssl for direct module access (from ViewAnglePred level)
        self.view_ssl_module = self.accelerator.unwrap_model(self.view_ssl)

        # Re-cast encoder to spatial variant (from Spatial level)
        self.encoder = _ensure_spatial_encoder(self.encoder)

    def _init_policy(self):
        cfg = self.cfg
        if self.policy is None:
            self.policy = hydra.utils.instantiate(cfg.model).to(cfg.device)
        if not hasattr(self, 'policy_optim') or self.policy_optim is None:
            self.policy_optim = self.policy.configure_optimizers(
                weight_decay=cfg.optim.weight_decay,
                learning_rate=cfg.optim.lr,
                betas=cfg.optim.betas,
            )
        self.lr_scheduler = None
        (self.policy, self.policy_optim) = self.accelerator.prepare(self.policy, self.policy_optim)

    def _setup_loaders(self):
        """Data pipeline: SO3Proprio embedding + PolicySlicerDatasetSO3Proprio."""
        cfg = self.cfg
        train_data, test_data = split_traj_datasets(
            self.dataset, train_fraction=cfg.train_fraction, random_seed=cfg.seed
        )
        use_libero_goal = self.use_libero_goal
        train_data = TrajectoryEmbeddingDatasetSO3Proprio(
            self.encoder, train_data, device=cfg.device, embed_goal=use_libero_goal
        )
        test_data = TrajectoryEmbeddingDatasetSO3Proprio(
            self.encoder, test_data, device=cfg.device, embed_goal=use_libero_goal
        )

        obs_window   = cfg.data.obs_window
        action_chunk = cfg.data.action_chunk
        train_data = PolicySlicerDatasetSO3Proprio(train_data, obs_window=obs_window, action_chunk=action_chunk)
        test_data  = PolicySlicerDatasetSO3Proprio(test_data,  obs_window=obs_window, action_chunk=action_chunk)

        loader_kwargs = {
            "batch_size":  cfg.batch_size,
            "num_workers": cfg.get('num_workers', 0),
            "pin_memory":  cfg.get('pin_memory', False),
        }
        if torch.cuda.device_count() > 1:
            assert loader_kwargs["batch_size"] % self.accelerator.num_processes == 0
            loader_kwargs["batch_size"] //= self.accelerator.num_processes

        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **loader_kwargs)
        self.test_loader  = torch.utils.data.DataLoader(test_data,  shuffle=False, **loader_kwargs)
        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader  = self.accelerator.prepare(self.test_loader)
        print(f'SO3+proprio loaders: dataset={len(self.dataset)} train={len(train_data)} test={len(test_data)}')

        ds = self.dataset
        if getattr(ds, 'normalize_actions', False) and ds.act_mean is not None:
            self._act_mean = ds.act_mean.numpy()
            self._act_std  = ds.act_std.numpy()
        else:
            self._act_mean = None
            self._act_std  = None

        warmup_steps = cfg.get('lr_warmup_steps', 0)
        if warmup_steps > 0:
            total_steps = len(self.train_loader) * cfg.epochs
            from transformers.optimization import get_cosine_schedule_with_warmup
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.policy_optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

    def _init_tracker(self):
        if self.wandb_run is None:
            cfg = self.cfg
            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            self.accelerator.init_trackers(
                project_name=cfg.wandb.project,
                config=wandb_cfg,
                init_kwargs={"wandb": {
                    "reinit": False,
                    "settings": {"start_method": "thread"},
                    "entity": cfg.wandb.entity,
                }},
            )
            self.wandb_run = self.accelerator.get_tracker("wandb", unwrap=True)
            self.logger.info("wandb run url: %s", self.wandb_run.get_url())
            check_wandb()

    # -----------------------------------------------------------------------
    # Snapshot helpers
    # -----------------------------------------------------------------------

    def save_snapshot(self, version='last'):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            payload = {}
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    payload[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    payload[k] = self.__dict__[k]
                if 'optim' in k:
                    payload[k + '_state_dict'] = payload[k].state_dict()
                    del payload[k]
            with (self.save_path / f"policy_{version}.pt").open("wb") as f:
                torch.save(payload, f)
            print('saved', self.save_path / f"policy_{version}.pt")


    # -----------------------------------------------------------------------
    # Param / model utilities
    # -----------------------------------------------------------------------

    def count_params(self):
        total = 0
        for key, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module):
                param_cnt = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total += param_cnt
                param_cnt_all = sum(p.numel() for p in module.parameters())
                print(key, f'req_grad: {param_cnt} all: {param_cnt_all}')
        print('total req_grad', total)

    def set_model_train(self):
        if self.cfg.get('train_encoder', False):
            self.encoder.train()
        else:
            self.encoder.eval()
        self.policy.train()

    def set_model_eval(self):
        self.encoder.eval()
        self.policy.eval()

    # -----------------------------------------------------------------------
    # Eval / checkpoint helpers
    # -----------------------------------------------------------------------

    def setup_eval(self):
        self.epochs = 0
        assert 'cbet_path' in self.cfg, 'expected `cbet_path` in cfg'
        ckpt = load_model(Path(self.cfg.cbet_path))
        if isinstance(ckpt, dict):
            if isinstance(ckpt['policy'], torch.nn.Module):
                self.policy = ckpt['policy']
                if self.cfg.get('load_optim', False):
                    self.load_policy_state(ckpt)
            else:
                raise NotImplementedError(f'type(ckpt[policy])', type(ckpt['policy']))
        elif isinstance(ckpt, torch.nn.Module):
            self.policy = ckpt
        else:
            raise NotImplementedError(f'type(ckpt)', type(ckpt))
        self.videorecorder = None

    def load_policy_state(self, payload):
        for i, opt_state in enumerate(payload['policy_optim/optimizers']):
            self.policy_optim.optimizer.optimizers[i].load_state_dict(opt_state)
        print('loaded policy_optim state dict')
        (self.policy, self.policy_optim) = self.accelerator.prepare(self.policy, self.policy_optim)

    def save_model(self, version='best'):
        self._keys_to_save = ['policy', 'policy_optim']
        self.save_snapshot(version)

    def save_best_model(self):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        env_name = self.env_name
        save = False
        if env_name == "metaworld_bc":
            save = (
                len(self.reward_history) == 1
                or max(self.reward_history[:-1]) < self.reward_history[-1]
            )
        elif env_name == "pusht":
            save = (
                len(self.metrics_history) == 1
                or max(x["final coverage mean"] for x in self.metrics_history[:-1])
                   < self.metrics_history[-1]["final coverage mean"]
            )
        elif env_name == "blockpush":
            save = (
                len(self.metrics_history) == 1
                or max(x["entered mean"] for x in self.metrics_history[:-1])
                   < self.metrics_history[-1]["entered mean"]
            )
        elif env_name in ("libero_goal", "kitchen-v0"):
            save = (
                len(self.reward_history) == 1
                or max(self.reward_history[:-1]) < self.reward_history[-1]
            )
        if save and self.mode in ['train', 'tune']:
            self.save_model(version='best')

    # -----------------------------------------------------------------------
    # Forward / training step
    # -----------------------------------------------------------------------

    def _goal_for_obs(self, obs_v, goal):
        if self.policy.goal_dim == 0:
            return None
        elif self.use_libero_goal:
            v = self.cfg.get('policy_view_idx', 0)
            return goal[:, v] if goal.ndim == 3 else goal
        elif getattr(self, 'use_prop_goal', False):
            return goal[:, 0, :].to(obs_v.device)
        else:
            return torch.zeros(obs_v.shape[0], self.policy.goal_dim, device=obs_v.device)

    def _warp_to_canonical_spatial(self, spatial_v: torch.Tensor) -> torch.Tensor:
        """spatial_v: [N, T, 512, H, W] → canonical_spatial: [N, T, 512, H, W]."""
        from models.ssl import rot6d_to_matrix
        m = self.view_ssl_module
        N, T, C, H, W = spatial_v.shape

        pred_rot6d_raw = m.angle_head(spatial_v)                               # [N, 6]
        R_pred         = rot6d_to_matrix(pred_rot6d_raw)                        # [N, 3, 3]
        pred_rot6d     = torch.cat([R_pred[:, :, 0], R_pred[:, :, 1]], dim=-1)  # [N, 6]

        if hasattr(m, 'projector') and m.projector is not None:
            rot_lat = m.projector(pred_rot6d)
        else:
            rot_lat = self.view_projector(pred_rot6d)

        spatial_flat   = spatial_v.reshape(N * T, C, H, W)
        rot_lat_flat   = rot_lat.unsqueeze(1).expand(-1, T, -1).reshape(N * T, -1)
        canonical_flat = m.forward_dynamics(spatial_flat, rot_lat_flat)         # [N*T, C, H, W]
        return canonical_flat.reshape(N, T, C, H, W)

    @staticmethod
    def _flatten_spatial_tokens(spatial_v: torch.Tensor) -> torch.Tensor:
        """[N, T, 512, H, W] → [N, T, H*W, 512]."""
        N, T, C, H, W = spatial_v.shape
        return spatial_v.permute(0, 1, 3, 4, 2).reshape(N, T, H * W, C).contiguous()

    def forward(self, data):
        obs_gap, act, goal, spatial, state, *_ = data
        v = self.cfg.get('policy_view_idx', 0)
        spatial_v         = spatial[:, :, v]
        canonical_spatial = self._warp_to_canonical_spatial(spatial_v)
        obs_tokens        = self._flatten_spatial_tokens(canonical_spatial)
        goal_v            = self._goal_for_obs(obs_gap[:, :, v], goal)
        return self.forward_policy((obs_tokens, state, goal_v, act))

    def forward_policy(self, data):
        obs_tokens, state, goal_v, act = data
        return self.policy(obs_tokens, state, goal_v, act)

    def opt_step(self):
        max_grad_norm = self.cfg.get('max_grad_norm', None)
        if max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.policy_optim.step()
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def opt_zerograd(self):
        self.policy_optim.zero_grad(set_to_none=True)

    def train_step(self, data, grad_acc_step=1):
        with self.accelerator.autocast():
            _, loss, loss_dict = self.forward(data)
        if grad_acc_step > 1:
            loss = loss / grad_acc_step
        self.accelerator.backward(loss)
        if (self.step + 1) % grad_acc_step == 0:
            self.opt_step()
            self.opt_zerograd()
        if self.accelerator.is_main_process:
            lr_log = {}
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                lr_log["train/lr"] = self.lr_scheduler.get_last_lr()[0]
            self.wandb_run.log({"train/{}".format(x): y for (x, y) in {**loss_dict, **lr_log}.items()})
            if (self.step + 1) == len(self.train_loader) or (self.cfg.get('debug', False) and (self.step + 1) > self.cfg.get('num_steps', 16)):
                self.logger.info(f"epoch {self.epoch} loss: {loss_dict}")

    def train(self):
        for step, data in enumerate(tqdm.tqdm(self.train_loader, desc='train step')):
            self.step = step
            self.train_step(data)
            if self.cfg.get('debug', False) and step > self.cfg.get('num_steps', 16):
                break

    # -----------------------------------------------------------------------
    # Eval
    # -----------------------------------------------------------------------

    def eval_actions(self):
        """Test-set BC loss with canonical spatial + proprio."""
        self.set_model_eval()
        device = self.cfg.device
        v = self.cfg.get('policy_view_idx', 0)
        action_diff = 0.0
        num_batches = 0
        with torch.no_grad():
            for data in self.test_loader:
                obs_gap, act, goal, spatial, state, *_ = (x.to(device) for x in data)
                spatial_v         = spatial[:, :, v]
                canonical_spatial = self._warp_to_canonical_spatial(spatial_v)
                obs_tokens        = self._flatten_spatial_tokens(canonical_spatial)
                goal_v            = self._goal_for_obs(obs_gap[:, :, v], goal)
                _, loss, loss_dict = self.policy(obs_tokens, state, goal_v, act)
                action_diff += loss_dict.get('flow_matching_loss', 0.0)
                num_batches += 1
                if self.accelerator.is_main_process:
                    self.wandb_run.log({'eval/' + k: vv for k, vv in loss_dict.items()})
                if self.cfg.get('debug', False):
                    break
        if num_batches > 0:
            action_diff /= num_batches
        if self.accelerator.is_main_process:
            self.wandb_run.log({'eval/epoch_wise_flow_matching_loss': action_diff})
            self.logger.info(f'eval flow_matching_loss: {action_diff}')
        return 0.0, action_diff, 0, 0, 0, 0

    def eval_rollout(self, num_evals, epoch, num_eval_per_goal, final_eval=False):
        self.set_model_eval()
        avg_reward, completion_id_list, max_coverage, final_coverage = self.eval_on_env(
            num_evals, epoch, num_eval_per_goal, final_eval
        )

        if self.env_name in ["pusht", "blockpush"]:
            metrics = compute_metrics(self.env_name, max_coverage, final_coverage)
            self.wandb_run.log(metrics)
            self.logger.info(f'eval_on_env metrics {self.env_name}: {metrics}')
            self.metrics_history.append(metrics)

        if self.env_name == "pusht":
            self.reward_history.append(self.metrics_history[-1]["final coverage mean"])
        elif self.env_name == "blockpush":
            self.reward_history.append(self.metrics_history[-1]["entered mean"])
        else:
            self.reward_history.append(avg_reward)

        self.wandb_run.log({"eval_on_env": avg_reward})
        self.logger.info(f'epoch {epoch} eval_on_env: {avg_reward}')
        self.save_best_model()

        if (epoch + 1) % self.cfg.save_every == 0 and self.mode in ['train', 'tune']:
            self.save_model(version=f'epoch{epoch}')

        if final_eval is False:
            with open("{}/completion_idx_{}.json".format(self.save_path, epoch), "wb") as fp:
                pickle.dump(completion_id_list, fp)
        else:
            if self.env_name == "pusht":
                final_eval_on_env = max(x["final coverage mean"] for x in self.metrics_history)
            elif self.env_name == "blockpush":
                final_eval_on_env = max(x["entered mean"] for x in self.metrics_history)
            elif self.env_name == "kitchen-v0":
                final_eval_on_env = avg_reward
            else:
                final_eval_on_env = max(self.reward_history)

            self.wandb_run.log({"final_eval_on_env": final_eval_on_env})
            self.logger.info(f'final_eval_on_env: {final_eval_on_env}')
            self.logger.info(f'reward_history: {self.reward_history}, {sum(self.reward_history)/len(self.reward_history)}')

            if self.mode == 'train':
                with open("{}/completion_idx_final.json".format(self.save_path), "wb") as fp:
                    pickle.dump(completion_id_list, fp)
                self.save_model(version='last')

    @torch.no_grad()
    def eval_on_env(self, num_evals, epoch, num_eval_per_goal, final_eval=False):
        """Rollout with canonical spatial + proprio.

        Spatial deque holds RAW spatial maps; canonicalization happens once per
        re-plan step from the accumulated deque window.
        """
        if self.cfg.get('parallel_eval', False):
            raise NotImplementedError("Parallel eval not implemented for Spatial+Canonical")

        cfg, env = self.cfg, self.env
        v_idx    = cfg.get('policy_view_idx', 0)
        n_act    = cfg.get('n_action_steps', cfg.data.action_chunk)
        T_obs    = cfg.data.obs_window
        device   = cfg.device

        set_seed_everywhere(cfg.seed)
        env.seed(cfg.seed)

        def embed_single(raw_obs):
            t = torch.as_tensor(raw_obs, dtype=torch.float32).unsqueeze(0).to(device)
            gap, spatial = self.encoder(t)
            return gap[0, v_idx], spatial[0, v_idx]

        if self.policy.goal_dim == 0:
            goal_fn = lambda goal_idx: None
        elif getattr(self, 'use_prop_goal', False):
            def goal_fn(goal_idx):
                pos = env._env._target_pos
                return torch.as_tensor(pos, dtype=torch.float32, device=device)
        else:
            empty = torch.zeros(self.policy.goal_dim, device=device)
            goal_fn = lambda goal_idx: empty

        if not hasattr(env, 'get_state'):
            raise AttributeError('Spatial+Canonical eval_on_env requires env.get_state()')

        avg_reward, completion_id_list = 0.0, []
        avg_max_coverage, avg_final_coverage = [], []

        for goal_idx in range(num_evals):
            if self.videorecorder is not None:
                self.videorecorder.init(enabled=True)
            for eval_idx in range(num_eval_per_goal):
                # Cycle rollout cameras across trials only when mixed_view is set (matches the
                # original eval: multi-view SR when True, single fixed view when False).
                if hasattr(env, 'incr_view') and getattr(env, 'mixed_view', False):
                    env.incr_view()

                spatial_stack = deque(maxlen=T_obs)
                state_stack   = deque(maxlen=T_obs)
                this_obs      = env.reset(goal_idx=goal_idx)
                _, first_spatial = embed_single(this_obs)
                first_state = torch.as_tensor(env.get_state(), dtype=torch.float32, device=device)
                for _ in range(T_obs):
                    spatial_stack.append(first_spatial)
                    state_stack.append(first_state)

                goal_enc = goal_fn(goal_idx)
                goal_enc = goal_enc.to(device) if goal_enc is not None else None

                done, step, total_reward = False, 0, 0
                max_reward, max_success  = 0.0, 0
                act_chunk, chunk_cursor  = None, n_act

                while not done:
                    if chunk_cursor >= n_act:
                        spatial_w  = torch.stack(list(spatial_stack)).float().to(device)   # [T, C, H, W]
                        state_w    = torch.stack(list(state_stack)).float().to(device)     # [T, state_dim]
                        canonical  = self._warp_to_canonical_spatial(spatial_w.unsqueeze(0))  # [1, T, C, H, W]
                        obs_tokens = self._flatten_spatial_tokens(canonical)               # [1, T, 49, C]
                        goal_arg   = goal_enc.unsqueeze(0) if goal_enc is not None else None
                        act_chunk  = self.policy.select_action(
                            obs_tokens,
                            state_w.unsqueeze(0),
                            goal_arg,
                        )[0]
                        chunk_cursor = 0

                    curr_action  = act_chunk[chunk_cursor].cpu().numpy()
                    chunk_cursor += 1
                    if self._act_mean is not None:
                        curr_action = curr_action * self._act_std + self._act_mean

                    this_obs, reward, done, info = env.step(curr_action)
                    _, new_spatial = embed_single(this_obs)
                    spatial_stack.append(new_spatial)
                    new_state = info.get('state', env.get_state())
                    state_stack.append(torch.as_tensor(new_state, dtype=torch.float32, device=device))

                    if self.videorecorder is not None and goal_idx % max(1, cfg.num_final_evals // cfg.num_env_evals) == 0:
                        self.videorecorder.record(info['image'])

                    step         += 1
                    total_reward += reward
                    max_reward    = max(max_reward, reward)
                    max_success   = max(max_success, info.get('all_completions_ids', 0))
                    if cfg.get('debug', False):
                        break

                is_mimicgen   = cfg.env.gym.id not in ["pusht", "blockpush", "libero_goal", "kitchen-v0"]
                episode_score = _get_episode_score(cfg.env.gym.id, total_reward, max_reward, info)
                avg_reward   += episode_score
                avg_max_coverage.append(max_reward)
                avg_final_coverage.append(max_reward)
                completion_id_list.append(max_success if is_mimicgen else info.get('all_completions_ids', 0))

                _view_idx = getattr(env, 'view_idx', -1)
                _cam_az   = env.views[_view_idx] if hasattr(env, 'views') and _view_idx >= 0 else 'na'
                print(
                    f'[EVAL_TRIAL] task={cfg.dataset.task_name} view_idx={_view_idx} cam_az={_cam_az} '
                    f'goal={goal_idx} trial={eval_idx} reward={total_reward:.4f} '
                    f'success={int(episode_score > 0)} ep_score={episode_score:.4f}',
                    flush=True,
                )

                if cfg.get('debug', False):
                    break

            if self.mode in ['train', 'eval'] and self.videorecorder is not None:
                self.videorecorder.save(f'eval_{epoch}_{goal_idx}.mp4')
            if cfg.get('debug', False):
                break

        return (avg_reward / (num_evals * num_eval_per_goal), completion_id_list,
                avg_max_coverage, avg_final_coverage)

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def run(self):
        for epoch in tqdm.trange(self.epochs, desc='train epoch'):
            self.epoch = epoch
            if self.accelerator.is_main_process:
                if epoch > 0 and epoch % self.cfg.eval_on_env_freq == 0:
                    self.eval_rollout(
                        num_evals=self.cfg.num_env_evals,
                        epoch=epoch,
                        num_eval_per_goal=self.cfg.num_final_eval_per_goal,
                    )
            if epoch % self.cfg.eval_freq == 0:
                self.eval_actions()
            if self.mode in ['train', 'tune']:
                self.set_model_train()
                self.train()
            if self.cfg.get('debug', False):
                break

        if self.accelerator.is_main_process and self.cfg.get('do_final_eval', True):
            self.eval_rollout(
                num_evals=self.cfg.num_final_evals,
                epoch=self.epochs,
                num_eval_per_goal=self.cfg.num_final_eval_per_goal,
                final_eval=True
            )

        self.save_model(version='last')
        self.close()

    def close(self):
        if self.accelerator.is_main_process:
            self.env.close()
            if hasattr(self, 'env_live') and self.env_live is not None:
                self.env_live.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", version_base="1.2")
def main(cfg):
    debug = cfg.get('debug', False)
    apply_debug_config_policy(cfg)
    set_env_vars()
    set_seed_everywhere(cfg.seed)

    module = cfg.get('module', 'canonpolicyspatial').lower()
    if module == 'canonpolicyspatial':
        trainer = PolicyTrainer(cfg)
    else:
        raise RuntimeError(
            f'train_policy: unknown module "{module}". '
            f'This file only supports: canonpolicyspatial'
        )
    trainer.run()


if __name__ == "__main__":
    main()
