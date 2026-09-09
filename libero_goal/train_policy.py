# Policy training and rollout evaluation for LIBERO Goal (CANON).
#
# DiTFlowPolicy trainer (primary):  PolicyTrainerDiTFlow
# VQ-BeT trainer (add-on):          PolicyTrainerVQBeT
#
# Usage:
#   DiT:    python train_policy.py --config-path configs --config-name policy \
#           module=canonpolicyditflow +encoder_path=... +debug=true
#   VQ-BeT: python train_policy.py --config-path configs --config-name policy_vqbet \
#           module=canonpolicyvqbet +encoder_path=... +debug=true

import einops
import json
import math
import os
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf, open_dict

from utils.video import VideoRecorder
import pickle
from datasets.core import TrajectoryEmbeddingDataset, split_traj_datasets, PolicySlicerDataset
from datasets.vqbet_repro import TrajectorySlicerDataset

from accelerate.logging import get_logger
from utils.trainer import check_wandb, apply_debug_config_policy
from utils import set_env_vars, set_seed_everywhere
from utils.metrics import _get_episode_score, compute_metrics
from utils.checkpoint import load_model, load_snapshot_safe, load_encoder

from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs


# ---------------------------------------------------------------------------
# Canonical view
# ---------------------------------------------------------------------------

# Canonical view azimuth (degrees) for LIBERO-Goal. The self-calibrating angle
# head warps every camera view back to this reference frame before the policy.
CANONICAL_AZIMUTH_DEG = 0.0

def update_dict(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] += value
        else:
            dict1[key] = value


def divide_dict(dic, denom):
    for key, value in dic.items():
        dic[key] = value / denom
    return dic


def reward_dicts_update(reward_dicts, angle, goal, reward):
    reward_dict = reward_dicts[-1]
    if angle not in reward_dict:
        reward_dict[angle] = {goal: reward}
    else:
        reward_dict[angle][goal] = reward


# ---------------------------------------------------------------------------
# PolicyTrainer (base)
# ---------------------------------------------------------------------------

class PolicyTrainer:

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

    def count_params(self):
        total = 0
        for key, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module):
                param_cnt = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total += param_cnt
                param_cnt_all = sum(p.numel() for p in module.parameters())
                print(key, f'req_grad: {param_cnt} all: {param_cnt_all}')
        print('total req_grad', total)

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

        (
            self.policy,
            self.policy_optim,
        ) = self.accelerator.prepare(self.policy, self.policy_optim)

    def _init_encoder(self):

        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(self.cfg.encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.encoder = self.accelerator.prepare(self.encoder)

    def _setup_loaders(self):

        cfg = self.cfg
        train_data, test_data = split_traj_datasets(
            self.dataset,
            train_fraction=cfg.train_fraction,
            random_seed=cfg.seed,
        )

        use_libero_goal = self.use_libero_goal or self.use_prop_goal
        embed_goal = self.use_libero_goal and not self.use_prop_goal
        self.embed_dataset = cfg.data.get('embed', True)
        if self.embed_dataset:
            train_data = TrajectoryEmbeddingDataset(
                self.encoder, train_data, device=cfg.device, embed_goal=embed_goal
            )
            test_data = TrajectoryEmbeddingDataset(
                self.encoder, test_data, device=cfg.device, embed_goal=embed_goal
            )
        traj_slicer_kwargs = {
            "window": cfg.data.window_size,
            "action_window": cfg.data.action_window_size,
            "vqbet_get_future_action_chunk": cfg.data.vqbet_get_future_action_chunk,
            "future_conditional": (cfg.data.goal_conditional == "future"),
            "min_future_sep": cfg.data.action_window_size,
            "future_seq_len": cfg.data.future_seq_len,
            "use_libero_goal": use_libero_goal,
        }

        loader_kwargs = {
            "batch_size": self.cfg.batch_size,
            "num_workers": self.cfg.get('num_workers', 0),
            "pin_memory": self.cfg.get('pin_memory', False),
        }
        if torch.cuda.device_count() > 1:
            assert loader_kwargs["batch_size"] % self.accelerator.num_processes == 0, (
                "Batch size must be divisible by the number of processes. "
                f"Got {loader_kwargs['batch_size']} and {self.accelerator.num_processes}."
            )
            loader_kwargs["batch_size"] = loader_kwargs["batch_size"] // self.accelerator.num_processes
        train_shuffle = False if self.mode in ['tune'] else True
        eval_shuffle = False
        train_data = TrajectorySlicerDataset(train_data, **traj_slicer_kwargs)
        test_data = TrajectorySlicerDataset(test_data, **traj_slicer_kwargs)
        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=train_shuffle, **loader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_data, shuffle=eval_shuffle, **loader_kwargs)
        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader = self.accelerator.prepare(self.test_loader)

        print(f'dataset: {len(self.dataset)} train_data: {len(train_data)} test_data: {len(test_data)}')
        print(f'train_loader: {len(self.train_loader)} test_loader: {len(self.test_loader)}')

    def _init_tracker(self):

        if self.wandb_run is None:
            cfg = self.cfg
            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)

            self.accelerator.init_trackers(
                project_name=cfg.wandb.project,
                config=wandb_cfg,
                init_kwargs={
                    "wandb": {
                        "reinit": False,
                        "settings": {"start_method": "thread"},
                        "entity": cfg.wandb.entity,
                    },
                },
            )

            self.wandb_run = self.accelerator.get_tracker("wandb", unwrap=True)
            self.logger.info("wandb run url: %s", self.wandb_run.get_url())
            check_wandb()

    def __init__(self, cfg):

        self.logger = get_logger(__name__, log_level="DEBUG" if cfg.get('debug', False) else "INFO")
        self.cfg = cfg
        self.mode = cfg.get('mode', 'train')
        self.epoch = 0
        self.epochs = cfg.epochs
        self.use_libero_goal = cfg.data.get("use_libero_goal", False)
        self.use_prop_goal = cfg.data.get("use_prop_goal", False)

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

        # LIBERO rollout-eval views: single source of truth = the policy
        # camera_configs.json `pool` (rollout azimuths), overriding the config default.
        if cfg.env.gym.id == 'libero_goal':
            _cc = Path(cfg.dataset.data_directory) / 'camera_configs.json'
            with open(_cc) as _f:
                _cam = json.load(_f)
            _pool = _cam['pool'] if isinstance(_cam, dict) else _cam
            with open_dict(cfg):
                cfg.env.gym.view_indices = [float(c['azimuth']) for c in _pool]

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
            self.reward_history = []

        self.count_params()

    def setup_eval(self):
        self.epochs = 0
        assert 'cbet_path' in self.cfg, f'expected `cbet_path` in cfg'
        ckpt = load_model(Path(self.cfg.cbet_path))
        if isinstance(ckpt, dict):
            if isinstance(ckpt['policy'], torch.nn.Module):
                self.policy = ckpt['policy']; print('loaded policy nn.Module')
                if self.cfg.get('load_optim', False):
                    self.load_policy_state(ckpt)
            else:
                raise NotImplemented(f'type(ckpt[policy])', type(ckpt['policy']))
        elif isinstance(ckpt, torch.nn.Module):
            self.policy = ckpt
        else:
            raise NotImplementedError(f'type(ckpt)', type(ckpt))
        self.videorecorder = None

    def load_policy_state(self, payload):

        for i, opt_state in enumerate(payload['policy_optim/optimizers']):
            self.policy_optim.optimizer.optimizers[i].load_state_dict(opt_state)

        print('loaded policy_optim state dict')

        (
            self.policy,
            self.policy_optim,
        ) = self.accelerator.prepare(self.policy, self.policy_optim)

    def get_last_frame_live_libero(self, idx, view):

        if not hasattr(self, 'env_live') or self.env_live is None:
            self.cfg_env_live = self.cfg.env.copy()
            with open_dict(self.cfg_env_live):
                self.cfg_env_live.gym.mixed_view = False
                self.cfg_env_live.gym.view_indices = view
            self.env_live = hydra.utils.instantiate(self.cfg_env_live.gym)
            self.env_live.seed(self.cfg.seed)
        else:
            if self.env_live.env.env.view_angle != view:
                with open_dict(self.cfg_env_live):
                    self.cfg_env_live.gym.view_indices = view
                self.env_live = hydra.utils.instantiate(self.cfg_env_live.gym)
                self.env_live.seed(self.cfg.seed)

        num_demos_per_task = len(self.dataset.demos) // len(self.dataset.task_names)
        task_idx = idx // num_demos_per_task
        self.env_live.reset(task_idx)

        state = self.dataset.get_states(idx, -1)
        obs = self.env_live.env.regenerate_obs_from_state(state)['agentview_image']
        obs = torch.flip(torch.Tensor(obs), dims=[0])
        return obs.permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0  # 1 V C H W

    def get_last_frame_live(self, idx, view):
        if self.cfg.env.gym.id == 'libero_goal':
            return self.get_last_frame_live_libero(idx, view)

    def set_random_views(self, env):
        random_views = self.dataset.get_view_angles_list()
        if self.env_name == 'libero_goal':
            env.views = random_views
        elif self.env_name == 'kitchen-v0':
            env.env.env.env.views = random_views
        else:
            raise RuntimeError(f'set_random_views not implemented for {self.env_name}')
        return random_views

    @torch.no_grad()
    def eval_on_env(
        self,
        num_evals,
        epoch,
        num_eval_per_goal,
        final_eval=False
    ):
        if self.cfg.get('parallel_eval', False):
            return self._eval_on_env_vec(num_evals, epoch, num_eval_per_goal, final_eval)

        def get_goals_cache(view=None):
            goals_cache = []
            with torch.no_grad():
                demos_source = self.dataset.demos if hasattr(self.dataset, 'demos') else self.dataset
                num_demos_per_task = len(demos_source) // len(self.dataset.task_names)
                for i in range(len(self.dataset.task_names)):
                    task_start = i * num_demos_per_task
                    task_end = (i + 1) * num_demos_per_task

                    found_idx = None
                    if view is not None and hasattr(self.dataset, 'view_list'):
                        for d_idx in range(task_start, task_end):
                            if abs(self.dataset.view_list[d_idx] - view) < 1e-3:
                                found_idx = d_idx
                                break

                    if found_idx is None:
                        found_idx = task_start

                    frames = self.dataset.get_frames(found_idx, [-1], view)
                    if frames is None or self.cfg.get('live_render_goal', False):
                        last_obs = self.get_last_frame_live(found_idx, view)
                    else:
                        last_obs = frames[0]

                    last_obs = last_obs.to(self.cfg.device)
                    embd = self.encoder(last_obs)[0]  # V E
                    embd = einops.rearrange(embd, "V E -> (V E)")
                    goals_cache.append(embd)
                    if self.cfg.get('debug', False):
                        break
            return goals_cache

        def embed(enc, obs):
            obs = (
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
            )  # 1 V C H W
            result = enc(obs)
            result = einops.rearrange(result, "1 V E -> (V E)")
            return result

        cfg = self.cfg
        env = self.env
        encoder = self.encoder
        set_seed_everywhere(cfg.seed)
        env.seed(cfg.seed)
        if self.cfg.get('random_angle_eval', False) and hasattr(self.dataset, 'continuous'):
            random_views = self.set_random_views(env)
            print(f'env.views set from dataset continuous view angles: {random_views}')
        if self.use_libero_goal:
            if hasattr(env, 'mixed_view') and env.mixed_view:
                if not hasattr(self, 'goals_caches'):
                    views_to_cache = env.views[:1] if self.cfg.get('debug', False) else env.views
                    print(f'Generating goals caches for views: {env.views} | views_to_cache: {views_to_cache}')
                    self.goals_caches = {view: get_goals_cache(view) for view in tqdm.tqdm(views_to_cache)}
                print(f'Using goals caches for views: {self.goals_caches.keys()}')
                def goal_fn(goal_idx, view):
                    return self.goals_caches[view][goal_idx]
            else:
                goals_cache = get_goals_cache(env.views[0])
                def goal_fn(goal_idx):
                    return goals_cache[goal_idx]
        elif self.use_prop_goal:
            def goal_fn(goal_idx):
                return env._env._target_pos.copy()  # [3] numpy float64
        else:
            empty_tensor = torch.zeros(1)
            def goal_fn(goal_idx):
                return empty_tensor
        print('env.views', env.views)

        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []

        if hasattr(env, 'mixed_view') and env.mixed_view:
            self.reward_dicts.append({})

        print('eval goals')
        iterator = range(num_evals)
        for goal_idx in iterator:
            if self.videorecorder is not None:
                self.videorecorder.init(enabled=True)
            for eval_idx in range(num_eval_per_goal):
                if hasattr(env, 'incr_view'):
                    env.incr_view()
                    print(f'(goal_idx, i) = ({goal_idx}, {eval_idx}) \tenv.incr_view(): {env.views[env.view_idx]}')
                obs_stack = deque(maxlen=cfg.eval_window_size)
                this_obs = env.reset(goal_idx=goal_idx)  # V C H W
                assert (
                    this_obs.min() >= 0 and this_obs.max() <= 1
                ), "expect 0-1 range observation"
                this_obs_enc = embed(encoder, this_obs)
                obs_stack.append(this_obs_enc)
                done, step, total_reward = False, 0, 0
                max_reward = 0
                max_success = 0

                if self.use_libero_goal and getattr(env, 'mixed_view', None):
                    goal_obs = goal_fn(goal_idx, env.views[env.view_idx])
                else:
                    goal_obs = goal_fn(goal_idx)

                while not done:
                    obs = torch.stack(tuple(obs_stack)).float().to(cfg.device)
                    if self.use_libero_goal:
                        goal = torch.as_tensor(goal_obs, dtype=torch.float32, device=cfg.device)
                        goal = goal.unsqueeze(0).repeat(obs.shape[0], 1)
                    else:
                        goal = torch.zeros((obs.shape[0], obs.shape[1]))
                        goal = torch.as_tensor(goal, dtype=torch.float32, device=cfg.device)

                    if self.use_libero_goal and self.cfg.env.gym.single_view is False:
                        obs = einops.rearrange(obs, "N (V E) -> N V E", V=self.cfg.env.views)
                        goal = einops.rearrange(goal, "N (V E) -> N V E", V=self.cfg.env.views)

                    policy_input = [obs.unsqueeze(0), goal.unsqueeze(0), None]
                    if self.cfg.dataset.get('aux_view_inverse', False):
                        if hasattr(self.dataset, 'get_aux_view_single_by_view'):
                            aux_view = self.dataset.get_aux_view_single_by_view(env.views[env.view_idx])
                            aux_view = aux_view.to(cfg.device)
                        else:
                            aux_view = env.views[env.view_idx]
                        policy_input.append(aux_view)
                    elif (self.cfg.dataset.get('aux_view', None) is not None
                          or getattr(self, '_needs_view_angle', False)
                          or (hasattr(self, 'dataset') and hasattr(self.dataset, 'aux_view') and self.dataset.aux_view is not None)):
                        policy_input.append(env.views[env.view_idx])
                    action, _, _ = self.forward_policy(policy_input)

                    action = action[0]
                    if cfg.action_window_size > 1:
                        action_list.append(action[-1].cpu().detach().numpy())
                        if len(action_list) > cfg.action_window_size:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)
                        curr_action = (
                            np.sum(curr_action, axis=0)[0] / curr_action.shape[0]
                        )
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[1:], np.zeros((1, a_chunk.shape[1])))
                                )
                            )
                        action_list = new_action_list
                    else:
                        curr_action = action[-1, 0, :].cpu().detach().numpy()

                    this_obs, reward, done, info = env.step(curr_action)
                    this_obs_enc = embed(encoder, this_obs)
                    obs_stack.append(this_obs_enc)
                    if self.videorecorder is not None and goal_idx % (cfg.num_final_evals // cfg.num_env_evals) == 0:
                        self.videorecorder.record(info["image"])
                    step += 1
                    total_reward += reward
                    max_reward = max(max_reward, reward)
                    max_success = max(max_success, info.get("all_completions_ids", 0))
                    if self.cfg.get('debug', False):
                        break

                is_mimicgen = cfg.env.gym.id not in ["pusht", "blockpush", "libero_goal", "kitchen-v0"]
                episode_score = _get_episode_score(cfg.env.gym.id, total_reward, max_reward, info)

                avg_reward += episode_score
                if cfg.env.gym.id == "pusht":
                    env.env._seed += 1
                    avg_max_coverage.append(info["max_coverage"])
                    avg_final_coverage.append(info.get("final_coverage", 0.0))
                elif is_mimicgen:
                    avg_max_coverage.append(max_reward)
                    avg_final_coverage.append(max_reward)
                elif cfg.env.gym.id == "blockpush":
                    avg_max_coverage.append(info.get("moved", 0))
                    avg_final_coverage.append(info.get("entered", 0))

                if is_mimicgen:
                    completion_id_list.append(max_success)
                else:
                    completion_id_list.append(info.get("all_completions_ids", 0))

                if hasattr(env, 'mixed_view') and env.mixed_view:
                    reward_dicts_update(self.reward_dicts, angle=env.views[env.view_idx], goal=goal_idx, reward=episode_score if is_mimicgen else total_reward)

                print(f'goal {goal_idx} eval {eval_idx} total_reward {total_reward} max_reward {max_reward}')

                _view_idx = getattr(env, 'view_idx', -1)
                _cam_az = env.views[_view_idx] if hasattr(env, 'views') and _view_idx >= 0 else 'na'
                print(
                    f'[EVAL_TRIAL] env={cfg.env.gym.id} cam_az={_cam_az} '
                    f'goal={goal_idx} trial={eval_idx} reward={total_reward:.4f} '
                    f'success={int(episode_score > 0)}',
                    flush=True,
                )

                if self.cfg.get('debug', False):
                    break

            if self.mode in ['train', 'eval'] and self.videorecorder is not None:
                self.videorecorder.save(f'eval_{epoch}_{goal_idx}.mp4')
                print(f'saved video: {self.videorecorder.dir_name}/eval_{epoch}_{goal_idx}.mp4')
            if self.cfg.get('debug', False):
                break
        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    @torch.no_grad()
    def _eval_on_env_vec(self, num_evals, epoch, num_eval_per_goal, final_eval=False):
        """
        Parallel rollout evaluation using SubprocVecLiberoEnv.
        """
        from libero_goal.envs.vec_env import SubprocVecLiberoEnv

        cfg      = self.cfg
        encoder  = self.encoder
        device   = cfg.device
        n_workers = min(cfg.get('num_eval_workers', 10), num_evals * num_eval_per_goal)
        total_eps = num_evals * num_eval_per_goal

        set_seed_everywhere(cfg.seed)
        if self.use_libero_goal:
            goals_cache = []
            with torch.no_grad():
                for i in range(len(self.dataset.task_names)):
                    num_demos_per_task = len(self.dataset.demos) // len(self.dataset.task_names)
                    task_start = i * num_demos_per_task
                    frames = self.dataset.get_frames(task_start, [-1])
                    if frames is None or cfg.get('live_render_goal', False):
                        last_obs = self.get_last_frame_live(task_start)
                    else:
                        last_obs = frames[0]
                    last_obs = last_obs.to(device)
                    embd = encoder(last_obs)[0]   # [V, E]
                    embd = einops.rearrange(embd, "V E -> (V E)")
                    goals_cache.append(embd)
        else:
            goals_cache = [torch.zeros(1, device=device)] * num_evals

        aux_view_base = None
        if cfg.dataset.get('aux_view_inverse', False):
            view = cfg.env.gym.view_indices
            if hasattr(self.dataset, 'get_aux_view_single_by_view'):
                aux_view_base = self.dataset.get_aux_view_single_by_view(view).to(device)
            else:
                aux_view_base = view

        spec       = SubprocVecLiberoEnv.spec_from_cfg(cfg.env.gym)
        asset_path = os.environ.get('ASSET_PATH', str(Path(cfg.get('asset_path', '.')).resolve()))
        vec = SubprocVecLiberoEnv(n_workers, **spec, asset_path=asset_path)
        vec.seed([cfg.seed + i for i in range(n_workers)])
        # Parallel eval must cover camera views like the single-process path (previously it did
        # not: every episode ran at the initial view). Fetch the view list once; each batch
        # assigns each worker a precomputed view = (flat job index) % n_views via set_view, which
        # reproduces the single-process pairing (every goal sees every view evenly). A shared
        # incr_view counter would instead give every worker the same view per batch, confounding
        # the per-view / ID-OOD breakdown with goal identity.
        _mixed_view = bool(getattr(cfg.env.gym, 'mixed_view', False))
        _views_list = vec.get_views()[0] if _mixed_view else []

        all_jobs   = [(g, e) for g in range(num_evals) for e in range(num_eval_per_goal)]
        is_mimicgen = cfg.env.gym.id not in ["pusht", "blockpush", "libero_goal", "kitchen-v0"]

        avg_reward         = 0.0
        completion_id_list = []
        avg_max_coverage   = []
        avg_final_coverage = []

        try:
            for batch_start in range(0, total_eps, n_workers):
                batch_jobs = all_jobs[batch_start: batch_start + n_workers]
                n_batch    = len(batch_jobs)

                # mixed_view: assign each worker its precomputed view before reset so the parallel
                # eval covers env.views with the same goal-balanced pairing as single-process.
                _batch_cam_az = ['na'] * n_batch
                if _mixed_view:
                    _vidx = vec.set_view([(batch_start + k) % len(_views_list) for k in range(n_batch)])
                    _batch_cam_az = [_views_list[v] if 0 <= v < len(_views_list) else 'na' for v in _vidx]

                obs_list = vec.reset([g for g, _ in batch_jobs])
                obs_arr  = np.stack(obs_list)
                enc_init = encoder(torch.as_tensor(obs_arr).float().to(device))  # [n, V, E]

                obs_stacks = []
                for i in range(n_batch):
                    emb   = einops.rearrange(enc_init[i], "V E -> (V E)")
                    stack = deque(maxlen=cfg.eval_window_size)
                    stack.append(emb)
                    obs_stacks.append(stack)

                active        = [True]  * n_batch
                total_rewards = [0.0]   * n_batch
                max_rewards   = [0.0]   * n_batch
                max_successes = [0]     * n_batch
                episode_infos = [None]  * n_batch
                action_lists  = [[]     for _ in range(n_batch)]

                while any(active):
                    actions_for_step = [np.zeros(cfg.env.act_dim, dtype=np.float32)] * n_batch
                    for i in range(n_batch):
                        if not active[i]:
                            continue

                        obs  = torch.stack(list(obs_stacks[i])).float().to(device)
                        goal = goals_cache[batch_jobs[i][0]]
                        goal = goal.unsqueeze(0).repeat(obs.shape[0], 1)

                        if self.use_libero_goal and cfg.env.gym.single_view is False:
                            obs  = einops.rearrange(obs,  "N (V E) -> N V E", V=cfg.env.views)
                            goal = einops.rearrange(goal, "N (V E) -> N V E", V=cfg.env.views)

                        policy_input = [obs.unsqueeze(0), goal.unsqueeze(0), None]
                        if aux_view_base is not None:
                            policy_input.append(aux_view_base)

                        action, _, _ = self.forward_policy(policy_input)
                        action = action[0]

                        if cfg.action_window_size > 1:
                            action_lists[i].append(action[-1].cpu().detach().numpy())
                            if len(action_lists[i]) > cfg.action_window_size:
                                action_lists[i] = action_lists[i][1:]
                            curr = np.array(action_lists[i])
                            curr_action = np.sum(curr, axis=0)[0] / curr.shape[0]
                            action_lists[i] = [
                                np.concatenate((a[1:], np.zeros((1, a.shape[1])))) for a in action_lists[i]
                            ]
                        else:
                            curr_action = action[-1, 0, :].cpu().detach().numpy()

                        actions_for_step[i] = curr_action

                    obs_list, rewards, dones, infos = vec.step(actions_for_step)

                    active_idx = [i for i in range(n_batch) if active[i]]
                    if active_idx:
                        active_obs = np.stack([obs_list[i] for i in active_idx])
                        enc_new    = encoder(torch.as_tensor(active_obs).float().to(device))
                        for k, i in enumerate(active_idx):
                            obs_stacks[i].append(einops.rearrange(enc_new[k], "V E -> (V E)"))

                    for i in range(n_batch):
                        if not active[i]:
                            continue
                        total_rewards[i] += rewards[i]
                        max_rewards[i]    = max(max_rewards[i], rewards[i])
                        max_successes[i]  = max(max_successes[i], infos[i].get('all_completions_ids', 0))
                        episode_infos[i]  = infos[i]
                        if dones[i]:
                            active[i] = False

                    if cfg.get('debug', False):
                        break

                for i in range(n_batch):
                    g_idx, e_idx = batch_jobs[i]
                    e_score = _get_episode_score(cfg.env.gym.id, total_rewards[i], max_rewards[i], episode_infos[i])
                    avg_reward += e_score

                    print(
                        f'[EVAL_TRIAL] env={cfg.env.gym.id} cam_az={_batch_cam_az[i]} '
                        f'goal={g_idx} trial={e_idx} reward={total_rewards[i]:.4f} '
                        f'success={int(e_score > 0)}',
                        flush=True,
                    )

                    if final_eval:
                        print(f'goal {g_idx} eval {e_idx} reward {total_rewards[i]}')

                    if cfg.env.gym.id == "pusht":
                        avg_max_coverage.append(episode_infos[i].get("max_coverage", 0))
                        avg_final_coverage.append(episode_infos[i].get("final_coverage", 0.0))
                    elif is_mimicgen:
                        avg_max_coverage.append(max_rewards[i])
                        avg_final_coverage.append(max_rewards[i])
                    elif cfg.env.gym.id == "blockpush":
                        avg_max_coverage.append(episode_infos[i].get("moved", 0))
                        avg_final_coverage.append(episode_infos[i].get("entered", 0))

                    if is_mimicgen:
                        completion_id_list.append(max_successes[i])
                    else:
                        completion_id_list.append(
                            episode_infos[i].get('all_completions_ids', 0) if episode_infos[i] else 0
                        )

                if cfg.get('debug', False):
                    break

        finally:
            vec.close()

        return (
            avg_reward / total_eps,
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    def _eval_preprocess(self, obs, goal):
        """Decode raw obs/goal tensors to embeddings before eval_actions assertion."""
        if self.use_libero_goal:
            if not self.embed_dataset and goal.ndim != 4:
                goal = self.encoder(goal)
        elif self.use_prop_goal:
            pass
        else:
            goal = torch.zeros_like(obs).to(self.cfg.device)
        if not self.embed_dataset and obs.ndim != 4:
            obs = self.encoder(obs)
        return obs, goal

    @torch.no_grad()
    def eval_actions(self):

        self.set_model_eval()

        if hasattr(self, 'models'):
            assert isinstance(self.models, list)
            for m in self.models:
                m.eval()

        device = self.cfg.device
        total_loss = 0
        action_diff = 0
        action_diff_tot = 0
        action_diff_mean_res1 = 0
        action_diff_mean_res2 = 0
        action_diff_max = 0

        with torch.no_grad():
            for data in self.test_loader:
                obs, act, goal, *rest = (x.to(device) for x in data)

                obs, goal = self._eval_preprocess(obs, goal)
                assert obs.ndim == 4, f"expect N T V E here, got {obs.ndim}: {obs.shape}"

                policy_input = [obs, goal, act, *rest]
                _, loss, loss_dict = self.forward_policy(policy_input)

                total_loss += loss.detach()
                action_diff += loss_dict.get("action_diff", loss_dict.get("diffusion_loss", 0))
                action_diff_tot += loss_dict.get("action_diff_tot", loss_dict.get("diffusion_loss", 0))
                action_diff_mean_res1 += loss_dict.get("action_diff_mean_res1", 0)
                action_diff_mean_res2 += loss_dict.get("action_diff_mean_res2", 0)
                action_diff_max += loss_dict.get("action_diff_max", 0)
                if self.accelerator.is_main_process:
                    self.wandb_run.log({"eval/{}".format(x): y for (x, y) in loss_dict.items()})
                if self.cfg.get('debug', False):
                    print(f"Test loss: {total_loss / len(self.test_loader)}")
                    break

            self.logger.info(f"eval:\n{loss_dict}")

        print(f"Test loss: {total_loss / len(self.test_loader)}")

        n_batches = len(self.test_loader)
        action_diff /= n_batches
        action_diff_tot /= n_batches
        action_diff_mean_res1 /= n_batches
        action_diff_mean_res2 /= n_batches
        action_diff_max /= n_batches

        if self.accelerator.is_main_process:

            self.wandb_run.log({"eval/epoch_wise_action_diff": action_diff})
            self.wandb_run.log({"eval/epoch_wise_action_diff_tot": action_diff_tot})
            self.wandb_run.log({"eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1})
            self.wandb_run.log({"eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2})
            self.wandb_run.log({"eval/epoch_wise_action_diff_max": action_diff_max})

            log_dict = {
                "epoch_wise_action_diff": action_diff,
                "epoch_wise_action_diff_tot": action_diff_tot,
                "epoch_wise_action_diff_mean_res1": action_diff_mean_res1,
                "epoch_wise_action_diff_mean_res2": action_diff_mean_res2,
                "epoch_wise_action_diff_max": action_diff_max,
            }
            self.logger.info(f"eval:\n{log_dict}")

        return total_loss, action_diff, action_diff_tot, action_diff_mean_res1, action_diff_mean_res2, action_diff_max

    def save_best_model(self):

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:

            env_name, reward_history, metrics_history = self.env_name, self.reward_history, self.metrics_history
            save = False

            if env_name == "pusht":
                if len(metrics_history) == 1 or max([x["final coverage mean"] for x in metrics_history[:-1]]) < metrics_history[-1]["final coverage mean"]:
                    save = True
            elif env_name == "blockpush":
                if len(metrics_history) == 1 or max([x["entered mean"] for x in metrics_history[:-1]]) < metrics_history[-1]["entered mean"]:
                    save = True
            elif env_name == "libero_goal":
                if len(reward_history) == 1 or max(reward_history[:-1]) < reward_history[-1]:
                    save = True
            elif env_name == "kitchen-v0":
                if len(reward_history) == 1 or max(reward_history[:-1]) < reward_history[-1]:
                    save = True

            if save and self.mode in ['train', 'tune']:
                self.save_model(version='best')

    def save_model(self, version='best'):
        self._keys_to_save = ['policy', 'policy_optim']
        self.save_snapshot(version)

    def eval_rollout(self, num_evals, epoch, num_eval_per_goal, final_eval=False):

        self.set_model_eval()

        if hasattr(self, 'models'):
            assert isinstance(self.models, list)
            for m in self.models:
                m.eval()

        avg_reward, completion_id_list, max_coverage, final_coverage = self.eval_on_env(num_evals, epoch, num_eval_per_goal, final_eval)

        if self.env_name in ["pusht", "blockpush"]:
            metrics = compute_metrics(self.env_name, max_coverage, final_coverage)
            self.wandb_run.log(metrics)
            self.logger.info(f'eval_on_env metrics {self.env_name}: {metrics}')
            self.metrics_history.append(metrics)

        if self.env_name == "pusht":
            self.reward_history.append(self.metrics_history[-1]["final coverage mean"])
        elif self.env_name == "blockpush":
            self.reward_history.append(self.metrics_history[-1]["entered mean"])
        elif self.env_name in ["libero_goal", "kitchen-v0"]:
            self.reward_history.append(avg_reward)
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
                final_eval_on_env = max([x["final coverage mean"] for x in self.metrics_history])
            elif self.env_name == "blockpush":
                final_eval_on_env = max([x["entered mean"] for x in self.metrics_history])
            elif self.env_name == "libero_goal":
                final_eval_on_env = max(self.reward_history)
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

    def _get_precision(self):
        amp = self.cfg.get('amp', False)
        if isinstance(amp, bool):
            return 'fp16' if amp is True else 'no'
        if isinstance(amp, str) and amp in ['no', 'fp16', 'bf16', 'fp8']:
            return amp
        raise NotImplementedError(f'amp: {amp}')

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

    def forward_policy(self, data):

        obs, goal, act, *rest = data
        # obs [N T V E], goal [N T V E], act [N T A]

        if obs.ndim == 4:
            obs = einops.rearrange(obs, "N T V E -> N T (V E)")
            goal = einops.rearrange(goal, "N T V E -> N T (V E)")

        return self.policy(obs, goal, act)

    def encode_goal(self, data):

        obs, _, goal, *_ = data

        if self.use_libero_goal:
            if not self.embed_dataset and goal.ndim != 4:
                with torch.no_grad():
                    goal = self.encoder(goal)
        elif self.use_prop_goal:
            pass
        else:
            goal = torch.zeros(obs.shape).to(self.cfg.device)

        return goal

    def forward(self, data):

        obs, act, goal, *rest = data

        if not self.embed_dataset and obs.ndim != 4:
            with torch.no_grad():
                obs = self.encoder(obs)  # N T V C H W -> N T V E

        total_loss = 0
        total_loss_dict = {}
        final_action = None

        goal = self.encode_goal(data)

        model_views = self.cfg.model.get('views', 1)
        try:
            is_multiview = len(model_views) > 1
        except TypeError:
            is_multiview = isinstance(model_views, int) and model_views > 1

        if is_multiview:
            predicted_act, loss, loss_dict = self.forward_policy((obs, goal, act, *rest))
            total_loss = loss
            total_loss_dict = loss_dict
            num_views_processed = 1
        else:
            num_views = obs.shape[2]
            for v in range(num_views):
                predicted_act, loss, loss_dict = self.forward_policy((obs[:, :, v], goal[:, :, v], act, *rest))
                total_loss += loss
                update_dict(total_loss_dict, loss_dict)
            num_views_processed = num_views

        return None, total_loss / num_views_processed, divide_dict(total_loss_dict, num_views_processed)

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

    def set_model_train(self):

        if self.cfg.get('train_encoder', False):
            self.encoder.train()
        else:
            self.encoder.eval()

        self.policy.train()

        if hasattr(self, 'models'):
            assert isinstance(self.models, list)
            for m in self.models:
                m.train()

    def set_model_eval(self):

        self.encoder.eval()
        self.policy.eval()

        if hasattr(self, 'models'):
            assert isinstance(self.models, list)
            for m in self.models:
                m.eval()

    def train(self):

        for step, data in enumerate(tqdm.tqdm(self.train_loader, desc='train step')):
            self.step = step
            self.train_step(data)
            if self.cfg.get('debug', False) and step > self.cfg.get('num_steps', 16):
                print('debug mode: exit after few steps')
                break

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
                print('debug mode: exit after 1 epoch')
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
# VQ-BeT CANON trainer
# ---------------------------------------------------------------------------

class PolicyTrainerVQBeT(PolicyTrainer):
    """
    Self-calibrating VQ-BeT policy trainer for TrainerViewInvarRotationPredictionSO2ViewOnlyAnglePred.

    Loads encoder + view_projector + view_ssl from a full DynaMo snapshot, uses
    view_ssl.angle_head to predict camera azimuth θ, builds an inverse SO(2)
    rotation, projects it through view_projector, then warps observations to the
    canonical (0°) frame before passing them to the VQ-BeT policy.

    Snapshot keys required: encoder, view_projector, view_ssl
    Config:  module: canonpolicyvqbet

    Dispatch string: canonpolicyvqbet
    """

    _needs_view_angle = True  # tells base eval_on_env to append env.views[view_idx]

    def __init__(self, cfg):
        super().__init__(cfg)
        self.count_params()

    def _init_encoder(self):
        self.cfg.encoder_path = self.cfg.encoder_path.replace('encoder_epoch', 'snapshot_epoch')
        snapshot = load_encoder(self.cfg.encoder_path)

        self.encoder = snapshot['encoder']
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder = self.accelerator.prepare(self.encoder)
        self.encoder.eval()

        self._init_view_projector(snapshot)
        self._init_view_ssl(snapshot)

    def _init_view_projector(self, snapshot):
        assert 'view_projector' in snapshot, (
            f"Snapshot loaded from '{self.cfg.encoder_path}' is missing key 'view_projector'. "
            f"Module '{self.cfg.get('module', '?')}' requires a full snapshot (snapshot_epoch*.pt), "
            f"not an encoder-only file (encoder_epoch*.pt). "
            f"Available keys: {list(snapshot.keys())}"
        )
        self.view_projector = snapshot['view_projector']
        for param in self.view_projector.parameters():
            param.requires_grad = False
        self.view_projector = self.accelerator.prepare(self.view_projector)
        self.view_projector.eval()

    def _init_view_ssl(self, snapshot):
        assert 'view_ssl' in snapshot, (
            f"Snapshot loaded from '{self.cfg.encoder_path}' is missing key 'view_ssl'. "
            f"Module '{self.cfg.get('module', '?')}' requires a full snapshot (snapshot_epoch*.pt), "
            f"not an encoder-only file (encoder_epoch*.pt). "
            f"Available keys: {list(snapshot.keys())}"
        )
        self.view_ssl = snapshot['view_ssl']
        for param in self.view_ssl.parameters():
            param.requires_grad = False
        self.view_ssl = self.accelerator.prepare(self.view_ssl)
        self.view_ssl.eval()
        # Cache unwrapped module for direct sub-module access (angle_head, etc.)
        self.view_ssl_module = self.accelerator.unwrap_model(self.view_ssl)

    def forward(self, data):
        """
        policy training: given input x_{v_2}, convert v_2 back to v_1 using trained rotation predictor.

        obs_enc [N, T, V=1, E]
        goal [N, T, V=1, E]
        act [N, T, A]
        """
        assert self.embed_dataset is True
        obs_enc, act, goal = data[0], data[1], data[2]
        goal = self.encode_goal(data)
        _, loss, loss_dict = self.forward_policy((obs_enc[:, :, 0], goal[:, :, 0], act))
        return None, loss, loss_dict

    def forward_policy(self, data):
        """
        Thread data[3] (GT view angle) to encode_rotation for metric accumulation.

        obs_enc [N, T, E]
        goal_enc [N, T, E]
        act [N, T, A]
        rotation_feat (optional): GT view angle for metric logging
        """
        rotation_feat = data[3] if len(data) > 3 else None
        obs_enc, goal_enc, act = data[0], data[1], data[2]
        if obs_enc.ndim == 4:
            obs_enc = obs_enc[:, :, 0]
        if goal_enc is not None and goal_enc.ndim == 4:
            goal_enc = goal_enc[:, :, 0]
        rotation_latent = self.encode_rotation(rotation_feat, obs_enc)
        obs_enc_pred, goal_enc_pred = self.forward_obs_goal(obs_enc, goal_enc, rotation_latent)
        return PolicyTrainer.forward_policy(self, (obs_enc_pred, goal_enc_pred, act))

    def encode_rotation(self, rotation_feat, obs_enc=None):
        """
        Predict absolute azimuthal angle from obs features via angle_head, then
        project the inverse SO(2) rotation through view_projector.

        obs_enc: [B, T, E]
        Returns: [B, 1, D]  projected rotation latent for canonical-frame dynamics.
        """
        assert obs_enc is not None, 'obs_enc required for angle prediction'
        with torch.no_grad():
            sincos_pred = self.view_ssl_module.angle_head(obs_enc)  # [B, 2]
            theta_pred = torch.atan2(sincos_pred[:, 0], sincos_pred[:, 1])  # [B]
            so2_inv = self._build_so2_flat(-theta_pred)  # [B, 4]
            lat = self.view_projector(so2_inv)  # [B, D]

        if isinstance(rotation_feat, (int, float)):
            _ds = getattr(self, 'dataset', None)
            canonical_az_deg = float(
                getattr(_ds, 'canonical_view', {}).get('azimuth', CANONICAL_AZIMUTH_DEG)
            )
            theta_gt = torch.tensor(
                (rotation_feat - canonical_az_deg) * (math.pi / 180.0),
                dtype=theta_pred.dtype, device=theta_pred.device,
            )
            delta = theta_pred - theta_gt
            err_rad = torch.atan2(
                torch.sin(delta), torch.cos(delta),
            ).abs()  # [B], wrapped to [0, π]
            if not hasattr(self, '_angle_err_buf'):
                self._angle_err_buf = []
            self._angle_err_buf.append(err_rad.mean().item() * (180.0 / math.pi))
            if not hasattr(self, '_angle_cos_buf'):
                self._angle_cos_buf = []
            self._angle_cos_buf.append(torch.cos(delta).mean().item())

        return lat.unsqueeze(1)  # [B, 1, D]

    @staticmethod
    def _build_so2_flat(delta_theta: torch.Tensor) -> torch.Tensor:
        """delta_theta [B] → flattened SO(2) matrix [B, 4]: [cos, -sin, sin, cos]."""
        c, s = torch.cos(delta_theta), torch.sin(delta_theta)
        return torch.stack([c, -s, s, c], dim=-1)

    def forward_obs_goal(self, obs_enc, goal_enc, rotation_latent):
        """
        Vectorized forward_view() for obs_enc + goal_enc.
        """
        if obs_enc.ndim == 4:
            obs_enc = obs_enc[:, :, 0]   # [N, T, V=1, E] -> [N, T, E]
        if goal_enc.ndim == 4:
            goal_enc = goal_enc[:, :, 0]
        obs_goal_enc = torch.cat([obs_enc, goal_enc], dim=0)  # [N * 2, T, E]
        if rotation_latent.ndim == 3:
            rotation_latent = rotation_latent[:, 0]  # [N, V=1, D] => [N, D]
        if obs_goal_enc.shape[0] != rotation_latent.shape[0]:
            rotation_latent = rotation_latent.repeat(2, 1)
        obs_goal_enc_pred = self.forward_view(obs_goal_enc, rotation_latent)
        obs_enc_pred, goal_enc_pred = torch.split(obs_goal_enc_pred, obs_enc.shape[0], dim=0)

        return obs_enc_pred, goal_enc_pred

    def forward_view(self, obs_enc, rotation_latent):
        """
        Use view_ssl.forward_dynamics to reversely transform obs_enc of new view back to base reference view.
        Inlined from ViewReverseSO2.forward_view (ndim==4 guard) + ViewReverse.forward_view (rearrange).
        """
        if obs_enc.ndim == 4:
            obs_enc = obs_enc[:, :, 0]
        view_enc = einops.rearrange(obs_enc, "B T E -> B 1 T E")
        view_enc_pred = self.forward_viewsteps_vectorized(view_enc, rotation_latent)
        return einops.rearrange(view_enc_pred, "B V T E -> B (T V) E")

    def forward_viewsteps_vectorized(self, obs_enc, rotation_latent):
        """
        For CANON_RotationSO2-style view_ssl: concatenate rotation_latent and call forward_dynamics.
        obs_enc:         [N, T=1, V=T_traj, E]
        rotation_latent: [N, D]
        """
        N, T, V, E = obs_enc.shape
        obs_flat = obs_enc.reshape(N * V, T, E)
        rot_flat = rotation_latent.unsqueeze(1).repeat(1, T * V, 1).reshape(N * V, T, -1)
        forward_dyn_input = torch.cat([obs_flat, rot_flat], dim=-1)  # [N*V, T, E+D]
        view_obs_enc_pred = self.view_ssl.forward_dynamics(forward_dyn_input)  # [N*V, T, E]
        return view_obs_enc_pred.reshape(obs_enc.shape)

    def eval_on_env(self, *args, **kwargs):
        """Override to log mean angle-head error and cosine similarity across eval rollouts."""
        self._angle_err_buf = []
        self._angle_cos_buf = []
        result = super().eval_on_env(*args, **kwargs)
        if self._angle_err_buf:
            mean_err = sum(self._angle_err_buf) / len(self._angle_err_buf)
            mean_cos = sum(self._angle_cos_buf) / len(self._angle_cos_buf)
            n = len(self._angle_err_buf)
            msg = (
                f'angle_head mean abs error: {mean_err:.2f}  cosine sim: {mean_cos:.4f} '
                f'(n={n} steps)'
            )
            print(msg)
            self.logger.info(msg)
            if hasattr(self, 'wandb_run') and self.wandb_run is not None:
                self.wandb_run.log({
                    'eval_on_env/angle_err_deg': mean_err,
                    'eval_on_env/angle_cos_sim': mean_cos,
                })
        return result


# ---------------------------------------------------------------------------
# DiTFlow CANON trainer
# ---------------------------------------------------------------------------

class PolicyTrainerDiTFlow(PolicyTrainer):
    """
    DiTFlowPolicy + view-branch canonicalization via angle prediction.

    Loads encoder + view_projector + view_ssl from a full DynaMo snapshot.
    Uses view_ssl.angle_head to predict camera azimuth θ, warps encoder features
    to the canonical (0°) frame, then feeds them to the DiTFlow policy.

    Data pipeline uses PolicySlicerDataset (obs_window / action_chunk).
    Rollout uses action-chunk re-planning.

    Snapshot keys required: encoder, view_projector, view_ssl
    Config:  module: canonpolicyditflow

    Dispatch string: canonpolicyditflow
    """

    def _init_encoder(self):
        """Load encoder + view_projector + view_ssl (all frozen) from snapshot; cache view_ssl_module."""
        assert 'encoder_path' in self.cfg, 'encoder_path must be set in config'
        self.cfg.encoder_path = self.cfg.encoder_path.replace(
            'encoder_epoch', 'snapshot_epoch'
        )
        snapshot = load_encoder(self.cfg.encoder_path)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = snapshot['encoder']
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.encoder = self.accelerator.prepare(self.encoder)

        # ── View-branch modules ───────────────────────────────────────────────
        for attr in ('view_projector', 'view_ssl'):
            m = snapshot[attr]
            _ddp = torch.nn.parallel.DistributedDataParallel
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

        # Cache unwrapped view_ssl for direct sub-module access (angle_head, projector, etc.)
        self.view_ssl_module = self.accelerator.unwrap_model(self.view_ssl)
        print(f'PolicyTrainerDiTFlow: loaded encoder + view branch from {self.cfg.encoder_path}')

    def _init_policy(self):
        """Identical to base. LR scheduler is created in _setup_loaders."""
        super()._init_policy()

    def _setup_loaders(self):
        """Use PolicySlicerDataset with obs_window / action_chunk parameters."""
        cfg = self.cfg
        train_data, test_data = split_traj_datasets(
            self.dataset,
            train_fraction=cfg.train_fraction,
            random_seed=cfg.seed,
        )

        use_libero_goal = self.use_libero_goal
        self.embed_dataset = True  # always pre-embed for DiTFlow
        train_data = TrajectoryEmbeddingDataset(
            self.encoder, train_data, device=cfg.device, embed_goal=use_libero_goal
        )
        test_data = TrajectoryEmbeddingDataset(
            self.encoder, test_data, device=cfg.device, embed_goal=use_libero_goal
        )

        obs_window   = cfg.data.obs_window
        action_chunk = cfg.data.action_chunk
        train_data = PolicySlicerDataset(train_data, obs_window=obs_window, action_chunk=action_chunk)
        test_data  = PolicySlicerDataset(test_data,  obs_window=obs_window, action_chunk=action_chunk)

        loader_kwargs = {
            "batch_size": cfg.batch_size,
            "num_workers": cfg.get('num_workers', 0),
            "pin_memory": cfg.get('pin_memory', False),
        }
        if torch.cuda.device_count() > 1:
            assert loader_kwargs["batch_size"] % self.accelerator.num_processes == 0
            loader_kwargs["batch_size"] //= self.accelerator.num_processes

        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **loader_kwargs)
        self.test_loader  = torch.utils.data.DataLoader(test_data,  shuffle=False, **loader_kwargs)
        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader  = self.accelerator.prepare(self.test_loader)
        print(f'dataset: {len(self.dataset)} train: {len(train_data)} test: {len(test_data)}')

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
                self.policy_optim,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            print(f'DiTFlow LR scheduler: cosine warmup={warmup_steps}, total={total_steps} ({len(self.train_loader)} steps/epoch × {cfg.epochs} epochs)')

    def _select_view(self, enc: torch.Tensor) -> torch.Tensor:
        """
        Select a single view from a multi-view encoding tensor.

        enc can be:
          [N, T, V, E]  → [N, T, E]   (batch of windows)
          [N, V, E]     → [N, E]      (batch of single frames)
        """
        v = self.cfg.get('policy_view_idx', 0)
        n_views = enc.shape[2] if enc.ndim == 4 else enc.shape[1]
        assert v < n_views, (
            f'policy_view_idx={v} is out of range for enc with {n_views} views '
            f'(shape={tuple(enc.shape)}). Set policy_view_idx correctly in your config.'
        )
        if enc.ndim == 4:   # [N, T, V, E]
            return enc[:, :, v, :]
        elif enc.ndim == 3:  # [N, V, E]
            return enc[:, v, :]
        else:
            raise ValueError(f'_select_view: unexpected enc.ndim={enc.ndim}, shape={enc.shape}')

    def forward(self, data):
        """
        data from PolicySlicerDataset: (obs, act, goal, *rest)
          obs  [N, T_obs, V, E]
          act  [N, T_act, A]
          goal [N, V, E]
        """
        obs, act, goal, *_ = data

        obs_v = self._select_view(obs)           # [N, T_obs, E]
        obs_canon = self._warp_to_canonical(obs_v)   # [N, T_obs, E]

        if self.policy.goal_dim == 0:
            goal_canon = None
        elif self.use_libero_goal:
            goal_v     = self._select_view(goal)
            goal_canon = self._warp_to_canonical(goal_v.unsqueeze(1)).squeeze(1)  # [N, E]
        elif self.use_prop_goal:
            goal_canon = goal[:, 0, :].to(obs_v.device)
        else:
            goal_canon = torch.zeros(obs_v.shape[0], self.policy.goal_dim, device=obs_v.device)

        return self.forward_policy((obs_canon, goal_canon, act))

    def forward_policy(self, data):
        obs_v, goal_v, act = data
        return self.policy(obs_v, goal_v, act)

    @torch.no_grad()
    def _warp_to_canonical(self, enc: torch.Tensor) -> torch.Tensor:
        """
        enc: [N, T, E]: features at arbitrary azimuth θ
        Returns: [N, T, E]: canonical-frame features

        1. angle_head(enc)                    → sincos [N, 2]
        2. Build R(-θ) flat                   → so2_inv [N, 4]
        3. projector(so2_inv)                 → rot_lat [N, D]
        4. forward_dynamics(cat([enc,rot_lat])) → canonical_feat [N, T, E]
        """
        m       = self.view_ssl_module
        N, T, E = enc.shape
        sincos  = m.angle_head(enc)                               # [N, 2]: sin(θ), cos(θ)
        s, c    = sincos[:, 0], sincos[:, 1]
        so2_inv = torch.stack([c, s, -s, c], dim=1)               # R(-θ) flat [N, 4]
        rot_lat = m.projector(so2_inv)                            # [N, D]
        enc_flat     = enc.reshape(N * T, 1, E)                   # [N*T, 1, E]
        rot_lat_flat = rot_lat.unsqueeze(1).expand(-1, T, -1).reshape(N * T, 1, -1)  # [N*T, 1, D]
        out_flat = m.forward_dynamics(torch.cat([enc_flat, rot_lat_flat], dim=-1))    # [N*T, 1, E]
        return out_flat.reshape(N, T, E)                          # [N, T, E]

    # ── View-warp hooks ──────────────────────────────────────────────────────

    def _warp_goal_emb(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: [E]  →  canonical [E]"""
        f = emb.unsqueeze(0).unsqueeze(0)             # [1, 1, E]
        f = self._warp_to_canonical(f)                # [1, 1, E]
        return f[0, 0]                                # [E]

    def _warp_obs_window(self, obs_window: torch.Tensor) -> torch.Tensor:
        """obs_window: [T, E]  →  canonical [T, E]  (serial eval hook)"""
        f = obs_window.unsqueeze(0)                   # [1, T, E]
        f = self._warp_to_canonical(f)                # [1, T, E]
        return f[0]                                   # [T, E]

    def _warp_obs_window_batch(self, obs_windows: torch.Tensor) -> torch.Tensor:
        """obs_windows: [B, T, E]  →  canonical [B, T, E]  (parallel eval hook)"""
        return self._warp_to_canonical(obs_windows)   # [B, T, E]

    def eval_actions(self):
        """Evaluate policy loss on the test set."""
        self.set_model_eval()
        device = self.cfg.device
        total_loss = 0.0
        action_diff = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in self.test_loader:
                obs, act, goal, *_ = (x.to(device) for x in data)
                obs_v = self._select_view(obs)
                if self.policy.goal_dim == 0:
                    goal_v = None
                elif self.use_libero_goal:
                    goal_v = self._select_view(goal)
                else:
                    goal_v = torch.zeros(obs_v.shape[0], self.policy.goal_dim, device=device)
                _, loss, loss_dict = self.policy(obs_v, goal_v, act)
                action_diff += loss_dict.get('flow_matching_loss', 0.0)
                num_batches += 1
                if self.accelerator.is_main_process:
                    self.wandb_run.log({'eval/' + k: v for k, v in loss_dict.items()})
                if self.cfg.get('debug', False):
                    break

        if num_batches > 0:
            action_diff /= num_batches

        if self.accelerator.is_main_process:
            self.wandb_run.log({'eval/epoch_wise_flow_matching_loss': action_diff})
            self.logger.info(f'eval flow_matching_loss: {action_diff}')

        return total_loss, action_diff, 0, 0, 0, 0

    @torch.no_grad()
    def _eval_on_env_vec(self, num_evals, epoch, num_eval_per_goal, final_eval=False):
        """
        Parallel rollout evaluation using SubprocVecLiberoEnv.
        """
        from libero_goal.envs.vec_env import SubprocVecLiberoEnv

        cfg      = self.cfg
        encoder  = self.encoder
        v_idx    = cfg.get('policy_view_idx', 0)
        n_act    = cfg.get('n_action_steps', cfg.data.action_chunk)
        T_obs    = cfg.data.obs_window
        act_dim  = cfg.env.act_dim
        n_workers = cfg.get('num_eval_workers', 10)
        device   = cfg.device

        total_eps = num_evals * num_eval_per_goal
        n_workers = min(n_workers, total_eps)

        set_seed_everywhere(cfg.seed)
        if self.use_libero_goal:
            task_names     = self.dataset.task_names
            goals_cache    = []
            demos_per_task = len(self.dataset) // len(task_names)
            for i in range(len(task_names)):
                demo_idx  = i * demos_per_task
                items     = self.dataset.get_frames(demo_idx, [-1])
                last_obs  = items[0].to(device)
                enc       = encoder(last_obs)             # [1, V, E]
                goals_cache.append(self._warp_goal_emb(enc[0, v_idx]))  # [E]
        else:
            goals_cache = [torch.zeros(1, device=device)] * num_evals

        spec       = SubprocVecLiberoEnv.spec_from_cfg(cfg.env.gym)
        asset_path = str(Path(cfg.get('asset_path', '.')).resolve())
        asset_path = os.environ.get('ASSET_PATH', asset_path)

        vec = SubprocVecLiberoEnv(n_workers, **spec, asset_path=asset_path)
        vec.seed([cfg.seed + i for i in range(n_workers)])
        # Fetch the view list once; each batch assigns each worker a precomputed view =
        # (flat job index) % n_views via set_view, reproducing the single-process goal-balanced
        # pairing (every goal sees every view evenly) and logging the view per trial. A shared
        # incr_view counter would give every worker the same view per batch → per-view/ID-OOD
        # breakdown confounded with goal identity.
        _mixed_view = bool(getattr(self.env, 'mixed_view', False))
        _views_list = vec.get_views()[0] if _mixed_view else []

        all_jobs = [(g, e) for g in range(num_evals) for e in range(num_eval_per_goal)]

        avg_reward         = 0.0
        completion_id_list = []
        avg_max_coverage   = []
        avg_final_coverage = []
        is_mimicgen = cfg.env.gym.id not in ["pusht", "blockpush", "libero_goal", "kitchen-v0"]

        if hasattr(self, 'reward_dicts') and hasattr(self.env, 'mixed_view') and self.env.mixed_view:
            self.reward_dicts.append({})

        try:
            for batch_start in range(0, total_eps, n_workers):
                batch_jobs = all_jobs[batch_start : batch_start + n_workers]
                n_batch    = len(batch_jobs)

                _batch_cam_az = ['na'] * n_batch
                if _mixed_view:
                    _vidx = vec.set_view([(batch_start + k) % len(_views_list) for k in range(n_batch)])
                    _batch_cam_az = [_views_list[v] if 0 <= v < len(_views_list) else 'na' for v in _vidx]

                obs_list = vec.reset([g for g, _ in batch_jobs])

                obs_arr = np.stack(obs_list)
                t       = torch.as_tensor(obs_arr).float().to(device)
                enc_init = encoder(t)  # [n, V, E]

                obs_stacks = [deque(maxlen=T_obs) for _ in range(n_batch)]
                for i in range(n_batch):
                    emb = enc_init[i, v_idx]   # [E]
                    for _ in range(T_obs):
                        obs_stacks[i].append(emb)

                active         = [True]  * n_batch
                total_rewards  = [0.0]   * n_batch
                max_rewards    = [0.0]   * n_batch
                max_successes  = [0]     * n_batch
                act_chunks     = [None]  * n_batch
                chunk_cursors  = [n_act] * n_batch
                episode_infos  = [None]  * n_batch

                while any(active):

                    plan_idx = [i for i in range(n_batch)
                                if active[i] and chunk_cursors[i] >= n_act]
                    if plan_idx:
                        obs_wins = torch.stack([
                            torch.stack(list(obs_stacks[i])).float()
                            for i in plan_idx
                        ]).to(device)
                        obs_wins   = self._warp_obs_window_batch(obs_wins)
                        goal_sub   = torch.stack([goals_cache[batch_jobs[i][0]]
                                                  for i in plan_idx]).to(device)
                        new_chunks = self.policy.select_action(obs_wins, goal_sub)  # [n_plan, C, A]
                        for k, i in enumerate(plan_idx):
                            act_chunks[i]    = new_chunks[k]
                            chunk_cursors[i] = 0

                    actions = []
                    for i in range(n_batch):
                        if active[i]:
                            actions.append(act_chunks[i][chunk_cursors[i]].cpu().numpy())
                            chunk_cursors[i] += 1
                        else:
                            actions.append(np.zeros(act_dim, dtype=np.float32))

                    obs_list, rewards, dones, infos = vec.step(actions)

                    active_idx = [i for i in range(n_batch) if active[i]]
                    if active_idx:
                        active_obs = np.stack([obs_list[i] for i in active_idx])
                        t_new      = torch.as_tensor(active_obs).float().to(device)
                        enc_new    = encoder(t_new)
                        for k, i in enumerate(active_idx):
                            obs_stacks[i].append(enc_new[k, v_idx])

                    for i in range(n_batch):
                        if not active[i]:
                            continue
                        total_rewards[i] += rewards[i]
                        max_rewards[i]    = max(max_rewards[i], rewards[i])
                        max_successes[i]  = max(max_successes[i],
                                               infos[i].get('all_completions_ids', 0))
                        episode_infos[i]  = infos[i]
                        if dones[i]:
                            active[i] = False

                    if cfg.get('debug', False):
                        break

                for i in range(n_batch):
                    g_idx = batch_jobs[i][0]
                    e_score = _get_episode_score(
                        cfg.env.gym.id, total_rewards[i], max_rewards[i], episode_infos[i]
                    )
                    avg_reward += e_score

                    print(
                        f'[EVAL_TRIAL] env={cfg.env.gym.id} cam_az={_batch_cam_az[i]} '
                        f'goal={g_idx} trial={batch_jobs[i][1]} reward={total_rewards[i]:.4f} '
                        f'success={int(e_score > 0)}',
                        flush=True,
                    )

                    if final_eval:
                        print(f'goal {g_idx} eval {batch_jobs[i][1]} reward {total_rewards[i]}')

                    if cfg.env.gym.id == "pusht":
                        avg_max_coverage.append(episode_infos[i].get("max_coverage", 0))
                        avg_final_coverage.append(episode_infos[i].get("final_coverage", 0.0))
                    elif is_mimicgen:
                        avg_max_coverage.append(max_rewards[i])
                        avg_final_coverage.append(max_rewards[i])
                    elif cfg.env.gym.id == "blockpush":
                        avg_max_coverage.append(episode_infos[i].get("moved", 0))
                        avg_final_coverage.append(episode_infos[i].get("entered", 0))

                    if is_mimicgen:
                        completion_id_list.append(max_successes[i])
                    else:
                        completion_id_list.append(episode_infos[i].get('all_completions_ids', 0) if episode_infos[i] else 0)

                if cfg.get('debug', False):
                    break

        finally:
            vec.close()

        return (
            avg_reward / total_eps,
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    @torch.no_grad()
    def eval_on_env(self, num_evals, epoch, num_eval_per_goal, final_eval=False):
        """
        Rollout with action-chunk re-planning.
        """
        if self.cfg.get('parallel_eval', False):
            return self._eval_on_env_vec(num_evals, epoch, num_eval_per_goal, final_eval)

        cfg      = self.cfg
        env      = self.env
        encoder  = self.encoder
        v_idx    = cfg.get('policy_view_idx', 0)
        n_act    = cfg.get('n_action_steps', cfg.data.action_chunk)
        T_obs    = cfg.data.obs_window

        set_seed_everywhere(cfg.seed)
        env.seed(cfg.seed)

        def embed_single(raw_obs):
            """raw_obs: V C H W  →  E  (selected view, float32 on device)"""
            t = torch.as_tensor(raw_obs, dtype=torch.float32).unsqueeze(0).to(cfg.device)
            enc = encoder(t)   # [1, V, E]
            return enc[0, v_idx]  # [E]

        if self.use_libero_goal:
            task_names = self.dataset.task_names
            goals_cache = []
            demos_per_task = len(self.dataset) // len(task_names)
            for i, _ in enumerate(task_names):
                demo_idx = i * demos_per_task
                items    = self.dataset.get_frames(demo_idx, [-1])
                last_obs = items[0].to(cfg.device)
                enc      = encoder(last_obs)
                goals_cache.append(self._warp_goal_emb(enc[0, v_idx]))  # [E]
            goal_fn = lambda goal_idx: goals_cache[goal_idx]
        else:
            if self.policy.goal_dim == 0:
                goal_fn = lambda goal_idx: None
            elif self.use_prop_goal:
                def goal_fn(goal_idx):
                    pos = env._env._target_pos
                    return torch.as_tensor(pos, dtype=torch.float32, device=cfg.device)
            else:
                empty = torch.zeros(self.policy.goal_dim, device=cfg.device)
                goal_fn = lambda goal_idx: empty

        print('env.views', env.views)

        avg_reward       = 0.0
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []

        if hasattr(env, 'mixed_view') and env.mixed_view:
            self.reward_dicts.append({})

        for goal_idx in range(num_evals):
            if self.videorecorder is not None:
                self.videorecorder.init(enabled=True)

            for eval_idx in range(num_eval_per_goal):
                if hasattr(env, 'incr_view'):
                    env.incr_view()
                    print(f'(goal_idx, i) = ({goal_idx}, {eval_idx}) \tenv.incr_view(): {env.views[env.view_idx]}')

                obs_stack  = deque(maxlen=T_obs)
                this_obs   = env.reset(goal_idx=goal_idx)
                assert this_obs.min() >= 0 and this_obs.max() <= 1
                first_enc  = embed_single(this_obs)
                for _ in range(T_obs):
                    obs_stack.append(first_enc)

                goal_enc_raw = goal_fn(goal_idx)
                goal_enc = goal_enc_raw.to(cfg.device) if goal_enc_raw is not None else None

                done, step, total_reward = False, 0, 0
                max_reward   = 0.0
                max_success  = 0
                act_chunk    = None
                chunk_cursor = n_act  # force re-plan on first step

                while not done:
                    if chunk_cursor >= n_act:
                        obs_window = torch.stack(list(obs_stack)).float().to(cfg.device)  # [T_obs, E]
                        obs_window = self._warp_obs_window(obs_window)
                        goal_arg = goal_enc.unsqueeze(0) if goal_enc is not None else None
                        act_chunk  = self.policy.select_action(
                            obs_window.unsqueeze(0),   # [1, T_obs, E]
                            goal_arg,                  # [1, E] or None
                        )[0]                           # [action_chunk, A]
                        chunk_cursor = 0

                    curr_action = act_chunk[chunk_cursor].cpu().numpy()
                    chunk_cursor += 1
                    if self._act_mean is not None:
                        curr_action = curr_action * self._act_std + self._act_mean

                    this_obs, reward, done, info = env.step(curr_action)
                    obs_stack.append(embed_single(this_obs))

                    if self.videorecorder is not None and goal_idx % max(1, cfg.num_final_evals // cfg.num_env_evals) == 0:
                        self.videorecorder.record(info['image'])

                    step         += 1
                    total_reward += reward
                    max_reward    = max(max_reward, reward)
                    max_success   = max(max_success, info.get('all_completions_ids', 0))

                    if cfg.get('debug', False):
                        break

                is_mimicgen = cfg.env.gym.id not in ["pusht", "blockpush", "libero_goal", "kitchen-v0"]
                episode_score = _get_episode_score(cfg.env.gym.id, total_reward, max_reward, info)

                avg_reward += episode_score
                if cfg.env.gym.id == "pusht":
                    env.env._seed += 1
                    avg_max_coverage.append(info["max_coverage"])
                    avg_final_coverage.append(info.get("final_coverage", 0.0))
                elif is_mimicgen:
                    avg_max_coverage.append(max_reward)
                    avg_final_coverage.append(max_reward)
                elif cfg.env.gym.id == "blockpush":
                    avg_max_coverage.append(info.get("moved", 0))
                    avg_final_coverage.append(info.get("entered", 0))

                if is_mimicgen:
                    completion_id_list.append(max_success)
                else:
                    completion_id_list.append(info.get('all_completions_ids', 0))

                if hasattr(env, 'mixed_view') and env.mixed_view:
                    reward_dicts_update(self.reward_dicts, angle=env.views[env.view_idx], goal=goal_idx, reward=episode_score if is_mimicgen else total_reward)

                if final_eval:
                    print(f'goal {goal_idx} eval {eval_idx} reward {total_reward}')
                if cfg.get('debug', False):
                    break

            if self.mode in ['train', 'eval'] and self.videorecorder is not None:
                self.videorecorder.save(f'eval_{epoch}_{goal_idx}.mp4')
                print(f'saved video: {self.videorecorder.dir_name}/eval_{epoch}_{goal_idx}.mp4')

        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", version_base="1.2")
def main(cfg):

    debug = cfg.get('debug', False)
    apply_debug_config_policy(cfg)
    set_env_vars()
    if not debug:
        print('OmegaConf.to_yaml(cfg)\n', OmegaConf.to_yaml(cfg))

    set_seed_everywhere(cfg.seed)

    module = cfg.get('module', 'canonpolicyditflow').lower()
    if module == 'canonpolicyditflow':
        trainer = PolicyTrainerDiTFlow(cfg)
    elif module == 'canonpolicyvqbet':
        trainer = PolicyTrainerVQBeT(cfg)
    else:
        raise RuntimeError(
            f'Unknown module: {module!r}. '
            f'Supported: canonpolicyditflow, canonpolicyvqbet'
        )
    trainer.run()


if __name__ == "__main__":
    main()
