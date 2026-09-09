import os
import tqdm
import utils
import hydra
import torch
import einops
import datasets
import torch.distributed
from pathlib import Path
from datetime import timedelta
from omegaconf import OmegaConf, open_dict
from accelerate import Accelerator
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs


def apply_debug_config_encoder(cfg):
    """If debug=true, override training configs with fast debug values."""
    if not cfg.get('debug', False):
        print(OmegaConf.to_yaml(cfg))
        return
    with open_dict(cfg):
        cfg.env.dataset.subset_fraction = 0.02
        cfg.batch_size = 64
        cfg.num_epochs = 1
        cfg.save_every_epochs = 1
        # cfg.num_steps = 2
    os.environ['WANDB_MODE'] = 'disabled'


def apply_debug_config_policy(cfg):
    """If debug=true, override policy training configs with fast debug values."""
    if not cfg.get('debug', False):
        return
    with open_dict(cfg):
        cfg.epochs = 2
        cfg.batch_size = 4
        cfg.dataset.subset_fraction = 0.1
        cfg.num_env_evals = 1
        cfg.num_final_evals = 1
        cfg.num_final_eval_per_goal = 1
        cfg.eval_freq = 1
        cfg.eval_on_env_freq = 1


def check_wandb():
    if os.environ.get('WANDB_MODE') == 'offline':
        os.environ['WANDB_DIR'] = os.getcwd()

def clear(debug: bool, dirs: list):
    if debug:
        for d in dirs:
            print(f'rm -rf {d}')

class Workspace:
    """Minimal workspace stub for CANON public release."""
    def __init__(self, cfg, work_dir):
        self.cfg = cfg
        self.work_dir = work_dir


class EarlyStop:
    def __init__(self, patience: int = 10, delta: float = 0.001, comparator: callable = min):
        '''
            patience: number of iteration threshold to tolerate
            delta: difference of new metric and stored previous metric
            compartor: function to compare stored against new metric
        '''
        self.previous = None
        self.patience = patience
        self.delta = delta
        self.wait = 0
        self.comparator = comparator
    def update(self, new_metric):
        if self.previous is None:
            self.previous = new_metric
        else:
            if (self.comparator(self.previous, new_metric) == self.previous) or \
            (abs(self.previous - new_metric) > self.delta):
                self.wait += 1
            else:
                self.wait = 0
            self.previous = new_metric
        if self.wait > self.patience:
            return True
        return False


class EarlyStopCallback:
    """Stop when every tracked metric has plateaued over a window."""

    def __init__(
        self,
        metric: str = None,
        window_epochs: int = 100,
        threshold: float = 0.1,
        min_epochs: int = 50,
        mode: str = "max",
        patience: int = 1,
        metrics_config: dict = None,
    ):
        """
        Args:
            metric: (legacy) single metric name (e.g., 'eff_rank')
            window_epochs: check improvement over last N epochs
            threshold: (legacy) minimum improvement for single metric
            min_epochs: don't stop before this epoch (safety buffer)
            mode: (legacy) "max" or "min"
            patience: number of checks to tolerate without improvement
            metrics_config: dict of {metric_name: {threshold, mode, weight}} for multi-metric.
                           Example: {
                               'eff_rank': {'threshold': 0.1, 'mode': 'max', 'weight': 1.0},
                               'can_var_loss': {'threshold': 0.001, 'mode': 'min', 'weight': 2.0},
                               'view_loss': {'threshold': 0.001, 'mode': 'min', 'weight': 1.0},
                           }
                           Higher weight = more critical to stop.
        """
        self.metric = metric
        self.window_epochs = window_epochs
        self.threshold = threshold
        self.min_epochs = min_epochs
        self.mode = mode
        self.patience = patience
        self.wait = 0

        # Multi-metric support
        self.metrics_config = metrics_config or {}
        if self.metric and not self.metrics_config:
            # Fallback to legacy single-metric mode
            self.metrics_config = {
                metric: {'threshold': threshold, 'mode': mode, 'weight': 1.0}
            }

    def _check_metric_plateau(self, metric_name: str, config: dict, metric_history: dict) -> bool:
        """Check if a single metric has plateaued. Returns True if plateaued."""
        if metric_name not in metric_history:
            return False

        history = metric_history[metric_name]
        if len(history) < self.window_epochs:
            return False

        current_val = history[-1]
        window_start_val = history[-self.window_epochs]
        improvement = current_val - window_start_val

        mode = config.get('mode', self.mode)
        threshold = config.get('threshold', self.threshold)

        if mode == "max":
            has_improvement = improvement > threshold
        else:  # "min"
            has_improvement = improvement < -threshold

        return not has_improvement  # True if NO improvement (plateaued)

    def on_epoch_end(self, epoch: int, metric_history: dict) -> bool:
        """Check if multiple metrics have plateaued using weighted criteria."""
        if epoch < self.min_epochs:
            return False  # too early to stop

        if not self.metrics_config:
            return False

        # Check each metric and compute plateau score (0-1, higher = more plateaued)
        plateau_scores = {}
        valid_metrics = 0

        for metric_name, config in self.metrics_config.items():
            is_plateaued = self._check_metric_plateau(metric_name, config, metric_history)
            weight = config.get('weight', 1.0)
            plateau_scores[metric_name] = (is_plateaued, weight)

            if metric_name in metric_history and len(metric_history[metric_name]) >= self.window_epochs:
                valid_metrics += 1

        if valid_metrics == 0:
            return False  # not enough data yet

        # Weighted plateau assessment: if ANY critical metric (high weight) plateaus, OR
        # if majority of metrics (by weight) plateau, stop
        total_weight = sum(w for _, w in plateau_scores.values())
        plateau_weight = sum(w for is_p, w in plateau_scores.values() if is_p)
        plateau_ratio = plateau_weight / total_weight if total_weight > 0 else 0

        # Stop if >50% of weighted metrics have plateaued
        metrics_plateau = plateau_ratio > 0.5

        if metrics_plateau:
            self.wait += 1
        else:
            self.wait = 0

        should_stop = self.wait >= self.patience

        if epoch % 50 == 0 or should_stop:
            status_str = " | ".join([
                f"{m}={metric_history.get(m, [-1])[-1]:.4f} ({'plateau' if p else 'improving'})"
                for m, (p, _) in plateau_scores.items()
                if m in metric_history
            ])
            print(
                f"EarlyStopCallback: epoch {epoch}, {status_str}, "
                f"plateau_ratio={plateau_ratio:.2f} (threshold=0.50), "
                f"wait={self.wait}/{self.patience}"
            )

        return should_stop


class Trainer:
    def _init_accelerator(self):

        cfg = self.cfg
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=cfg.timeout_seconds)
        )
        dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.effective_batch_size = self.cfg.batch_size
        self.accelerator = Accelerator(
            log_with="wandb", kwargs_handlers=[process_group_kwargs, dist_kwargs]
        )
        # self.logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
        utils.set_seed_everywhere(cfg.seed)
        if self.cfg.get('full_precision', False):
            utils.set_full_precision()

        self.job_num, self.work_dir = utils.get_hydra_jobnum_workdir()

        if self.cfg.get('debug', False):
            self.work_dir = Path(self.work_dir).parent.parent / 'test'
            print('using debug dir', self.work_dir)
            os.makedirs(self.work_dir, exist_ok=True)

        # all processes use the work_dir from the main process
        if torch.distributed.is_initialized():
            objs = [str(self.work_dir)]
            torch.distributed.broadcast_object_list(objs, 0)
            self.work_dir = Path(objs[0])
        self.accelerator.wait_for_everyone()

    def count_params(self):

        total = 0
        for key, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module) and hasattr(module, '_parameters'):
                param_cnt = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total += param_cnt
                param_cnt_all = sum(p.numel() for p in module.parameters())
                print(key, f'req_grad: {param_cnt} all: {param_cnt_all}')
        print('total req_grad', total)

    def __init__(self, cfg):

        self.cfg = cfg
        self.mode = cfg.get('mode', 'train')
        self.logger = get_logger(__name__)
        self.early_stop = self.cfg.get('early_stop', None)
        if self.early_stop is not None:
            self.early_stop = EarlyStop()

        self._init_accelerator()

        self.logger.info("Saving to {}".format(self.work_dir))
        os.chdir(self.work_dir); print('self.work_dir', self.work_dir)
        self.work_dir = Path(os.getcwd())  # get the absolute path

        self._init_tracker(cfg)

        # Create the model
        self.encoder = None
        self.projector = None
        self.ssl = None
        self._init_encoder()
        self._init_projector()
        self._init_ssl()

        self._keys_to_save = [
            "encoder",
            "projector",
            "encoder_optim",
            "projector_optim",
            "ssl",
            "ssl_optim",
            "epoch",
        ]

        self.dataset = hydra.utils.instantiate(cfg.env.dataset)
        self._setup_loaders(batch_size=self.cfg.batch_size)

        self.count_params()

        self.workspace: Workspace = hydra.utils.instantiate(
            self.cfg.env.get('workspace', None),
            cfg=self.cfg,
            work_dir=self.work_dir,
            _recursive_=False,
        )
        if self.workspace is not None:
            self.workspace.set_dataset(self.dataset)

        self.log_components = OrderedDict()
        self.epoch = 0

        # Metric history for early stopping and debugging
        self.metric_history = {
            'eff_rank': [],
            'can_var_loss': [],
            'loss': [],
        }
        self.epoch_metrics = {}

        # Register callbacks from config
        self.callbacks = []
        if cfg.get('early_stop_config', None):
            stop_cfg = cfg.early_stop_config
            self.callbacks.append(
                EarlyStopCallback(
                    metric=stop_cfg.get('metric', 'eff_rank'),
                    window_epochs=stop_cfg.get('window_epochs', 100),
                    threshold=stop_cfg.get('threshold', 0.1),
                    min_epochs=stop_cfg.get('min_epochs', 50),
                    mode=stop_cfg.get('mode', 'max'),
                    patience=stop_cfg.get('patience', 1),
                )
            )

    def _init_tracker(self, cfg):
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_cfg["effective_batch_size"] = self.effective_batch_size
        wandb_cfg["save_path"] = str(self.work_dir)
        self.accelerator.init_trackers(
            project_name=cfg.project,
            config=wandb_cfg,
            init_kwargs={
                "wandb": {
                    "reinit": False,
                    "settings": {"start_method": "thread"},
                },
            },
        )
        if self.accelerator.is_main_process:
            self.wandb_run = self.accelerator.get_tracker("wandb", unwrap=True)
            self.logger.info("wandb run url: %s", self.wandb_run.get_url())
            check_wandb()

    def _init_encoder(self):
        if self.encoder is None:  # possibly already initialized from snapshot
            self.encoder = hydra.utils.instantiate(self.cfg.encoder)
            if self.cfg.sync_bn:
                self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.encoder
                )

        if not hasattr(self, 'encoder_optim'):
            self.encoder_optim = torch.optim.AdamW(
                params=self.encoder.parameters(),
                lr=self.cfg.get('resnet_lr', self.cfg.ssl_lr),
                weight_decay=self.cfg.get('resnet_weight_decay', self.cfg.ssl_weight_decay),
                betas=tuple(self.cfg.betas),
            )

        (self.encoder, self.encoder_optim) = self.accelerator.prepare(self.encoder, self.encoder_optim)
        if self.accelerator.is_main_process and self.cfg.get('debug', False):
            self.wandb_run.watch(self.encoder)

    def _init_projector(self):
        if self.projector is None:  # possibly already initialized from snapshot
            self.projector = hydra.utils.instantiate(
                self.cfg.projector, _recursive_=False
            )
            self.projector_optim: torch.optim.Optimizer = (
                self.projector.configure_optimizers(
                    lr=self.cfg.ssl_lr,
                    weight_decay=self.cfg.ssl_weight_decay,
                    betas=tuple(self.cfg.betas),
                )
            )
        (
            self.projector,
            self.projector_optim,
        ) = self.accelerator.prepare(self.projector, self.projector_optim)

    def _init_ssl(self):
        if self.ssl is None:
            self.ssl = hydra.utils.instantiate(
                self.cfg.ssl,
                encoder=self.encoder,
                projector=self.projector,
            )
            self.ssl_optim = self.ssl.optimizers

    def _split_and_slice_dataset(self, dataset):
        kwargs = {
            "train_fraction": self.cfg.train_fraction,
            "random_seed": self.cfg.seed,
            "window_size": self.cfg.window_size,
            "future_conditional": (self.cfg.goal_conditional == "future"),
            "min_future_sep": self.cfg.min_future_sep,
            "future_seq_len": self.cfg.goal_seq_len,
            "num_extra_predicted_actions": self.cfg.num_extra_predicted_actions,
        }
        return datasets.core.get_train_val_sliced(dataset, **kwargs)

    def _setup_loaders(self, batch_size=None, pin_memory=True, num_workers=None):
        self.train_set, self.test_set = self._split_and_slice_dataset(self.dataset)
        # Close any HDF5 handle opened during slicing so each DataLoader worker opens its
        # own: h5py handles are unsafe to share across forked/spawned worker processes.
        if getattr(self.dataset, '_hdf5_file', None) is not None:
            self.dataset._hdf5_file.close()
            self.dataset._hdf5_file = None
        if num_workers is None:
            num_workers = self.cfg.num_workers
            if self.cfg.get('debug', False):
                print('debug mode: set num_workers=0')
                num_workers = 0
        kwargs = {
            "batch_size": batch_size or self.cfg.batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if self.accelerator.num_processes > 1 and num_workers > 0:
            # DDP initializes CUDA before DataLoader workers are created.
            # fork() inherits CUDA state → SIGABRT. Use spawn instead:
            # workers start fresh with no CUDA state. persistent_workers=True
            # amortizes the spawn startup cost across epochs.
            kwargs["multiprocessing_context"] = "spawn"
            kwargs["persistent_workers"] = True
        # scale batch size by number of gpus
        assert kwargs["batch_size"] % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Got {kwargs['batch_size']} and {self.accelerator.num_processes}."
        )
        kwargs["batch_size"] = kwargs["batch_size"] // self.accelerator.num_processes
        # Pass an explicit seeded generator so accelerate.prepare() does not replace
        # sampler.generator=None with torch.Generator() (unseeded from system entropy),
        # which would break per-run reproducibility despite set_seed_everywhere().
        train_generator = torch.Generator().manual_seed(self.cfg.seed)
        self.train_loader = DataLoader(self.train_set, shuffle=True, generator=train_generator, **kwargs)
        self.test_loader = DataLoader(self.test_set, shuffle=False, **kwargs)
        print(f'dataset: {len(self.dataset)} train_set: {len(self.train_set)} test_set: {len(self.test_set)}')
        print(f'train_loader: {len(self.train_loader)} test_loader: {len(self.test_loader)}')

        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader = self.accelerator.prepare(self.test_loader)

    def train_step(self, data):
        (obs_enc, obs_proj, *_, ssl_loss, ssl_loss_components) = self.forward(data)
        self.log_append("ssl_train", len(obs_enc), ssl_loss_components)
        self.backprop(ssl_loss)
        self.opt_step()
        self.opt_zerograd()

    def train(self):
        self.set_model_train()
        self.ssl.adjust_beta(self.epoch, self.cfg.num_epochs)
        pbar = tqdm.tqdm(
            self.train_loader,
            desc=f"epoch {self.epoch}",
            disable=not self.accelerator.is_main_process,
            ncols=80,
        )
        self.num_steps = self.cfg.get('num_steps', None)
        for step, data in enumerate(pbar):
            # break; print('save random init snapshot at epoch 0')
            self.step = step
            self.train_step(data)
            if self.num_steps and step == self.num_steps: # up to specified number of training steps
                print(f'terminate: reached num_steps {self.num_steps}')
                break

    def forward(self, data):

        obs, _, *_ = data

        with self.accelerator.autocast():
            (
                obs_enc,
                obs_proj,
                ssl_loss,
                ssl_loss_components,
            ) = self.ssl.forward(obs)

        return (obs_enc, obs_proj, ssl_loss, ssl_loss_components)

    def backprop(self, ssl_loss):
        self.accelerator.backward(ssl_loss)

        if self.cfg.clip_grad_norm:
            self.accelerator.clip_grad_norm_(
                self.encoder.parameters(), self.cfg.clip_grad_norm
            )
            self.accelerator.clip_grad_norm_(
                self.projector.parameters(), self.cfg.clip_grad_norm
            )
            self.accelerator.clip_grad_norm_(
                self.ssl.parameters(), self.cfg.clip_grad_norm
            )

    def opt_step(self):
        self.encoder_optim.step()
        self.projector_optim.step()
        self.ssl.step()

    def opt_zerograd(self):
        self.encoder_optim.zero_grad(set_to_none=True)
        self.projector_optim.zero_grad(set_to_none=True)
        self.ssl.zero_grad(set_to_none=True)

    @staticmethod
    def _ssl_health_metrics(x: torch.Tensor, labels: torch.Tensor = None) -> dict:
        """x: [N, E] flat feature matrix. Returns SSL health metrics."""
        stds = x.std(dim=0)
        norms = x.norm(dim=-1)
        x_norm = F.normalize(x, p=2, dim=-1)
        N = x.shape[0]
        if N < 2:
            return {}  # sample statistics undefined for < 2 observations

        # Per-dim std diagnostics
        mean_std = stds.mean()
        min_std = stds.min()
        max_std = stds.max()
        num_active = (stds > 0.01).float().sum()

        # Norm diagnostics
        mean_norm = norms.mean()
        min_norm = norms.min()
        max_norm = norms.max()
        norm_cv = norms.std() / (norms.mean() + 1e-8)  # scale consistency

        # Average pairwise cosine similarity (collapse → approaches 1)
        idx1 = torch.randint(0, N, (min(N, 512),), device=x.device)
        idx2 = torch.randint(0, N, (min(N, 512),), device=x.device)
        mask = idx1 != idx2
        avg_cos_sim = (x_norm[idx1[mask]] * x_norm[idx2[mask]]).sum(dim=-1).mean()

        # Uniformity (Wang & Isola 2020): log avg exp(-2||zi-zj||²) on unit sphere
        # Healthy: ≈ −2 to −3. Near 0 → collapsed to a patch.
        sub = x_norm[:min(N, 512)]
        sq_dists = torch.cdist(sub, sub).pow(2)
        off = sq_dists.triu(diagonal=1).reshape(-1)
        off = off[off > 0]
        uniformity = torch.log(torch.exp(-2.0 * off).mean() + 1e-8)

        # Average nearest-neighbour distance (k=1): local state discriminability
        dists_nn = sq_dists.sqrt()
        dists_nn.fill_diagonal_(float("inf"))
        avg_nn_dist = dists_nn.min(dim=1).values.mean()

        # kNN accuracy (k=5): fraction of frames whose 5 nearest neighbours share
        # the same label. Requires optional labels [N] passed by the caller.
        # Distinct from avg_nn_dist: this measures label consistency, not distance.
        if labels is not None:
            sub_labels = labels[:min(N, 512)]
            k = min(5, sub_labels.shape[0] - 1)
            knn_idx = dists_nn.topk(k, largest=False, dim=1).indices  # (M, k)
            knn_labels = sub_labels[knn_idx]                           # (M, k)
            correct = (knn_labels == sub_labels.unsqueeze(1)).float().mean()
            knn_acc = correct
        else:
            knn_acc = None

        # Effective rank (Roy & Vetterli 2007): exp(entropy of normalised eigenvalues)
        # Penalises top-heavy spectra more sharply than participation ratio.
        x_c = x - x.mean(dim=0)
        cov = x_c.T @ x_c / (x_c.shape[0] - 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
        p = eigvals / (eigvals.sum() + 1e-8)
        effective_rank = torch.exp(-(p * torch.log(p + 1e-10)).sum())

        return {
            "mean_std": mean_std,
            "min_std": min_std,
            "max_std": max_std,
            "num_active_dims": num_active,
            "mean_norm": mean_norm,
            "min_norm": min_norm,
            "max_norm": max_norm,
            "norm_cv": norm_cv,
            "avg_cos_sim": avg_cos_sim,
            "uniformity": uniformity,
            "avg_nn_dist": avg_nn_dist,
            "effective_rank": effective_rank,
            **({"knn_acc": knn_acc} if knn_acc is not None else {}),
        }

    @staticmethod
    def _build_task_labels(loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Build a [num_trajectories] integer task-label tensor from the loader's dataset.

        Requires dataset._index (list of (task_name, ...)) and dataset.task_names.
        Returns None when the dataset does not expose these attributes.
        Works correctly only when the loader does not shuffle (standard for eval).
        """
        ds = getattr(loader, 'dataset', None)
        if ds is None or not (hasattr(ds, '_index') and hasattr(ds, 'task_names')):
            return None
        t2i = {t: i for i, t in enumerate(ds.task_names)}
        return torch.tensor([t2i[item[0]] for item in ds._index], dtype=torch.long)

    @staticmethod
    def _expand_task_labels(
        labels_all: torch.Tensor,
        ptr: int,
        obs_enc: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Slice [ptr:ptr+N] from labels_all and expand to [N*T*V] to match flat features."""
        N = obs_enc.shape[0]
        TV = obs_enc[0].numel() // obs_enc.shape[-1]
        lbl = labels_all[ptr:ptr + N].unsqueeze(1).expand(N, TV).reshape(-1)
        return lbl.to(device)

    # @profile
    def eval(self):
        self.set_model_eval()
        if self.cfg.get('eval_offline', False):
            # env-specific offline eval
            self.workspace.set_models(
                encoder=self.encoder,
                projector=self.projector,
            )
            offline_eval_results = self.workspace.run_offline_eval()
            if self.accelerator.is_main_process:
                self.log_append("env_offline_eval", 1, offline_eval_results)
        loss_avg = 0
        task_labels_all = self._build_task_labels(self.test_loader)
        lbl_ptr = 0

        with utils.eval_mode(
            self.encoder,
            self.projector,
            no_grad=True,
        ):
            # eval on test set
            for step, data in enumerate(tqdm.tqdm(self.test_loader)):
                (
                    obs_enc,
                    obs_proj,
                    *rest,
                    ssl_loss,
                    ssl_loss_components,
                ) = self.forward(data)
                ssl_loss = self.accelerator.gather_for_metrics(ssl_loss).mean()
                ssl_loss_components = utils.reduce_dict(
                    torch.mean,
                    self.accelerator.gather_for_metrics(ssl_loss_components),
                )
                loss_avg += ssl_loss.detach()
                if (hasattr(self, 'actor') or hasattr(self, 'policy')) and hasattr(self, 'forward_action'):
                    act_loss = self.forward_action((obs_enc, obs_proj), data)
                    total_loss = ssl_loss + act_loss
                    ssl_loss_components['action_loss'] = act_loss
                    ssl_loss_components['total_loss'] = total_loss
                    loss_avg += act_loss.detach()

                self.log_append(
                    "ssl_eval",
                    len(data[0]),
                    ssl_loss_components,
                )
                # Store can_var_loss for early stopping (D4 criterion)
                if self.mode in ['train', 'tune']:
                    if 'view_can_var_loss' in ssl_loss_components and 'can_var_loss' not in self.epoch_metrics:
                        val = ssl_loss_components['view_can_var_loss']
                        self.epoch_metrics['can_var_loss'] = val.item() if isinstance(val, torch.Tensor) else val
                    if 'total_loss' not in self.epoch_metrics:
                        self.epoch_metrics['loss'] = ssl_loss.item()

                if self.mode in ['train', 'tune']:
                    flat_obs_enc = self.accelerator.gather_for_metrics(obs_enc)
                    # NOTE: ViewOnly trainers return obs_enc as [N, V, T, E] (swapped),
                    # but the rearrange is a flat reshape so axis labels don't affect the result.
                    flat_obs_enc = einops.rearrange(flat_obs_enc, "N T V E -> (N T V) E")

                    enc_labels = None
                    if task_labels_all is not None:
                        enc_labels = self.accelerator.gather_for_metrics(
                            self._expand_task_labels(task_labels_all, lbl_ptr, obs_enc, obs_enc.device)
                        )

                    enc_health = self._ssl_health_metrics(flat_obs_enc, enc_labels)
                    self.log_append(
                        "metrics",
                        len(flat_obs_enc),
                        {"obs_enc/" + k: v for k, v in enc_health.items()},
                    )
                    # Store eff_rank for early stopping
                    if 'effective_rank' in enc_health and 'eff_rank' not in self.epoch_metrics:
                        self.epoch_metrics['eff_rank'] = enc_health['effective_rank'].item()

                    flat_obs_proj = self.accelerator.gather_for_metrics(obs_proj)
                    flat_obs_proj = einops.rearrange(flat_obs_proj, "N T V Z -> (N T V) Z")
                    self.log_append(
                        "metrics",
                        len(flat_obs_proj),
                        {"obs_proj/" + k: v for k, v in self._ssl_health_metrics(flat_obs_proj, enc_labels).items()},
                    )
                    lbl_ptr += len(data[0])
                if self.cfg.get('debug', False) and step > self.cfg.get('num_steps', 2):
                    print(f'eval(): break at step {step}: debug {self.cfg.get("debug",False)}')
                    break
                if step + 1 == len(self.test_loader):
                    print('ssl_eval', ssl_loss_components)
        return (loss_avg / len(self.test_loader))

    def set_model_train(self):
        self.encoder.train()
        self.projector.train()
        self.ssl.train()

    def set_model_eval(self):
        self.encoder.eval()
        self.projector.eval()
        self.ssl.eval()

    def resume_from_snapshot(self, snapshot_path: str):
        """Resume training from a saved snapshot, loading state dicts (DDP-safe).

        Restores model weights, optimizer states, and epoch number from the
        given snapshot file.  Works correctly even when models have already
        been wrapped by ``accelerator.prepare`` (loads via ``state_dict``
        rather than replacing objects).
        """
        snapshot_path = Path(snapshot_path)
        assert snapshot_path.exists(), f"Snapshot not found: {snapshot_path}"
        self.logger.info("Resuming from snapshot: %s", snapshot_path)
        payload = torch.load(snapshot_path, map_location="cpu")

        for k, v in payload.items():
            if k not in self.__dict__:
                self.logger.warning("Snapshot key '%s' not found on trainer, skipping", k)
                continue
            current = self.__dict__[k]
            if isinstance(v, torch.nn.Module) and isinstance(current, torch.nn.Module):
                # DDP-safe: load state dict, handling possible 'module.' prefix mismatch
                src_sd = v.state_dict()
                try:
                    current.load_state_dict(src_sd)
                except RuntimeError:
                    try:
                        # Strategy 1: add leading module. prefix (whole model is DDP-wrapped)
                        adjusted = {f"module.{kk}": vv for kk, vv in src_sd.items()}
                        current.load_state_dict(adjusted)
                    except RuntimeError:
                        # Strategy 2: handle per-submodule DDP wrapping by accelerate.prepare.
                        # Snapshot saved without .module. (properly unwrapped); current model
                        # has .module. inserted after each accelerate-prepared submodule name.
                        # Normalize both sides by stripping .module. then remap.
                        current_sd = current.state_dict()
                        def _norm(k):
                            return k.replace('.module.', '.')
                        src_norm = {_norm(kk): vv for kk, vv in src_sd.items()}
                        adjusted2 = {dk: src_norm[_norm(dk)] for dk in current_sd if _norm(dk) in src_norm}
                        if set(adjusted2.keys()) == set(current_sd.keys()):
                            current.load_state_dict(adjusted2)
                        else:
                            raise
                self.logger.info("  loaded model  %s", k)
            elif isinstance(v, torch.optim.Optimizer) and isinstance(current, torch.optim.Optimizer):
                current.load_state_dict(v.state_dict())
                self.logger.info("  loaded optim  %s", k)
            elif isinstance(v, list) and isinstance(current, list):
                for i in range(min(len(v), len(current))):
                    if hasattr(v[i], "state_dict") and hasattr(current[i], "load_state_dict"):
                        current[i].load_state_dict(v[i].state_dict())
                self.logger.info("  loaded list   %s", k)
            elif isinstance(v, (int, float, str)):
                self.__dict__[k] = v
                self.logger.info("  loaded scalar %s = %s", k, v)
            else:
                self.logger.warning("  skipped unsupported type for '%s': %s", k, type(v).__name__)

        # Advance epoch so we don't re-train the last saved epoch
        self.epoch += 1
        self.logger.info("Resuming from epoch %d", self.epoch)

    def run(self):
        # ── Resume from snapshot if requested ────────────────────────────────
        resume_path = self.cfg.get("resume_path", None)
        if resume_path is not None:
            self.resume_from_snapshot(resume_path)

        if self.cfg.get('save_epoch0_snapshot', False):
            print(f"Saving snapshot at epoch 0")
            self.save_snapshot()
            return 0.0

        self.train_iterator = tqdm.trange(
            self.epoch,
            self.cfg.num_epochs,
            disable=not self.accelerator.is_main_process,
            ncols=80,
        )
        self.train_iterator.set_description("training")
        # Reset the log.
        self.log_components = OrderedDict()
        for epoch in self.train_iterator:
            self.epoch = epoch

            self.train()

            loss_avg = None
            if self.mode in ['train', 'tune'] and len(self.test_loader) > 0:
                with torch.no_grad():
                    loss_avg = self.eval()

            if self.accelerator.is_main_process:
                self.flush_log(step=self.epoch, iterator=self.train_iterator)

            should_save = (
                (self.cfg.get('save_every_epochs', None) and (epoch + 1) % self.cfg.save_every_epochs == 0) or
                (epoch + 1 == self.cfg.num_epochs) or
                (self.cfg.get('save_at_epochs', None) and (epoch + 1) in self.cfg.save_at_epochs)
            )
            if should_save:
                self.save_snapshot(iter=self.epoch)

            # ── Update metric history and check callbacks ──
            if self.mode in ['train', 'tune'] and self.epoch_metrics:
                for metric_name, metric_val in self.epoch_metrics.items():
                    if metric_name not in self.metric_history:
                        self.metric_history[metric_name] = []
                    self.metric_history[metric_name].append(metric_val)
                # Clear epoch metrics for next epoch
                self.epoch_metrics = {}

            # early_stop break must be broadcast to all processes; if only main breaks,
            # non-main processes continue into the next epoch's gather_for_metrics → deadlock.
            should_stop = False
            if self.accelerator.is_main_process:
                # Check old early_stop first
                if self.early_stop is not None and loss_avg is not None and self.early_stop.update(loss_avg):
                    print(f'early stop threshold {self.early_stop.patience}; epoch {epoch}')
                    should_stop = True

                # Check new callbacks
                for callback in self.callbacks:
                    if callback.on_epoch_end(self.epoch, self.metric_history):
                        print(f'Stopping triggered by {callback.__class__.__name__} at epoch {epoch}')
                        should_stop = True
                        break

            if self.accelerator.num_processes > 1:
                stop_tensor = torch.tensor(int(should_stop), device=self.accelerator.device)
                torch.distributed.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            if should_stop:
                break

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def save_snapshot(self, iter='', interval='epoch'):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:

            payload = {}
            # Unwrap top-level DDP models.
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    payload[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    payload[k] = self.__dict__[k]

            _ddp_stash = {}  # {stash_key: (obj, attr_or_name, original_ddp)}

            def _collect_ddp(module):
                # Pattern A: __dict__ bypass (DDP stored directly in __dict__)
                for attr, val in list(vars(module).items()):
                    if isinstance(val, torch.nn.parallel.DistributedDataParallel):
                        key = (id(module), 'dict', attr)
                        _ddp_stash[key] = (module, attr, val)
                        module.__dict__[attr] = self.accelerator.unwrap_model(val)
                # Pattern B: registered submodules in _modules
                for name, submod in list(module._modules.items()):
                    if isinstance(submod, torch.nn.parallel.DistributedDataParallel):
                        key = (id(module), 'mod', name)
                        _ddp_stash[key] = (module, name, submod)
                        module._modules[name] = self.accelerator.unwrap_model(submod)
                    elif submod is not None:
                        _collect_ddp(submod)

            for k, obj in payload.items():
                if isinstance(obj, torch.nn.Module):
                    _collect_ddp(obj)

            try:
                with (self.work_dir / "snapshot.pt").open("wb") as f:
                    torch.save(payload, f)
                with (self.work_dir / "encoder.pt").open("wb") as f:
                    torch.save(payload["encoder"], f)
                with (self.work_dir / f"snapshot_{interval}{iter}.pt").open("wb") as f:
                    torch.save(payload, f)
                with (self.work_dir / f"encoder_{interval}{iter}.pt").open("wb") as f:
                    torch.save(payload["encoder"], f)
                print('saved', self.work_dir / f"encoder_{interval}{iter}.pt")
            finally:
                # Restore the DDP wrappers we temporarily swapped out above,
                # even if a save fails: otherwise training objects stay corrupted.
                for (module_id, storage, name), (obj, attr_or_name, original) in _ddp_stash.items():
                    if storage == 'mod':
                        obj._modules[attr_or_name] = original
                    else:
                        obj.__dict__[attr_or_name] = original

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value),
            )

    def flush_log(self, step, iterator=None):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        for key, value in self.log_components.items():
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Set the iterator status
            log_key, name_key = key.split("/", 1)
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        if iterator is not None:
            iterator.set_postfix_str(postfix)
        self.accelerator.log(log_components, step=step)
        self.logger.info(f"[{self.job_num}] Epoch {self.epoch}: {log_components}")
        self.log_components = OrderedDict()

