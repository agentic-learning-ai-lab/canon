"""Encoder SSL pretraining for MetaWorld (SO(3)-6D rotation prediction + canonical-warp SpatialTransformer).
Trainer: CanonEncoder.
Usage: python train_encoder.py --config-path configs --config-name encoder +debug=true
"""

import hydra
import torch
import torch.nn.functional as F
import tqdm

import utils
# NOTE: trainer.py lives in utils/; run this script with the project root on sys.path.
from utils.trainer import Trainer, clear, apply_debug_config_encoder




class CanonEncoder(Trainer):
    """SO(3)-6D encoder with spatial forward_dynamics (prevents GAP-collapse).

    Trains only the view dynamics branch (encoder + view_projector + view_ssl).
    The motion branch (projector / ssl) is removed after construction.

    Key design:
      - ``forward_dynamics`` is called with ``(spatial, rot_lat)`` instead of
        ``cat([gap, rot_lat])``. Outputs canonical_spatial of shape [N, C, H, W].
      - ``L_view`` and ``L_canonical`` are computed on **spatial features**
        (per-position cosine, averaged), not GAP. This prevents the encoder
        collapse (eff_rank ~3) caused by GAP-level alignment losses.

    Rotation representation: 6D (rot6d only, no log_dist).

    Data format (MetaworldGoalMultiViewDataset):
        obs  [B, T, 2, C, H, W]
        aux  dict: rot6d_v1v2, rot6d_v1_can, rot6d_v2_can (each [B, 6]);
             is_canonical, is_can_v1, is_can_v2 (each [B] bool)

    Config pointers:
        encoder._target_:        models.encoder.resnet18_spatial
        view_projector._target_: models.projector.MLP with input_dim=6
        view_ssl._target_:       models.ssl.CANON_RotationSO3AnglePred_6D_Spatial
        trainer._target_:        this class
    """

    def __init__(self, cfg):
        # Bootstrap the Trainer base (accelerator, encoder, dataset, loaders).
        # CANON is view-only: _init_projector / _init_ssl are overridden to no-ops
        # below, so no DynaMo motion (inverse/forward-dynamics) branch is built.
        Trainer.__init__(self, cfg)

        # Initialise view branch modules.
        self.view_projector = None
        self._init_view_projector()
        self.view_ssl = None
        self._init_view_ssl()

        # Encoder + view branch only (no DynaMo motion branch).
        self._keys_to_save = [
            "encoder", "encoder_optim",
            "view_projector", "view_projector_optim",
            "view_ssl", "view_ssl_optim",
            "epoch",
        ]
        self.count_params()

    # ------------------------------------------------------------------
    # Module initialisation
    # ------------------------------------------------------------------

    def _init_view_projector(self):
        config = (self.cfg.view_projector).copy()

        if self.view_projector is None:
            self.view_projector = hydra.utils.instantiate(
                config, _recursive_=False
            )
            self.view_projector_optim: torch.optim.Optimizer = (
                self.view_projector.configure_optimizers(
                    lr=self.cfg.ssl_lr,
                    weight_decay=self.cfg.ssl_weight_decay,
                    betas=tuple(self.cfg.betas),
                )
            )
            (
                self.view_projector,
                self.view_projector_optim,
            ) = self.accelerator.prepare(self.view_projector, self.view_projector_optim)

    def _init_view_ssl(self):
        # Instantiate view_ssl and register its optimizers (from TrainerViewInvar).
        if self.view_ssl is None:
            self.view_ssl = hydra.utils.instantiate(
                self.cfg.view_ssl,
                encoder=self.encoder,
                projector=self.view_projector,
            )
            self.view_ssl_optim = self.view_ssl.optimizers

        # Cache unwrapped module for direct sub-module access (from SO2ViewOnlyAnglePred).
        if self.view_ssl is not None:
            self.view_ssl_module = self.accelerator.unwrap_model(self.view_ssl)

    def _init_projector(self):
        # CANON is view-only: no DynaMo inverse-dynamics (motion) projector.
        # Overrides Trainer._init_projector so cfg.projector is not required.
        self.projector = None
        self.projector_optim = None

    def _init_ssl(self):
        # CANON is view-only: no DynaMo forward-dynamics (motion) SSL branch.
        # Overrides Trainer._init_ssl so cfg.ssl is not required.
        self.ssl = None
        self.ssl_optim = None

    # ------------------------------------------------------------------
    # Train / eval loop overrides
    # ------------------------------------------------------------------

    def train(self):
        """Override Trainer.train(); skips self.ssl.adjust_beta() (ssl is None)."""
        self.set_model_train()
        # skip self.ssl.adjust_beta; no motion branch
        pbar = tqdm.tqdm(
            self.train_loader,
            desc=f"epoch {self.epoch}",
            disable=not self.accelerator.is_main_process,
            ncols=80,
        )
        self.num_steps = self.cfg.get('num_steps', None)
        for step, data in enumerate(pbar):
            self.step = step
            self.train_step(data)
            if self.num_steps and step + 1 >= self.num_steps:
                print(f'terminate: reached num_steps {self.num_steps}')
                break

    def set_model_train(self):
        # _ViewOnlyMixin body:
        self.encoder.train()
        self.view_projector.train()
        self.view_ssl.forward_dynamics.train()
        # SO2ViewOnlyAnglePred addition:
        self.view_ssl_module.angle_head.train()

    def set_model_eval(self):
        # _ViewOnlyMixin body:
        self.encoder.eval()
        self.view_projector.eval()
        self.view_ssl.forward_dynamics.eval()
        # SO2ViewOnlyAnglePred addition:
        self.view_ssl_module.angle_head.eval()

    def backprop(self, loss):
        self.accelerator.backward(loss)
        if self.cfg.clip_grad_norm:
            # Clip all trainable modules including angle_head.
            for module in (self.encoder, self.view_projector,
                           self.view_ssl_module.forward_dynamics,
                           self.view_ssl_module.angle_head):
                self.accelerator.clip_grad_norm_(module.parameters(), self.cfg.clip_grad_norm)

    def opt_step(self):
        # View-only: skip Trainer's motion-branch optimizers entirely.
        self.encoder_optim.step()
        self.view_projector_optim.step()
        self.view_ssl.step()

    def opt_zerograd(self):
        # View-only: skip Trainer's motion-branch optimizers entirely.
        self.encoder_optim.zero_grad(set_to_none=True)
        self.view_projector_optim.zero_grad(set_to_none=True)
        self.view_ssl.zero_grad(set_to_none=True)

    def eval(self):
        # Execution order follows the original MRO:
        #   Trainer.eval → _eval_angle_prediction (SO3_6D variant) → _eval_temporal_std
        loss_avg = Trainer.eval(self)
        self._eval_angle_prediction()
        self._eval_temporal_std()
        return loss_avg

    # ------------------------------------------------------------------
    # Loss helper
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_cosine_loss(
        pred_spatial: torch.Tensor, target_spatial: torch.Tensor
    ) -> torch.Tensor:
        """Per-position cosine loss on spatial maps [N, C, H, W].

        Reshapes to [N*H*W, C] and computes cosine over C; targets detached.
        Aligning per position requires the encoder to keep rich per-position
        features (i.e. high spatial-channel rank), not just rich on average.
        """
        N, C, H, W = pred_spatial.shape
        pred   = pred_spatial.permute(0, 2, 3, 1).reshape(N * H * W, C)
        target = target_spatial.permute(0, 2, 3, 1).reshape(N * H * W, C)
        return 1 - F.cosine_similarity(pred, target.detach(), dim=-1).mean()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data):
        from models.ssl import rot6d_to_matrix, geodesic_loss

        obs, _, _, aux = data

        rot6d_v1v2   = aux['rot6d_v1v2']
        rot6d_v1_can = aux['rot6d_v1_can']
        rot6d_v2_can = aux['rot6d_v2_can']
        is_canonical = aux['is_canonical']

        with self.accelerator.autocast():

            # ── ENCODER (resnet18_spatial returns gap + spatial) ────────────
            obs_enc, obs_spatial = self.encoder(obs)
            # obs_enc:     [B, T, V, E]
            # obs_spatial: [B, T, V, C, H, W]

            f_v1 = obs_enc[:, :, 0]                          # [B, T, E] (GAP; for regularizers)
            f_v2 = obs_enc[:, :, 1]
            f_v1_sp = obs_spatial[:, :, 0]                   # [B, T, C, H, W]
            f_v2_sp = obs_spatial[:, :, 1]

            # ── ANGLE PREDICTION: 6D rot6d only (uses detached spatial) ────
            pred_both = self.view_ssl_module.angle_head(
                torch.cat([f_v1_sp.detach(), f_v2_sp.detach()], dim=0)
            )                                                # [2B, 6]
            pred_v1, pred_v2 = pred_both.chunk(2, dim=0)

            L_angle = (
                geodesic_loss(pred_v1, rot6d_v1_can)
                + geodesic_loss(pred_v2, rot6d_v2_can)
            ) / 2

            # ── ROTATION LATENTS (6-dim rot6d only) ─────────────────────────
            lat_v1v2, lat_v1_can, lat_v2_can = self.view_projector(
                torch.cat([rot6d_v1v2, rot6d_v1_can, rot6d_v2_can], dim=0)
            ).chunk(3, dim=0)                                 # [B, D] each

            # ── BATCHED SPATIAL FORWARD_DYNAMICS (3 pairs, T flattened) ─────
            B, T, V, C, H, W = obs_spatial.shape
            E = f_v1.shape[-1]
            D = lat_v1v2.shape[-1]

            f_v1_flat = f_v1_sp.reshape(B * T, C, H, W)       # [B*T, C, H, W]
            f_v2_flat = f_v2_sp.reshape(B * T, C, H, W)

            def _rep_lat(lat):
                # rot_lat is per-sample (B, D); replicate across the T window.
                return lat.unsqueeze(1).expand(-1, T, -1).reshape(B * T, D)

            spatial_in = torch.cat([f_v1_flat, f_v1_flat, f_v2_flat], dim=0)
            rot_in     = torch.cat([
                _rep_lat(lat_v1v2),
                _rep_lat(lat_v1_can),
                _rep_lat(lat_v2_can),
            ], dim=0)

            f_pred_all_sp = self.view_ssl_module.forward_dynamics(spatial_in, rot_in)
            # [3*B*T, C, H, W]  -> split back to 3 outputs
            f_v2_pred_sp, f_v1_can_pred_sp, f_v2_can_pred_sp = (
                f_pred_all_sp.reshape(3, B * T, C, H, W).unbind(0)
            )

            # ── L_VIEW on spatial (per-position cosine) ─────────────────────
            L_view = self._spatial_cosine_loss(f_v2_pred_sp, f_v2_flat)

            # ── L_CANONICAL on spatial (per-position cosine, symmetric) ─────
            oversample_w       = self.cfg.get('canonical_oversample_weight', 1.0)

            # Symmetric cosine (each side as detached target alternately), per position.
            v1_flat = f_v1_can_pred_sp.permute(0, 2, 3, 1).reshape(B * T * H * W, C)
            v2_flat = f_v2_can_pred_sp.permute(0, 2, 3, 1).reshape(B * T * H * W, C)
            sim_can = F.cosine_similarity(
                torch.cat([v1_flat, v2_flat], dim=0),
                torch.cat([v2_flat, v1_flat], dim=0).detach(),
                dim=-1,
            )                                                  # [2*B*T*H*W]

            if oversample_w != 1.0:
                # Per-sample canonical weight, broadcast to (T,H,W) positions, then dup for both sides.
                w_b   = torch.where(
                    is_canonical,
                    f_v1.new_full((B,), oversample_w),
                    f_v1.new_ones(B),
                )                                              # [B]
                w_bt  = w_b.unsqueeze(1).expand(-1, T).reshape(B * T)
                w_bthw = w_bt.unsqueeze(1).expand(-1, H * W).reshape(B * T * H * W)
                w_2bthw = w_bthw.repeat(2)
                L_canonical = (w_2bthw * (1 - sim_can)).mean()
            else:
                L_canonical = (1 - sim_can).mean()

            # ── Covariance reg on GAP encoder features (auxiliary regulariser) ─
            L_cov = self.view_ssl_module._covariance_reg_loss(obs_enc)

            # ── Rotation-latent VICReg regs (unchanged) ─────────────────────
            all_rot_latents = torch.cat([lat_v1v2, lat_v1_can, lat_v2_can], dim=0)
            fd_covariance_loss = self.view_ssl_module._covariance_reg_loss(all_rot_latents)
            fd_variance_loss   = self.view_ssl_module._variance_reg_loss(all_rot_latents)

            # ── Encoder + canonical variance/covariance regs (on GAP-derived) ─
            if self.cfg.get('use_enc_can_variance_reg', False):
                # canonical_gap = AvgPool(canonical_spatial) over H*W
                f_v1_can_pred_gap = f_v1_can_pred_sp.mean(dim=(-1, -2))   # [B*T, C]
                f_v2_can_pred_gap = f_v2_can_pred_sp.mean(dim=(-1, -2))   # [B*T, C]
                enc_variance_loss = self.view_ssl_module._variance_reg_loss(obs_enc)
                can_gap_concat = torch.cat([f_v1_can_pred_gap, f_v2_can_pred_gap], dim=0)
                can_variance_loss   = self.view_ssl_module._variance_reg_loss(can_gap_concat)
                can_covariance_loss = self.view_ssl_module._covariance_reg_loss(can_gap_concat)
            else:
                enc_variance_loss = can_variance_loss = can_covariance_loss = (
                    obs_enc.new_zeros(1).squeeze()
                )


            # ── Loss aggregation ─────────────────────────────────────────────
            cov_coef       = self.cfg.get('cov_loss_coef',       1.0)
            angle_coef     = self.cfg.get('angle_loss_coef',     0.1)
            canonical_coef = self.cfg.get('canonical_loss_coef', 1.0)
            warmup_epochs  = self.cfg.get('loss_warmup_epochs',  5)

            if self.epoch < warmup_epochs:
                L_total = (L_view + cov_coef * L_cov
                          + fd_covariance_loss + fd_variance_loss
                          + enc_variance_loss + can_variance_loss + can_covariance_loss)
            else:
                L_total = (L_view
                         + cov_coef       * L_cov
                         + angle_coef     * L_angle
                         + canonical_coef * L_canonical
                         + fd_covariance_loss + fd_variance_loss
                         + enc_variance_loss + can_variance_loss + can_covariance_loss)

        loss_components = {
            "total_loss":           L_total,
            "view_loss":            L_view,
            "canonical_loss":       L_canonical,
            "angle_loss":           L_angle,
            "cov_loss":             L_cov,
            "fd_covariance_loss":   fd_covariance_loss,
            "fd_variance_loss":     fd_variance_loss,
            "enc_variance_loss":    enc_variance_loss,
            "can_variance_loss":    can_variance_loss,
            "can_covariance_loss":  can_covariance_loss,
        }
        obs_proj = lat_v1v2.unsqueeze(1).unsqueeze(1)            # log-shaped placeholder
        return (obs_enc, obs_proj, L_total, loss_components)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_angle_prediction(self):
        """Evaluate AnglePredictionHeadSO3 (6D variant) against GT SO(3) rotation.

        Logs per-view: geodesic error (deg), azimuth MAE (deg), elevation MAE (deg),
        and 6D cosine similarity (no distance: pred has shape [B, 6]).
        """
        import math
        from models.ssl import rot6d_to_matrix

        self.encoder.eval()
        self.view_ssl_module.angle_head.eval()

        metrics = {k: [] for k in [
            'geo_v1', 'geo_v2', 'az_v1', 'az_v2',
            'el_v1', 'el_v2', 'cos6d_v1', 'cos6d_v2',
        ]}

        for step, data in enumerate(self.test_loader):
            obs, _, _, aux = data
            rot6d_v1_can_gt = aux['rot6d_v1_can']    # [B, 6]
            rot6d_v2_can_gt = aux['rot6d_v2_can']    # [B, 6]

            obs_enc, obs_spatial = self.encoder(obs)
            f_v1_spatial = obs_spatial[:, :, 0]
            f_v2_spatial = obs_spatial[:, :, 1]

            pred_both = self.view_ssl_module.angle_head(
                torch.cat([f_v1_spatial, f_v2_spatial], dim=0)
            )  # [2B, 6]
            pred_v1, pred_v2 = pred_both.chunk(2, dim=0)  # [B, 6]

            for pred, gt_rot6d, suffix in [
                (pred_v1, rot6d_v1_can_gt, 'v1'),
                (pred_v2, rot6d_v2_can_gt, 'v2'),
            ]:
                R_pred = rot6d_to_matrix(pred[:, :6])
                R_gt   = rot6d_to_matrix(gt_rot6d)

                trace = (R_pred.transpose(-2, -1) @ R_gt).diagonal(dim1=-2, dim2=-1).sum(-1)
                geo_rad = torch.acos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))
                geo_deg = geo_rad * 180 / math.pi

                el_pred = torch.asin(-R_pred[:, 2, 0].clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / math.pi
                az_pred = torch.atan2(R_pred[:, 1, 0], R_pred[:, 0, 0]) * 180 / math.pi
                el_gt   = torch.asin(-R_gt[:, 2, 0].clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / math.pi
                az_gt   = torch.atan2(R_gt[:, 1, 0], R_gt[:, 0, 0]) * 180 / math.pi
                az_mae  = torch.abs(az_pred - az_gt)
                el_mae  = torch.abs(el_pred - el_gt)

                cos6d = F.cosine_similarity(pred[:, :6], gt_rot6d, dim=-1)

                for key, val in [
                    (f'geo_{suffix}',   geo_deg),
                    (f'az_{suffix}',    az_mae),
                    (f'el_{suffix}',    el_mae),
                    (f'cos6d_{suffix}', cos6d),
                ]:
                    metrics[key].append(self.accelerator.gather_for_metrics(val))

            if self.cfg.get('debug', False) and step > self.cfg.get('num_steps', 2):
                break

        if self.accelerator.is_main_process:
            def _mean(k):
                return torch.cat(metrics[k]).mean().item()

            self.logger.info(
                f"angle_pred  v1 geodesic:{_mean('geo_v1'):.2f}°  "
                f"az_mae:{_mean('az_v1'):.2f}°  el_mae:{_mean('el_v1'):.2f}°  "
                f"cos6d:{_mean('cos6d_v1'):.4f}  |  "
                f"v2 geodesic:{_mean('geo_v2'):.2f}°  "
                f"az_mae:{_mean('az_v2'):.2f}°  el_mae:{_mean('el_v2'):.2f}°  "
                f"cos6d:{_mean('cos6d_v2'):.4f}"
            )
            self.log_append("angle_pred", 1, {
                "geodesic_deg_v1": _mean('geo_v1'),
                "geodesic_deg_v2": _mean('geo_v2'),
                "az_mae_deg_v1":   _mean('az_v1'),
                "az_mae_deg_v2":   _mean('az_v2'),
                "el_mae_deg_v1":   _mean('el_v1'),
                "el_mae_deg_v2":   _mean('el_v2'),
                "cos6d_v1":        _mean('cos6d_v1'),
                "cos6d_v2":        _mean('cos6d_v2'),
            })

    @torch.no_grad()
    def _eval_temporal_std(self):
        """Log mean temporal std of GAP features across timesteps, averaged over samples and views.

        Low values (<0.05) indicate VICReg data starvation: features encode trajectory identity
        rather than robot state. Target: >0.10 (cf. SO2 255-traj ≈ 0.17–0.28).
        """
        self.encoder.eval()
        temporal_stds = []

        for step, data in enumerate(self.test_loader):
            obs, _, _, _ = data
            obs_enc, _ = self.encoder(obs)    # [B, T, V, E]
            if obs_enc.shape[1] < 2:
                break                          # need ≥2 timesteps for meaningful std
            std_per_sample = obs_enc.std(dim=1)              # [B, V, E]
            mean_std = std_per_sample.mean(dim=-1)           # [B, V]
            temporal_stds.append(self.accelerator.gather_for_metrics(mean_std))
            if self.cfg.get('debug', False) and step > 2:
                break

        if not temporal_stds:
            return
        if self.accelerator.is_main_process:
            all_stds = torch.cat(temporal_stds, dim=0)       # [N, V]
            mean_per_view = all_stds.mean(dim=0)             # [V]
            overall = all_stds.mean().item()
            self.logger.info(
                "temporal_std  overall:{:.4f}  {}".format(
                    overall,
                    "  ".join(f"v{i}:{mean_per_view[i].item():.4f}"
                              for i in range(mean_per_view.shape[0]))
                )
            )
            log_dict = {"temporal_std_overall": overall}
            for i in range(mean_per_view.shape[0]):
                log_dict[f"temporal_std_v{i}"] = mean_per_view[i].item()
            self.log_append("temporal_std", 1, log_dict)


@hydra.main(version_base="1.2", config_path="configs", config_name="encoder")
def main(cfg):

    apply_debug_config_encoder(cfg)

    module = cfg.get('module', 'canonencoder')

    if module == 'canonencoder':
        trainer = CanonEncoder(cfg)
    else:
        raise ValueError(f"Unknown module {module}")

    trainer.run()

    clear(cfg.get('debug', False), [utils.get_hydra_jobnum_workdir()[1]])  # clean up the debug working dir

if __name__ == "__main__":
    main()
