# Encoder SSL pretraining for LIBERO Goal (SO(2) rotation prediction + canonical-warp forward dynamics).
# Trainer: CanonEncoder
# Usage: python train_encoder.py --config-path configs --config-name encoder +debug=true
import hydra
import torch
import torch.nn.functional as F
import tqdm

import utils
from utils.trainer import Trainer, clear, apply_debug_config_encoder


# theta is a view's yaw relative to the (JSON-defined) canonical azimuth, so the
# canonical view has theta 0 by definition. A view is canonical iff theta == this.
CANONICAL_THETA = 0.0




class CanonEncoder(Trainer):
    """
    View-only SO(2) trainer with angle prediction and self-supervised canonical loss.

    Module roles:
        view_projector  → MLP(4 → D)            maps SO(2) flat [B, 4] → rotation latent [B, D]
                          Shared with view_ssl as view_ssl.projector (same object).
        view_ssl        → CANON_RotationSO2AnglePred
                          Houses forward_dynamics (transformer), view_projector (above),
                          and angle_head (AnglePredictionHead). Trainer accesses
                          sub-modules directly via view_ssl_module; view_ssl.forward()
                          is NOT called (trainer drives all dynamics explicitly).

    Rotation representation: 4-dim SO(2) flat [cos θ, −sin θ, sin θ, cos θ]:
        Built from GT angles via _build_so2_flat(delta_theta).
        Inverse R(−θ) = _build_so2_flat(-theta) = [cos θ, sin θ, −sin θ, cos θ].

    Dynamics calls: all frame-independent (T′=1 per timestep, batched over B×T):
        forward_dynamics(cat(f_src [B*T,1,E], rotation_latent [B*T,1,D])) → f_tgt [B*T,1,E]
        Three passes batched in one call: v1→v2, v1→0°, v2→0°.

    Data format (LiberoGoalMultiViewDatasetAngle):
        obs   [B, T, 2, C, H, W]: (v1, v2)
        aux   dict: theta_v1, theta_v2 (per-view yaw relative to canonical, radians)

    Config pointers:
        view_projector._target_: MLP with input_dim=4
        view_ssl._target_:       models.ssl.CANON_RotationSO2AnglePred

    Losses:
        L_view       cosine(forward_dynamics(f_v1, R(θ₂−θ₁)), f_v2)     cross-view, same t
        L_canonical  symmetric cosine(forward_dynamics(f_v1, R(−θ₁)),
                                      forward_dynamics(f_v2, R(−θ₂)))    self-supervised
        L_angle      MSE(angle_head(f_v1), gt_sincos_v1) + v2   backprops into encoder
        L_cov        covariance regularisation on obs_enc

    Schedule:
        epoch < loss_warmup_epochs : L_view + cov_coef·L_cov
        epoch ≥ loss_warmup_epochs : above + angle_coef·L_angle + canonical_coef·L_canonical
    """

    def __init__(self, cfg):
        # ── from Trainer ────────────────────────────────────────────────────
        # CANON is view-only: _init_projector / _init_ssl are overridden to no-ops
        # below, so no DynaMo motion (inverse/forward-dynamics) branch is built.
        super().__init__(cfg)

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
        # ── from TrainerViewInvar ────────────────────────────────────────────
        if self.view_ssl is None:
            self.view_ssl = hydra.utils.instantiate(
                self.cfg.view_ssl,
                encoder=self.encoder,
                projector=self.view_projector,
            )
            self.view_ssl_optim = self.view_ssl.optimizers

        # ── from CanonEncoder ────────────────────────────────────────────────
        if self.view_ssl is not None:
            # DynaMoSSL prepares its sub-modules internally; no extra accelerator.prepare.
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

    def train(self):
        """Override Trainer.train() to skip self.ssl.adjust_beta() (ssl is None)."""
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
        # ── from _ViewOnlyMixin ──────────────────────────────────────────────
        self.encoder.train()
        self.view_projector.train()
        self.view_ssl.forward_dynamics.train()
        # ── from CanonEncoder ────────────────────────────────────────────────
        self.view_ssl_module.angle_head.train()

    def set_model_eval(self):
        # ── from _ViewOnlyMixin ──────────────────────────────────────────────
        self.encoder.eval()
        self.view_projector.eval()
        self.view_ssl.forward_dynamics.eval()
        # ── from CanonEncoder ────────────────────────────────────────────────
        self.view_ssl_module.angle_head.eval()

    def backprop(self, loss):
        self.accelerator.backward(loss)
        if self.cfg.clip_grad_norm:
            # Clip all trainable modules: consistent with _ViewOnlyMixin.backprop()
            # which clips encoder, view_projector, and forward_dynamics.
            # angle_head is also clipped here since it receives gradients from L_angle.
            for module in (self.encoder, self.view_projector,
                           self.view_ssl_module.forward_dynamics,
                           self.view_ssl_module.angle_head):
                self.accelerator.clip_grad_norm_(module.parameters(), self.cfg.clip_grad_norm)

    def opt_step(self):
        # ── from _ViewOnlyMixin (complete override; no motion-branch super() call) ──
        self.encoder_optim.step()
        self.view_projector_optim.step()
        self.view_ssl.step()

    def opt_zerograd(self):
        # ── from _ViewOnlyMixin (complete override; no motion-branch super() call) ──
        self.encoder_optim.zero_grad(set_to_none=True)
        self.view_projector_optim.zero_grad(set_to_none=True)
        self.view_ssl.zero_grad(set_to_none=True)

    @staticmethod
    def _build_so2_flat(delta_theta: torch.Tensor) -> torch.Tensor:
        """delta_theta [B] → flattened SO(2) matrix [B, 4]: [cos, -sin, sin, cos]."""
        c, s = torch.cos(delta_theta), torch.sin(delta_theta)
        return torch.stack([c, -s, s, c], dim=-1)

    @staticmethod
    def _cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Asymmetric cosine loss: gradients flow through pred only (target detached)."""
        return 1 - F.cosine_similarity(pred, target.detach(), dim=-1).mean()

    def forward(self, data):
        obs, _, _, aux = data
        # obs: [B, T, 2, C, H, W];  aux: dict of per-view supervision (see dataset)

        theta_v1 = aux['theta_v1']   # [B] absolute yaw in radians
        theta_v2 = aux['theta_v2']   # [B]

        with self.accelerator.autocast():

            # ── ENCODER ──────────────────────────────────────────────────────
            obs_enc = self.encoder(obs)   # [B, T, V=2, E=512]
            f_v1    = obs_enc[:, :, 0]    # [B, T, E]
            f_v2    = obs_enc[:, :, 1]    # [B, T, E]

            # ── ANGLE PREDICTION (both views in one forward pass) ────────────
            # angle_head: [B, T, E] → [B, 2] sincos (attention-pooled internally).
            # No stop-grad: L_angle backprops into the encoder (keeps features angle-discriminative).
            sincos_both = self.view_ssl_module.angle_head(
                torch.cat([f_v1, f_v2], dim=0)                 # [2B, T, E]
            )                                                            # [2B, 2]
            sincos_v1, sincos_v2 = sincos_both.chunk(2, dim=0)          # [B, 2] each
            gt_sc_v1  = torch.stack([torch.sin(theta_v1), torch.cos(theta_v1)], dim=-1)
            gt_sc_v2  = torch.stack([torch.sin(theta_v2), torch.cos(theta_v2)], dim=-1)
            L_angle   = (F.mse_loss(sincos_v1, gt_sc_v1) + F.mse_loss(sincos_v2, gt_sc_v2)) / 2

            # ── ROTATION REPRESENTATIONS (4-dim SO(2) flat) ──────────────────
            delta_R_v1v2 = self._build_so2_flat(theta_v2 - theta_v1)  # [B, 4]  v1 → v2
            delta_R_v1v0 = self._build_so2_flat(-theta_v1)             # [B, 4]  v1 → 0°
            delta_R_v2v0 = self._build_so2_flat(-theta_v2)             # [B, 4]  v2 → 0°

            # Project all 3 SO(2) flats in one forward pass → rotation latents
            # sequential: lat_v1v2 = self.view_projector(delta_R_v1v2)
            #             lat_v1v0 = self.view_projector(delta_R_v1v0)
            #             lat_v2v0 = self.view_projector(delta_R_v2v0)
            lat_v1v2, lat_v1v0, lat_v2v0 = self.view_projector(
                torch.cat([delta_R_v1v2, delta_R_v1v0, delta_R_v2v0], dim=0)  # [3B, 4]
            ).chunk(3, dim=0)                                                   # [B, D] each

            # ── BATCHED DYNAMICS (3 pairs, frame-independent T′=1) ────────────
            # Each (sample, timestep) treated as an independent sequence of length 1.
            B, T, E = f_v1.shape
            D = lat_v1v2.shape[-1]
            f_v1_flat = f_v1.reshape(B * T, 1, E)
            f_v2_flat = f_v2.reshape(B * T, 1, E)

            # Broadcast per-sample latent to per-(sample × timestep): [B, D] → [B*T, 1, D]
            def _rep_lat(lat):
                return lat.unsqueeze(1).expand(-1, T, -1).reshape(B * T, 1, D)

            # Concatenate obs + rotation latent → forward_dynamics input [E+D]
            inp_all = torch.cat([
                torch.cat([f_v1_flat, _rep_lat(lat_v1v2)], dim=-1),   # v1 → v2
                torch.cat([f_v1_flat, _rep_lat(lat_v1v0)], dim=-1),   # v1 → 0°
                torch.cat([f_v2_flat, _rep_lat(lat_v2v0)], dim=-1),   # v2 → 0°
            ], dim=0)  # [3*B*T, 1, E+D]

            f_pred_all = self.view_ssl_module.forward_dynamics(inp_all)  # [3*B*T, 1, E]

            f_v2_pred, f_v1_can, f_v2_can = f_pred_all.reshape(3, B * T, 1, E).unbind(0)
            f_v2_pred = f_v2_pred.reshape(B, T, E)
            f_v1_can  = f_v1_can.reshape(B, T, E)
            f_v2_can  = f_v2_can.reshape(B, T, E)

            # ── LOSSES ───────────────────────────────────────────────────────
            L_view = self._cosine_loss(f_v2_pred, f_v2)

            oversample_w       = self.cfg.get('canonical_oversample_weight', 1.0)
            # Lazily compute the canonical mask (needed only for canonical oversampling)
            if oversample_w != 1.0:
                is_can_v1 = theta_v1 == CANONICAL_THETA   # [B] bool
                is_can_v2 = theta_v2 == CANONICAL_THETA   # [B] bool
                is_canonical = is_can_v1 | is_can_v2       # [B] bool

            # Symmetric cosine loss: both directions in one call (mean over 2B·T
            # = (mean_v1 + mean_v2) / 2). L_cov prevents representational collapse.
            # sequential: L_canonical = (self._cosine_loss(f_v1_can, f_v2_can)
            #                          + self._cosine_loss(f_v2_can, f_v1_can)) / 2
            if oversample_w != 1.0:
                w_bt = torch.where(
                    is_canonical.unsqueeze(1).expand(-1, T).reshape(-1),
                    f_v1.new_full((B * T,), oversample_w),
                    f_v1.new_ones(B * T),
                )
                w_2bt = w_bt.repeat(2)
                sim_can = F.cosine_similarity(
                    torch.cat([f_v1_can, f_v2_can], dim=0).reshape(2 * B * T, E),
                    torch.cat([f_v2_can, f_v1_can], dim=0).reshape(2 * B * T, E).detach(),
                    dim=-1,
                )
                L_canonical = (w_2bt * (1 - sim_can)).mean()
            else:
                L_canonical = 1 - F.cosine_similarity(
                    torch.cat([f_v1_can,  f_v2_can], dim=0),
                    torch.cat([f_v2_can,  f_v1_can], dim=0).detach(),
                    dim=-1,
                ).mean()

            L_cov = self.view_ssl_module._covariance_reg_loss(obs_enc)

            # VICReg on rotation latent (obs_proj): consistent with Methods 1&2
            # which apply this via cov_reg_obs_proj in CANON_RotationSO2.forward().
            # Stack all 3 rotation latents [3B, D] for richer batch statistics.
            all_rot_latents = torch.cat([lat_v1v2, lat_v1v0, lat_v2v0], dim=0)  # [3B, D]
            fd_covariance_loss = self.view_ssl_module._covariance_reg_loss(all_rot_latents)
            fd_variance_loss   = self.view_ssl_module._variance_reg_loss(all_rot_latents)
            # enc/can variance reg: dataset-specific. Off by default (LIBERO: low visual
            # diversity → opposes L_canonical, halves eff_rank 14.3→6.1). Enable for
            # MetaWorld via use_enc_can_variance_reg: true in YAML (high diversity →
            # constraint satisfied naturally, no conflict; eff_rank 7→14+).
            if self.cfg.get('use_enc_can_variance_reg', False):
                enc_variance_loss = self.view_ssl_module._variance_reg_loss(obs_enc)
                can_variance_loss = self.view_ssl_module._variance_reg_loss(
                    torch.cat([f_v1_can, f_v2_can], dim=0)
                )
                # Covariance reg on canonical predictions: L_canonical (cosine sim) aligns
                # direction but does not decorrelate: all 512 dims can collapse to a low-rank
                # subspace while still satisfying cosine similarity. This directly prevents that.
                can_covariance_loss = self.view_ssl_module._covariance_reg_loss(
                    torch.cat([f_v1_can, f_v2_can], dim=0)
                )
            else:
                enc_variance_loss = can_variance_loss = can_covariance_loss = obs_enc.new_zeros(1).squeeze()


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
        obs_proj = lat_v1v2.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, D]; for logging only; not used in dynamics forward pass
        return (obs_enc, obs_proj, L_total, loss_components)

    def eval(self):
        loss_avg = super().eval()
        self._eval_angle_prediction()
        return loss_avg

    @torch.no_grad()
    def _eval_angle_prediction(self):
        """Evaluate AnglePredictionHead (view_ssl_module.angle_head) against GT absolute yaw.

        Logs per-view MAE (degrees) and cosine similarity of (sin,cos) vectors.
        ~40s extra for one test-loader pass.
        """
        self.encoder.eval()
        self.view_ssl_module.angle_head.eval()

        angle_errors_v1, angle_errors_v2 = [], []
        cosine_sims_v1, cosine_sims_v2 = [], []

        for step, data in enumerate(self.test_loader):
            obs, _, _, aux = data
            # obs: [B, T, V, C, H, W];  aux: dict of per-view supervision
            theta_v1_gt = aux['theta_v1']  # [B]
            theta_v2_gt = aux['theta_v2']  # [B]

            obs_enc = self.encoder(obs)    # [B, T, V, E]
            f_v1 = obs_enc[:, :, 0]        # [B, T, E]
            f_v2 = obs_enc[:, :, 1]        # [B, T, E]

            sincos_both = self.view_ssl_module.angle_head(
                torch.cat([f_v1, f_v2], dim=0)  # [2B, T, E]
            )  # [2B, 2]
            sincos_v1, sincos_v2 = sincos_both.chunk(2, dim=0)  # [B, 2] each

            for sincos_pred, gt_theta, err_list, sim_list in [
                (sincos_v1, theta_v1_gt, angle_errors_v1, cosine_sims_v1),
                (sincos_v2, theta_v2_gt, angle_errors_v2, cosine_sims_v2),
            ]:
                gt_sincos = torch.stack([torch.sin(gt_theta), torch.cos(gt_theta)], dim=-1)
                theta_pred = torch.atan2(sincos_pred[:, 0], sincos_pred[:, 1])
                err = torch.abs(torch.atan2(
                    torch.sin(theta_pred - gt_theta),
                    torch.cos(theta_pred - gt_theta),
                ))
                # Gather across all processes so metrics cover the full test set
                err_list.append(self.accelerator.gather_for_metrics(err))
                sim_list.append(self.accelerator.gather_for_metrics(
                    F.cosine_similarity(sincos_pred, gt_sincos, dim=-1)))

            if self.cfg.get('debug', False) and step > self.cfg.get('num_steps', 2):
                break

        # Only main process logs: non-main processes have the same gathered
        # tensors but don't need to log.
        if self.accelerator.is_main_process:
            mae_v1 = torch.rad2deg(torch.cat(angle_errors_v1).mean()).item()
            mae_v2 = torch.rad2deg(torch.cat(angle_errors_v2).mean()).item()
            cos_v1 = torch.cat(cosine_sims_v1).mean().item()
            cos_v2 = torch.cat(cosine_sims_v2).mean().item()

            self.logger.info(
                f"angle_pred  v1 MAE: {mae_v1:.2f}° cos_sim: {cos_v1:.4f}  "
                f"v2 MAE: {mae_v2:.2f}° cos_sim: {cos_v2:.4f}"
            )
            self.log_append("angle_pred", 1, {
                "mae_deg_v1": mae_v1,
                "mae_deg_v2": mae_v2,
                "cos_sim_v1": cos_v1,
                "cos_sim_v2": cos_v2,
            })


@hydra.main(version_base="1.2", config_path="configs", config_name="encoder")
def main(cfg):

    apply_debug_config_encoder(cfg)

    module = cfg.get('module', 'canonencoder')

    dispatch = {
        'canonencoder': CanonEncoder,
    }

    if module not in dispatch:
        raise ValueError(f"Unknown module '{module}'. Available: {list(dispatch)}")

    trainer = dispatch[module](cfg)
    trainer.run()

    clear(cfg.get('debug', False), [utils.get_hydra_jobnum_workdir()[1]])  # clean up the debug working dir

if __name__ == "__main__":
    main()
