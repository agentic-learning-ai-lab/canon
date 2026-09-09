"""
CANON SSL modules.

Naming: `DynaMoSSL` is the SSL core inherited from DynaMo; everything CANON adds on top carries
the `CANON_` prefix.

Class dependency chain (single linear chain; the two leaves are the shipped encoders):

    nn.Module
    └── DynaMoSSL                                   upstream DynaMo SSL core: builds the
        │                                           forward-dynamics transformer + its optimizer,
        │                                           covariance/variance regularisers, step/zero_grad.
        └── CANON_RotationSO2                       view dynamics on *encoded* inputs (encoder runs
            │                                       outside this module) + SO(2) rotation conditioning.
            └── CANON_RotationSO2AnglePred          + per-view angle-prediction head.
                │                                   ← LIBERO-Goal encoder  (configs/encoder.yaml)
                │                                     LIBERO is genuinely SO(2): azimuth alone
                │                                     defines the viewpoint pool.
                └── CANON_RotationSO3AnglePred_6D_Spatial
                                                    + SO(3)-6D angle head and a SpatialTransformer
                                                      forward-dynamics over the layer4 7x7 map.
                                                    ← MetaWorld encoder    (configs/encoder.yaml)
                                                      NOTE: MetaWorld's SO(3) encoder *descends from*
                                                      the LIBERO SO(2) one: it refines it rather than
                                                      replacing it.

⚠️  __init__ is RNG-load-bearing. Construction proceeds base-first and each level *builds then
    discards* the previous level's module (see the cascade comment in the MetaWorld class). Those
    throwaway builds consume torch RNG, so the trained weights depend on them. Never "tidy" one
    away: doing so shifts the global RNG stream and silently produces a different encoder.

The trainers do NOT call forward() on these modules; they drive `forward_dynamics`, `angle_head`
and the `_covariance_reg_loss`/`_variance_reg_loss` helpers directly.
"""

import math
import inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

accelerator = Accelerator()


# ─────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average): used when ema_beta is set in config
# ─────────────────────────────────────────────────────────────────────────────

class EMA(nn.Module):
    def __init__(self, src_model: nn.Module, beta: float, copy: bool = True):
        super().__init__()
        self.model = deepcopy(src_model) if copy else src_model
        self.model.eval()
        self.model.requires_grad_(False)
        self.beta = beta

    def step(self, src_model):
        for ema_p, src_p in zip(self.model.parameters(), src_model.parameters()):
            ema_p.data.mul_(self.beta).add_(src_p.data, alpha=1.0 - self.beta)
            ema_p.requires_grad_(False)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.model(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# TransformerEncoder (nanoGPT-style, causal)
# Based on: https://github.com/karpathy/nanoGPT
# ─────────────────────────────────────────────────────────────────────────────

class _LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn   = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj   = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                               dropout_p=self.dropout if self.training else 0,
                                               is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y   = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class _TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = _LayerNorm(config.n_embd, bias=config.bias)
        self.attn = _CausalSelfAttention(config)
        self.ln_2 = _LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class TransformerEncoderConfig:
    block_size: int = 10
    input_dim:  int = 512
    n_layer:    int = 3
    n_head:     int = 4
    n_embd:     int = 256
    output_dim: int = 512
    dropout:  float = 0.0
    bias:      bool = True


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Linear(config.input_dim, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([_TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=_LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.output_head = nn.Linear(config.n_embd, config.output_dim, bias=True)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            # GPT-2 scaled init on residual projections: the attention output (c_proj) AND the MLP
            # residual output. The original block named the MLP output 'c_proj' so this single check
            # caught both; the nn.Sequential refactor renamed the MLP output to 'mlp.2', so it must be
            # matched explicitly to preserve std = 0.02/sqrt(2*n_layer) (otherwise the MLP branch is
            # initialized ~sqrt(2*n_layer)x too large, slowing encoder feature decorrelation).
            if pn.endswith("c_proj.weight") or pn.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, target=None):
        device = x.device
        b, t, d = x.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        output = self.output_head(x)
        if target is None:
            return output
        return output, F.mse_loss(output, target)

    def configure_optimizers(self, weight_decay, lr, betas, device_type=None):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(
            [{"params": decay_params,   "weight_decay": weight_decay},
             {"params": nodecay_params, "weight_decay": 0.0}],
            lr=lr, betas=betas, **extra_args,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DynaMoSSL base: VICReg losses + vectorized forward-dynamics
# ─────────────────────────────────────────────────────────────────────────────

def off_diag(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def off_diag_cov_loss(x: torch.Tensor) -> torch.Tensor:
    cov = torch.cov(einops.rearrange(x, "... E -> E (...)"))
    return off_diag(cov).square().mean()


def variance_reg_loss(x: torch.Tensor) -> torch.Tensor:
    x_flat = einops.rearrange(x, "... E -> (...) E")
    std = torch.sqrt(x_flat.var(dim=0) + 1e-4)
    return torch.clamp(1 - std, min=0).mean()


class DynaMoSSL(nn.Module):
    """Upstream DynaMo SSL core (inherited, not a CANON contribution).

    Builds the forward-dynamics transformer + its optimizer, and provides the covariance/variance
    regularisers and the optimizer step/zero_grad plumbing that every CANON subclass reuses.
    Never instantiated directly in this release: both encoders use a CANON_* subclass.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        window_size: int,
        feature_dim: int,
        projection_dim: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        covariance_reg_coef: float = 0.04,
        dynamics_loss_coef: float = 1.0,
        ema_beta: Optional[float] = None,
        beta_scheduling: bool = False,
        projector_use_ema: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        separate_single_views: bool = True,
        cov_reg_obs_proj: bool = False,
        variance_reg_coef: float = 1.0,
    ):
        nn.Module.__init__(self)
        self.__dict__["encoder"]   = encoder
        self.__dict__["projector"] = projector
        forward_dynamics_cfg = TransformerEncoderConfig(
            block_size=window_size,
            input_dim=feature_dim + projection_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            output_dim=feature_dim,
        )
        self.forward_dynamics = TransformerEncoder(forward_dynamics_cfg)
        self.forward_dynamics_optimizer = self.forward_dynamics.configure_optimizers(
            weight_decay=weight_decay, lr=lr, betas=betas,
        )
        self.optimizers = [self.forward_dynamics_optimizer]
        self.forward_dynamics, self.forward_dynamics_optimizer = accelerator.prepare(
            self.forward_dynamics, self.forward_dynamics_optimizer,
        )
        self.covariance_reg_coef = covariance_reg_coef
        self.dynamics_loss_coef  = dynamics_loss_coef
        self.ema_beta            = ema_beta
        self.beta_scheduling     = beta_scheduling
        self.projector_use_ema   = projector_use_ema
        if self.ema_beta is not None:
            self.ema_encoder = EMA(self.encoder, self.ema_beta)
            if self.projector_use_ema:
                self.ema_projector = EMA(self.projector, self.ema_beta)
        self.separate_single_views = separate_single_views
        self.cov_reg_obs_proj      = cov_reg_obs_proj
        self.variance_reg_coef     = variance_reg_coef

    def _covariance_reg_loss(self, obs_enc: torch.Tensor):
        return off_diag_cov_loss(obs_enc) * self.covariance_reg_coef

    def _variance_reg_loss(self, x: torch.Tensor):
        return variance_reg_loss(x) * self.variance_reg_coef

    def adjust_beta(self, epoch: int, max_epoch: int):
        if (self.ema_beta is None) or not self.beta_scheduling or (max_epoch == 0):
            return
        self.ema_encoder.beta = 1.0 - 0.5 * (
            1.0 + np.cos(np.pi * epoch / max_epoch)
        ) * (1.0 - self.ema_beta)
        if self.projector_use_ema:
            self.ema_projector.beta = self.ema_encoder.beta

    def step(self):
        self.forward_dynamics_optimizer.step()
        if self.ema_beta is not None:
            self.ema_encoder.step(self.encoder)
            if self.projector_use_ema:
                self.ema_projector.step(self.projector)

    def zero_grad(self, set_to_none):
        self.forward_dynamics_optimizer.zero_grad(set_to_none=set_to_none)


# ─────────────────────────────────────────────────────────────────────────────
# CANON_RotationSO2: encoded-input view dynamics + SO(2) rotation conditioning
# ─────────────────────────────────────────────────────────────────────────────

class CANON_RotationSO2(DynaMoSSL):
    """View forward-dynamics on *encoded* inputs, conditioned on a relative rotation.

    Two differences from the DynaMo core it extends:
      1. The encoder runs *outside* this module (forward() takes `obs_enc`, not raw pixels), so
         the trainer can share one encoder pass across the motion and view branches.
      2. The dynamics are conditioned on a rotation latent (the SO(2) structure is built by the
         trainer and projected here), which is what makes the representation view-equivariant.

    Never instantiated directly; it is the shared base of both shipped encoders.
    """

    def __init__(self, cov_reg_loss: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cov_reg_loss = cov_reg_loss

    def convert_obs(self, obs_enc, obs=None):
        if self.ema_beta is not None:
            obs_target = self.ema_encoder(obs)
            obs_proj = self.ema_projector(obs_enc) if self.projector_use_ema else self.projector(obs_enc)
        else:
            obs_target = obs_enc
            obs_proj   = self.projector(obs_enc)
        return obs_enc, obs_proj, obs_target

    def forward(self, obs_enc, obs: torch.Tensor = None):
        obs_enc, obs_proj, obs_target = self.convert_obs(obs_enc, obs)
        covariance_loss = self._covariance_reg_loss(obs_enc) if self.cov_reg_loss else torch.tensor(0.0).to(obs_enc.device)
        dynamics_loss, dynamics_loss_components = self._forward_dyn_loss_vectorized(
            obs_enc, obs_proj, obs_target, self.separate_single_views
        )
        total_loss = dynamics_loss + covariance_loss
        loss_components = {"total_loss": total_loss, **dynamics_loss_components, "covariance_loss": covariance_loss}
        if self.cov_reg_obs_proj:
            fd_cov = self._covariance_reg_loss(obs_proj)
            fd_var = self._variance_reg_loss(obs_proj)
            total_loss += fd_cov + fd_var
            loss_components.update({"fd_covariance_loss": fd_cov, "fd_variance_loss": fd_var})
        return obs_enc, obs_proj, total_loss, loss_components

    def _forward_dyn_loss_vectorized(
        self,
        obs_enc:    torch.Tensor,
        obs_proj:   torch.Tensor,
        obs_target: torch.Tensor,
        separate_single_views: bool = True,
    ):
        N, T, V, E = obs_enc.shape

        if separate_single_views:
            obs_enc_j    = obs_enc[:, :-1]
            obs_proj_i   = obs_proj[:, 1:]
            obs_target_j = obs_target[:, 1:]
        else:
            v_indices_j  = torch.roll(torch.arange(V), shifts=-1, dims=0)
            obs_enc_j    = obs_enc[:, :-1].index_select(dim=2, index=v_indices_j)
            obs_proj_i   = obs_proj[:, 1:]
            obs_target_j = obs_target[:, 1:].index_select(dim=2, index=v_indices_j)

        forward_dyn_input = torch.cat([obs_enc_j, obs_proj_i], dim=-1)
        flat_input = einops.rearrange(forward_dyn_input, 'N T_prime V F -> (N V) T_prime F')
        obs_enc_pred_flat = self.forward_dynamics(flat_input)
        obs_enc_pred = einops.rearrange(obs_enc_pred_flat, '(N V) T_prime E -> N T_prime V E', N=N, V=V)

        similarity = F.cosine_similarity(obs_enc_pred, obs_target_j.detach(), dim=-1)
        loss = (1 - similarity.mean()) * self.dynamics_loss_coef / V
        return loss, {"dynamics_loss_total": loss}


# ─────────────────────────────────────────────────────────────────────────────
# Angle prediction heads
# ─────────────────────────────────────────────────────────────────────────────

class AnglePredictionHead(nn.Module):
    """Attention-pooled MLP: [B, T, E] → [B, 2] sincos (L2-normalized).

    Input: f [B, T, E]: encoder features over a trajectory window.
    """

    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.mlp  = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 2),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        attn_w = F.softmax(self.attn(f), dim=1)
        f_pool = (attn_w * f).sum(dim=1)
        return F.normalize(self.mlp(f_pool), dim=-1)

    def configure_optimizers(self, lr, weight_decay, betas):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


class AnglePredictionHeadSO3(nn.Module):
    """SO(3)+dist angle head: conv channel reduction + flatten projection.

    Input : x  [B, T, in_channels, H, W]  spatial maps from resnet18_spatial
    Output: [B, output_dim]               6D rotation (rot6d) or 7D (rot6d + log_dist)
    """

    def __init__(
        self,
        in_channels:      int = 512,
        reduced_channels: int = 128,
        spatial_size:     int = 7,
        output_dim:       int = 7,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1), nn.ReLU(),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=1), nn.ReLU(),
        )
        flat_dim    = reduced_channels * spatial_size * spatial_size
        self.proj   = nn.Linear(flat_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.conv(x).flatten(1)
        x = x.reshape(B, T, -1).mean(1)
        return self.proj(x)

    def configure_optimizers(self, lr, weight_decay, betas):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


# ─────────────────────────────────────────────────────────────────────────────
# LIBERO Goal CANON class: CANON_RotationSO2AnglePred
# ─────────────────────────────────────────────────────────────────────────────

class CANON_RotationSO2AnglePred(CANON_RotationSO2):
    """CANON encoder for LIBERO-Goal: SO(2) rotation conditioning + per-view angle head.

    Adds a self-calibrating angle-prediction head on top of the rotation-conditioned view dynamics:
    the head predicts each view's azimuth, which supplies the angle supervision that anchors the
    canonical (0°) frame. LIBERO is genuinely SO(2) (azimuth alone defines the viewpoint pool) so
    the rotation representation is the 4-dim flat SO(2) matrix [cos, -sin, sin, cos] (built by the
    trainer and projected here).

    The trainer calls `angle_head` directly; forward() would only handle dynamics.
    ← Hydra _target_ of libero_goal/configs/encoder.yaml
    """

    def __init__(self, *args, feat_dim: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.angle_head = AnglePredictionHead(input_dim=feat_dim)
        self.angle_head_optimizer = torch.optim.AdamW(
            self.angle_head.parameters(),
            lr=kwargs.get('lr', 1e-4),
            weight_decay=kwargs.get('weight_decay', 0.0),
            betas=kwargs.get('betas', (0.9, 0.999)),
        )
        self.angle_head, self.angle_head_optimizer = accelerator.prepare(
            self.angle_head, self.angle_head_optimizer
        )
        self.optimizers.append(self.angle_head_optimizer)

    def convert_obs(self, obs_enc: torch.Tensor, rotation_feat: torch.Tensor):
        obs_target = obs_enc
        rotation_latent = self.projector(rotation_feat)
        B, V, T, E = obs_enc.shape
        rotation_latent = einops.repeat(rotation_latent, 'b d -> b v t d', v=V, t=T)
        return obs_enc, rotation_latent, obs_target

    def step(self):
        super().step()
        self.angle_head_optimizer.step()

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none)
        self.angle_head_optimizer.zero_grad(set_to_none=set_to_none)


# ─────────────────────────────────────────────────────────────────────────────
# SO(3) utilities
# ─────────────────────────────────────────────────────────────────────────────

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt recovery of SO(3) matrix from 6D representation (Zhou et al. 2019)."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_loss(pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
    """Geodesic (angular) loss between two rotation matrices in 6D representation."""
    R_pred = rot6d_to_matrix(pred_6d)
    R_gt   = rot6d_to_matrix(target_6d)
    R_diff = R_pred.transpose(-2, -1) @ R_gt
    trace  = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# SpatialTransformer: bidirectional transformer over spatial tokens (7×7 map)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialTransformer(nn.Module):
    """Spatial-input forward_dynamics: bidirectional transformer over spatial tokens.

    Replaces the GAP-input TransformerEncoder used by the base DynaMoSSL.
    The 7×7 layer4 feature map (49 tokens) + 1 rotation register = 50 tokens.
    Standard random init (no skip / no zero-init) to avoid encoder collapse.

    Input : (spatial [N, C, H, W], rot_lat [N, D])
    Output:  canonical_spatial [N, C, H, W]
    """

    def __init__(
        self,
        channels:          int = 512,
        d_model:           int = 128,
        rot_lat_dim:       int = 64,
        n_layer:           int = 4,
        n_head:            int = 4,
        dim_feedforward:   int = None,
        dropout:         float = 0.0,
        n_spatial_tokens:  int = 49,
    ):
        super().__init__()
        self.channels = channels
        self.d_model   = d_model
        self.n_spatial_tokens = n_spatial_tokens
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.in_proj  = nn.Linear(channels, d_model)
        self.rot_proj = nn.Linear(rot_lat_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_spatial_tokens + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.out_proj = nn.Linear(d_model, channels)

    def forward(self, spatial: torch.Tensor, rot_lat: torch.Tensor) -> torch.Tensor:
        N, C, H, W = spatial.shape
        toks_spatial = self.in_proj(spatial.flatten(2).transpose(1, 2))
        tok_rot      = self.rot_proj(rot_lat).unsqueeze(1)
        toks = torch.cat([toks_spatial, tok_rot], dim=1) + self.pos_embed
        toks_out = self.transformer(toks)
        out_tokens = self.out_proj(toks_out[:, :H * W])
        return out_tokens.transpose(1, 2).reshape(N, C, H, W)

    def configure_optimizers(self, weight_decay, lr, betas):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=tuple(betas))


# ─────────────────────────────────────────────────────────────────────────────
# MetaWorld CANON class: CANON_RotationSO3AnglePred_6D_Spatial
# ─────────────────────────────────────────────────────────────────────────────

class CANON_RotationSO3AnglePred_6D_Spatial(CANON_RotationSO2AnglePred):
    """CANON encoder for MetaWorld: SO(3)-6D angle head + SpatialTransformer view dynamics.

    Refines the LIBERO SO(2) encoder it descends from, because MetaWorld's cameras vary in
    elevation and distance as well as azimuth:
      - the angle head becomes `AnglePredictionHeadSO3` predicting a 6D rotation (rot6d only,
        no log_dist), replacing the SO(2) head;
      - forward dynamics become a `SpatialTransformer` over the layer4 spatial map
        [N, 512, 7, 7], replacing the GAP-input TransformerEncoder (standard random init, no skip).

    ← Hydra _target_ of metaworld_bench/configs/encoder.yaml

    ⚠️  RNG-LOAD-BEARING CONSTRUCTION CASCADE: do not "optimise" the throwaway builds away.
        Each stage builds a module and the next stage discards it; every build consumes torch RNG,
        so the trained weights depend on this exact sequence:
            DynaMoSSL.__init__          builds forward_dynamics = TransformerEncoder   -> discarded below
            CANON_RotationSO2AnglePred  builds angle_head       = AnglePredictionHead  -> discarded below
            (inlined SO3 stage)         pops SO2 optim, builds  AnglePredictionHeadSO3
            (spatial stage)             pops fd  optim, builds  SpatialTransformer
        Removing a discarded build shifts the global RNG stream and silently yields a *different*
        encoder: this exact class of change caused a real reproduction discrepancy before.
    """

    def __init__(
        self,
        *args,
        feat_dim:        int = 512,
        rotation_dim:    int = 6,
        spatial_d_model: int = 128,
        **kwargs,
    ):
        # Builds the SO(2) angle head (RNG): intentionally discarded by the SO(3) stage below.
        super().__init__(*args, feat_dim=feat_dim, **kwargs)

        # ── SO(3)-6D angle head (inlined verbatim from the former SO3AnglePred stage) ──
        self.optimizers.pop()  # remove SO2 angle_head_optimizer
        new_head = AnglePredictionHeadSO3(
            in_channels=feat_dim, reduced_channels=128, spatial_size=7,
            output_dim=rotation_dim,
        )
        new_optimizer = new_head.configure_optimizers(
            lr=kwargs.get('lr', 1e-4),
            weight_decay=kwargs.get('weight_decay', 0.0),
            betas=kwargs.get('betas', (0.9, 0.999)),
        )
        self.angle_head, self.angle_head_optimizer = accelerator.prepare(new_head, new_optimizer)
        self.optimizers.append(self.angle_head_optimizer)

        # ── SpatialTransformer forward dynamics ──
        rot_lat_dim = kwargs.get('projection_dim')
        assert rot_lat_dim is not None, (
            "view_ssl config must specify projection_dim "
            "(set to ${view_projector.output_dim} in the Hydra config)"
        )

        old_idx = next((i for i, opt in enumerate(self.optimizers)
                        if opt is self.forward_dynamics_optimizer), None)
        if old_idx is not None:
            self.optimizers.pop(old_idx)

        n_layer = kwargs.get('n_layer', 4)
        n_head  = kwargs.get('n_head',  4)
        dropout = kwargs.get('dropout', 0.0)

        new_fd = SpatialTransformer(
            channels=feat_dim, d_model=spatial_d_model, rot_lat_dim=rot_lat_dim,
            n_layer=n_layer, n_head=n_head, dropout=dropout,
        )
        new_fd_optim = new_fd.configure_optimizers(
            weight_decay=kwargs.get('weight_decay', 0.0),
            lr=kwargs.get('lr', 1e-4),
            betas=kwargs.get('betas', (0.9, 0.999)),
        )
        self.forward_dynamics, self.forward_dynamics_optimizer = accelerator.prepare(new_fd, new_fd_optim)
        self.optimizers.append(self.forward_dynamics_optimizer)
