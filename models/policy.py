"""
DiT + Flow Matching policy for robot learning.

DiTFlowPolicy  : base diffusion transformer (flat obs tokens). Used for LIBERO Goal CANON.
DiTFlowSpatial : spatial-map obs tokens + separate state tokens. Used for MetaWorld CANON.

Architecture: Diffusion Transformer (DiT) as the vector-field network,
trained with the Optimal-Transport Conditional Flow Matching (OT-CFM) objective.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal time-step embedding. t: [N] in [0,1], d: embedding dim (even)."""
    assert d % 2 == 0
    half = d // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning (Peebles & Xie, 2023)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        )
        x_n = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x   = x + gate1.unsqueeze(1) * self.drop(self.attn(x_n, x_n, x_n)[0])
        x_n = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x   = x + gate2.unsqueeze(1) * self.drop(self.mlp(x_n))
        return x


class DiTFlowPolicy(nn.Module):
    """Diffusion Transformer with flow-matching objective.

    forward(obs, goal, act):
        Training  (act is not None): returns (None, loss, loss_dict)
        Inference (act is None)    : returns (pred_act, None, {})
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        action_chunk: int,
        obs_window: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        dropout: float = 0.0,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.obs_dim      = obs_dim
        self.goal_dim     = goal_dim
        self.act_dim      = act_dim
        self.action_chunk = action_chunk
        self.obs_window   = obs_window
        self.d_model      = d_model
        self.num_inference_steps = num_inference_steps

        self.obs_proj  = nn.Linear(obs_dim,  d_model)
        self.goal_proj = nn.Linear(goal_dim, d_model) if goal_dim > 0 else None
        self.act_proj  = nn.Linear(act_dim,  d_model)
        self.obs_pos_emb = nn.Parameter(torch.zeros(1, obs_window, d_model))
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.blocks   = nn.ModuleList([DiTBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.act_out  = nn.Linear(d_model, act_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.act_out.weight)
        nn.init.zeros_(self.act_out.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.trunc_normal_(self.obs_pos_emb, std=0.02)

    def _predict_vector_field(self, obs, goal, act, t):
        obs_tok = self.obs_proj(obs) + self.obs_pos_emb
        act_tok = self.act_proj(act)
        t_emb   = self.time_mlp(sinusoidal_embedding(t, self.d_model))
        if self.goal_proj is not None:
            goal_tok = self.goal_proj(goal).unsqueeze(1)
            tokens = torch.cat([obs_tok, goal_tok, act_tok], dim=1)
        else:
            tokens = torch.cat([obs_tok, act_tok], dim=1)
        for block in self.blocks:
            tokens = block(tokens, t_emb)
        tokens = self.out_norm(tokens)
        return self.act_out(tokens[:, -self.action_chunk:, :])

    def compute_loss(self, obs, goal, act):
        N, device, dtype = obs.shape[0], obs.device, obs.dtype
        t    = torch.rand(N, device=device, dtype=dtype)
        x0   = torch.randn_like(act)
        t_bc = t.view(N, 1, 1)
        xt   = (1.0 - t_bc) * x0 + t_bc * act
        u    = act - x0
        v    = self._predict_vector_field(obs, goal, xt, t)
        loss = F.mse_loss(v, u)
        return loss, {"flow_matching_loss": loss.item()}

    @torch.no_grad()
    def select_action(self, obs, goal):
        N, device, dtype = obs.shape[0], obs.device, obs.dtype
        x  = torch.randn(N, self.action_chunk, self.act_dim, device=device, dtype=dtype)
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = torch.full((N,), step * dt, device=device, dtype=dtype)
            x = x + dt * self._predict_vector_field(obs, goal, x, t)
        return x

    def forward(self, obs, goal, act):
        if act is not None:
            loss, loss_dict = self.compute_loss(obs, goal, act)
            return None, loss, loss_dict
        return self.select_action(obs, goal), None, {}

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=learning_rate, betas=tuple(betas),
        )


class DiTFlowSpatial(DiTFlowPolicy):
    """DiTFlow with spatial feature-map obs tokens + separate state tokens.

    Token sequence: [obs_spatial (T_obs * n_spatial) | state (T_obs) | goal? | act (T_act)]

    Used by MetaWorld CANON: resnet18_spatial emits [N, V, 512, 7, 7];
    policy receives 49 spatial tokens per timestep instead of one GAP token.
    """

    def __init__(
        self,
        spatial_channel_dim: int,
        n_spatial_tokens: int,
        state_dim: int,
        goal_dim: int,
        act_dim: int,
        action_chunk: int,
        obs_window: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.0,
        num_inference_steps: int = 10,
        obs_dim: Optional[int] = None,  # absorbed from Hydra config; unused
    ):
        super().__init__(
            obs_dim=spatial_channel_dim,
            goal_dim=goal_dim,
            act_dim=act_dim,
            action_chunk=action_chunk,
            obs_window=obs_window,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            num_inference_steps=num_inference_steps,
        )
        self.spatial_channel_dim = spatial_channel_dim
        self.n_spatial_tokens    = n_spatial_tokens
        self.state_dim           = state_dim

        # 2D positional embedding [1, T_obs, n_spatial, d_model]
        self.obs_pos_emb = nn.Parameter(torch.zeros(1, obs_window, n_spatial_tokens, d_model))
        nn.init.trunc_normal_(self.obs_pos_emb, std=0.02)

        # Separate state token projection (one token per timestep)
        self.state_proj    = nn.Linear(state_dim, d_model)
        self.state_pos_emb = nn.Parameter(torch.zeros(1, obs_window, d_model))
        nn.init.trunc_normal_(self.state_pos_emb, std=0.02)
        nn.init.xavier_uniform_(self.state_proj.weight)
        nn.init.zeros_(self.state_proj.bias)

    def _predict_vector_field(self, obs, state, goal, act, t):
        # obs: [N, T, n_spatial, spatial_channel_dim]
        N, T, S, C = obs.shape
        obs_tok = self.obs_proj(obs) + self.obs_pos_emb   # [N, T, S, d]
        obs_tok = obs_tok.reshape(N, T * S, -1)           # [N, T*S, d]
        state_tok = self.state_proj(state) + self.state_pos_emb  # [N, T, d]
        act_tok   = self.act_proj(act)
        t_emb     = self.time_mlp(sinusoidal_embedding(t, self.d_model))
        if self.goal_proj is not None:
            goal_tok = self.goal_proj(goal).unsqueeze(1)
            tokens = torch.cat([obs_tok, state_tok, goal_tok, act_tok], dim=1)
        else:
            tokens = torch.cat([obs_tok, state_tok, act_tok], dim=1)
        for block in self.blocks:
            tokens = block(tokens, t_emb)
        tokens = self.out_norm(tokens)
        return self.act_out(tokens[:, -self.action_chunk:, :])

    def compute_loss(self, obs, state, goal, act):
        N, device, dtype = obs.shape[0], obs.device, obs.dtype
        t    = torch.rand(N, device=device, dtype=dtype)
        x0   = torch.randn_like(act)
        t_bc = t.view(N, 1, 1)
        xt   = (1.0 - t_bc) * x0 + t_bc * act
        u    = act - x0
        v    = self._predict_vector_field(obs, state, goal, xt, t)
        loss = F.mse_loss(v, u)
        return loss, {"flow_matching_loss": loss.item()}

    @torch.no_grad()
    def select_action(self, obs, state, goal):
        N, device, dtype = obs.shape[0], obs.device, obs.dtype
        x  = torch.randn(N, self.action_chunk, self.act_dim, device=device, dtype=dtype)
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = torch.full((N,), step * dt, device=device, dtype=dtype)
            x = x + dt * self._predict_vector_field(obs, state, goal, x, t)
        return x

    def forward(self, obs, state, goal, act):
        if act is not None:
            loss, loss_dict = self.compute_loss(obs, state, goal, act)
            return None, loss, loss_dict
        return self.select_action(obs, state, goal), None, {}
