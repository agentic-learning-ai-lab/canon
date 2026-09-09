"""
MLP projector used as view_projector in both MetaWorld and LIBERO Goal CANON encoders.
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[256, 64], output_dim=2):
        super().__init__()
        layers = []
        if len(hidden_sizes) == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_sizes[0]))
            layers.append(nn.LayerNorm(hidden_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(1, len(hidden_sizes)):
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                layers.append(nn.LayerNorm(hidden_sizes[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self, weight_decay, lr, betas):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )


