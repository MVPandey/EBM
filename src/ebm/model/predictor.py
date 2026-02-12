"""Latent predictor MLP that maps (z_context, z) to z_pred."""

import torch
from torch import Tensor, nn

from ebm.utils.config import ArchitectureConfig


class LatentPredictor(nn.Module):
    """
    3-layer MLP with residual connections.

    Maps concat(z_context, z) to a prediction in the target encoder's
    representation space. Intentionally limited capacity so it cannot
    ignore the latent variable z.
    """

    def __init__(self, cfg: ArchitectureConfig | None = None) -> None:
        """
        Initialize predictor.

        Args:
            cfg: Architecture config. Uses defaults if None.

        """
        super().__init__()
        if not cfg:
            cfg = ArchitectureConfig()

        input_dim = cfg.d_model + cfg.d_latent
        hidden_dim = cfg.predictor_hidden
        output_dim = cfg.d_model

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_context: Tensor, z: Tensor) -> Tensor:
        """
        Predict target representation from context and latent.

        Args:
            z_context: (B, d_model) from context encoder.
            z: (B, d_latent) sampled latent variable.

        Returns:
            (B, d_model) predicted target representation.

        """
        x = self.input_proj(torch.cat([z_context, z], dim=-1))
        x = self.act(x)
        residual = x
        x = self.hidden(x)
        x = self.act(x)
        x = self.norm(x + residual)
        return self.output_proj(x)
