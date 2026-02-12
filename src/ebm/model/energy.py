"""Energy function for the JEPA model."""

from torch import Tensor


def energy_fn(z_pred: Tensor, z_target: Tensor) -> Tensor:
    """
    Compute per-sample energy as squared L2 distance.

    Args:
        z_pred: (B, d_model) predicted representation.
        z_target: (B, d_model) target representation (should be detached).

    Returns:
        (B,) per-sample energy values.

    """
    return ((z_pred - z_target) ** 2).sum(dim=-1)
