import torch

from ebm.model.predictor import LatentPredictor
from ebm.utils.config import ArchitectureConfig

SMALL_CFG = ArchitectureConfig(d_model=32, d_latent=16, predictor_hidden=64)


def test_output_shape():
    pred = LatentPredictor(SMALL_CFG)
    z_context = torch.randn(4, 32)
    z = torch.randn(4, 16)
    out = pred(z_context, z)
    assert out.shape == (4, 32)


def test_output_matches_d_model():
    cfg = ArchitectureConfig(d_model=64, d_latent=32, predictor_hidden=128)
    pred = LatentPredictor(cfg)
    out = pred(torch.randn(2, 64), torch.randn(2, 32))
    assert out.shape == (2, 64)


def test_gradients_flow_through_both_inputs():
    pred = LatentPredictor(SMALL_CFG)
    z_context = torch.randn(2, 32, requires_grad=True)
    z = torch.randn(2, 16, requires_grad=True)
    out = pred(z_context, z)
    out.sum().backward()
    assert z_context.grad is not None
    assert z.grad is not None


def test_different_z_gives_different_output():
    pred = LatentPredictor(SMALL_CFG)
    pred.eval()
    z_context = torch.randn(1, 32)
    z1 = torch.randn(1, 16)
    z2 = torch.randn(1, 16)
    out1 = pred(z_context, z1)
    out2 = pred(z_context, z2)
    assert not torch.allclose(out1, out2)
