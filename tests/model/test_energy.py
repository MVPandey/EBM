import torch

from ebm.model.energy import energy_fn


def test_energy_shape():
    z_pred = torch.randn(4, 32)
    z_target = torch.randn(4, 32)
    e = energy_fn(z_pred, z_target)
    assert e.shape == (4,)


def test_energy_zero_for_identical():
    z = torch.randn(3, 64)
    e = energy_fn(z, z)
    assert torch.allclose(e, torch.zeros(3))


def test_energy_positive():
    z_pred = torch.randn(5, 32)
    z_target = torch.randn(5, 32)
    e = energy_fn(z_pred, z_target)
    assert (e >= 0).all()


def test_energy_increases_with_distance():
    z_target = torch.zeros(1, 32)
    z_near = torch.ones(1, 32) * 0.1
    z_far = torch.ones(1, 32) * 10.0
    e_near = energy_fn(z_near, z_target)
    e_far = energy_fn(z_far, z_target)
    assert e_far > e_near


def test_energy_gradient_flows():
    z_pred = torch.randn(2, 32, requires_grad=True)
    z_target = torch.randn(2, 32)
    e = energy_fn(z_pred, z_target)
    e.sum().backward()
    assert z_pred.grad is not None
