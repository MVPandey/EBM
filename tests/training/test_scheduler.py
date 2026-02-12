import torch

from ebm.training.scheduler import create_lr_scheduler, get_ema_momentum
from ebm.utils.config import TrainingConfig


def _make_optimizer(lr: float = 3e-4) -> torch.optim.AdamW:
    model = torch.nn.Linear(10, 10)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def test_lr_starts_near_zero():
    cfg = TrainingConfig(warmup_steps=100)
    opt = _make_optimizer()
    sched = create_lr_scheduler(opt, cfg, total_steps=1000)
    assert sched.get_last_lr()[0] < 1e-6


def test_lr_reaches_peak_after_warmup():
    cfg = TrainingConfig(warmup_steps=100)
    opt = _make_optimizer(lr=3e-4)
    sched = create_lr_scheduler(opt, cfg, total_steps=1000)
    for _ in range(100):
        opt.step()
        sched.step()
    lr = sched.get_last_lr()[0]
    assert lr > 2e-4


def test_lr_decays_after_warmup():
    cfg = TrainingConfig(warmup_steps=10)
    opt = _make_optimizer(lr=3e-4)
    sched = create_lr_scheduler(opt, cfg, total_steps=100)
    for _ in range(10):
        opt.step()
        sched.step()
    lr_at_warmup = sched.get_last_lr()[0]
    for _ in range(50):
        opt.step()
        sched.step()
    lr_later = sched.get_last_lr()[0]
    assert lr_later < lr_at_warmup


def test_ema_momentum_start():
    cfg = TrainingConfig(ema_momentum_start=0.996, ema_momentum_end=1.0)
    m = get_ema_momentum(0, 1000, cfg)
    assert m == 0.996


def test_ema_momentum_end():
    cfg = TrainingConfig(ema_momentum_start=0.996, ema_momentum_end=1.0)
    m = get_ema_momentum(1000, 1000, cfg)
    assert abs(m - 1.0) < 1e-6


def test_ema_momentum_midpoint():
    cfg = TrainingConfig(ema_momentum_start=0.0, ema_momentum_end=1.0)
    m = get_ema_momentum(500, 1000, cfg)
    assert abs(m - 0.5) < 1e-6
