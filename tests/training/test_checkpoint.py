import torch

from ebm.training.checkpoint import CheckpointManager, _CheckpointData


def _make_data(epoch: int = 0, val_energy: float = 1.0) -> _CheckpointData:
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return _CheckpointData(model=model, optimizer=optimizer, epoch=epoch, step=0, val_energy=val_energy)


def test_save_creates_file(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=3)
    path = mgr.save(_make_data(epoch=0, val_energy=0.5))
    assert path is not None
    assert path.exists()


def test_keeps_top_k(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=2)
    mgr.save(_make_data(epoch=0, val_energy=1.0))
    mgr.save(_make_data(epoch=1, val_energy=0.8))
    mgr.save(_make_data(epoch=2, val_energy=0.5))

    checkpoint_files = list(tmp_path.glob('*.pt'))
    assert len(checkpoint_files) == 2


def test_rejects_worse_than_top_k(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=2)
    mgr.save(_make_data(epoch=0, val_energy=0.5))
    mgr.save(_make_data(epoch=1, val_energy=0.3))
    path = mgr.save(_make_data(epoch=2, val_energy=0.9))
    assert path is None


def test_load_restores_model(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=3)
    data = _make_data(epoch=5, val_energy=0.5)
    original_state = {k: v.clone() for k, v in data.model.state_dict().items()}
    path = mgr.save(data)
    assert path is not None

    new_model = torch.nn.Linear(4, 4)
    checkpoint = CheckpointManager.load(path, new_model)
    assert checkpoint['epoch'] == 5
    for k, v in new_model.state_dict().items():
        assert torch.allclose(v, original_state[k])
