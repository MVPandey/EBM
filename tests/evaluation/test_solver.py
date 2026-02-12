import torch
from torch.utils.data import DataLoader, Dataset

from ebm.evaluation.solver import solve_batch, solve_dataset
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.utils.config import ArchitectureConfig, TrainingConfig

SMALL_ARCH = ArchitectureConfig(
    d_model=32,
    n_layers=1,
    n_heads=4,
    d_ffn=64,
    d_latent=16,
    predictor_hidden=32,
    decoder_layers=1,
    decoder_heads=2,
    decoder_d_cell=16,
)
SMALL_TRAIN = TrainingConfig(langevin_steps=3, n_chains=2)
SMALL_INFERENCE = InferenceConfig(n_steps=3, n_chains=2)


def _make_batch(b: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    puzzle = torch.zeros(b, 10, 9, 9)
    puzzle[:, 0] = 1.0
    solution = torch.zeros(b, 9, 9, 9)
    solution[:, :, :, 0] = 1.0
    mask = torch.zeros(b, 9, 9)
    return puzzle, solution, mask


def test_solve_batch_shape():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    puzzle, _, mask = _make_batch(4)
    result = solve_batch(model, puzzle, mask, SMALL_INFERENCE)
    assert result.shape == (4, 9, 9)
    assert (result >= 1).all()
    assert (result <= 9).all()


def test_solve_dataset_returns_lists():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    puzzle, solution, mask = _make_batch(4)

    class SimpleDS(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {'puzzle': puzzle[index], 'solution': solution[index], 'mask': mask[index]}

    loader = DataLoader(SimpleDS(), batch_size=2)
    preds, _solutions, _masks = solve_dataset(model, loader, SMALL_INFERENCE)
    assert len(preds) == 2
    assert preds[0].shape == (2, 9, 9)
