import torch

from ebm.model.decoder import SudokuDecoder
from ebm.utils.config import ArchitectureConfig

SMALL_CFG = ArchitectureConfig(d_model=32, d_latent=16, decoder_layers=1, decoder_heads=2, decoder_d_cell=16)


def test_output_shape():
    dec = SudokuDecoder(SMALL_CFG)
    z_context = torch.randn(4, 32)
    z = torch.randn(4, 16)
    puzzle = torch.zeros(4, 10, 9, 9)
    mask = torch.zeros(4, 9, 9)
    logits = dec(z_context, z, puzzle, mask)
    assert logits.shape == (4, 9, 9, 9)


def test_clue_enforcement():
    """Given clues should produce extremely high logits at the correct digit."""
    dec = SudokuDecoder(SMALL_CFG)
    z_context = torch.randn(1, 32)
    z = torch.randn(1, 16)

    puzzle = torch.zeros(1, 10, 9, 9)
    mask = torch.zeros(1, 9, 9)

    # Set cell (0,0) as a clue with digit 5 (channel 5 in puzzle)
    puzzle[0, 5, 0, 0] = 1.0  # digit 5
    mask[0, 0, 0] = 1.0  # mark as given

    logits = dec(z_context, z, puzzle, mask)
    # Digit 5 is at index 4 (0-indexed: digits 1-9 map to channels 0-8)
    assert logits[0, 0, 0].argmax() == 4


def test_empty_cells_not_clamped():
    """Empty cells should have normal (non-huge) logit values."""
    dec = SudokuDecoder(SMALL_CFG)
    z_context = torch.randn(1, 32)
    z = torch.randn(1, 16)
    puzzle = torch.zeros(1, 10, 9, 9)
    puzzle[:, 0] = 1.0  # all cells empty
    mask = torch.zeros(1, 9, 9)

    logits = dec(z_context, z, puzzle, mask)
    assert logits.abs().max() < 1e4


def test_gradients_flow():
    dec = SudokuDecoder(SMALL_CFG)
    z_context = torch.randn(2, 32, requires_grad=True)
    z = torch.randn(2, 16, requires_grad=True)
    puzzle = torch.zeros(2, 10, 9, 9)
    mask = torch.zeros(2, 9, 9)

    logits = dec(z_context, z, puzzle, mask)
    logits.sum().backward()
    assert z_context.grad is not None
    assert z.grad is not None
