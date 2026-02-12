import torch

from ebm.model.encoder import SudokuEncoder, SudokuPositionalEncoding
from ebm.utils.config import ArchitectureConfig

SMALL_CFG = ArchitectureConfig(d_model=32, n_layers=2, n_heads=4, d_ffn=64)


def test_positional_encoding_shape():
    pos_enc = SudokuPositionalEncoding(d_model=32)
    x = torch.randn(2, 81, 32)
    out = pos_enc(x)
    assert out.shape == (2, 81, 32)


def test_positional_encoding_adds_structure():
    """Different cells should get different positional encodings."""
    pos_enc = SudokuPositionalEncoding(d_model=32)
    zeros = torch.zeros(1, 81, 32)
    out = pos_enc(zeros)  # (1, 81, 32)
    # Cell 0 (row=0, col=0, box=0) and cell 80 (row=8, col=8, box=8) should differ
    assert not torch.allclose(out[0, 0], out[0, 80])


def test_same_row_share_row_embedding():
    """Cells in the same row should share the row embedding component."""
    pos_enc = SudokuPositionalEncoding(d_model=32)
    # Cells 0 and 1 are both in row 0
    row_ids = pos_enc.row_ids
    assert isinstance(row_ids, torch.Tensor)
    row_embed_0 = pos_enc.row_embed(row_ids[0])
    row_embed_1 = pos_enc.row_embed(row_ids[1])
    assert torch.allclose(row_embed_0, row_embed_1)


def test_encoder_output_shape():
    enc = SudokuEncoder(input_channels=10, cfg=SMALL_CFG)
    x = torch.randn(4, 10, 9, 9)
    out = enc(x)
    assert out.shape == (4, 32)


def test_encoder_solution_channels():
    """Target encoder uses 9 input channels."""
    enc = SudokuEncoder(input_channels=9, cfg=SMALL_CFG)
    x = torch.randn(4, 9, 9, 9)
    out = enc(x)
    assert out.shape == (4, 32)


def test_encoder_deterministic():
    enc = SudokuEncoder(input_channels=10, cfg=SMALL_CFG)
    enc.eval()
    x = torch.randn(2, 10, 9, 9)
    out1 = enc(x)
    out2 = enc(x)
    assert torch.allclose(out1, out2)


def test_encoder_batch_size_one():
    enc = SudokuEncoder(input_channels=10, cfg=SMALL_CFG)
    x = torch.randn(1, 10, 9, 9)
    out = enc(x)
    assert out.shape == (1, 32)


def test_encoder_gradients_flow():
    enc = SudokuEncoder(input_channels=10, cfg=SMALL_CFG)
    x = torch.randn(2, 10, 9, 9, requires_grad=True)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
