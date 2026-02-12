import torch

from ebm.model.constraints import GROUP_INDICES, constraint_penalty


def test_penalty_shape():
    probs = torch.softmax(torch.randn(4, 9, 9, 9), dim=-1)
    penalty = constraint_penalty(probs)
    assert penalty.shape == (4,)


def test_penalty_nonnegative():
    probs = torch.softmax(torch.randn(8, 9, 9, 9), dim=-1)
    penalty = constraint_penalty(probs)
    assert (penalty >= 0).all()


def test_perfect_solution_low_penalty():
    """A valid Sudoku solution should have near-zero penalty."""
    # Build a valid Sudoku grid (one-hot)
    grid = torch.zeros(1, 9, 9, 9)
    base = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [3, 4, 5, 6, 7, 8, 0, 1, 2],
        [6, 7, 8, 0, 1, 2, 3, 4, 5],
        [1, 2, 0, 4, 5, 3, 7, 8, 6],
        [4, 5, 3, 7, 8, 6, 1, 2, 0],
        [7, 8, 6, 1, 2, 0, 4, 5, 3],
        [2, 0, 1, 5, 3, 4, 8, 6, 7],
        [5, 3, 4, 8, 6, 7, 2, 0, 1],
        [8, 6, 7, 2, 0, 1, 5, 3, 4],
    ]
    for r in range(9):
        for c in range(9):
            grid[0, r, c, base[r][c]] = 1.0

    penalty = constraint_penalty(grid)
    assert penalty.item() < 1e-6


def test_duplicate_digit_has_penalty():
    """Putting the same digit in every cell of a row violates constraints."""
    probs = torch.zeros(1, 9, 9, 9)
    # Row 0: all cells predict digit 0 with probability 1.0
    probs[0, 0, :, 0] = 1.0
    penalty = constraint_penalty(probs)
    assert penalty.item() > 0


def test_group_indices_shape():
    assert GROUP_INDICES.shape == (27, 9)


def test_group_indices_cover_all_cells():
    """Each of the 27 groups should reference 9 unique cells."""
    for g in range(27):
        cells = GROUP_INDICES[g].tolist()
        assert len(set(cells)) == 9
        assert all(0 <= c < 81 for c in cells)


def test_gradient_flows():
    probs = torch.softmax(torch.randn(2, 9, 9, 9, requires_grad=True), dim=-1)
    penalty = constraint_penalty(probs)
    penalty.sum().backward()
