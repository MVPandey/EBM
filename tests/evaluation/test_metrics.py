import torch

from ebm.evaluation.metrics import EvalMetrics, _constraint_satisfaction, evaluate


def _valid_grid() -> torch.Tensor:
    """A valid 9x9 Sudoku solution."""
    rows = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 1, 5, 6, 4, 8, 9, 7],
        [5, 6, 4, 8, 9, 7, 2, 3, 1],
        [8, 9, 7, 2, 3, 1, 5, 6, 4],
        [3, 1, 2, 6, 4, 5, 9, 7, 8],
        [6, 4, 5, 9, 7, 8, 3, 1, 2],
        [9, 7, 8, 3, 1, 2, 6, 4, 5],
    ]
    return torch.tensor(rows, dtype=torch.long)


def _solution_onehot(grid: torch.Tensor) -> torch.Tensor:
    """Convert (9,9) integer grid to (9,9,9) one-hot."""
    onehot = torch.zeros(9, 9, 9)
    for r in range(9):
        for c in range(9):
            onehot[r, c, grid[r, c] - 1] = 1.0
    return onehot


def test_perfect_prediction():
    grid = _valid_grid()
    pred = grid.unsqueeze(0)
    solution = _solution_onehot(grid).unsqueeze(0)
    mask = torch.zeros(1, 9, 9)

    result = evaluate([pred], [solution], [mask])
    assert result.cell_accuracy == 1.0
    assert result.puzzle_accuracy == 1.0
    assert result.n_puzzles == 1


def test_all_wrong():
    grid = _valid_grid()
    pred = torch.ones(1, 9, 9, dtype=torch.long) * 9
    solution = _solution_onehot(grid).unsqueeze(0)
    mask = torch.zeros(1, 9, 9)

    result = evaluate([pred], [solution], [mask])
    assert result.cell_accuracy < 0.2
    assert result.puzzle_accuracy == 0.0


def test_clues_ignored_in_cell_accuracy():
    grid = _valid_grid()
    pred = torch.ones(1, 9, 9, dtype=torch.long) * 9
    solution = _solution_onehot(grid).unsqueeze(0)
    mask = torch.ones(1, 9, 9)

    result = evaluate([pred], [solution], [mask])
    assert result.cell_accuracy == 1.0


def test_constraint_satisfaction_valid():
    grid = _valid_grid().unsqueeze(0)
    sat, total = _constraint_satisfaction(grid)
    assert sat == 27
    assert total == 27


def test_constraint_satisfaction_invalid():
    grid = torch.ones(1, 9, 9, dtype=torch.long)
    sat, total = _constraint_satisfaction(grid)
    assert sat == 0
    assert total == 27


def test_evaluate_returns_eval_metrics():
    grid = _valid_grid()
    pred = grid.unsqueeze(0)
    solution = _solution_onehot(grid).unsqueeze(0)
    mask = torch.zeros(1, 9, 9)
    result = evaluate([pred], [solution], [mask])
    assert isinstance(result, EvalMetrics)


def test_multiple_batches():
    grid = _valid_grid()
    pred = grid.unsqueeze(0)
    solution = _solution_onehot(grid).unsqueeze(0)
    mask = torch.zeros(1, 9, 9)

    result = evaluate([pred, pred, pred], [solution, solution, solution], [mask, mask, mask])
    assert result.n_puzzles == 3
    assert result.puzzle_accuracy == 1.0
