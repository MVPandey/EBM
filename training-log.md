# Training Log

Documenting runs, bugs, and learnings as we iterate toward replicating [Kona 1.0's 96.2% accuracy](https://logicalintelligence.com/blog/energy-based-model-sudoku-demo).

---

## Run 1: MVP Validation (100K samples, 20 epochs)

**Date:** Feb 11, 2026
**Config:** 100K samples, batch_size=512, 20 epochs, lr=3e-4
**Hardware:** RTX 5090

### Bugs Found During This Run

Five critical bugs were found and fixed before meaningful training could begin.

**1. Solution tensor encoding mismatch**
Solution was stored as `(digit, row, col)` (channel-first), but `argmax(dim=-1)` took argmax along the column axis. This meant the model was training on scrambled targets. Fixed by switching to `(row, col, digit)` and adding `permute(0, 3, 1, 2)` before the target encoder.

**2. Decode loss included given cells**
Cross-entropy was computed over all 81 cells, but given cells had 1e6 logits from hard enforcement — near-zero loss. With ~40% of cells given, the average loss dropped from 2.2 to 1.25 purely from the easy population. Empty cells never learned. Fixed by masking to compute loss on empty cells only.

**3. VICReg applied to wrong tensor**
VICReg was applied to `z_pred` (predictor output), but this didn't prevent the context/target encoders from collapsing. The target encoder (EMA of context encoder) collapsed to a point, making energy trivially zero. Fixed by applying VICReg to `z_context` (context encoder output), which propagates to the target encoder via EMA.

**4. LR warmup exceeded total steps**
With 100K samples and 5 epochs: only 880 total steps, but warmup was set to 2000. LR never reached its peak, and `CosineAnnealingLR(T_max=-1120)` got a negative period. Fixed by capping warmup to `total_steps // 5`.

**5. Checkpoint metric was inverted**
Checkpoints saved models with lowest energy. But low energy = collapsed representations (z_pred trivially matching z_target when both map to the same point). Models with best cell accuracy had the highest energy and were never saved. Fixed by checkpointing on cell accuracy (higher = better).

### Results After Fixes

| Metric | Epoch 0 | Epoch 20 |
|--------|---------|----------|
| Cell accuracy | 11.1% (random) | 13.9% |
| z_variance | > 0 | > 0 (no collapse) |
| Energy loss | decreasing | stable gradient signal |
| Decode loss | decreasing | real signal from empty cells |

Key finding: 25% relative improvement over random, representations not collapsing. Limited by data (100K) and epochs (20).

### Hyperparameters That Worked

- `decode_loss_weight=1.0` (was 0.1 — too low, decoder had no influence)
- `vicreg_var_weight=1.0` (was 25.0 — dominated total loss)
- `vicreg_cov_weight=0.01` (was 1.0 — too aggressive)
- VICReg target: `z_context` (not `z_pred`)
- Checkpoint metric: `cell_accuracy` (not energy)

---

## Run 2: Full Scale (9M samples, 20 epochs)

**Date:** Feb 11, 2026
**Config:** 9M samples (8M train / 500K val), batch_size=512, 20 epochs, lr=3e-4
**Hardware:** RTX 5090
**Estimated time:** ~8 hours (~24 min/epoch)