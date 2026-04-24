# Design of Experiments: `LSTMBaseline` on ESC-50 (mel + delta segments)

This document defines a **phased experiment plan** to map how **LSTMBaseline** performance depends on **architecture**, **regularization**, and **training** settings exposed in `LSTM baseline.ipynb` via `HYPERPARAMS`. Spectrogram settings are treated as **fixed** unless you add an explicit extension phase (noted at the end).

**Primary responses (record per run):**

| Response | Definition |
|----------|------------|
| `test_acc_final` | Segment-level test accuracy after the last epoch (as in the notebook) |
| `test_acc_best` | Best test accuracy across epochs (recommended to add when automating) |
| `train_test_gap` | `train_acc_final − test_acc_final` (overfitting indicator) |
| `params` | Total trainable parameters |
| `wall_time` | Total training time per fold (optional) |

**Evaluation protocol:** keep **ESC-50 5-fold cross-validation** as in the project. For screening, **one held-out fold** (e.g. fold 1) is acceptable; promote only promising configs to full 5-fold.

---

## 1. Factors and levels (maps to `HYPERPARAMS`)

| Factor | Key in `HYPERPARAMS` | Type | Levels to study | Rationale |
|--------|----------------------|------|-----------------|-----------|
| Segment policy | `segment_variants` → `short` / `long` | categorical | `short`, `long` | Changes sequence length \(T\) and segment count; interacts strongly with LSTM depth |
| LSTM width | `lstm.hidden_size` | numeric | 64, 128, 256, 512 | Capacity vs. overfitting and GPU memory |
| LSTM depth | `lstm.num_layers` | ordinal | 1, 2, 3 | Temporal abstraction; deeper stacks need more data and regularization |
| Direction | `lstm.bidirectional` | binary | `false`, `true` | Forward-only vs. full sequence context |
| Inter-layer LSTM dropout | `lstm.dropout` | numeric | 0.0, 0.2, 0.5 | Active only when `num_layers > 1` |
| MLP width | `classifier.hidden_dim` | numeric | 128, 256, 512 | Readout capacity after pooling over time (last step) |
| MLP dropout | `classifier.dropout` | numeric | 0.3, 0.5, 0.7 | Stabilizes the classifier head |
| Learning rate | per-variant `learning_rate` | numeric | log-spaced around defaults | SGD is sensitive; common grid: `5e-4`, `1e-3`, `2e-3`, `5e-3`, `1e-2` (clip to stable subset per variant) |
| Batch size | `training.batch_size` | numeric | 256, 1000, 2000 | Noise vs. speed; must fit memory |
| Weight decay | `training.weight_decay` | numeric | 0.0, `1e-3`, `5e-3` | L2 regularization on all weights |

**Fixed for Phases 1–2 (recommended):** `spectrogram.*`, `training.momentum` (0.9), `num_classes` (50), optimizer type (SGD + Nesterov as in notebook). **Epoch budget:** keep variant defaults (`short` → 300, `long` → 150) for comparability, or fix both to the same cap (e.g. 150) when comparing only architecture.

---

## 2. Phase 0 — Reference baseline

Single run with the **current default** `HYPERPARAMS` from the notebook (one fold for smoke, then 5-fold for reporting).

| Run ID | segment | hidden | layers | bi | lstm_do | clf_h | clf_do | lr | batch | wd |
|--------|---------|--------|--------|-----|---------|-------|--------|-----|-------|-----|
| R0 | `short` | 256 | 2 | yes | 0.2 | 512 | 0.5 | variant default | 1000 | 0.001 |

Duplicate R0 on `long` if you need a second reference (`R0-L`).

---

## 3. Phase 1 — Screening (architecture × direction × segment)

**Goal:** see which **capacity / depth / direction** region is worth refining. **Fix** regularization and training to defaults: `lstm.dropout=0.2`, `classifier`: hidden 512, dropout 0.5, `batch_size=1000`, `weight_decay=0.001`, learning rate = variant default.

| Run ID | segment | hidden | layers | bi |
|--------|---------|--------|--------|-----|
| S01 | short | 128 | 1 | no |
| S02 | short | 128 | 1 | yes |
| S03 | short | 256 | 2 | no |
| S04 | short | 256 | 2 | yes |
| S05 | short | 512 | 2 | yes |
| S06 | short | 256 | 3 | yes |
| S07 | long | 128 | 1 | yes |
| S08 | long | 256 | 2 | yes |
| S09 | long | 512 | 2 | yes |
| S10 | long | 256 | 3 | yes |

**Analysis:** plot `test_acc_best` (or final) vs. `(hidden, layers)` for each `segment`; note **train–test gap**. Pick 2–3 “corners”: small/fast (128,1), balanced (256,2+bi), large (512,2 or 256,3).

---

## 4. Phase 2 — Regularization and head (on best segment from Phase 1)

Assume Phase 1 picks **`short`** and architecture **`hidden=256`, `layers=2`, `bi=yes`** (adjust rows if your winner differs).

**Goal:** characterize **dropout** and **MLP width** without changing LSTM core.

| Run ID | lstm_do | clf_h | clf_do |
|--------|---------|-------|--------|
| T01 | 0.0 | 512 | 0.5 |
| T02 | 0.2 | 512 | 0.5 |
| T03 | 0.5 | 512 | 0.5 |
| T04 | 0.2 | 128 | 0.5 |
| T05 | 0.2 | 256 | 0.5 |
| T06 | 0.2 | 512 | 0.3 |
| T07 | 0.2 | 512 | 0.7 |

If **`layers=1`**, skip `lstm_do` sweeps (dropout has no effect); instead add a duplicate **Phase 2b** repeating T04–T07 only.

---

## 5. Phase 3 — Training dynamics (LR × batch × weight decay)

Take the **best config from Phase 2** and vary **optimization**. Use a **fixed epoch cap** (e.g. 150 for both segments) so runs are comparable.

| Run ID | lr | batch | wd |
|--------|-----|-------|-----|
| O01 | 1e-3 | 1000 | 1e-3 |
| O02 | 2e-3 | 1000 | 1e-3 |
| O03 | 5e-3 | 1000 | 1e-3 |
| O04 | 1e-2 | 1000 | 1e-3 |
| O05 | 2e-3 | 256 | 1e-3 |
| O06 | 2e-3 | 2000 | 1e-3 |
| O07 | 2e-3 | 1000 | 0 |
| O08 | 2e-3 | 1000 | 5e-3 |

**Caution:** very large `lr` with large `batch_size` can diverge; if loss spikes, drop `lr` or reduce `batch_size`.

---

## 6. Phase 4 — Confirmation (full 5-fold)

For **each** configuration you intend to report in the paper or compare to Piczak CNN:

1. Train **all 5 folds**.
2. Report **mean ± std** of clip-level or segment-level metric (be consistent with the notebook).
3. Keep **seed** fixed (`SEED = 99`) unless you add a variance study.

| Run ID | Description | Folds |
|--------|-------------|-------|
| C1 | Best overall from Phases 1–3 | 1–5 |
| C2 | Second-best (ablation) | 1–5 |

---

## 7. Optional extension — spectrogram factors (separate block)

If front-end capacity matters, run a **small** add-on grid **after** LSTM hyperparameters are settled (each cell = full Phase 4 on winner architecture only):

| Run ID | `n_mels` | `n_fft` | `hop_length` |
|--------|----------|---------|----------------|
| F01 | 60 | 1024 | 512 (baseline) |
| F02 | 80 | 1024 | 512 |
| F03 | 60 | 2048 | 512 |

Changing `hop_length` changes \(T\); treat as a **new input distribution** and re-tune `lr` lightly (Phase 3 mini-grid).

---

## 8. Master run log (template)

Copy this table into a spreadsheet while executing experiments.

| Run ID | Phase | fold(s) | segment | H | L | bi | lstm_do | clf_h | clf_do | lr | batch | wd | epochs | test_acc_best | test_acc_final | train_acc_final | gap | params | notes |
|--------|-------|---------|---------|---|---|-----|---------|-------|--------|-----|-------|-----|--------|---------------|----------------|-----------------|-----|--------|-------|
| R0 | 0 | 1 | short | 256 | 2 | Y | 0.2 | 512 | 0.5 | 0.002 | 1000 | 0.001 | 300 | | | | | | | |
| S01 | 1 | 1 | short | 128 | 1 | N | 0.2 | 512 | 0.5 | 0.002 | 1000 | 0.001 | 300 | | | | | | | |
| … | | | | | | | | | | | | | | | | | | | | |

**Legend:** H = `hidden_size`, L = `num_layers`, bi = bidirectional (Y/N).

---

## 9. Interpretation checklist

- **Main effects:** which single factor moved `test_acc_best` most between phases?
- **Interactions:** does **larger `hidden_size`** help only when `bi=yes` or `layers≥2`?
- **Overfitting:** if `gap` grows when capacity grows, increase **`classifier.dropout`** or **`weight_decay`** before adding width.
- **Segment policy:** `long` often yields more segments per clip; compare **per-epoch wall time** and **final metric**, not only peak accuracy.

This plan balances **coverage** (architecture range) with **run count** (screening on one fold, then full CV on finalists).
