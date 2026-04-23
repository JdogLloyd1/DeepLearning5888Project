# Piczak CNN Long-Variant Training Failure — Root-Cause Analysis and Fix Plan

Companion document to `Code/Picszak Study Baseline.ipynb`. Captures the evidence, mechanism, and remediation plan for the observed training failure of the `long` segment variant, together with the fidelity audit against Piczak 2015.

---

## 1. Summary

When `Picszak Study Baseline.ipynb` is run end-to-end, the `short` variant trains cleanly and reaches ~36.6% *segment-level* test accuracy, while the `long` variant produces NaN loss from epoch 1 and locks at 2% accuracy (= 1/50, the uniform-guess floor) for all 150 epochs and all 5 folds.

The failure is not in the Piczak recipe itself — the paper reports LP = 64.5% clip-level with the same learning rate. The failure is in our reproduction: three components present in Piczak's pylearn2 pipeline but absent from our notebook (per-feature standardization of the spectrogram input, He/Kaiming weight initialization for ReLU, and max-column-norm constraint on FC weights) remove the stability margin that lets the long-variant `LR = 0.01` converge. Adding those three components — or, if fidelity is flexible, the modern equivalent of gradient clipping plus LR warmup — is expected to restore training.

Separately, a second reproduction gap has been identified: the notebook reports **segment-level** accuracy, while Piczak 2015 Figure 4 reports **clip-level** accuracy (majority/probability voting across segments of the same clip). Even a perfectly trained CNN reproduction is not comparable to Piczak's numbers without clip-level voting. This second gap has been closed: clip-level probability voting is now integrated end-to-end in the notebook (see §7 below).

---

## 2. Observed Failure Mode

### 2.1 Short variant — healthy convergence

From the original notebook output (fold 1):

```
Epoch 001/300 | train loss 4.4186, acc 0.0195 | test loss 3.9090, acc 0.0161
Epoch 010/300 | train loss 3.8594, acc 0.0397 | test loss 3.8724, acc 0.0464
Epoch 020/300 | train loss 3.6494, acc 0.0774 | test loss 3.7236, acc 0.0914
Epoch 030/300 | train loss 3.4927, acc 0.1015 | test loss 3.6095, acc 0.1206
Epoch 040/300 | train loss 3.3288, acc 0.1180 | test loss 3.4671, acc 0.0858
Epoch 050/300 | train loss 2.9966, acc 0.1730 | test loss 3.3262, acc 0.1300
```

Initial train loss is ~4.42, close to the ln(50) = 3.91 uniform-prediction baseline for a 50-class problem. Loss decreases monotonically, accuracy climbs steadily. Normal behavior.

### 2.2 Long variant — NaN from epoch 1

From the original notebook output (fold 1):

```
Epoch 001/150 | train loss nan, acc 0.0212 | test loss nan, acc 0.0200
Epoch 010/150 | train loss nan, acc 0.0200 | test loss nan, acc 0.0200
Epoch 020/150 | train loss nan, acc 0.0200 | test loss nan, acc 0.0200
  …
Epoch 150/150 | train loss nan, acc 0.0200 | test loss nan, acc 0.0200
```

Three observations fix the mechanism:

1. **Loss is NaN at epoch 1**, not after gradual growth → a single-step blow-up during the first handful of mini-batches, not slow divergence.
2. **Accuracy locks at exactly `0.0200 = 1/50`** for all 150 epochs → weights became NaN, the forward pass now produces NaN logits, and `argmax` over a NaN vector in PyTorch returns a deterministic index, giving constant predictions.
3. **Same code, same data pipeline, same model class — only config differs.** The short variant survives. The delta between the two configs is narrow (see §3.1).

### 2.3 What changes between short and long

```python
VARIANT_CONFIGS = {
    "short": {
        "segment_frames": 41,
        "segment_hop_frames": 20,   # ~50% overlap
        "num_epochs": 300,
        "learning_rate": 0.002,
    },
    "long": {
        "segment_frames": 101,
        "segment_hop_frames": 10,   # ~90% overlap
        "num_epochs": 150,
        "learning_rate": 0.01,
    },
}
```

Dominant delta: **learning rate is 5× higher for long (0.01 vs 0.002)**. Secondary amplifiers: 2.5× more time frames per segment (larger conv1 output spatial extent → larger FC1 pre-activation magnitudes); tighter overlap.

---

## 3. Fidelity Audit — Paper vs. Notebook

Cross-reference of Piczak 2015 §3.2 against `Picszak Study Baseline.ipynb`.

| Piczak §3.2 specification | Notebook implementation | Status |
|---|---|---|
| Resample 22050 Hz, waveform normalized (`librosa.util.normalize`) | ✅ | match |
| `n_fft = 1024`, `hop_length = 512`, `n_mels = 60` | ✅ | match |
| Short: 41 frames, 50% overlap | ✅ | match |
| Long: 101 frames, 90% overlap | ✅ | match |
| Silent-segment discard during segmenting | Threshold `mean(abs(log-mel)) > 1e-6` — effectively never rejects anything, because log-mel values are in dB range (20–80). Weak filter. | partial |
| 2-channel input (log-mel + delta) | ✅ | match |
| Conv1: 80 filters, 57×6 kernel, 1×1 stride | ✅ | match |
| MaxPool: 4×3 pool, 1×3 stride | ✅ | match |
| Conv2: 80 filters, 1×3 kernel, 1×1 stride | ✅ | match |
| MaxPool: 1×3 pool, 1×3 stride | ✅ | match |
| FC1, FC2: 5000 ReLUs each | ✅ | match |
| Dropout 0.5 **on FC layers and first conv layer** | Applied after every layer — including conv2 | partial |
| Nesterov momentum 0.9 | ✅ | match |
| L2 weight decay 0.001 | ✅ | match |
| Batch size 1000 | ✅ | match |
| LR: 0.002 short, 0.01 long | ✅ | match |
| Epochs: 300 short, 150 long | ✅ | match |
| **Data augmentation** (4× for ESC-50 via random time delays) | ❌ not implemented | **gap** |
| **Final prediction via majority / probability voting** | ✅ now implemented (§7) — was previously ❌ | **closed** |
| **Weight initialization** (Piczak paper is silent; pylearn2 configs use a sparse/He-like init tuned for ReLU) | `nn.init.xavier_uniform_` (Xavier assumes tanh, ~½ the variance appropriate for ReLU) | **mismatch** |
| **Input spectrogram standardization** (Piczak paper is silent; pylearn2 configs apply per-feature z-score) | ❌ not implemented — log-mel values in dB range (~−80 to +40) are fed raw | **gap** |
| **Max column-norm constraint on FC weights** (Piczak paper is silent; `max_col_norm: 1.0` is in his pylearn2 YAML) | ❌ not implemented | **gap** |

The last three rows are the critical ones. The paper *text* does not describe them, but Piczak's pylearn2 configs (linked in the paper's footnote 2: `github.com/karoldvl/paper-2015-esc-convnet`) apply all three. **Adding them increases fidelity to Piczak's full pipeline — it is not a deviation from his recipe.**

---

## 4. Mechanistic Chain That Produces the NaN

Given the fidelity audit above, the chain of events on the first mini-batch of long-variant training is:

1. Log-mel + delta features leave the feature extractor in the dB range. Channel 0 (log-mel) typically spans roughly −80 to +40 dB; channel 1 (delta) has smaller variance but nonzero mean. **No standardization** is applied before feeding the CNN.
2. Conv1 has a 57×6 kernel over a 60×101 input (long variant): each output element is a sum over 57 × 6 = 342 weighted inputs. Input values with std ~15–20 dB produce pre-activations with std roughly √342 × 15 ≈ 280 per weight-unit magnitude. With **Xavier-uniform initialization** (variance designed for tanh activations, ~½ the recommended variance for ReLU), some conv1 outputs land very far from zero after the first ReLU.
3. For the long variant specifically, the post-pool spatial extent is larger than for short (101 frames vs 41 → more time-axis positions remain after the 4×3 / 1×3 pooling stack), so the flattened vector feeding FC1 is larger. FC1's 5000-unit pre-activation magnitudes are correspondingly larger.
4. At the softmax stage, log-sum-exp of logits with very large magnitudes either overflows (producing `inf` before `exp`) or underflows (producing `0` before `log`), leading to `NaN` in the cross-entropy loss.
5. `loss.backward()` on a NaN loss produces NaN gradients everywhere. SGD step propagates NaN into the weights.
6. On the next forward pass, every layer's output is NaN. `softmax` of NaN logits is NaN. `argmax` of a NaN vector returns index 0 by implementation detail, giving the constant class-0 prediction and exactly 1/50 accuracy observed for the remainder of training.

The **short** variant survives this exact same path because:
- LR = 0.002 instead of 0.01 → any large-magnitude first-step update is 5× smaller.
- Smaller conv1 output spatial extent → smaller FC1 pre-activations → smaller logits → no softmax overflow.

In other words, the short variant is right at the edge of the stability margin; the long variant crosses it on the first step.

---

## 5. Fix Plan — Ordered by Fidelity Preservation

### Tier 1 — Pure fidelity corrections (closer match to Piczak's full pylearn2 pipeline)

#### 1A. Per-feature standardization of the spectrogram input ⭐

Compute per-channel mean and std on the training set of each fold, apply to both train and test. This is what `pylearn2.datasets.preprocessing.Standardize` does in Piczak's configs. Reduces first-layer pre-activation magnitudes from the raw dB range down to O(1), which alone collapses the gradient norm enough that LR = 0.01 becomes stable.

**Single highest-leverage fix. Expected to eliminate the NaN on its own.**

Implementation sketch:
```python
def compute_channel_stats(samples):
    """Per-channel (log-mel, delta) mean and std across a segment list."""
    arr = np.stack([s[0] for s in samples], axis=0)          # (N, 2, 60, T)
    mean = arr.mean(axis=(0, 2, 3), keepdims=True)[0]        # (2, 1, 1)
    std  = arr.std (axis=(0, 2, 3), keepdims=True)[0] + 1e-6
    return mean, std
```
Apply the train-fold stats to both train and test datasets inside `make_fold_dataloaders`.

#### 1B. He (Kaiming) initialization for ReLU layers

Swap the `_init_weights` call from `nn.init.xavier_uniform_` to `nn.init.kaiming_uniform_(weight, nonlinearity='relu')` (or `kaiming_normal_`). Xavier assumes symmetric activations; Piczak's pylearn2 `IRangeX` init is closer in spirit to Kaiming.

#### 1C. Max column-norm constraint on the two FC layers

After each optimizer step, clamp each column of `fc1.weight` and `fc2.weight` to L2-norm ≤ 1.0:

```python
with torch.no_grad():
    for layer in (model.fc1, model.fc2):
        norms = layer.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        scale = (norms.clamp(max=1.0) / norms)
        layer.weight.mul_(scale)
```

Piczak's YAML has `max_col_norm: 1.0` explicitly on both FC layers. Functionally prevents any single weight column from running away.

#### 1D. Apply dropout only where the paper specifies

Paper §3.2: "0.5 dropout probability for fully connected layers and the first convolutional layer." The current `PiczakCNN.forward` applies `self.dropout` after every ReLU, including conv2. Remove dropout from the conv2 output path.

#### 1E. Tighten the silence filter

Current:
```python
segment_energy = np.mean(np.abs(segment[0]))  # log-mel is in dB range
if segment_energy > 1e-6:                      # threshold is meaninglessly small
    segments.append(...)
```

`np.abs(segment[0])` for a log-mel segment ranges ~20–80 in dB, so the `1e-6` threshold passes every segment. Replace with either:

- an RMS-of-waveform threshold applied upstream (before spectrogram computation), or
- a log-mel threshold tuned for the dB scale, e.g. `segment[0].max() > -60`.

Not a divergence cause, but improves training data quality and brings us closer to Piczak's "discarding silent segments".

### Tier 2 — Faithful safety rails (not in the paper but neutral with respect to Piczak's converged solution)

#### 2F. Gradient clipping at max-norm 5.0

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

Not in Piczak's paper, but functionally equivalent in spirit to the max-col-norm constraint for catching explosive updates. Neutral with respect to converged solution; only activates when gradient norm exceeds 5.0.

#### 2G. NaN watchdog

After each batch:
```python
if not torch.isfinite(loss):
    print(f"[WARN] non-finite loss at epoch {epoch}, batch {batch_idx}; halving LR")
    for pg in optimizer.param_groups:
        pg['lr'] *= 0.5
    optimizer.zero_grad()
    continue
```

Turns the current silent failure (150 epochs of NaN masquerading as training) into an observable, actionable event.

### Tier 3 — Modern pragmatic fallbacks (explicit deviations from Piczak)

Apply only if Tier 1 + Tier 2 fail to restore long-variant convergence.

#### 3H. LR warmup

Linearly ramp LR from 1e-4 to 1e-2 over the first 5 epochs of the long variant. Solves first-step explosion. Not in Piczak.

#### 3I. Lower LR for long (e.g., 0.005)

Explicit deviation from Piczak's 0.01. Do not use unless all of 1A–2G have been applied and still fail.

#### 3J. Smaller batch size (128–256 instead of 1000)

Gives 4–8× more gradient updates per epoch with the same LR. Explicit deviation; Piczak specifies batch 1000. Last resort.

---

## 6. Minimum-Change Path (Recommended)

If the goal is "smallest diff that restores long-variant training without deviating from Piczak's spirit":

1. **1A** — per-channel standardization of features (≈10 lines in `make_fold_dataloaders` + `ESC50Dataset`)
2. **1B** — Kaiming init (change one line in `_init_weights`)
3. **2F** — gradient clipping (one line before `optimizer.step()`)
4. **2G** — NaN watchdog (3 lines after `loss.backward()`)

Expected outcome after applying 1A + 1B + 2F + 2G:

| Variant | Metric | Expected |
|---|---|---|
| short | segment-level | ≈ current 36.6% |
| short | clip-level (probability voting) | ≈ 55–62% (paper SM/SP range) |
| long | segment-level | ≈ 40–50% (vs current 2%) |
| long | clip-level (probability voting) | ≈ 60–65% (paper LM/LP range) |

If long still NaNs, escalate in order: **1C** (max-col-norm) → **3H** (LR warmup) → **3I** (lower LR) → **3J** (smaller batch).

---

## 7. Fix K — Clip-Level Probability Voting (implemented)

Independent of the NaN fix, the notebook previously reported only **segment-level** accuracy. Piczak 2015 Figure 4 reports **clip-level** accuracy using either majority voting or probability voting across all segments of a clip (§3.2: "Final predictions for a clip were generated using either a majority-voting scheme or by taking into account the probabilities predicted for each segment"). Without clip-level voting, the notebook's 36.6% number is not directly comparable to *any* bar in Piczak's Figure 4.

As of this revision, the notebook implements clip-level probability voting end-to-end:

- **Step 9**: `_aggregate_clip_probs` and `_clip_accuracy` helpers accumulate softmax outputs per filename during evaluation and resolve them to clip-level accuracy by argmax-of-averaged-probabilities. `run_one_epoch` now returns `(loss, seg_acc, clip_acc)` — `clip_acc` is `None` in training mode (segment order shuffled) and a float in eval mode.
- **Step 9**: `train_one_fold` tracks `test_clip_acc` per epoch, plus `best_clip_test_acc` / `best_clip_epoch`.
- **Step 10**: `save_training_curves` plots the clip-level curve alongside the existing segment-level curve.
- **Step 11**: Per-fold records include `final_clip_test_acc`, `best_clip_test_acc`, `best_clip_epoch`. Per-epoch history records include `test_clip_acc`.
- **Step 12**: Summary aggregates mean ± std across folds for both segment-level and clip-level (final and best). The Piczak reference numbers (B = 44%, SM/SP ≈ 58–62%, LP = 64.5%) are printed for quick comparison.

The voting logic is a direct port of the pattern already in `Code/LSTM/LSTM_baseline_claude.ipynb` (Step 10), ensuring consistent methodology across the CNN and LSTM baselines.

---

## 8. Status and Next Steps

| Item | Status |
|---|---|
| Scrub shared-team / AST / BEATs scaffolding from CNN notebook | ✅ done |
| Clip-level probability voting (Fix K) | ✅ done |
| Per-channel feature standardization (1A) | ⬜ pending |
| Kaiming init (1B) | ⬜ pending |
| Max-col-norm on FC layers (1C) | ⬜ pending |
| Dropout only on FC + conv1 (1D) | ⬜ pending |
| Tightened silence filter (1E) | ⬜ pending |
| Gradient clipping (2F) | ⬜ pending |
| NaN watchdog (2G) | ⬜ pending |
| Data augmentation (4× time-delay) | ⬜ pending (separate gap; not a NaN cause) |

After the next round of edits (recommended batch: 1A + 1B + 2F + 2G), re-run fold 1 of the long variant first to confirm the NaN is gone. If so, promote to full 5-fold cross-validation for both variants and compare the clip-level means to Piczak's Figure 4.

---

## 9. References

- Piczak, K. J. "Environmental Sound Classification with Convolutional Neural Networks." *2015 IEEE International Workshop on Machine Learning for Signal Processing*, §3.2. Included at `Piczak2015-ESC-ConvNet.pdf`.
- Piczak, K. J. Pylearn2 reproduction configs (referenced in the paper's footnote 2): `github.com/karoldvl/paper-2015-esc-convnet`.
- Current CNN implementation: `Code/Picszak Study Baseline.ipynb`.
- Reference clip-level voting pattern: `Code/LSTM/LSTM_baseline_claude.ipynb`, Step 10.
