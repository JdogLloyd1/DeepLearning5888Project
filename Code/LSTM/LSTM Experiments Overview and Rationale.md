# LSTM Model Experiments — ESC-50 Environmental Sound Classification

## Overview

This document describes the LSTM baseline architecture, the rationale behind each design decision, the factors and levels under investigation, and the 9-run Option B experiment plan with guidance on what to look for in the results.

All experiments use the shared data pipeline (log-mel spectrogram + delta, 5-fold cross-validation). The Stage 1 screening grid runs on the `short` segment variant (41 frames, ~0.95 s, 50% overlap) by default, with the `long` variant (101 frames, ~2.3 s, 90% overlap) available via the opt-in Stage 2 cells. Results are characterized relative to the Piczak 2015 reference points on ESC-50 (all clip-level):

| Reference | Clip-level Accuracy |
|---|---|
| Random-forest baseline (**B**) | 0.4400 |
| CNN short variant (SM / SP) | ~0.58–0.62 |
| CNN long variant (LM / **LP**) | 0.6450 |
| Human accuracy (for context) | 0.8130 |

Our own CNN reproduction (`Picszak Study Baseline.ipynb`) lands at **~36.6% segment-level** on the short variant, which is not directly comparable to the paper's clip-level numbers — see **Evaluation Metric** below.

---

## Notebook Structure

The LSTM work is split across two notebooks in `Code/LSTM/`:

| Notebook | Purpose | When to run |
|---|---|---|
| `LSTM baseline.ipynb` | Smoke-test the full pipeline on the **Run 1 baseline architecture** (hidden=128, 2L, bi, final) for **both** the `short` and `long` variants on fold 1. | After any pipeline change, to confirm no NaN and clip-level > 0.44 on both variants before committing compute to the full grid. |
| `LSTM experiments.ipynb` | Full 9-architecture Stage 1 screening grid, 5-fold promotion of the winner, and the optional Stage 2 long-variant cells. | After `LSTM baseline.ipynb` passes. |

**Run-name convention (both notebooks).** The segment variant is **always** suffixed onto the run name so short and long artifacts never collide and the variant is unambiguous in every downstream CSV / plot / folder. Examples:

- Stage 1 grid row: `LSTM_Run1_Baseline_short`, `LSTM_Run3_CapacityUp_long`
- 5-fold promotion: `LSTM_Run1_Baseline_short_fold3`
- Phase 2 smoke test: `LSTM_Run1_Baseline_long_extension`
- Stage 2a long screening: `LSTM_Run3_CapacityUp_long`
- Stage 2b long 5-fold: `LSTM_Run3_CapacityUp_long_fold3`

Every summary DataFrame also carries explicit `variant` and `base_model_name` columns.

---

## Evaluation Metric — Segment vs. Clip

Each audio clip is split into multiple overlapping spectrogram segments and the model scores each segment independently. There are two ways to report accuracy:

- **Segment-level accuracy.** Fraction of segments classified correctly. This is what the per-batch training loop tracks and what our CNN reproduction reports as "~36.6%". It treats every segment as an independent sample.
- **Clip-level accuracy (probability voting).** Softmax probabilities are averaged across all segments from the same clip and the argmax of the averaged vector becomes the clip-level prediction. This is what Piczak 2015 reports and is the **paper-comparable metric**.

The LSTM training loop tracks both at every epoch (`test_acc` = segment, `clip_test_acc` = clip). Segment-level is useful for diagnosing whether the model is learning anything at all; clip-level is what matters for the final comparison with the literature. All "what to look for" accuracy ranges below refer to **clip-level** unless stated otherwise.

---

## Baseline Architecture

### Input Representation

Each audio clip produces a feature tensor of shape `(2, 60, T)` — two channels (log-mel spectrogram and its delta), 60 mel bins, and T time frames. The LSTM receives this as a sequence of T timesteps, where each timestep is a flattened vector of `2 × 60 = 120` features. This treats the spectrogram as a time series, which is the natural inductive bias of a recurrent model.

### Architecture Choices

**2-layer LSTM.** A single layer captures low-level temporal patterns but lacks the capacity to compose them into higher-level representations. Three or more layers risk overfitting on a dataset as small as ESC-50 (2000 clips total). Two layers is the standard starting point for sequence classification tasks of this scale.

**Hidden size 128.** Large enough to represent meaningful temporal dynamics across 50 classes, small enough to avoid severe overfitting. The Piczak CNN uses 5000-unit fully connected layers and still overfits heavily — a more conservative LSTM hidden size is appropriate given the sequential inductive bias does some of the representational work.

**Bidirectional.** Environmental sound clips do not have a causal structure the way speech does — a dog bark at second 2 is equally informative whether you're reading the sequence forward or backward. Bidirectionality lets the model incorporate future context at every timestep, which consistently helps on audio classification tasks. The cost is doubling the hidden state dimensionality fed to the classifier.

**Final hidden state pooling.** The default summary of the sequence uses the last-layer hidden states from the forward and backward passes concatenated, giving a `hidden_size × 2 = 256`-dimensional vector. This is the most common approach and forms the baseline from which mean pooling is tested.

**Forget gate bias initialized to 1.0.** A well-established LSTM training trick. Setting the forget gate bias high at initialization encourages the model to remember information across long spans early in training, before the gates have learned useful patterns. This reduces vanishing gradient problems in early epochs.

**Gradient clipping (max norm = 1.0).** LSTMs on spectrograms can produce unstable gradients, particularly with higher learning rates or deeper networks. The NaN losses observed in the Piczak "long" variant with SGD at LR=0.01 are an example of this failure mode. Clipping prevents exploding gradients without requiring a very low learning rate.

**Adam optimizer, LR = 1e-3, cosine LR decay.** Adam is more robust than SGD for LSTMs because its adaptive per-parameter learning rates handle the varying gradient magnitudes across gates and layers. Cosine decay gradually reduces the LR over training, which tends to improve final accuracy compared to a flat LR.

### Baseline Config Summary

| Parameter | Value |
|---|---|
| `hidden_size` | 128 |
| `num_layers` | 2 |
| `bidirectional` | True |
| `pooling` | final |
| `dropout` | 0.3 |
| `learning_rate` | 1e-3 |
| `num_epochs` | 60 |
| `batch_size` | 64 |

---

## Factors and Levels

Four architectural factors are varied. All other parameters are held constant at baseline values.

| Factor | Baseline | Levels Tested |
|---|---|---|
| `hidden_size` | 128 | 64, 128, 256 |
| `num_layers` | 2 | 1, 2, 3 |
| `bidirectional` | True | True, False |
| `pooling` | final | final, mean |

### Why These Four Factors

**Hidden size** directly controls model capacity. On a 50-class problem with limited data, the relationship between capacity and accuracy is non-monotonic — too small and the model underfits, too large and it overfits. The three levels bracket a reasonable range.

**Number of layers** controls representational depth. One layer is often sufficient for simple temporal patterns; two adds compositionality; three risks over-parameterization on ESC-50's small training set.

**Bidirectional** is a binary architectural choice with a clear hypothesis (environmental sounds benefit from future context) but a real cost (doubles the parameter count in the output projection). It's worth testing explicitly rather than assuming.

**Pooling strategy** changes how the sequence is summarized. Final-state pooling relies on the LSTM to carry all relevant information forward to the last timestep, which may fail for sounds that occur mid-clip. Mean pooling averages contributions from every timestep, which is more robust to temporal position but may dilute salient events with silence.

### Why Dropout and Learning Rate Are Held Fixed

Dropout and learning rate are **training hyperparameters** — they affect how well a given architecture is optimized, but they do not change what the architecture fundamentally is. The purpose of this experiment set is to characterize *architectural* differences between LSTM variants, so varying them would introduce a confound: if Run 3 (hidden=256) underperforms Run 1, it would be impossible to tell whether larger capacity genuinely doesn't help, or whether 256-unit networks simply need a lower learning rate and more dropout to train well.

Varying these properly would require tuning them *per architecture* — a 256-unit bidirectional LSTM has different optimization needs than a 64-unit unidirectional one. Doing that rigorously would require a grid search nested inside each of the 9 runs, which is outside scope. The right approach is to fix reasonable values known to work for LSTMs on small classification tasks (LR=1e-3, dropout=0.3), hold them constant across all runs, and note in the write-up that these were not tuned per architecture. This is a standard and legitimate design choice for architecture comparison studies.

**Debugging note.** If any run produces NaN loss or completely fails to learn, the first corrective step is to halve the learning rate (to 5e-4) before concluding the architecture is the problem. Unstable loss is almost always an optimization issue, not an architectural one, and is not grounds for discarding a run from the comparison.

---

## Experiment Plan — Option B (9 Runs)

Runs 1–7 are One-Factor-At-a-Time (OFAT) to isolate main effects. Runs 8–9 probe two specific interactions identified by domain reasoning.

### Staged Execution

The 9-run matrix is executed in stages to balance coverage with compute budget. Every stage below lives in **`LSTM experiments.ipynb`** and each cell can be skipped independently by not running it. Before these stages, `LSTM baseline.ipynb` should have been run once to confirm the pipeline is healthy on both `short` and `long`.

| Stage | What | Cost (short-equivalent runs) | Notebook Cell (in `LSTM experiments.ipynb`) |
|---|---|---|---|
| **1 — Short screening** | 9 architectures × `short` × fold 1 | ~9 | `Step 13` |
| **1b — Short 5-fold promotion** | Short winner + Run 1 × 5 folds | ~10 | `Step 14` |
| **Phase 2 — Long smoke test** | Short winner × `long` × fold 1 | ~3 | `Step 15` |
| **2a — Long screening (optional)** | 9 architectures × `long` × fold 1 | ~30 | `Step 16` |
| **2b — Long 5-fold promotion (optional)** | Long winner × `long` × 5 folds | ~16 | `Step 17` |

**Decision rule between Phase 2 and Stage 2.** If the Phase 2 smoke test shows long clip-level accuracy exceeds short by ≥ ~3 points absolute, proceed to Stage 2a/2b to get the paper-comparable LM/LP number. If not, stop and report the short variant as the final answer — the paper notes long gives only "slight improvements" over short and does not always justify the ~3.3× compute cost.

**Enabling a full 9 × 2 sweep up front.** If compute is not a constraint, set `VARIANT_LIST = ['short', 'long']` in the Stage 1 cell and skip Phase 2 / Stage 2a entirely — Stage 1b will then promote whichever overall winner emerges, and Stage 2b can separately promote the long winner if desired.

---

### Run 1 — Baseline

| hidden | layers | bidir | pooling |
|---|---|---|---|
| 128 | 2 | True | final |

**Purpose.** Establishes the reference point for all comparisons. Every other run should be interpreted relative to this one.

**What to look for.** Smooth loss convergence without NaN. Clip-level test accuracy should land in the **~40–55% range** based on LSTM results on ESC-50 in the literature; much lower suggests an implementation problem, much higher would be a surprisingly strong result worth investigating. Segment-level accuracy will typically be 5–15 points lower than clip-level. Useful checkpoints: clip-level ≥ 44% beats Piczak's random-forest baseline; clip-level ≈ 58–62% matches the Piczak CNN on SM/SP. Look at whether training accuracy pulls far ahead of test accuracy by epoch 60 — if the gap is large, the model is overfitting and subsequent capacity increases (Run 3, Run 5) are unlikely to help.

---

### Run 2 — Capacity Down

| hidden | layers | bidir | pooling |
|---|---|---|---|
| 64 | 2 | True | final |

**Purpose.** Tests whether the baseline is already over-parameterized for this dataset.

**What to look for.** If test accuracy is close to or matches Run 1, the baseline has more capacity than the data can use and smaller models are preferable (faster training, less overfitting). If test accuracy drops noticeably, capacity is a binding constraint. Also compare the training/test accuracy gap — a smaller gap here than Run 1 would confirm overfitting is a concern at 128 units.

---

### Run 3 — Capacity Up

| hidden | layers | bidir | pooling |
|---|---|---|---|
| 256 | 2 | True | final |

**Purpose.** Tests whether more capacity improves generalization or just increases overfitting.

**What to look for.** The most likely outcome on a small dataset is that test accuracy does not improve over Run 1 but training accuracy climbs higher, indicating overfitting. If test accuracy does improve, it suggests the task has sufficient complexity to reward larger models. Compare the epoch at which test accuracy peaks — if it peaks earlier than Run 1 and then declines, early stopping would help and should be noted as a recommendation.

---

### Run 4 — Depth Down

| hidden | layers | bidir | pooling |
|---|---|---|---|
| 128 | 1 | True | final |

**Purpose.** Tests whether the second LSTM layer is contributing meaningful representational depth or adding unnecessary parameters.

**What to look for.** A single-layer LSTM trains faster and has fewer parameters. If performance is similar to Run 1, two layers is not justified. If performance is clearly worse, the second layer is learning something useful — likely composing temporal patterns across different timescales. Also check convergence speed: single-layer models often converge faster because gradients flow more directly.

---

### Run 5 — Depth Up

| hidden | layers | bidir | pooling |
|---|---|---|---|
| 128 | 3 | True | final |

**Purpose.** Tests whether additional depth beyond two layers is beneficial or harmful on this dataset.

**What to look for.** Three-layer LSTMs on small datasets frequently underperform two-layer ones due to overfitting and harder optimization. If test accuracy drops relative to Run 1, note the depth ceiling for ESC-50. If training loss is noticeably slower to decrease, the added depth is creating optimization difficulty. This result, combined with Run 4, gives you a complete picture of depth effects.

---

### Run 6 — Unidirectional

| hidden | layers | bidir | pooling |
|---|---|---|---|
| 128 | 2 | False | final |

**Purpose.** Tests the value of future context. Also a meaningful parameter reduction — a unidirectional 2-layer LSTM has roughly half the recurrent parameters of the bidirectional version.

**What to look for.** A significant accuracy drop would confirm that environmental sounds genuinely benefit from bidirectional context — a point worth making in the write-up, since it contrasts with speech processing where causal (unidirectional) models are standard. A small drop or no drop would suggest the LSTM is not effectively using backward context, possibly because the forward pass is sufficient for the short segments used here (41 frames in the short variant).

---

### Run 7 — Mean Pooling

| hidden | layers | bidir | pooling |
|---|---|---|---|
| 128 | 2 | True | mean |

**Purpose.** Tests whether averaging across all timesteps produces a better sequence summary than relying on the final hidden state to carry all information forward.

**What to look for.** Mean pooling is expected to outperform final-state pooling for sounds that don't necessarily occur at the end of a clip — which is the common case in ESC-50. If mean pooling is better, it should be adopted as the default for all subsequent experiments (or at minimum, Run 9 becomes more informative). Also note whether mean pooling affects convergence speed — because the gradient signal is distributed across all timesteps rather than concentrated at the end, it often trains more stably.

---

### Run 8 — High Capacity + Mean Pooling

| hidden | layers | bidir | pooling |
|---|---|---|---|
| 256 | 1 | True | mean |

**Purpose.** Probes an interaction: does mean pooling compensate for reduced depth by preserving more information across timesteps? A deeper model can rely on later layers to carry information forward to the final state; a shallower model may benefit more from retaining information at every timestep via mean pooling.

**What to look for.** If this run outperforms both Run 3 (256 units, final pooling) and Run 4 (1 layer, final pooling) individually, there is a meaningful interaction — the combination of high capacity and mean pooling achieves something neither does alone. If performance is roughly the average of the two, the effects are additive. Either finding is worth reporting. This run also represents a practically interesting configuration: it is shallower and therefore easier to train, with the depth "replaced" by wider hidden states and better pooling.

---

### Run 9 — Unidirectional + Mean Pooling

| hidden | layers | bidir | pooling |
|---|---|---|---|
| 128 | 2 | False | mean |

**Purpose.** Probes whether mean pooling closes the gap between unidirectional and bidirectional models. The hypothesis is that bidirectionality helps primarily because it gives every timestep access to future context — but mean pooling over a unidirectional model's output sequence also provides a form of global context aggregation that may partially substitute for it.

**What to look for.** If this run matches or approaches Run 7 (bidirectional + mean pooling), mean pooling is doing the heavy lifting and bidirectionality provides limited additional benefit — a result that would simplify the recommended architecture considerably. If there is still a meaningful gap between Run 9 and Run 7, bidirectionality has value beyond what pooling can recover, and both components should be retained in any final recommended configuration.

---

## Summary Table

| Run | hidden | layers | bidir | pooling | Purpose | Question |
|---|---|---|---|---|---|---|
| 1 | 128 | 2 | T | final | Baseline | — |
| 2 | 64 | 2 | T | final | Capacity Down | Does smaller capacity hurt? |
| 3 | 256 | 2 | T | final | Capacity Up | Does larger capacity help? |
| 4 | 128 | 1 | T | final | Depth Down | Is the second layer earning its place? |
| 5 | 128 | 3 | T | final | Depth Up | Does extra depth help or hurt? |
| 6 | 128 | 2 | F | final | Unidirectional | How much does future context matter? |
| 7 | 128 | 2 | T | mean | Mean Pooling | Does mean pooling beat final-state? |
| 8 | 256 | 1 | T | mean | High Capacity + Mean Pooling | Does mean pooling substitute for depth? |
| 9 | 128 | 2 | F | mean | Unidirectional + Mean Pooling | Does mean pooling substitute for bidirectionality? |

---

## Long-Variant Notes

### What the Paper Actually Says

Piczak 2015 §3.3 reports that models operating on long segments (101 frames, ~2.3 s, 90% overlap) gave **"slight improvements"** over short (41 frames, ~0.95 s, 50% overlap), but the gain is smaller than the gain from adding a second convolutional layer in the short model. Concretely, the paper's LP variant reaches **64.5% clip-level** vs. ~58–62% for SM/SP — roughly a 3–6 point absolute improvement. The paper also flags that long trains more slowly and is more prone to overfitting on a dataset of this size.

### What Happened in Our CNN Reproduction

Our own `Picszak Study Baseline.ipynb` run of the CNN long variant collapsed to ~2% accuracy — essentially random. The most likely cause is optimizer instability: the CNN notebook inherits SGD at LR = 0.01 without gradient clipping, and the longer sequences produce larger gradient norms that drive the loss to NaN early in training. This is a reproducible pitfall, not an indictment of the long variant as such.

### Why the LSTM Long Variant Should Be More Stable

The LSTM baseline deliberately differs from the CNN in three ways that collectively remove the failure mode above:

- **Adam optimizer** instead of SGD (adaptive per-parameter learning rates, robust to gradient-magnitude mismatches between gates).
- **Gradient clipping at max-norm 1.0**, which is the specific defense against the exploding gradients that broke the CNN long run.
- **Cosine LR decay** starting from a conservative 1e-3, so the effective LR is always well below the SGD LR that destabilized the CNN.

Two distinct checks confirm these changes are working before committing to a full Stage 2 sweep:

1. **`LSTM baseline.ipynb`** — trains Run 1 on **both** `short` and `long` at fold 1. First guardrail: if `long` NaN's here, stop and debug the pipeline.
2. **Phase 2 smoke test (`LSTM experiments.ipynb` → Step 15)** — trains the Stage 1 short winner architecture on `long` at fold 1, to check whether the winning architecture specifically benefits from longer sequences.

### Decision Summary for the Long Variant

- **Skip long entirely:** acceptable if compute is tight. Report short 5-fold numbers and note that long was out of scope. The paper's ~3–6 point gap is real but small.
- **Run Phase 2 only (`LSTM experiments.ipynb` → Step 15):** cheap (~3 short-equivalent runs). Gives you one data point on the winning architecture and either confirms long is worth a full sweep or rules it out.
- **Run Stage 2a + 2b (`LSTM experiments.ipynb` → Steps 16 and 17):** the proper reproduction of Piczak's LM/LP column. ~46 short-equivalent runs total, meaningful only if Phase 2 showed a positive delta.

In all cases, the final write-up should report **clip-level** mean ± std across 5 folds and compare against B (44%), SM/SP (~60%), and LP (64.5%) from the paper.