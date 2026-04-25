# AST Model — ESC-50 Environmental Sound Classification

## Overview

This experiment uses the actual pretrained AST model from Hugging Face (`MIT/ast-finetuned-audioset-10-10-0.4593`), which is a transformer-based architecture pretrained on large-scale audio datasets and then fine-tuned for classification.

---

## Why AST?

The original Piczak CNN baseline relies on convolutional layers trained from scratch using log-mel spectrogram inputs. While CNNs are strong for local spectral pattern recognition, they are limited in modeling long-range temporal dependencies across an entire audio clip.

The **Audio Spectrogram Transformer (AST)** improves on this by:

- using self-attention instead of convolution
- learning global relationships across the full spectrogram
- leveraging large-scale pretraining on AudioSet before ESC-50 fine-tuning

This is especially valuable for ESC-50 because many environmental sounds depend on long-range context rather than short local events. Examples include rain, sea waves, chainsaws, helicopters, and crowd noise.

Rather than learning all audio features from only 2,000 ESC-50 clips, AST begins with strong pretrained representations and only adapts them to the final classification task.

This makes AST a strong transfer learning experiment and a meaningful extension beyond the original Piczak CNN.

---

## Model Architecture

### Input Representation

Each ESC-50 audio clip is:

- loaded as raw waveform
- converted to mono if necessary
- resampled to 16 kHz
- passed through the Hugging Face AST feature extractor

Unlike the CNN pipelines, AST does **not** use manually segmented mel spectrogram windows.

Instead, the full 5-second clip is processed directly into transformer-ready patch embeddings.

This means evaluation is naturally performed at the clip level, which is directly comparable to the literature.

---

## Real AST Architecture

The pretrained model used is:

```text
MIT/ast-finetuned-audioset-10-10-0.4593
```

This model includes:

- patch embedding layer
- transformer encoder blocks
- pretrained AudioSet feature representations
- classification head replaced for 50 ESC-50 classes

The classifier head is fine-tuned while optionally allowing the full backbone to update.

### Why Fine-Tuning Instead of Training from Scratch?

Training a transformer from scratch on ESC-50 would severely overfit because:

- only 2,000 total clips exist
- transformer models have very high parameter counts

Fine-tuning pretrained AST allows:

- much faster convergence
- significantly better generalization
- stronger performance than shallow models trained from scratch

---

## AST Configuration Summary

| Parameter | Value |
|---|---|
| pretrained model | MIT/ast-finetuned-audioset-10-10-0.4593 |
| sample rate | 16000 |
| batch size | 8 |
| learning rate | 1e-5 |
| weight decay | 1e-4 |
| epochs | 10 |
| freeze backbone | False |
| optimizer | AdamW |
| number of classes | 50 |

---

## Why These Settings?

### Small Learning Rate (1e-5)

Because AST is pretrained, large learning rates would destroy useful pretrained representations.

Fine-tuning transformers requires very small learning rates to preserve learned features while adapting the classifier.

This is very different from CNN training from scratch.

---

### Small Batch Size (8)

Transformer models require significantly more GPU memory than CNNs.

A batch size of 8 balances:

- stable gradients
- GPU memory limits
- practical Colab execution time

---

### Weight Decay (1e-4)

Weight decay helps reduce overfitting during fine-tuning and improves generalization.

This is especially important because AST can memorize ESC-50 quickly.

---

### Full Fine-Tuning

The backbone was not frozen:

```python
freeze_backbone = False
```

This allows the transformer to adapt fully to ESC-50 rather than only retraining the classifier head.

This usually improves final accuracy at the cost of faster overfitting.

---

## Training Results (Fold 1)

## Loss Curves

### Observed Behavior

Train loss:

- started at ~1.8
- dropped rapidly to near zero by epoch 3–4

Test loss:

- started at ~0.58
- dropped quickly to ~0.22
- stabilized with almost no increase afterward

This indicates:

- optimization was successful
- pretrained features transferred extremely well
- no major instability occurred during training

Unlike the CNN long variant that collapsed to random performance, AST training remained highly stable throughout.

---

## Accuracy Curves

### Observed Behavior

Train accuracy:

- started near 70%
- reached nearly 100% by epoch 3–4

Test accuracy:

- started around 92%
- stabilized around 94%

This is a very strong result for the first fold.

The model learned quickly and generalization remained strong even after train accuracy saturated.

---

## Interpretation of Results

AST:

- ~94% test accuracy
- strong transfer learning performance
- transformer pretrained on AudioSet

---

### Compared to Piczak CNN

Reference values:

| Reference | Clip-Level Accuracy |
|---|---|
| Random forest baseline | 44% |
| Piczak CNN short | ~58–62% |
| Piczak CNN long (LP) | 64.5% |
| Human accuracy | 81.3% |

AST fold 1 result:

```text
~94%
```

This significantly exceeds:

- the original CNN
- reported human accuracy

This is expected because AST benefits from:

- much larger pretraining datasets
- stronger transformer architecture
- transfer learning unavailable in 2015

This does not invalidate the comparison—it strengthens the argument that pretrained transformers substantially outperform shallow CNN baselines.

---

## Overfitting Discussion

Although train accuracy reached nearly 100%, test accuracy remained stable around 94%.

This suggests:

- some memorization exists
- but generalization remains strong


The gap is acceptable because:

- pretrained representations are already strong
- the test loss remains low and stable

If additional folds show similar behavior, this result is reliable.

---

### Final 5-Fold Performance

The final summary across all five folds was:

| Metric | Result |
|---|---:|
| Mean Final Test Accuracy | **95.25%** |
| Standard Deviation | **± 2.05%** |
| Mean Best Test Accuracy | **95.65%** |
| Standard Deviation (Best) | **± 2.05%** |
| Mean Test Loss | **0.1708** |
| Standard Deviation (Loss) | **0.0635** |

These values were generated directly from the saved fold summary files.

---

## Interpretation of Results

The most important result is the mean test accuracy of **95.25%**, which demonstrates extremely strong generalization across all five folds. The relatively small standard deviation (±2.05%) indicates that performance was stable and consistent across different train/test splits rather than being dependent on one unusually favorable fold.

This confirms that the strong Fold 1 result (~94%) was not an outlier and that the pretrained AST model performs reliably across the full dataset.

The low mean test loss (0.1708) also supports this conclusion, showing that predictions remained confident and well-calibrated across folds.

Unlike earlier placeholder AST and BEATs experiments, where train accuracy improved but test accuracy plateaued due to underpowered architectures or overfitting, the real AST model maintained both high training performance and strong generalization.

---

## Comparison to Literature

### Original Piczak CNN (2015)

The original Piczak CNN baseline reported:

| Model | Accuracy |
|---|---:|
| Random Forest Baseline | 44% |
| Piczak CNN Short | ~58–62% |
| Piczak CNN Long (LP) | 64.5% |

The Piczak model was one of the first successful deep learning approaches for ESC-50 and established CNNs as a strong baseline for environmental sound classification.

Compared to this baseline, the real AST result of:

```text
95.25%
```

represents a very large improvement of more than **30 percentage points** over the strongest Piczak CNN result.

---