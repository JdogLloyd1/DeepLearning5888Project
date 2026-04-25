# BEATs Model — ESC-50 Environmental Sound Classification

## Overview

This experiment uses the pretrained BEATs model from Microsoft’s official implementation (`BEATs_iter3_plus_AS2M.pt`), which is a transformer-based architecture pretrained on large-scale audio datasets and then adapted for ESC-50 classification.

---

## Why BEATs?

The original Piczak CNN baseline relies on convolutional layers trained from scratch using log-mel spectrogram inputs. While CNNs are effective for local spectral pattern recognition, they are limited in capturing long-range temporal dependencies and broader contextual relationships across an entire audio clip.

The BEATs model improves on this by:

- using transformer-based self-attention instead of only convolution
- learning global relationships across the full audio sequence
- leveraging large-scale pretraining on AudioSet and AS2M before ESC-50 fine-tuning

This is particularly useful for ESC-50 because many environmental sounds depend on long-duration context rather than short isolated events. Examples include rainfall, sea waves, helicopters, vacuum cleaners, chainsaws, and crowd noise.

Rather than learning all features from only 2,000 ESC-50 clips, BEATs begins with strong pretrained audio representations and only adapts them to the final classification task.

This makes BEATs a strong transfer learning experiment and a meaningful extension beyond the original Piczak CNN.

---

## Model Architecture

## Input Representation

Each ESC-50 audio clip is:

- loaded as raw waveform
- converted to mono if necessary
- resampled to the required sample rate
- passed through the pretrained BEATs feature extractor

Unlike the CNN and LSTM pipelines, BEATs does **not** rely on manually segmented mel spectrogram windows.

Instead, the full 5-second clip is processed directly through the transformer backbone to generate pretrained audio representations.

This means evaluation is naturally performed at the **clip level**, which is directly comparable to the literature.

---

## Real BEATs Architecture

The pretrained checkpoint used was:

```text
BEATs_iter3_plus_AS2M.pt
```

This model includes:

- transformer encoder blocks
- pretrained large-scale audio representations
- self-supervised pretraining on AudioSet and AS2M
- a new classification head added for 50 ESC-50 classes

The backbone provides strong pretrained audio embeddings, while a new classifier head is trained for ESC-50.

---

## Why Fine-Tuning Instead of Training from Scratch?

Training a transformer from scratch on ESC-50 would severely overfit because:

- only 2,000 total clips exist
- transformer models have extremely high parameter counts

Fine-tuning pretrained BEATs allows:

- much faster convergence
- significantly stronger generalization
- substantially better performance than shallow models trained from scratch

This is the primary reason real BEATs strongly outperforms the earlier placeholder “fake BEATs.”

---

## BEATs Configuration Summary

| Parameter | Value |
|---|---|
| pretrained checkpoint | BEATs_iter3_plus_AS2M.pt |
| batch size | 4 |
| learning rate | small fine-tuning LR |
| optimizer | AdamW |
| number of classes | 50 |
| folds | 5 |
| training style | transfer learning |
| evaluation | clip-level accuracy |

---

## Why These Settings?

### Small Learning Rate

Because BEATs is pretrained, large learning rates would destroy useful pretrained representations.

Fine-tuning transformer backbones requires very small learning rates to preserve learned features while adapting the final classification head.

This is very different from CNN/LSTM training from scratch.

---

### Small Batch Size (4)

Transformer models require significantly more GPU memory than CNNs and LSTMs.

A batch size of 4 balances:

- stable gradients
- GPU memory limitations
- practical Colab execution time

---

### Transfer Learning Instead of Full Training

The BEATs backbone already contains strong pretrained knowledge from large-scale audio pretraining.

Rather than relearning audio features from scratch, the model focuses on adapting these learned representations to ESC-50 classification.

This is the key reason for the major performance improvement.

---

## 5-Fold Cross Validation Results

To evaluate robustness, 5-fold cross validation was performed using the official ESC-50 fold structure. Each fold used one official fold as the test set and the remaining four folds for training.

This follows the standard evaluation protocol used in the original ESC-50 paper and allows direct comparison to literature.

---

## Final 5-Fold Performance

The final summary across all five folds was:

| Metric | Result |
|---|---:|
| Mean Final Test Accuracy | **94.40%** |
| Standard Deviation | **± 3.27%** |
| Mean Best Test Accuracy | **95.55%** |
| Standard Deviation (Best) | **± 2.45%** |
| Mean Test Loss | **0.2120** |
| Standard Deviation (Loss) | **0.1274** |
| Mean Training Time | **152.37 sec** |

These values were generated directly from the saved fold summary files.

---

## Interpretation of Results

The most important result is the mean test accuracy of **94.40%**, which demonstrates extremely strong generalization across all five folds.

The relatively small standard deviation (±3.27%) indicates that performance remained stable across different train/test splits rather than depending on one unusually favorable fold.

This confirms that BEATs performs reliably across the full dataset rather than only succeeding on individual folds.

The low mean test loss (0.2120) also supports this conclusion, showing that predictions remained confident and well-calibrated.

Unlike the earlier placeholder BEATs model, where train accuracy increased but test accuracy plateaued due to weak representations and overfitting, the real pretrained BEATs model maintained both strong training performance and strong generalization.

---

## Comparison to Literature

### Original Piczak CNN (2015)

The original Piczak CNN baseline reported:

| Model | Accuracy |
|---|---:|
| Random Forest Baseline | 44% |
| Piczak CNN Short | ~58–62% |
| Piczak CNN Long (LP) | 64.5% |

Compared to this baseline, the real BEATs result of:

```text
94.40%
```

represents a very large improvement of nearly **30 percentage points** over the strongest Piczak CNN result.

This clearly demonstrates the advantage of pretrained transformer-based models over CNNs trained from scratch.

---

## Why BEATs Performs Better

This large improvement is primarily due to **transfer learning**, not simply architecture changes.

The original Piczak CNN:

- trained from scratch on only 2,000 clips
- relied primarily on local convolutional filters
- had no large-scale pretraining

The real BEATs model:

- uses transformer self-attention across the full audio representation
- leverages large-scale AudioSet and AS2M pretraining
- begins with strong learned audio representations before ESC-50 fine-tuning

This experiment clearly shows that:

> pretraining is the most important factor driving performance improvement

The earlier placeholder “fake BEATs” model, which used flattened spectrograms and fully connected layers without pretraining, achieved only ~15–25% accuracy. Once true pretrained BEATs was introduced, performance increased to over 94%, demonstrating that architecture alone is not sufficient.

---

## Final Conclusion

The BEATs experiment produced one of the strongest performances of all tested models.

Using 5-fold cross validation, the model achieved:

```text
Mean Test Accuracy = 94.40% ± 3.27%
```

which substantially exceeds both the original Piczak CNN baseline and reported human accuracy on ESC-50.

These results demonstrate that pretrained transformer models provide major advantages in environmental sound classification through stronger feature representations, better long-range temporal modeling, and significantly improved generalization.

Among all architectures explored in the project, BEATs is one of the strongest final models and provides strong evidence that transfer learning is the most impactful improvement over the original research paper baseline.

