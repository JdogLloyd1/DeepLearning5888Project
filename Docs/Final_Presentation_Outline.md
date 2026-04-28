# Final Presentation Outline

## Story Spine

**Hypothesis:** Modern AI techniques will outperform the Piczak 2015 ESC-50 baseline.

**How we test it:** Use Piczak's environmental sound classification experiment as a historical anchor, then walk through three technical eras of neural-network development:

| Era | Models | Core Question |
|---|---|---|
| Training from scratch | Piczak CNN, LSTM | How far can models get when all audio features are learned from ESC-50 alone? |
| Transfer learning: pretrained CNN models | VGGish, YAMNet | What changes when the model brings audio knowledge from larger pretrained CNN-style models? |
| Transfer learning: pretrained audio transformers | AST, BEATs | What changes when pretrained audio representations use attention and large-scale audio encoders? |

**Main takeaway to build toward:** The largest gains come from pretrained representations and evaluation discipline, not simply from adding more architecture variants.

## Visual Language

Use a Tufte-inspired style: sparse text, direct labels, light gridlines, high data-to-ink ratio, and repeated visual grammar. Each model slide should behave like a compact "model card" with the same four elements:

| Element | Role |
|---|---|
| Architecture visual | Show what kind of model it is. |
| Change from prior era | Explain the technical evolution. |
| Result | Use final 5-fold accuracy as headline where available. |
| Takeaway | State what the model taught us. |

Use one recurring roadmap motif across the deck:

`Training from scratch -> Pretrained CNN models -> Pretrained audio transformers`

On transition slides, dim the previous era and highlight the new one.

## Reporting Standard

| Metric | Rule |
|---|---|
| Accuracy | Headline final 5-fold test accuracy where available. |
| Best accuracy | Small asterisk or footnote only when useful. |
| Segmented models | Use clip-level voting accuracy for Piczak-style comparisons. |
| Runtime | Include only if collected consistently on comparable hardware. |
| Baseline/CNN | Official team baseline should be the corrected clip-level voting rerun. |

## Slide Draft

### 1. Why This Project: Measuring The AI Acceleration

| Field | Draft |
|---|---|
| Purpose | Introduce the course-level motivation before narrowing into ESC-50. |
| Main message | Neural networks have evolved over decades, but the last decade made performance gains visible, measurable, and practical. |
| Visual | Course-relevant horizontal timeline: early neural nets/backprop -> deep learning revival -> ImageNet/CNNs -> Piczak 2015 -> transfer learning -> pretrained representations at scale. |
| On-slide text | "Can we make the evolution of neural networks measurable?" |
| Speaker point | This project uses a 2015 benchmark as a time capsule to observe what later neural-network techniques changed. |
| Requirements covered | Problem motivation; study purpose. |

### 2. Hypothesis And Test Design

| Field | Draft |
|---|---|
| Purpose | State the hypothesis and how the experiment will test it. |
| Main message | Modern AI techniques should produce better ESC-50 performance than the Piczak baseline. |
| Visual | Simple experiment schematic: 2015 benchmark -> model eras -> accuracy/loss/runtime comparison. Add an "in scope / out of scope" strip. |
| On-slide text | "Hypothesis: modern AI techniques improve environmental sound classification." |
| Speaker point | The comparison is not a model beauty contest. Each model represents a technical shift in how neural networks learn. |
| Requirements covered | Main question; metrics; other methods boundary. |

Suggested in-scope/out-of-scope strip:

| In Scope | Out Of Scope |
|---|---|
| CNN replication, LSTM, VGGish, YAMNet, AST, BEATs | Full classical ML survey, GNNs, generative audio, multimodal audio-visual models |

### 3. Evolution Roadmap

| Field | Draft |
|---|---|
| Purpose | Give the audience the map before the details. |
| Main message | The deck moves through three eras: from-scratch learning, pretrained CNN models, pretrained audio transformers. |
| Visual | Large roadmap with three era bands and model names placed under each band. |
| On-slide text | "The central axis: where does the model's audio knowledge come from?" |
| Speaker point | Era 1 learns from ESC-50 only. Eras 2 and 3 import learned representations from larger audio corpora. |
| Requirements covered | Methods overview; comparison framing. |

### 4. Piczak And ESC-50

| Field | Draft |
|---|---|
| Purpose | Introduce the original experiment and dataset. |
| Main message | ESC-50 made environmental sound classification benchmarkable, and Piczak's CNN is the historical reference point. |
| Visual | Dataset card plus a small spectrogram/waveform image: 2,000 clips, 50 classes, 5 folds, 5 seconds per clip. |
| On-slide text | "A controlled benchmark for machine listening." |
| Speaker point | The dataset is curated, balanced, labeled, and organized into official folds, which makes comparisons across methods meaningful. |
| Requirements covered | Dataset source and structure; motivation; validation structure. |

### 5. Era 1: Training From Scratch

| Field | Draft |
|---|---|
| Purpose | Transition from the historical setup to the first technical era. |
| Main message | In this era, the model must learn both the audio representation and the classifier from only 2,000 clips. |
| Visual | Roadmap motif with "Training from scratch" highlighted. Use a small data bottleneck graphic: small ESC-50 box feeding both feature learning and classifier learning. |
| On-slide text | "Constraint: all useful audio knowledge must come from ESC-50." |
| Speaker point | This is the important limitation that later transfer learning is designed to address. |
| Requirements covered | Methods appropriateness; experiment logic. |

### 6. Piczak CNN Baseline

| Field | Draft |
|---|---|
| Purpose | Explain the baseline architecture and corrected evaluation protocol. |
| Main message | The baseline converts audio to log-mel/delta spectrogram segments, classifies segments with a CNN, then aggregates segment predictions to clip-level voting. |
| Visual | Architecture pipeline: waveform -> log-mel + delta -> segments -> CNN -> clip-level voting. |
| On-slide text | "Baseline = Piczak CNN replication with clip-level voting." |
| Speaker point | The corrected rerun should be the official baseline because Piczak reported clip-level predictions, not raw segment-level accuracy. |
| Requirements covered | Preprocessing; normalization; methods; base/published comparison; validation. |

Current result treatment:

| Result Item | Deck Treatment |
|---|---|
| Published Piczak long probability voting | 64.5% clip-level accuracy |
| Current corrected team baseline | Pending Colab rerun |

### 7. LSTM Extension

| Field | Draft |
|---|---|
| Purpose | Show the from-scratch alternative architecture before shifting to transfer learning. |
| Main message | LSTM changes the modeling assumption from local spectrogram patterns to temporal sequence modeling while still learning from ESC-50 alone. |
| Visual | Side-by-side mini-diagram: CNN sees spectrogram patches; LSTM sees a sequence of 120-dimensional timesteps. |
| On-slide text | "What if temporal order is the missing ingredient?" |
| Speaker point | This slide should use the teammate-confirmed takeaway. Current evidence suggests it is an architecture extension within the same from-scratch regime. |
| Requirements covered | Multiple methods; optimization; results; comparison with baseline. |

Current result candidate:

| Model | Headline Result | Note |
|---|---:|---|
| LSTM Run8 CapacityMeanPool long | 56.8% final clip-level accuracy | Best 59.0%; teammate confirmation needed |

### 8. Era Shift: Transfer Learning With Pretrained CNN Models

| Field | Draft |
|---|---|
| Purpose | Explain why the next era matters before showing VGGish/YAMNet. |
| Main message | Transfer learning changes the source of knowledge: the model no longer learns audio features only from ESC-50. |
| Visual | Roadmap motif with "Pretrained CNN models" highlighted. Show a large pretraining corpus feeding a smaller ESC-50 fine-tuning box. |
| On-slide text | "Shift: learn the representation elsewhere, adapt it here." |
| Speaker point | ESC-50 is small. Transfer learning asks whether pretrained audio features can raise the performance ceiling. |
| Requirements covered | Methods rationale; appropriateness; other methods/pros-cons setup. |

### 9. VGGish

| Field | Draft |
|---|---|
| Purpose | Present early CNN-style pretrained audio transfer learning. |
| Main message | VGGish tests whether older pretrained audio embeddings improve over from-scratch models. |
| Visual | Model card: waveform/spectrogram-like input -> VGGish embedding -> classifier. Include frozen vs fine-tuned result as a small comparison. |
| On-slide text | "Pretraining helps only if the representation transfers well." |
| Speaker point | The framing needs teammate confirmation because frozen VGGish appears stronger than fine-tuned VGGish in current results. |
| Requirements covered | Method details; optimization; results; which method performed better. |

Current result candidates:

| VGGish Variant | Final 5-Fold Accuracy | Best 5-Fold Accuracy |
|---|---:|---:|
| Frozen | 56.45% | 60.25% |
| Fine-tuned | 53.15% | 53.50% |

### 10. YAMNet

| Field | Draft |
|---|---|
| Purpose | Present stronger CNN-style pretrained audio transfer learning. |
| Main message | YAMNet shows that domain-fit and pretrained model quality matter; transfer learning is not one-size-fits-all. |
| Visual | Model card plus small bar comparison against VGGish. |
| On-slide text | "Better pretrained audio representations produce a large jump." |
| Speaker point | This is the first major performance break from the from-scratch results. |
| Requirements covered | Method details; results; comparison; insight. |

Current result candidate:

| Model | Headline Result | Note |
|---|---:|---|
| YAMNet fine-tuned | 83.40% final accuracy | Best 84.45% |

### 11. Era Shift: Transfer Learning With Pretrained Audio Transformers

| Field | Draft |
|---|---|
| Purpose | Explain why AST and BEATs are a distinct period within transfer learning. |
| Main message | The next transfer-learning period combines external pretraining with attention-based or foundation-style audio encoders. |
| Visual | Roadmap motif with "Pretrained audio transformers" highlighted. Show local CNN receptive fields vs global attention over a clip. |
| On-slide text | "Shift: pretrained representations plus broader context." |
| Speaker point | We are still in transfer learning, but the pretrained backbone has evolved from CNN-style feature extraction to transformer-based audio representation learning. |
| Requirements covered | Methods rationale; model comparison framing. |

### 12. AST

| Field | Draft |
|---|---|
| Purpose | Present the spectrogram-transformer path. |
| Main message | AST fine-tunes a pretrained AudioSet spectrogram transformer on ESC-50 and achieves near-ceiling performance. |
| Visual | Model card: waveform -> AST feature extractor -> spectrogram patches -> transformer encoder -> 50-class head. |
| On-slide text | "AST: attention over audio spectrogram patches." |
| Speaker point | AST keeps a spectrogram-like representation but changes the backbone to a pretrained transformer. |
| Requirements covered | Method details; optimization; results; validation. |

Current result candidate:

| Model | Headline Result | Note |
|---|---:|---|
| AST | 95.25% final 5-fold accuracy | Best 95.65% |

### 13. BEATs And Final Takeaway

| Field | Draft |
|---|---|
| Purpose | Present BEATs, then synthesize the whole project. |
| Main message | BEATs confirms the final pattern: modern pretrained audio representations substantially outperform from-scratch baselines. |
| Visual | Left: compact BEATs model card. Right: final benchmark ladder from Piczak references through team models. |
| On-slide text | "The biggest shift was where the model learned its audio knowledge." |
| Speaker point | Architecture mattered, but the decisive improvement was moving from ESC-50-only learning to large-scale pretrained audio representations. |
| Requirements covered | Final model vs baseline/published; conclusions; best-performing methods; insights; alternatives/pros-cons. |

Current result candidate:

| Model | Headline Result | Note |
|---|---:|---|
| BEATs | 94.40% final 5-fold accuracy | Best 95.55%; mean training time 152.37 sec in current summary |

Final comparison ladder draft:

| Model / Reference | Final Accuracy |
|---|---:|
| Random forest reference | 44.0% |
| Piczak CNN published long probability voting | 64.5% |
| LSTM long | 56.8% |
| VGGish frozen | 56.45% |
| YAMNet fine-tuned | 83.40% |
| BEATs | 94.40% |
| AST | 95.25% |

Final alternatives/pros-cons callout:

| Alternative | Pros | Cons |
|---|---|---|
| Classical ML on engineered features | Fast, interpretable, low compute | Lower ceiling on complex audio patterns |
| GNNs | Useful for relational label or event graphs | ESC-50 does not naturally provide graph structure |
| Generative/self-supervised audio methods | Could improve data efficiency | Larger scope and compute requirement |
| Multimodal models | Useful when audio and video coexist | Not aligned with audio-only ESC-50 benchmark |

## Open Items

| Item | Owner / Note |
|---|---|
| Corrected Piczak CNN rerun with clip-level voting | Needed before final baseline slide is locked. |
| VGGish frozen vs fine-tuned framing | Teammate confirmation needed. |
| LSTM takeaway wording | Teammate confirmation needed. |
| Runtime comparison | Only include if measured consistently, ideally on the same Colab GPU class. |
| Piczak-era hardware/cost estimate | Optional; should be framed as contextual if hardware is not specified in the paper. |
