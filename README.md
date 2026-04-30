# Environmental Sound Classification Using Neural Nets

SYSEN 5888 Deep Learning

Cornell Engineering

Amine Amirouche, Jack Bush, Angela Li, Andrew Littlefield, Jonathan Lloyd 

This repository contains our deep learning project on environmental sound classification using the ESC-50 dataset. The goal is to compare model families across roughly the last decade of neural network development and document how architectural choices affect clip-level accuracy as a measure of performance.

The project includes:

- A Piczak-style CNN baseline and CNN variation studies
- Recurrent sequence modeling experiments with LSTMs
- Transfer learning experiments (for example VGGish and YAMNet)
- Transformer-based audio model explorations (for example AST and BEATs)
- Experiment outputs, plots, and fold-level result tables for reporting

## Project Goals

- Build strong baseline results on ESC-50
- Evaluate how newer architectures compare to classic CNNs
- Analyze trade-offs between model complexity, training stability, and accuracy
- Produce reproducible artifacts for presentation and final reporting

## Directory Overview

### `Code/`

Primary notebooks and experiment writeups.

- `Code/Picszak Study Baseline.ipynb`: baseline CNN implementation and evaluation.
- `Code/CNN Variations/`: controlled changes to CNN hyperparameters and architecture settings.
- `Code/LSTM/`: baseline + follow-up LSTM experiments and rationale documents.
- `Code/Transfer Learning/`: transfer learning workflows and notebooks for VGGish/YAMNet.
- `Code/AST/` and `Code/BEATS/`: transformer-based model experiments and summaries.

### `Plots and Data/`

Generated outputs from training/evaluation runs.

- `baseline/`: baseline fold curves and summary CSVs.
- `Variations/`: CNN variation curves and aggregate summary tables.
- `LSTM/`: LSTM run artifacts, fold outputs, and consolidated result sheets.
- `Transfer Learning/`: VGGish and YAMNet curves and per-fold metrics.
- `AST/` and `BEATs/`: experiment plots for transformer-based models.

### `Docs/`

Project documentation and presentation materials.

- Project planning documents and meeting notes
- Data pipeline notes for ESC-50 preprocessing
- Presentation outline and draft slide deck

### Root files

- `Piczak2015-ESC-ConvNet.pdf`: reference paper for the baseline architecture.
- `README.md`: this project overview and navigation guide.

## Notes for Contributors

- Most experimentation is notebook-driven and organized by model family.
- Final metrics are typically reported as fold-level CSV summaries in `Plots and Data/`.
- Keep new artifacts grouped under the corresponding model directory to preserve reproducibility.

