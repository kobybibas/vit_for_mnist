# Vision Transformer (ViT) for MNIST Sequences
This project investigates the use of Vision Transformers (ViTs) for sequence-based tasks on the MNIST dataset. 
The repository implements and compares Transformer-based models with classical architectures, evaluating tasks beyond single-digit classification.

## Features
1. MNIST digit classification using a simple CNN baseline
2. Self-attention regression for Predicting the sum of a sequence of MNIST images
3. [TODO] Cross-attention detection to determine if an anchor image exists in a given sequence


## Problem Definition
MNIST is a set of handwritten digit images. The primary goal is:
Given a sequence of MNIST images, predict the sum of all digits in the sequence.
The project benchmarks a Transformer-based architecture against fully connected models.

## Architecture
- Each image is encoded via a CNN to a 128-dimensional embedding
- Sequences (up to 10 images) are processed by a Transformer
- The model predicts a single float (sum) using MSE loss
- Output is compared to ground truth for regression and accuracy metrics

## Results
Two key metrics are evaluated:
1. MSE Loss
2. Accuracy (correct if the rounded prediction matches ground truth)

| Model | Validation Loss | Validation Accuracy |
| :-- | :-- | :-- |
| Fully Connected | 1.69 | 0.27 |
| Transformer | **1.18** | **0.66** |

## Visualizations

Example images from a sequence:
![Example images from a sequence](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/transformer__predictions_0.png?raw=true)

Transformer accuracy over training epochs:
![Transformer accuracy](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/transformer__accuracy_over_epochs.png?raw=true)

Fully connected accuracy over training epochs:
![Fully connected accuracy](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/fully_connected__accuracy_over_epochs.png?raw=true)


# Running the Main Scripts

This project uses the [uv](https://github.com/astral-sh/uv) package manager for Python. 
Install requirements:
```sh
uv sync
```

Running training and evaluation
```sh
uv run src/main_mnist_classification.py
uv run src/main_mnist_sum_prediction.py
uv run src/main_mnist_anchor_in_sequence.py
```

# TODOs
Self attention:
- [x] Visualizations of the sequence, ground truth and prediction
- [x] Add loss vs epoch figure
- [x] Add accuracy metric (if round(output)=GT)
- [x] Sequence of varying length (1-10)
- [x] Add output transformation for output to be between -1 and 1: (y_hat + 0.5) * y_max
- [x] Compare to traditional model (CNN + fc)

Cross attention:
- [X] Baseline
- [X] Sequence of varying length 
- [ ] Compare to traditional model (CNN + fc)
  - [ ] Run transformer, optimize hyperparams
  - [ ] Run FC with the same hyperparams