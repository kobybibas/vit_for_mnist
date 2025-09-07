# Vision Transformer (ViT) for MNIST Sequences
This project investigates the use of **Vision Transformers (ViTs)** for sequence-based tasks on the MNIST dataset. 
The repository implements and compares Transformer-based models with classical architectures, evaluating tasks beyond single-digit classification.

## Features
1. **Simple CNN baseline** for MNIST digit classification
2. **Self-attention regression**: Predicting the sum of a sequence of MNIST images
3. [TODO] **Cross-attention detection**: Determining if an anchor image exists in a given sequence


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
2. Accuracy: Rounded prediction matches ground truth

| Model | Validation Loss | Validation Accuracy |
| :-- | :-- | :-- |
| Fully Connected | *TBD* | *TBD* |
| Transformer | **1.17** | **0.64** |

## Visualizations

Example images from a sequence:
![Example images from a sequence](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/transformer__predictions_0.png?raw=true)

Loss versus training epochs:
![Fully connected accuracy](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/fully_connected__accuracy_over_epochs.png?raw=true)


Fully connected accuracy over training epochs:
![Fully connected accuracy](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/fully_connected__accuracy_over_epochs.png?raw=true)


# TODOs
Self attention:
- [x] Visualizations of the sequence, ground truth and prediction
- [x] Add loss vs epoch figure
- [x] Add accuracy metric (if round(output)=GT)
- [x] Sequence of varying length (1-10)
- [x] Add output transformation for output to be between -1 and 1: (y_hat + 0.5) * y_max
- [x] Compare to traditional model (CNN + fc)
- [ ] Histogram of error residual 

Cross attention:
- [ ] Baseline
- [ ] Sequence of varying length 
- [ ] Compare to traditional model (CNN + fc)