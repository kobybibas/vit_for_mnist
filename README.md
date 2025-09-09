# Vision Transformer for MNIST Sequences
This project investigates the use of Vision Transformers (ViTs) for sequence-based tasks on the MNIST dataset. 
The repository implements and compares Transformer-based models with classical architectures, evaluating tasks beyond single-digit classification.


## Features
1. MNIST digit classification using a simple CNN baseline
2. Self-attention regression for predicting the sum of a sequence of MNIST images
3. Cross-attention detection to determine if an anchor image exists in a given sequence


## Problem Definition
MNIST is a set of handwritten digit images. In this repository we focus on the following tasks:
- **Digit Classification**: Predict the class of a single MNIST digit.  
- **Sequence Regression**: Given a sequence of MNIST images, predict the sum of all digits in the sequence.  
- **Anchor-in-Sequence Classification**: Given an anchor MNIST digit and a sequence of digits, predict whether the anchor digit appears anywhere in the sequence.  

The project benchmarks Transformer-based architectures against fully connected models across these tasks.


## Architecture
- Each image is encoded via a CNN into a 128-dimensional embedding
- Sequences (up to 10 images) are processed by a Transformer
- Per task head:
  - Classification head for single-digit recognition
  - Self-attention layers and Regression head for predicting sums (MSE loss)
  - Cross-attention layer and binary classification head for anchor detection (cross-entropy loss)


## Results

### Sequence Regression (Sum Prediction)
Two key metrics are evaluated:
1. MSE Loss
2. Accuracy (correct if the rounded prediction matches ground truth)

| Model           | Validation Loss | Validation Accuracy |
| :-------------- | :-------------- | :------------------ |
| Fully Connected | 1.69            | 0.27                |
| Transformer     | **1.18**        | **0.66**            |

![Example images from a sequence](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/sequence_sum__transformer__predictions_0.png?raw=true)

Transformer accuracy over training epochs:
![Transformer accuracy](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/sequence_sum__transformer__accuracy_over_epochs.png?raw=true)



### Anchor-in-Sequence Classification
Evaluation metric: Binary classification accuracy

| Model           | Validation Loss | Validation Accuracy |
| :-------------- | :-------------- | :------------------ |
| Fully Connected | 0.64            | 0.77                |
| Transformer     | **0.12**        | **0.92**            |


![Example images from a sequence](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/anchor_in_seq__transformer__predictions_0.png?raw=true)

Transformer accuracy over training epochs:
![Transformer accuracy](https://github.com/kobybibas/vit_for_mnist/blob/main/figures/anchor_in_seq__transformer__accuracy_over_epochs.png?raw=true)



## Running the Main Scripts

This project uses the [uv](https://github.com/astral-sh/uv) package manager for Python.  
Install requirements:
```
uv sync
```

Running training and evaluation:
```
uv run src/main_mnist_classification.py
uv run src/main_mnist_sum_prediction.py
uv run src/main_mnist_anchor_in_sequence.py
```
