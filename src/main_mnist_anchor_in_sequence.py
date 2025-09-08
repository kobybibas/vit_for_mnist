# =============================
# Imports and Global Constants
# =============================

import binascii
import os.path as osp
from datetime import datetime
from typing import Any

import lightning as L
import pandas as pd
import torch
import torchvision
from lightning.pytorch.loggers import CSVLogger
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import _Image_fromarray

# Debug and training parameters
DEBUG = False
MAX_IMG_PER_SEQ = 10
NUM_WORKERS = 9
MAX_EPOCHS = 50
LR_STEPS = [30, 40]
LR = 1e-3
MODEL_TYPE = "transformer"
assert MODEL_TYPE in ["transformer", "fc"]


# =============================
# Dataset and Data Preparation
# =============================


class MNISTSequence(MNIST):
    """
    Custom MNIST dataset that returns a sequence of images and an anchor image.
    The label is True if the anchor digit appears in the sequence.
    """

    def get_sequence_length(self, index):
        # Deterministically set the sequence length
        return binascii.crc32(str(index).encode()) % MAX_IMG_PER_SEQ + 1

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        start = index
        end = start + self.get_sequence_length(index) + 1  # +1 for anchor image
        imgs, targets = self.data[start:end], self.targets[start:end]

        img_seq = []
        for img in imgs:
            img = _Image_fromarray(img.numpy(), mode="L")
            if self.transform is not None:
                img = self.transform(img)
            img_seq.append(img)

        img_seq = torch.stack(img_seq, dim=0)  # (IMG_PER_SEQ, 1, 28, 28)

        if self.target_transform is not None:
            for i in range(len(targets)):
                targets[i] = self.target_transform(targets[i])

        img_anchor = img_seq[0]  # (1, 28, 28)
        img_seq = img_seq[1:]  # (IMG_PER_SEQ, 1, 28, 28)

        target_anchor = targets[0]
        target_seq = targets[1:]
        label = target_anchor in target_seq
        return img_anchor, img_seq, label

    def __len__(self) -> int:
        return len(self.data) - MAX_IMG_PER_SEQ - 1


def pad_sequences(batch):
    """
    Pads sequences in a batch to the same length and creates attention masks.
    Returns anchors, padded sequences, attention masks, and labels as tensors.
    """
    anchors, sequences, labels = zip(*batch)
    max_len = (
        max(len(seq) for seq in sequences)
        if MODEL_TYPE == "transformer"
        else MAX_IMG_PER_SEQ
    )
    padded_seqs = []
    attention_masks = []
    pad_value = 0
    for seq in sequences:
        padded = F.pad(
            seq,
            (
                pad_value,
                pad_value,
                pad_value,
                pad_value,
                pad_value,
                pad_value,
                pad_value,
                max_len - len(seq),
            ),
        )
        mask = torch.ones(len(seq), dtype=torch.bool)
        mask = F.pad(mask, (0, max_len - len(seq)), value=False)
        padded_seqs.append(padded)
        attention_masks.append(mask)
    return (
        torch.stack(anchors),
        torch.stack(padded_seqs),
        torch.stack(attention_masks),
        torch.tensor(labels),
    )


def data_preparation():
    """
    Prepares and returns the training and validation datasets with transforms.
    """
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    val_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_set = MNISTSequence(
        root="data", train=True, download=True, transform=train_transform
    )
    val_set = MNISTSequence(
        root="data", train=False, download=True, transform=val_transform
    )
    return train_set, val_set


# =============================
# Model Definitions
# =============================


class ImageEncoder(nn.Module):
    """
    Simple CNN encoder for MNIST images.
    """

    def __init__(self, output_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1024, output_embed_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class TransformerModel(nn.Module):
    """
    Transformer-based model for sequence classification.
    """

    def __init__(self):
        super().__init__()
        self.embed_dim = 128
        self.encoder = ImageEncoder(self.embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.embed_dim, num_heads=4, batch_first=True
        )
        self.fc1 = nn.Linear(self.embed_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x_anchor, x_seq, attention_mask):
        if DEBUG:
            print(f"1. {x_anchor.shape=} {x_seq.shape=} {attention_mask.shape=}")
        z_anchor = self.encoder(x_anchor)
        if DEBUG:
            print(f"1. {z_anchor.shape=}")
        batch, seq, channel, height, width = x_seq.shape
        if DEBUG:
            print(f"1. {x_seq.shape=}")
        x = x_seq.view(batch * seq, channel, height, width)
        if DEBUG:
            print(f"2. {x.shape=}")
        z = self.encoder(x)
        if DEBUG:
            print(f"3. {z.shape=}")
        z = z.view(batch, seq, self.embed_dim)
        if DEBUG:
            print(f"4. {z.shape=}")
        z, _ = self.cross_attention(
            z_anchor.unsqueeze(1),  # Query
            z,  # Key
            z,  # Value
        )  # We pad with 0 instead of src_key_padding_mask=~attention_mask assuming the model would pick up that these are fake images
        if DEBUG:
            print(f"5. {z.shape=}")
        z = z.mean(dim=1)
        if DEBUG:
            print(f"6. {z.shape=}")
        z = F.relu(self.fc1(z))
        if DEBUG:
            print(f"7. {z.shape=}")
        z = self.fc2(z)
        if DEBUG:
            print(f"8. {z.shape=}")
        y_hat = F.sigmoid(z)
        return y_hat, z


class FullyConnectedModel(nn.Module):
    """
    Fully connected model for sequence classification (fixed input size).
    """

    def __init__(self):
        super().__init__()
        self.embed_dim = 128
        self.encoder = ImageEncoder(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim * MAX_IMG_PER_SEQ + self.embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x_anchor, x_seq, attention_mask):
        # attention_mask is not used, it is to be compatible with Transformer inputs
        z_anchor = self.encoder(x_anchor)
        batch, seq, channel, height, width = x_seq.shape
        if DEBUG:
            print(f"1. {x_seq.shape=}")
        x = x_seq.view(batch * seq, channel, height, width)
        if DEBUG:
            print(f"2. {x.shape=}")
        z = self.encoder(x)
        if DEBUG:
            print(f"3. {z.shape=}")
        z = z.view(batch, seq, self.embed_dim)
        if DEBUG:
            print(f"4. {z.shape=}")
        z = z.view(batch, -1)
        if DEBUG:
            print(f"5. {z.shape=}")
        z = torch.cat((z_anchor, z), dim=1)
        if DEBUG:
            print(f"6. {z.shape=}")
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        y_hat = F.sigmoid(z)
        return y_hat, z


# =============================
# Lightning Callbacks and Module
# =============================


class EpochMetricsCallback(L.Callback):
    """
    Callback to log and print metrics at the end of each validation epoch.
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        train_loss_mean, train_acc_mean = -1, -1
        if len(pl_module.training_step_loss_outputs) > 0:
            train_loss_mean = torch.stack(pl_module.training_step_loss_outputs).mean()
            train_acc_mean = torch.stack(
                pl_module.training_step_accuracy_outputs
            ).mean()
        val_loss_mean = torch.stack(pl_module.validation_step_loss_outputs).mean()
        val_acc_mean = torch.stack(pl_module.validation_step_accuracy_outputs).mean()
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        print(
            f"[Epoch {trainer.current_epoch}] Validation [Loss Acc]=[{val_loss_mean:.2f} {val_acc_mean:.2f}] Training [Loss Acc]=[{train_loss_mean:.2f} {train_acc_mean:.2f}] lr={lr:.2e}"
        )
        self.log("val_loss", val_loss_mean, on_step=False, on_epoch=True)
        self.log("val_accuracy", val_acc_mean, on_step=False, on_epoch=True)
        self.log("train_loss", train_loss_mean, on_step=False, on_epoch=True)
        self.log("train_accuracy", train_acc_mean, on_step=False, on_epoch=True)
        self.log("lr", lr, on_step=False, on_epoch=True)
        pl_module.training_step_loss_outputs.clear()
        pl_module.validation_step_loss_outputs.clear()


class LitModel(L.LightningModule):
    """
    PyTorch Lightning module for training and evaluation.
    """

    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.accuracy = Accuracy(task="binary")
        self.training_step_loss_outputs = []
        self.validation_step_loss_outputs = []
        self.training_step_accuracy_outputs = []
        self.validation_step_accuracy_outputs = []
        self.compute_loss = nn.BCEWithLogitsLoss()

    def get_random_seq_length(self, attention_mask):
        num_elements = len(attention_mask)
        random_index = torch.randint(0, num_elements, (1,))
        random_mask = attention_mask[random_index]
        random_seq_length = random_mask.sum(dim=1).item()
        return random_seq_length

    def forward(self, x_anchor, x_seq, attention_mask):
        y_hat, logits = self.model(x_anchor, x_seq, attention_mask)
        return y_hat.float().squeeze(), logits.float().squeeze()

    def training_step(self, batch, batch_idx):
        x_anchor, x_seq, attention_mask, y = batch
        y_hat, logits = self.forward(x_anchor, x_seq, attention_mask)
        loss = self.compute_loss(logits, y.float())
        accuracy_value = self.accuracy(y_hat, y)
        self.log(
            "train_random_seq_length",
            self.get_random_seq_length(attention_mask),
            on_step=True,
            on_epoch=False,
        )
        self.training_step_loss_outputs.append(loss)
        self.training_step_accuracy_outputs.append(accuracy_value)
        return loss

    def validation_step(self, batch, batch_idx):
        x_anchor, x_seq, attention_mask, y = batch
        y_hat, logits = self.forward(x_anchor, x_seq, attention_mask)
        loss = self.compute_loss(logits, y.float())
        accuracy_value = self.accuracy(y_hat, y)
        self.log(
            "val_random_seq_length",
            self.get_random_seq_length(attention_mask),
            on_step=True,
            on_epoch=False,
        )
        self.validation_step_loss_outputs.append(loss)
        self.validation_step_accuracy_outputs.append(accuracy_value)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_STEPS, gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_anchor, x_seq, attention_mask, y = batch
        y_hat, logits = self(x_anchor, x_seq, attention_mask)
        out_dir = self.logger.log_dir
        print(f"Logging to: {out_dir}")
        batch_size = x_seq.size(0)
        seq_len = x_seq.size(1)
        fig, axes = plt.subplots(
            batch_size, 2, figsize=(12, 2 * batch_size), squeeze=False
        )
        for i in range(batch_size):
            anchor_img = x_anchor[i].cpu()
            axes[i, 0].imshow(
                anchor_img.permute(1, 2, 0) * 0.5 + 0.5,
                cmap="gray" if anchor_img.size(0) == 1 else None,
            )
            axes[i, 0].set_title("Anchor Image")
            axes[i, 0].axis("off")
            seq_imgs = x_seq[i]
            grid = torchvision.utils.make_grid(seq_imgs.cpu(), nrow=seq_len, padding=2)
            axes[i, 1].imshow(grid.permute(1, 2, 0) * 0.5 + 0.5)
            axes[i, 1].set_title(f"Pred: {y_hat[i].item():.2f} | Label: {y[i].item()}")
            axes[i, 1].axis("off")
        plt.tight_layout()
        plt.savefig(osp.join(out_dir, f"predictions_{batch_idx}.png"))
        plt.close(fig)
        return y_hat


# =============================
# Main Training and Evaluation
# =============================


def main():
    """
    Main function to train, evaluate, and visualize the model and metrics.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mnist_anchor_in_seq_{now}_{MODEL_TYPE}"
    run_name = run_name + "_DEBUG" if DEBUG else run_name
    logger = CSVLogger("logs", name=run_name)

    train_set, val_set = data_preparation()
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        collate_fn=pad_sequences,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        collate_fn=pad_sequences,
    )

    # Initialize model
    model = TransformerModel() if MODEL_TYPE == "transformer" else FullyConnectedModel()
    lit_model = LitModel(model=model)

    # Train
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=1 if DEBUG else MAX_EPOCHS,
        accelerator=device,
        callbacks=[EpochMetricsCallback()],
        logger=logger,
        limit_train_batches=100 if DEBUG else None,
        limit_val_batches=10 if DEBUG else None,
        limit_test_batches=10 if DEBUG else None,
        limit_predict_batches=1,
    )

    print("Training model...")
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("Predicting model...")
    trainer.predict(lit_model, dataloaders=val_loader)

    # Visualize metrics over epochs
    out_dir = logger.log_dir
    log_df = pd.read_csv(osp.join(out_dir, "metrics.csv"))

    # Plot training and validation loss
    train_data = log_df.dropna(subset=["train_loss"])
    val_data = log_df.dropna(subset=["val_loss"])
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if len(train_data) > 0:
        ax.plot(
            train_data["epoch"],
            train_data["train_loss"],
            label="Training Loss",
            marker="o",
        )
    if len(val_data) > 0:
        ax.plot(
            val_data["epoch"], val_data["val_loss"], label="Validation Loss", marker="s"
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over Epochs")
    ax.legend()
    ax.grid(True)
    ax.set_yscale("log")
    ax.set_xlim(0, log_df["epoch"].max() + 1)
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, "loss_over_epochs.png"))
    plt.close(fig)

    # Plot training and validation accuracy
    train_data = log_df.dropna(subset=["train_accuracy"])
    val_data = log_df.dropna(subset=["val_accuracy"])
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if len(train_data) > 0:
        ax.plot(
            train_data["epoch"],
            train_data["train_accuracy"],
            label="Training Accuracy",
            marker="o",
        )
    if len(val_data) > 0:
        ax.plot(
            val_data["epoch"],
            val_data["val_accuracy"],
            label="Validation Accuracy",
            marker="s",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy over Epochs")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, log_df["epoch"].max() + 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, "accuracy_over_epochs.png"))
    plt.close(fig)

    # Plot histogram of sequence length
    train_data = log_df.dropna(subset=["train_random_seq_length"])
    val_data = log_df.dropna(subset=["val_random_seq_length"])
    bins = MAX_IMG_PER_SEQ
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    ax = axs[0]
    ax.hist(train_data["train_random_seq_length"], bins=bins, alpha=0.7, label="Train")
    ax.set_ylabel("Frequency")
    ax = axs[1]
    ax.hist(val_data["val_random_seq_length"], bins=bins, alpha=0.7, label="Validation")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Frequency")
    axs[0].set_title("Histogram of Sequence Lengths")
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, "histogram_sequence_length.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
