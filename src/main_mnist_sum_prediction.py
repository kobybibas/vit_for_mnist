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
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import _Image_fromarray

DEBUG = True
MAX_IMG_PER_SEQ = 10
NUM_WORKERS = 9
MAX_EPOCHS = 30
LR_STEPS = [20, 28]


class MNISTSequence(MNIST):
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        # Deterministically set the sequence length
        seq_length = binascii.crc32(str(index).encode()) % MAX_IMG_PER_SEQ + 1

        start = index
        end = start + seq_length

        imgs, target = self.data[start:end], self.targets[start:end].sum().item()

        img_seq = []
        for img in imgs:
            img = _Image_fromarray(img.numpy(), mode="L")

            if self.transform is not None:
                img = self.transform(img)

            img_seq.append(img)
        img_seq = torch.stack(img_seq, dim=0)  # (IMG_PER_SEQ, 1, 28, 28)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_seq, target

    def __len__(self) -> int:
        return len(self.data) - (MAX_IMG_PER_SEQ - 1)


def collate_fn(batch):
    # Pad sequences to max length in batch, use attention masks
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)

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

    return torch.stack(padded_seqs), torch.stack(attention_masks), torch.tensor(labels)


def data_preparation():
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


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1024, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Take sequence -> encoder of each image -> self attention -> relu
        self.encoder = ImageEncoder()

        self.embed_dim = 128
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.embed_dim, nhead=4, dropout=0.0, batch_first=True
            ),
            1,
        )

        self.fc1 = nn.Linear(self.embed_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x_seq, attention_mask):
        # Reshape to have a virtual larger batch, (B * IMG_PER_SEQ, C, H, W) then reshape again to feed the transformer
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
        z = self.self_attention(
            z  # , src_key_padding_mask=~attention_mask TODO: we pad with 0, assuming the model pick up that these are fake images.
        )  # Invert mask for transformer (True = ignore)
        if DEBUG:
            print(f"5. {z.shape=}, {attention_mask.shape=} {attention_mask=}")

        masked_encoded = z * attention_mask.unsqueeze(-1)
        z = masked_encoded.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        if DEBUG:
            print(f"6. {z.shape=}")
        z = F.relu(self.fc1(z))
        if DEBUG:
            print(f"7. {z.shape=}")
        z = F.relu(self.fc2(z))
        if DEBUG:
            print(f"8. {z.shape=}")
        return z


class EpochMetricsCallback(L.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Train
        train_loss_mean, train_acc_mean = -1, -1
        if len(pl_module.training_step_loss_outputs) > 0:
            train_loss_mean = torch.stack(pl_module.training_step_loss_outputs).mean()
            train_acc_mean = torch.stack(
                pl_module.training_step_accuracy_outputs
            ).mean()

        # Val
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
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.training_step_loss_outputs = []
        self.validation_step_loss_outputs = []
        self.training_step_accuracy_outputs = []
        self.validation_step_accuracy_outputs = []

    def compute_accuracy(self, y, y_hat):
        preds = y_hat.round()
        return (preds == y).float().mean()

    def get_random_seq_length(self, attention_mask):
        num_elements = len(attention_mask)
        random_index = torch.randint(0, num_elements, (1,))
        random_mask = attention_mask[random_index]
        random_seq_length = random_mask.sum(dim=1).item()
        return random_seq_length

    def forward(self, x, attention_mask):
        y_hat = self.model(x, attention_mask)
        return y_hat.float().squeeze()

    def training_step(self, batch, batch_idx):
        x, attention_mask, y = batch
        y_hat = self.forward(x, attention_mask)
        loss = nn.functional.mse_loss(y_hat, y.float())
        accuracy = self.compute_accuracy(y, y_hat)
        self.log(
            "train_random_seq_length",
            self.get_random_seq_length(attention_mask),
            on_step=True,
            on_epoch=False,
        )
        self.training_step_loss_outputs.append(loss)
        self.training_step_accuracy_outputs.append(accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, attention_mask, y = batch
        y_hat = self.forward(x, attention_mask)
        loss = nn.functional.mse_loss(y_hat, y.float())
        accuracy = self.compute_accuracy(y, y_hat)
        self.log(
            "val_random_seq_length",
            self.get_random_seq_length(attention_mask),
            on_step=True,
            on_epoch=False,
        )
        self.validation_step_loss_outputs.append(loss)
        self.validation_step_accuracy_outputs.append(accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_STEPS, gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, attention_mask, y = batch
        y_hat = self(x, attention_mask)

        out_dir = self.logger.log_dir
        print(f"Logging to: {out_dir}")
        imgs = x  # (B, IMG_PER_SEQ, 1, 28, 28)

        batch_size = imgs.size(0)

        fig, axes = plt.subplots(batch_size, 1, figsize=(12, 2 * batch_size))

        if batch_size == 1:
            axes = [axes]  # keep iterable

        for i in range(batch_size):
            seq_imgs = imgs[i]  # [seq_len, 1, 28, 28]

            grid = torchvision.utils.make_grid(seq_imgs.cpu(), nrow=seq_imgs.size(0))
            axes[i].imshow(grid.permute(1, 2, 0) * 0.5 + 0.5)
            axes[i].set_title(f"Pred: {y_hat[i].item():.2f} | Label: {y[i].item()}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(osp.join(out_dir, f"predictions_{batch_idx}.png"))
        plt.close(fig)

        return y_hat


def main():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger("logs", name=now)
    train_set, val_set = data_preparation()
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    # Initiate model
    model = TransformerModel()
    lit_model = LitModel(model=model)

    # Train
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=1 if DEBUG else MAX_EPOCHS,
        accelerator=device,
        callbacks=[EpochMetricsCallback()],
        logger=logger,
        # Fast dev mode
        limit_train_batches=100 if DEBUG else None,
        limit_val_batches=10 if DEBUG else None,
        limit_test_batches=10 if DEBUG else None,
        # Visualize a single batch
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

    # ----------------------------- #
    # Visualize metrics over epochs #
    # ----------------------------- #
    out_dir = logger.log_dir
    log_df = pd.read_csv(osp.join(out_dir, "metrics.csv"))

    # Plot training loss (skip NaN values)
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

    # Plot validation loss (skip NaN values)
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

    # Accuracy
    train_data = log_df.dropna(subset=["train_accuracy"])
    val_data = log_df.dropna(subset=["val_accuracy"])
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot training accuracy (skip NaN values)
    if len(train_data) > 0:
        ax.plot(
            train_data["epoch"],
            train_data["train_accuracy"],
            label="Training Accuracy",
            marker="o",
        )

    # Plot validation accuracy (skip NaN values)
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(train_data["train_random_seq_length"], bins=bins, alpha=0.7, label="Train")
    ax.hist(val_data["val_random_seq_length"], bins=bins, alpha=0.7, label="Validation")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Sequence Lengths")
    ax.legend()
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, "histogram_sequence_length.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
