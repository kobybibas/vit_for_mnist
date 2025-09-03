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

DEBUG = False
NUM_IMG_PER_SEQ = 3
NUM_WORKERS = 9
MAX_EPOCHS = 50
LR_STEPS = [20, 30]


class MNISTSequence(MNIST):
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        start = index
        end = start + NUM_IMG_PER_SEQ

        imgs, target = self.data[start:end], self.targets[start:end].sum().item()

        img_seq = []
        for img in imgs:
            img = _Image_fromarray(img.numpy(), mode="L")

            if self.transform is not None:
                img = self.transform(img)

            img_seq.append(img)
        img_seq = torch.stack(img_seq, dim=0)  # (NUM_IMG_PER_SEQ, 1, 28, 28)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_seq, target

    def __len__(self) -> int:
        # ensure every index has NUM_IMG_PER_SEQ images available
        return len(self.data) - (NUM_IMG_PER_SEQ - 1)


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

        self.self_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, dropout=0.0, batch_first=True
        )
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x_seq):
        # Reshape to have a virtual larger batch, (B * NUM_IMG_PER_SEQ, C, H, W) then reshape again to feed the transformer
        batch, seq, channel, height, width = x_seq.shape
        if DEBUG:
            print(f"1. {x_seq.shape=}")
        x = x_seq.view(batch * seq, channel, height, width)
        if DEBUG:
            print(f"2. {x.shape=}")
        z = self.encoder(x)
        if DEBUG:
            print(f"3. {z.shape=}")
        z = z.view(batch, seq, 128)
        if DEBUG:
            print(f"4. {z.shape=}")
        z, _ = self.self_attention(z, z, z)
        if DEBUG:
            print(f"5. {z.shape=}")
        z = z.mean(dim=1)
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

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat.float().squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y.float())
        accuracy = self.compute_accuracy(y, y_hat)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True)
        self.training_step_loss_outputs.append(loss)
        self.training_step_accuracy_outputs.append(accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y.float())
        accuracy = self.compute_accuracy(y, y_hat)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)
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
        x, y = batch
        y_hat = self(x)

        out_dir = self.logger.log_dir
        print(f"Logging to: {out_dir}")
        imgs = x  # (B, NUM_IMG_PER_SEQ, 1, 28, 28)

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
    )
    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
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
        limit_train_batches=1 if DEBUG else None,
        limit_val_batches=1 if DEBUG else None,
        limit_test_batches=1 if DEBUG else None,
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


if __name__ == "__main__":
    main()
