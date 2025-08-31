from typing import Any

import lightning as L
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import _Image_fromarray

DEBUG = True
NUM_IMG_PER_SEQ = 3
NUM_WORKERS = 9
MAX_EPOCHS = 20 if DEBUG is False else 1


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
        train_loss_mean = -1
        if len(pl_module.training_step_outputs) > 0:
            train_loss_mean = torch.stack(pl_module.training_step_outputs).mean()
            pl_module.log("training_epoch_mean", train_loss_mean)

        # Val
        val_loss_mean = torch.stack(pl_module.validation_step_outputs).mean()
        pl_module.log("validation_epoch_mean", val_loss_mean)

        lr = trainer.optimizers[0].param_groups[0]["lr"]

        print(
            f"[Epoch {trainer.current_epoch}] Validation [Loss]=[{val_loss_mean:.2f}] Training [Loss]=[{train_loss_mean:.2f}] lr={lr:.2e}"
        )

        pl_module.training_step_outputs.clear()
        pl_module.validation_step_outputs.clear()


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.float().squeeze()
        loss = nn.functional.mse_loss(y_hat, y.float())
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.float().squeeze()
        loss = nn.functional.mse_loss(y_hat, y.float())
        self.validation_step_outputs.append(loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     imgs, labels = batch  # imgs: [B, seq_len, 1, 28, 28], labels: [B]
    #     preds = self(batch).argmax(dim=-1)  # [B]

    #     batch_size = imgs.size(0)

    #     fig, axes = plt.subplots(batch_size, 1, figsize=(12, 2 * batch_size))

    #     if batch_size == 1:
    #         axes = [axes]  # keep iterable

    #     for i in range(batch_size):
    #         seq_imgs = imgs[i]  # [seq_len, 1, 28, 28]

    #         grid = torchvision.utils.make_grid(
    #             seq_imgs.cpu(), nrow=seq_imgs.size(0)
    #         )
    #         axes[i].imshow(grid.permute(1, 2, 0) * 0.5 + 0.5)
    #         axes[i].set_title(f"Pred: {preds[i].item()} | Label: {labels[i].item()}")
    #         axes[i].axis("off")

    #     plt.tight_layout()
    #     plt.show()

    #     return preds



def main():
    # Dataset
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
        max_epochs=MAX_EPOCHS,
        accelerator=device,
        callbacks=[EpochMetricsCallback()],
    )

    print("Training model...")
    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    print("Testing model...") # Todo, move to the end
    predictions = trainer.predict(lit_model, val_loader)


if __name__ == "__main__":
    main()
