import lightning as L
import torch
import torchvision
from lightning.pytorch.callbacks import Callback
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def data_preparation():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_set = MNIST(root="data", train=True, download=True, transform=transform)
    val_set = MNIST(root="data", train=False, download=True, transform=transform)
    return train_set, val_set


class PrintEpochCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(
            f"[{trainer.current_epoch}] Training Loss: {trainer.callback_metrics['train_loss']:.2f}"
        )


# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)  # flatten
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    # Dataset
    train_set, val_set = data_preparation()

    # Dataloader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Initiate model
    model = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 10))
    lit_model = LitModel(model=model)
    # Train
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=20,
        accelerator=device,
        callbacks=[PrintEpochCallback()],
    )
    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    main()
