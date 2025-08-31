import lightning as L
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST


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
    train_set = MNIST(root="data", train=True, download=True, transform=train_transform)
    val_set = MNIST(root="data", train=False, download=True, transform=val_transform)
    return train_set, val_set


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc1(x)


class EpochMetricsCallback(L.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        
        # Train
        train_mean, train_acc = -1, -1
        if len(pl_module.training_step_outputs) > 0:
            train_mean = torch.stack(pl_module.training_step_outputs).mean()
            pl_module.log("training_epoch_mean", train_mean)
            train_acc = pl_module.train_accuracy.compute()
            pl_module.log("training_epoch_acc", train_acc)

        # Val
        val_mean = torch.stack(pl_module.validation_step_outputs).mean()
        pl_module.log("validation_epoch_mean", val_mean)
        val_acc = pl_module.val_accuracy.compute()
        pl_module.log("validation_epoch_acc", val_acc)

        lr = trainer.optimizers[0].param_groups[0]["lr"]

        print(
            f"[Epoch {trainer.current_epoch}] Validation [Acc Loss]=[{val_acc:.2f} {val_mean:.2f}] Training [Acc Loss]=[{train_acc:.2f} {train_mean:.2f}] lr={lr:.2e}"
        )

        pl_module.training_step_outputs.clear()
        pl_module.validation_step_outputs.clear()



class LitModel(L.LightningModule):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.train_accuracy.update(logits, y)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.validation_step_outputs.append(loss)
        self.val_accuracy.update(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


def main():
    # Dataset
    train_set, val_set = data_preparation()
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=9, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=9, persistent_workers=True)

    # Initiate model
    model = Net()
    lit_model = LitModel(model=model)
    
    # Train
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=20,
        accelerator=device,
        callbacks=[EpochMetricsCallback()],
    )
    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    main()
