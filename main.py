import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


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


def main():
    print("Hello from vit-for-mnist!")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    a = torch.tensor([1, 2, 3]).to(device)
    b = torch.tensor([4, 5, 6]).to(device)
    c = a + b
    print(c)

    # Dataset
    train_set, val_set = data_preparation()

    # Dataloader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)


if __name__ == "__main__":
    main()
