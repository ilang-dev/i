import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


# Extremely simple MNIST MLP:
# input 28x28 image -> flatten to 784
# 784 -> 128 -> ReLU -> 64 -> ReLU -> 10 logits

BATCH_SIZE = 64
EPOCHS = 3
LR = 0.1


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    transform = transforms.ToTensor()  # pixels become floats in [0, 1]

    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)

        print("epoch", epoch + 1, "loss", total_loss / len(train_data))

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()

    print("test accuracy", correct / len(test_data))

    torch.save(model.state_dict(), "mnist_mlp.pt")

    # Also save raw arrays, useful for reimplementing inference elsewhere.
    np.savez(
        "mnist_mlp_weights.npz",
        fc1_weight=model.fc1.weight.detach().numpy(),
        fc1_bias=model.fc1.bias.detach().numpy(),
        fc2_weight=model.fc2.weight.detach().numpy(),
        fc2_bias=model.fc2.bias.detach().numpy(),
        fc3_weight=model.fc3.weight.detach().numpy(),
        fc3_bias=model.fc3.bias.detach().numpy(),
    )


if __name__ == "__main__":
    main()
