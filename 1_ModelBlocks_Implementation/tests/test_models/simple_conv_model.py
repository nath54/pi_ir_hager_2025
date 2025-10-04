"""
Simple convolutional model for testing.
"""

import torch
import torch.nn as nn


class SimpleConvModel(nn.Module):
    """A simple convolutional model with conv layers, batch norm, and ReLU."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.linear = nn.Linear(8192, 10)

    def forward(self, x):

        print("0 | x.shape = ", x.shape)

        x = self.conv1(x)

        print("1 | x.shape = ", x.shape)

        x = self.bn1(x)

        print("2 | x.shape = ", x.shape)

        x = self.relu(x)

        print("3 | x.shape = ", x.shape)

        x = self.pool(x)

        print("4 | x.shape = ", x.shape)

        x = self.conv2(x)

        print("5 | x.shape = ", x.shape)

        x = self.bn2(x)

        print("6 | x.shape = ", x.shape)

        x = self.relu(x)

        print("7 | x.shape = ", x.shape)

        x = x.view(x.size(0), -1)

        print("8 | x.shape = ", x.shape)

        x = self.linear(x)

        print("9 | x.shape = ", x.shape)

        return x


# if __name__ == "__main__":
#     # Test the model
#     model = SimpleConvModel()
#     x = torch.randn(2, 3, 32, 32)
#     output = model(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Model: {model}")
