"""
Simple linear model for testing.
"""

import torch
import torch.nn as nn


class SimpleLinearModel(nn.Module):
    """A simple linear model with one linear layer and ReLU activation."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=10, out_features=5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=5, out_features=25)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# if __name__ == "__main__":
#     # Test the model
#     model = SimpleLinearModel()
#     x = torch.randn(2, 10)
#     output = model(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Model: {model}")
