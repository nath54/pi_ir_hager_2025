"""
Simple convolutional model for testing.
"""

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer: nn.MaxPool2d = nn.MaxPool2d(kernel_size=3)

    def forward(self, x):

        x = self.layer(x)

        return x


# if __name__ == "__main__":
#     # Test the model
#     model = SimpleConvModel()
#     x = torch.randn(2, 3, 32, 32)
#     output = model(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Model: {model}")
