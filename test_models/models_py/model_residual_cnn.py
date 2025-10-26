"""
Model Name: Residual CNN

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels
    - k_h: Kernel height
    - k_w: Kernel width

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Conv2d (B, 1, 30, 10) -> (B, c0, 30, 10) with padding='same'
    - ReLU (B, c0, 30, 10) -> (B, c0, 30, 10)
    - Conv2d (B, c0, 30, 10) -> (B, c0, 30, 10) with padding='same'
    - Add residual connection via 1x1 conv on input
    - ReLU (B, c0, 30, 10) -> (B, c0, 30, 10)
    - GlobalAvgPool2d (B, c0, 30, 10) -> (B, c0)
    - Linear (B, c0) -> (B, 1)
"""

#
### Import Modules. ###
#
from torch import nn
from torch import Tensor


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(self, c0: int, k_h: int, k_w: int) -> None:

        #
        super().__init__()  # type: ignore

        #
        padding_h: int = k_h // 2
        #
        padding_w: int = k_w // 2
        #
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=(k_h, k_w), padding=(padding_h, padding_w))
        #
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=c0, out_channels=c0, kernel_size=(k_h, k_w), padding=(padding_h, padding_w))
        #
        self.conv_skip: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=1)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin: nn.Linear = nn.Linear(in_features=c0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.unsqueeze(1)
        #
        x_residual = self.conv_skip(x)
        #
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        #
        x = x + x_residual
        x = self.relu(x)
        #
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.lin(x)

        #
        return x
