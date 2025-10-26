"""
Model Name: Stacked Conv2D

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of channels in first layer
    - c1: Number of channels in second layer
    - k0: Kernel size for first layer
    - k1: Kernel size for second layer
    - p: Pool size

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Conv2d (B, 1, 30, 10) -> (B, c0, 30-k0+1, 10-k0+1)
    - ReLU (B, c0, 30-k0+1, 10-k0+1) -> (B, c0, 30-k0+1, 10-k0+1)
    - MaxPool2d (B, c0, 30-k0+1, 10-k0+1) -> (B, c0, H1, W1)
    - Conv2d (B, c0, H1, W1) -> (B, c1, H1-k1+1, W1-k1+1)
    - ReLU (B, c1, H1-k1+1, W1-k1+1) -> (B, c1, H1-k1+1, W1-k1+1)
    - GlobalAvgPool2d (B, c1, H1-k1+1, W1-k1+1) -> (B, c1)
    - Linear (B, c1) -> (B, 1)
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
    def __init__(self, c0: int, c1: int, k0: int, k1: int, p: int) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=k0)
        #
        self.relu1: nn.ReLU = nn.ReLU()
        #
        self.maxpool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=p)
        #
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=c0, out_channels=c1, kernel_size=k1)
        #
        self.relu2: nn.ReLU = nn.ReLU()
        #
        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin: nn.Linear = nn.Linear(in_features=c1, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.lin(x)

        #
        return x
