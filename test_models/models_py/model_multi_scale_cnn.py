"""
Model Name: Multi-Scale CNN

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels per branch

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Branch1: Conv2d kernel=(3,3) (B, 1, 30, 10) -> (B, c0, 28, 8)
    - Branch2: Conv2d kernel=(5,5) (B, 1, 30, 10) -> (B, c0, 26, 6)
    - Branch3: Conv2d kernel=(7,7) (B, 1, 30, 10) -> (B, c0, 24, 4)
    - ReLU on each branch
    - GlobalAvgPool2d on each branch -> (B, c0) each
    - Concatenate (B, c0) + (B, c0) + (B, c0) -> (B, 3*c0)
    - Linear (B, 3*c0) -> (B, 1)
"""

#
### Import Modules. ###
#
from torch import nn
from torch import Tensor
import torch


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(self, c0: int) -> None:

        #
        super().__init__()

        #
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=3)
        #
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=5)
        #
        self.conv3: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=7)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin: nn.Linear = nn.Linear(in_features=3 * c0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, X: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        X = X.unsqueeze(1)
        #
        X1 = self.relu(self.conv1(X))
        X1 = self.global_pool(X1)
        X1 = self.flatten(X1)
        #
        X2 = self.relu(self.conv2(X))
        X2 = self.global_pool(X2)
        X2 = self.flatten(X2)
        #
        X3 = self.relu(self.conv3(X))
        X3 = self.global_pool(X3)
        X3 = self.flatten(X3)
        #
        X = torch.cat([X1, X2, X3], dim=1)
        X = self.lin(X)

        #
        return X
