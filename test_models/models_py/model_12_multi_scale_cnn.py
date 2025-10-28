"""
Model Name: Multi-Scale CNN with Depth Scaling

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels per branch
    - depth: Number of convolutional layers per branch

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Branch1: Conv2d kernel=(3,3) repeated 'depth' times (B, 1, 30, 10) -> (B, c0, H1, W1)
    - Branch2: Conv2d kernel=(5,5) repeated 'depth' times (B, 1, 30, 10) -> (B, c0, H2, W2)
    - Branch3: Conv2d kernel=(7,7) repeated 'depth' times (B, 1, 30, 10) -> (B, c0, H3, W3)
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
    def __init__(self, c0: int = 4, depth: int = 1) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.c0: int = c0
        #
        self.depth: int = depth
        #
        self.conv1_branch: nn.ModuleList = self._make_conv_branch(kernel_size=3, out_channels=c0, depth=depth)
        #
        self.conv2_branch: nn.ModuleList = self._make_conv_branch(kernel_size=5, out_channels=c0, depth=depth)
        #
        self.conv3_branch: nn.ModuleList = self._make_conv_branch(kernel_size=7, out_channels=c0, depth=depth)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin: nn.Linear = nn.Linear(in_features=3 * c0, out_features=1)


    #
    ### Helper method to create convolutional branch. ###
    #
    def _make_conv_branch(self, kernel_size: int, out_channels: int, depth: int) -> nn.ModuleList:

        #
        layers: list[nn.Conv2d] = []
        #
        in_channels: int = 1
        #
        for i in range(depth):

            #
            conv_layer: nn.Conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=0
            )
            #
            layers.append(conv_layer)
            #
            in_channels = out_channels

        #
        return nn.ModuleList(layers)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.unsqueeze(1)
        #
        x1 = x
        #
        for conv in self.conv1_branch:
            x1 = self.relu(conv(x1))
        #
        x1 = self.global_pool(x1)
        x1 = self.flatten(x1)
        #
        x2 = x
        #
        for conv in self.conv2_branch:
            x2 = self.relu(conv(x2))
        #
        x2 = self.global_pool(x2)
        x2 = self.flatten(x2)
        #
        x3 = x
        #
        for conv in self.conv3_branch:
            x3 = self.relu(conv(x3))
        #
        x3 = self.global_pool(x3)
        x3 = self.flatten(x3)
        #
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.lin(x)

        #
        return x
