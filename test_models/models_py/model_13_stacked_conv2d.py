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
    - depth: Number of conv layers in the stack

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - For i in range(depth):
        - Conv2d (B, cin, Hin, Win) -> (B, cout, Hin-k+1, Win-k+1)
        - ReLU (B, cout, Hin-k+1, Win-k+1) -> (B, cout, Hin-k+1, Win-k+1)
        - MaxPool2d (B, cout, Hin-k+1, Win-k+1) -> (B, cout, Hout, Wout)
    - GlobalAvgPool2d (B, c_final, H, W) -> (B, c_final)
    - Linear (B, c_final) -> (B, 1)
"""

#
### Import Modules. ###
#
from torch import nn
from torch import Tensor
from typing import List


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(self, c0: int = 8, c1: int = 16, k0: int = 3, k1: int = 3, p: int = 2, depth: int = 2) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.depth: int = depth
        #
        self.conv_layers: nn.ModuleList = nn.ModuleList()
        #
        self.relu_layers: nn.ModuleList = nn.ModuleList()
        #
        self.pool_layers: nn.ModuleList = nn.ModuleList()

        #
        ### Build layers based on depth. ###
        #
        in_channels: int = 1
        current_channels: int = c0

        #
        for i in range(depth):
            #
            if i == 0:
                #
                out_channels: int = c0
            #
            elif i == depth - 1:
                #
                out_channels = c1
            #
            else:
                #
                out_channels = max(c0, c1 // 2)  # Intermediate layers use mid-size channels
            #
            self.conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k0 if i == 0 else k1))
            #
            self.relu_layers.append(nn.ReLU())
            #
            self.pool_layers.append(nn.MaxPool2d(kernel_size=p))
            #
            in_channels = out_channels


        #
        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin: nn.Linear = nn.Linear(in_features=in_channels, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.unsqueeze(1)

        #
        for i in range(self.depth):
            #
            x = self.conv_layers[i](x)
            #
            x = self.relu_layers[i](x)
            #
            x = self.pool_layers[i](x)

        #
        x = self.global_pool(x)
        #
        x = self.flatten(x)
        #
        x = self.lin(x)

        #
        return x