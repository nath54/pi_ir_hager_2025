"""
Model Name: 2D Convolution with Depth Scaling

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - depth: Number of convolutional layers
    - c0: Number of output channels for first layer
    - growth_rate: Multiplier for channel growth per layer
    - k_h: Kernel height
    - k_w: Kernel width
    - p: Pool size

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Conv2d layers (depth number of layers)
        - Conv2d (B, prev_channels, H, W) -> (B, curr_channels, H-k_h+1, W-k_w+1)
        - ReLU
        - MaxPool2d (B, curr_channels, H-k_h+1, W-k_w+1) -> (B, curr_channels, (H-k_h+1)//p, (W-k_w+1)//p)
    - Flatten (B, final_channels * final_H * final_W) -> (B, final_channels * final_H * final_W)
    - Linear (B, final_channels * final_H * final_W) -> (B, 1)
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
    def __init__(self, depth: int = 3, c0: int = 8, growth_rate: float = 1.5, k_h: int = 3, k_w: int = 3, p: int = 2) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.depth: int = depth
        #
        self.c0: int = c0
        #
        self.growth_rate: float = growth_rate
        #
        self.k_h: int = k_h
        #
        self.k_w: int = k_w
        #
        self.p: int = p

        #
        self.conv_layers: nn.ModuleList = nn.ModuleList()
        #
        self.relu_layers: nn.ModuleList = nn.ModuleList()
        #
        self.pool_layers: nn.ModuleList = nn.ModuleList()

        #
        in_channels: int = 1
        #
        H: int = 30
        #
        W: int = 10

        #
        for i in range(depth):

            #
            out_channels: int = int(c0 * (growth_rate ** i))
            #
            conv_layer: nn.Conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(k_h, k_w),
                padding="same"
            )
            #
            relu_layer: nn.ReLU = nn.ReLU()
            #
            pool_layer: nn.MaxPool2d = nn.MaxPool2d(kernel_size=p)

            #
            self.conv_layers.append(conv_layer)
            #
            self.relu_layers.append(relu_layer)
            #
            self.pool_layers.append(pool_layer)

            #
            in_channels = out_channels
            #
            H = H // p
            W = W // p
            #
            # H = (H - k_h + 1) // p
            # W = (W - k_w + 1) // p

        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        final_features: int = in_channels * H * W
        #
        self.lin: nn.Linear = nn.Linear(in_features=final_features, out_features=1)


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
            x = self.relu_layers[i](x)
            x = self.pool_layers[i](x)

        #
        x = self.flatten(x)
        x = self.lin(x)

        #
        return x