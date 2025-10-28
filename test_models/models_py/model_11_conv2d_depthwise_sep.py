"""
Model Name: Depthwise Separable Conv2D with Depth Scaling

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - k_h: Kernel height
    - k_w: Kernel width
    - c0: Number of output channels for first layer
    - p: Pool size
    - depth: Number of depthwise separable blocks to stack
    - channel_multiplier: Multiplier for channel growth across depth

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - [DepthwiseConv2d + PointwiseConv2d + ReLU + MaxPool2d] x depth
    - Flatten
    - Linear (B, final_features) -> (B, 1)
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
    def __init__(self, k_h: int = 3, k_w: int = 3, c0: int = 8, p: int = 2, depth: int = 1, channel_multiplier: int = 2) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.depth: int = depth
        #
        self.channel_multiplier: int = channel_multiplier
        #
        self.k_h: int = k_h
        #
        self.k_w: int = k_w
        #
        self.p: int = p
        #
        self.c0: int = c0

        #
        self.blocks: nn.ModuleList = nn.ModuleList()

        #
        in_channels: int = 1
        #
        current_h: int = 30
        #
        current_w: int = 10

        #
        for i in range(depth):

            #
            out_channels: int = c0 * (channel_multiplier ** i) if i > 0 else c0

            #
            depthwise: nn.Conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(k_h, k_w),
                groups=in_channels,
                padding=0
            )
            #
            pointwise: nn.Conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
            #
            relu: nn.ReLU = nn.ReLU()
            #
            maxpool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=p)

            #
            block: nn.Sequential = nn.Sequential(
                depthwise,
                pointwise,
                relu,
                maxpool
            )

            #
            self.blocks.append(block)

            #
            in_channels = out_channels
            #
            current_h = (current_h - k_h + 1) // p
            #
            current_w = (current_w - k_w + 1) // p

        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        final_features: int = in_channels * current_h * current_w
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
        for block in self.blocks:
            x = block(x)

        #
        x = self.flatten(x)
        x = self.lin(x)

        #
        return x