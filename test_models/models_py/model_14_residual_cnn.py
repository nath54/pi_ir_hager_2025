"""
Model Name: Residual CNN

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels
    - k_h: Kernel height
    - k_w: Kernel width
    - depth: Number of residual blocks

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Conv2d (B, 1, 30, 10) -> (B, c0, 30, 10) with padding='same'
    - ReLU (B, c0, 30, 10) -> (B, c0, 30, 10)
    For each residual block:
        - Conv2d (B, c0, 30, 10) -> (B, c0, 30, 10) with padding='same'
        - ReLU (B, c0, 30, 10) -> (B, c0, 30, 10)
        - Conv2d (B, c0, 30, 10) -> (B, c0, 30, 10) with padding='same'
        - Add residual connection
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
    def __init__(self, c0: int = 8, k_h: int = 3, k_w: int = 3, depth: int = 2) -> None:

        #
        super().__init__()  # type: ignore

        #
        padding_h: int = k_h // 2
        #
        padding_w: int = k_w // 2
        #
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=(k_h, k_w), padding=(padding_h, padding_w))
        #
        self.relu: nn.ReLU = nn.ReLU()

        #
        residual_blocks: list[nn.Module] = []
        #
        for _ in range(depth):
            #
            block: nn.Sequential = nn.Sequential(
                nn.Conv2d(in_channels=c0, out_channels=c0, kernel_size=(k_h, k_w), padding=(padding_h, padding_w)),
                nn.ReLU(),
                nn.Conv2d(in_channels=c0, out_channels=c0, kernel_size=(k_h, k_w), padding=(padding_h, padding_w)),
                nn.ReLU()
            )
            #
            residual_blocks.append(block)
        #
        self.residual_blocks: nn.ModuleList = nn.ModuleList(residual_blocks)
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
        x = self.conv1(x)
        x = self.relu(x)
        #
        for block in self.residual_blocks:
            #
            x_residual = x
            x = block(x)
            x = x + x_residual
        #
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.lin(x)

        #
        return x
