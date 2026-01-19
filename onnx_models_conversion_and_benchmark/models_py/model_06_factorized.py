"""
Model Name: Factorized Architecture

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Intermediate dimension for feature compression
    - depth: Number of depth layers to stack

Data variables:
    - B: Batch size

Architecture:
    - Feature compression block repeated 'depth' times
        - Linear along features (B, 30, 10) -> (B, 30, h0)
        - ReLU (B, 30, h0) -> (B, 30, h0)
        - Permute (B, 30, h0) -> (B, h0, 30)
        - Linear along time (B, h0, 30) -> (B, h0, 1)
        - ReLU (B, h0, 1) -> (B, h0, 1)
        - Squeeze (B, h0, 1) -> (B, h0)
    - Final Linear (B, h0) -> (B, 1)
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
    def __init__(self, h0: int = 8, depth: int = 1) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.h0: int = h0
        #
        self.depth: int = depth

        #
        self.feature_blocks: nn.ModuleList = nn.ModuleList()
        #
        for i in range(depth):
            #
            feature_block: nn.Sequential = nn.Sequential(
                nn.Linear(in_features=10 if i == 0 else h0, out_features=h0),
                nn.ReLU(),
                nn.Linear(in_features=30, out_features=1),
                nn.ReLU(),
            )
            #
            self.feature_blocks.append(feature_block)

        #
        self.final_lin: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        for i, block in enumerate(self.feature_blocks):
            #
            x = block[0](x)  # Linear along features
            x = block[1](x)  # ReLU
            #
            x = x.permute(0, 2, 1)  # Permute for time processing
            x = block[2](x)  # Linear along time
            x = block[3](x)  # ReLU
            #
            x = x.squeeze(-1)  # Squeeze time dimension
            #
            if i < len(self.feature_blocks) - 1:  # Don't apply final squeeze on last block
                x = x.unsqueeze(-1)  # Add dimension back for next block
                x = x.permute(0, 2, 1)  # Restore shape for next feature processing

        #
        x = self.final_lin(x)

        #
        return x