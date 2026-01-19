"""
Model Name: Global Average Pooling

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden dimension
    - depth: Number of hidden layers

Data variables:
    - B: Batch size

Architecture:
    - GlobalAvgPool (B, 30, 10) -> (B, 10)
    - Linear (B, 10) -> (B, h0)
    - ReLU (B, h0) -> (B, h0)
    - [Linear (B, h0) -> (B, h0) -> ReLU (B, h0) for depth - 1 times]
    - Linear (B, h0) -> (B, 1)
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
    def __init__(self, h0: int = 16, depth: int = 2) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.global_pool: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        #
        self.lin1: nn.Linear = nn.Linear(in_features=10, out_features=h0)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.hidden_layers: nn.ModuleList = nn.ModuleList()
        #
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.Linear(in_features=h0, out_features=h0))
            self.hidden_layers.append(nn.ReLU())
        #
        self.lin_final: nn.Linear = nn.Linear(in_features=h0, out_features=1)
        #
        self.depth: int = depth
        #
        self.h0: int = h0


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.permute(0, 2, 1)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.lin1(x)
        x = self.relu(x)
        #
        for layer in self.hidden_layers:
            x = layer(x)
        #
        x = self.lin_final(x)

        #
        return x