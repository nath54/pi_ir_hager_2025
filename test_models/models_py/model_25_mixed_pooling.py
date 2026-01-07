"""
Model Name: Mixed Pooling (Avg + Max) with Depth Scaling

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden dimension
    - depth: Number of hidden layers

Data variables:
    - B: Batch size

Architecture:
    - GlobalAvgPool (B, 30, 10) -> (B, 10)
    - GlobalMaxPool (B, 30, 10) -> (B, 10)
    - Concatenate -> (B, 20)
    - Linear (B, 20) -> (B, h0)
    - ReLU (B, h0) -> (B, h0)
    - [Optional: Additional hidden layers based on depth]
    - Linear (B, h0) -> (B, 1)
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
    def __init__(self, h0: int = 16, depth: int = 1) -> None:

        #
        super().__init__()  # type: ignore

        #
        # self.global_avg_pool: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        #
        # self.global_max_pool: nn.AdaptiveMaxPool1d = nn.AdaptiveMaxPool1d(1)
        #
        self.lin1: nn.Linear = nn.Linear(in_features=20, out_features=h0)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.hidden_layers: nn.ModuleList = nn.ModuleList()
        #
        # Add additional hidden layers based on depth parameter
        for i in range(depth - 1):
            self.hidden_layers.append(nn.Linear(in_features=h0, out_features=h0))
            self.hidden_layers.append(nn.ReLU())
        #
        self.lin2: nn.Linear = nn.Linear(in_features=h0, out_features=1)
        #
        self.depth: int = depth
        self.h0: int = h0


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        # x_perm = x.permute(0, 2, 1)
        #
        # x_avg = self.global_avg_pool(x_perm)
        # x_avg = x_avg.squeeze(-1)
        x_avg = x.mean(dim=1)
        #
        # x_max = self.global_max_pool(x_perm)
        # x_max = x_max.squeeze(-1)
        x_max = x.max(dim=1).values
        #
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.lin1(x)
        x = self.relu(x)
        #
        # Apply additional hidden layers if depth > 1
        for layer in self.hidden_layers:
            x = layer(x)
        #
        x = self.lin2(x)

        #
        return x