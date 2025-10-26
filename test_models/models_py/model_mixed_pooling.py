"""
Model Name: Mixed Pooling (Avg + Max)

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden dimension

Data variables:
    - B: Batch size

Architecture:
    - GlobalAvgPool (B, 30, 10) -> (B, 10)
    - GlobalMaxPool (B, 30, 10) -> (B, 10)
    - Concatenate -> (B, 20)
    - Linear (B, 20) -> (B, h0)
    - ReLU (B, h0) -> (B, h0)
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
    def __init__(self, h0: int) -> None:

        #
        super().__init__()

        #
        self.global_avg_pool: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        #
        self.global_max_pool: nn.AdaptiveMaxPool1d = nn.AdaptiveMaxPool1d(1)
        #
        self.lin1: nn.Linear = nn.Linear(in_features=20, out_features=h0)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.lin2: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, X: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        X_perm = X.permute(0, 2, 1)
        #
        X_avg = self.global_avg_pool(X_perm)
        X_avg = X_avg.squeeze(-1)
        #
        X_max = self.global_max_pool(X_perm)
        X_max = X_max.squeeze(-1)
        #
        X = torch.cat([X_avg, X_max], dim=1)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)

        #
        return X
