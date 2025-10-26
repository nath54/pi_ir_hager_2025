"""
Model Name: Global Statistics Extractor

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden dimension

Data variables:
    - B: Batch size

Architecture:
    - Compute statistics:
      - Mean along time (B, 30, 10) -> (B, 10)
      - Std along time (B, 30, 10) -> (B, 10)
      - Max along time (B, 30, 10) -> (B, 10)
      - Min along time (B, 30, 10) -> (B, 10)
    - Concatenate all statistics -> (B, 40)
    - Linear (B, 40) -> (B, h0)
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
        self.lin1: nn.Linear = nn.Linear(in_features=40, out_features=h0)
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
        X_mean = X.mean(dim=1)
        #
        X_std = X.std(dim=1)
        #
        X_max = X.max(dim=1)[0]
        #
        X_min = X.min(dim=1)[0]
        #
        X = torch.cat([X_mean, X_std, X_max, X_min], dim=1)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)

        #
        return X
