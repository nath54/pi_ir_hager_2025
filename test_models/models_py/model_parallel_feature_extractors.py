"""
Model Name: Parallel Feature Extractors

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden dimension

Data variables:
    - B: Batch size

Architecture:
    - Time-wise MaxPool (B, 30, 10) -> (B, 10)
    - Feature-wise MaxPool (B, 30, 10) -> (B, 30)
    - Concatenate -> (B, 40)
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
        super().__init__()  # type: ignore

        #
        self.time_pool: nn.AdaptiveMaxPool1d = nn.AdaptiveMaxPool1d(1)
        #
        self.feature_pool: nn.AdaptiveMaxPool1d = nn.AdaptiveMaxPool1d(1)
        #
        self.lin1: nn.Linear = nn.Linear(in_features=40, out_features=h0)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.lin2: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x_time = x.permute(0, 2, 1)
        x_time = self.time_pool(x_time)
        x_time = x_time.squeeze(-1)
        #
        x_feature = self.feature_pool(x)
        x_feature = x_feature.squeeze(-1)
        #
        x = torch.cat([x_time, x_feature], dim=1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        #
        return x
