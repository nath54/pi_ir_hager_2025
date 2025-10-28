"""
Model Name: Global Statistics Extractor

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden dimension
    - depth: Number of hidden layers

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
    - [Additional hidden layers if depth > 1]
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
        self.h0: int = h0
        #
        self.depth: int = depth
        #
        self.lin1: nn.Linear = nn.Linear(in_features=40, out_features=h0)
        #
        self.relu: nn.ReLU = nn.ReLU()

        #
        if depth > 1:
            #
            self.hidden_layers: nn.ModuleList = nn.ModuleList()
            #
            for _ in range(depth - 1):
                #
                self.hidden_layers.append(nn.Linear(in_features=h0, out_features=h0))
                #
                self.hidden_layers.append(nn.ReLU())
        #
        self.lin2: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x_mean = x.mean(dim=1)
        #
        x_std = x.std(dim=1)
        #
        x_max = x.max(dim=1)[0]
        #
        x_min = x.min(dim=1)[0]
        #
        x = torch.cat([x_mean, x_std, x_max, x_min], dim=1)
        x = self.lin1(x)
        x = self.relu(x)

        #
        if self.depth > 1:
            #
            for layer in self.hidden_layers:
                #
                x = layer(x)
        x = self.lin2(x)

        #
        return x