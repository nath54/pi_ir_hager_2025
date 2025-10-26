"""
Model Name: Factorized Architecture

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Intermediate dimension for feature compression

Data variables:
    - B: Batch size

Architecture:
    - Linear along features (B, 30, 10) -> (B, 30, h0)
    - ReLU (B, 30, h0) -> (B, 30, h0)
    - Permute (B, 30, h0) -> (B, h0, 30)
    - Linear along time (B, h0, 30) -> (B, h0, 1)
    - ReLU (B, h0, 1) -> (B, h0, 1)
    - Squeeze (B, h0, 1) -> (B, h0)
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
    def __init__(self, h0: int) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.lin1: nn.Linear = nn.Linear(in_features=10, out_features=h0)
        #
        self.relu1: nn.ReLU = nn.ReLU()
        #
        self.lin2: nn.Linear = nn.Linear(in_features=30, out_features=1)
        #
        self.relu2: nn.ReLU = nn.ReLU()
        #
        self.lin3: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = self.lin1(x)
        x = self.relu1(x)
        #
        x = x.permute(0, 2, 1)
        x = self.lin2(x)
        x = self.relu2(x)
        #
        x = x.squeeze(-1)
        x = self.lin3(x)

        #
        return x
