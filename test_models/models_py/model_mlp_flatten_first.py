"""
Model Name: MLP Flatten-First

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: First hidden dimension

Data variables:
    - B: Batch size

Architecture:
    - Flatten (B, 30, 10) -> (B, 300)
    - Linear (B, 300) -> (B, h0)
    - ReLU (B, h0) -> (B, h0)
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
        super().__init__()

        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin1: nn.Linear = nn.Linear(in_features=300, out_features=h0)
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
        X = self.flatten(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)

        #
        return X
