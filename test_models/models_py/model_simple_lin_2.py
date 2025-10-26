"""
Model Name: Simple Linear with 2 hidden dim

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: First Hidden Dimension
    - h1: Second Hidden Dimension

Data variables:
    - B: Batch size

Architecture:
    - Linear (B, 30, 10) -> (B, 30, h0)
    - ReLU (B, 30, h0) -> (B, 30, h0)
    - Flatten (B, 30, h0) -> (B, 30 * h0)
    - Linear (B, 30 * h0) -> (B, h1)
    - ReLU (B, h1) -> (B, h1)
    - Linear (B, h1) -> (B, 1)
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
    def __init__(self, h0: int, h1: int) -> None:

        #
        super().__init__()

        #
        self.lin1: nn.Linear = nn.Linear(in_features=10, out_features=h0)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin2: nn.Linear = nn.Linear(in_features=30 * h0, out_features=h1)
        #
        self.lin3: nn.Linear = nn.Linear(in_features=h1, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, X: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        X = self.lin1(X)
        X = self.relu(X)
        X = self.flatten(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)

        #
        return X

