"""
Model Name: Single-Layer Perceptron

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - None

Data variables:
    - B: Batch size

Architecture:
    - Flatten (B, 30, 10) -> (B, 300)
    - Linear (B, 300) -> (B, 1)
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
    def __init__(self) -> None:

        #
        super().__init__()

        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin: nn.Linear = nn.Linear(in_features=300, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, X: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        X = self.flatten(X)
        X = self.lin(X)

        #
        return X
