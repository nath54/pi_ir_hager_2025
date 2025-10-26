"""
Model Name: Bidirectional GRU

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden size

Data variables:
    - B: Batch size

Architecture:
    - BiGRU (B, 30, 10) -> (B, 30, 2*h0)
    - Take last timestep (B, 30, 2*h0) -> (B, 2*h0)
    - Linear (B, 2*h0) -> (B, 1)
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
        self.bigru: nn.GRU = nn.GRU(input_size=10, hidden_size=h0, batch_first=True, bidirectional=True)
        #
        self.lin: nn.Linear = nn.Linear(in_features=2 * h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, X: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        X, h_n = self.bigru(X)
        X = X[:, -1, :]
        X = self.lin(X)

        #
        return X
