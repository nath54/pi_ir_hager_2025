"""
Model Name: Simple RNN

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden size

Data variables:
    - B: Batch size

Architecture:
    - RNN (B, 30, 10) -> (B, 30, h0)
    - Take last timestep (B, 30, h0) -> (B, h0)
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
        self.rnn: nn.RNN = nn.RNN(input_size=10, hidden_size=h0, batch_first=True)
        #
        self.lin: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, X: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        X, _ = self.rnn(X)
        X = X[:, -1, :]
        X = self.lin(X)

        #
        return X
