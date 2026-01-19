"""
Model Name: LSTM (Long Short-Term Memory)

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden size
    - depth: Number of LSTM layers

Data variables:
    - B: Batch size

Architecture:
    - LSTM (B, 30, 10) -> (B, 30, h0)
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
    def __init__(self, h0: int = 16, depth: int = 1) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.lstm: nn.LSTM = nn.LSTM(input_size=10, hidden_size=h0, num_layers=depth, batch_first=True)
        #
        self.lin: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.lin(x)

        #
        return x
