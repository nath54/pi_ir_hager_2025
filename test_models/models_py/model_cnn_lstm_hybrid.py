"""
Model Name: CNN + LSTM Hybrid

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels
    - k0: Kernel size
    - h0: Hidden size for LSTM

Data variables:
    - B: Batch size

Architecture:
    - Permute (B, 30, 10) -> (B, 10, 30)
    - Conv1d (B, 10, 30) -> (B, c0, 30-k0+1)
    - ReLU (B, c0, 30-k0+1) -> (B, c0, 30-k0+1)
    - Permute (B, c0, 30-k0+1) -> (B, 30-k0+1, c0)
    - LSTM (B, 30-k0+1, c0) -> (B, 30-k0+1, h0)
    - Take last timestep (B, 30-k0+1, h0) -> (B, h0)
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
    def __init__(self, c0: int, k0: int, h0: int) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.conv1d: nn.Conv1d = nn.Conv1d(in_channels=10, out_channels=c0, kernel_size=k0)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.lstm: nn.LSTM = nn.LSTM(input_size=c0, hidden_size=h0, batch_first=True)
        #
        self.lin: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.lin(x)

        #
        return x
