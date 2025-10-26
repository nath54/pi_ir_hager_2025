"""
Model Name: 1D Feature Convolution

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels
    - k0: Kernel size

Data variables:
    - B: Batch size

Architecture:
    - Conv1d along features (B, 30, 10) -> (B, 30, 10-k0+1) with c0 applied
    - ReLU (B, 30, c0*(10-k0+1)) -> (B, 30, c0*(10-k0+1))
    - Flatten (B, 30, c0*(10-k0+1)) -> (B, 30*c0*(10-k0+1))
    - Linear (B, 30*c0*(10-k0+1)) -> (B, 1)
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
    def __init__(self, c0: int, k0: int) -> None:

        #
        super().__init__()

        #
        self.conv1d: nn.Conv1d = nn.Conv1d(in_channels=1, out_channels=c0, kernel_size=k0)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        L0: int = 10 - k0 + 1
        #
        self.lin: nn.Linear = nn.Linear(in_features=30 * c0 * L0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, X: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        B, T, F = X.shape
        X = X.unsqueeze(2)
        X = X.reshape(B * T, 1, F)
        #
        X = self.conv1d(X)
        X = self.relu(X)
        #
        X = X.view(B, -1)
        X = self.lin(X)

        #
        return X
