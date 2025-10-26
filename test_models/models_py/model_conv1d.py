"""
Model Name: 1D Temporal Convolution

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels
    - k0: Kernel size
    - p0: Pool size

Data variables:
    - B: Batch size

Architecture:
    - Permute (B, 30, 10) -> (B, 10, 30)
    - Conv1d (B, 10, 30) -> (B, c0, 30-k0+1)
    - ReLU (B, c0, 30-k0+1) -> (B, c0, 30-k0+1)
    - MaxPool1d (B, c0, 30-k0+1) -> (B, c0, (30-k0+1)//p0)
    - Flatten (B, c0, (30-k0+1)//p0) -> (B, c0 * ((30-k0+1)//p0))
    - Linear (B, c0 * ((30-k0+1)//p0)) -> (B, 1)
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
    def __init__(self, c0: int = 8, k0: int = 3, p0: int = 2) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.conv1d: nn.Conv1d = nn.Conv1d(in_channels=10, out_channels=c0, kernel_size=k0)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.maxpool: nn.MaxPool1d = nn.MaxPool1d(kernel_size=p0)
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        L0: int = 30 - k0 + 1
        #
        L_pooled: int = L0 // p0
        #
        self.lin: nn.Linear = nn.Linear(in_features=c0 * L_pooled, out_features=1)


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
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.lin(x)

        #
        return x