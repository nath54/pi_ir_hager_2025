"""
Model Name: Temporal Convolutional Network (TCN)

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - num_channels: Number of channels in each layer
    - kernel_size: Kernel size for convolutions
    - num_layers: Number of TCN layers

Data variables:
    - B: Batch size

Architecture:
    - Permute (B, 30, 10) -> (B, 10, 30)
    - TCN Layer 1: Dilated Conv1d, dilation=1
    - TCN Layer 2: Dilated Conv1d, dilation=2
    - TCN Layer 3: Dilated Conv1d, dilation=4
    - ... (num_layers total)
    - GlobalAvgPool1d -> (B, num_channels)
    - Linear (B, num_channels) -> (B, 1)
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
    def __init__(self, num_channels: int = 8, kernel_size: int = 3, num_layers: int = 3) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.num_layers: int = num_layers
        #
        self.tcn_layers: nn.ModuleList = nn.ModuleList()
        #
        in_channels: int = 10
        #
        for i in range(num_layers):
            #
            dilation: int = 2 ** i
            #
            padding: int = (kernel_size - 1) * dilation
            #
            self.tcn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding
                )
            )
            #
            in_channels = num_channels
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.global_pool: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        #
        self.lin: nn.Linear = nn.Linear(in_features=num_channels, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.permute(0, 2, 1)
        #
        for i in range(self.num_layers):
            #
            x = self.tcn_layers[i](x)
            x = x[:, :, :30]
            x = self.relu(x)
        #
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.lin(x)

        #
        return x
