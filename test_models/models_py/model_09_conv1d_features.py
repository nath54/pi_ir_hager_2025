"""
Model Name: 1D Feature Convolution (Deep Scalable Version)

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels for the first Conv1d layer
    - k0: Kernel size for all Conv1d layers
    - depth: Number of stacked Conv1d + ReLU blocks (minimum 1)

Data variables:
    - B: Batch size

Architecture:
    - Reshape input to apply convolutions independently per timestep:
        (B, 30, 10) -> (B*30, 1, 10)
    - [Conv1d -> ReLU] x depth
        - Each Conv1d uses same kernel size k0
        - Channel dimension grows as: 1 -> c0 -> 2*c0 -> 4*c0 -> ... (doubling each layer)
        - Feature dimension reduces by (k0 - 1) per layer
    - Flatten per timestep: (B*30, C_final, F_final) -> (B, 30 * C_final * F_final)
    - Linear (B, 30 * C_final * F_final) -> (B, 1)
"""

#
### Import Modules. ###
#
from torch import nn
from torch import Tensor
from typing import List


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(
        self,
        c0: int = 8,
        k0: int = 3,
        depth: int = 1
    ) -> None:

        #
        super().__init__()  # type: ignore

        #
        ### Validate depth. ###
        #
        if depth < 1:
            raise ValueError("Depth must be at least 1.")

        #
        ### Build deep convolutional layers. ###
        #
        self.depth: int = depth
        self.k0: int = k0

        #
        conv_layers: List[nn.Module] = []
        in_channels: int = 1
        out_channels: int = c0

        #
        for i in range(depth):
            #
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k0
                )
            )
            #
            conv_layers.append(nn.ReLU())
            #
            in_channels = out_channels
            out_channels *= 2  # Double channels at each depth level

        #
        self.conv_blocks: nn.Sequential = nn.Sequential(*conv_layers)

        #
        ### Determine final feature length and channel size after convolutions. ###
        #
        feat_len: int = 10
        for _ in range(depth):
            feat_len = feat_len - k0 + 1
        #
        final_channels: int = in_channels  # This is c0 * (2 ** (depth - 1))

        #
        ### Final layers. ###
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin: nn.Linear = nn.Linear(
            in_features=30 * final_channels * feat_len,
            out_features=1
        )


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        B, T, F = x.shape  # T = 30, F = 10
        #
        x = x.unsqueeze(2)              # (B, 30, 10) -> (B, 30, 1, 10)
        x = x.reshape(B * T, 1, F)      # (B*30, 1, 10)

        #
        x = self.conv_blocks(x)         # (B*30, C_final, F_final)

        #
        x = x.view(B, -1)               # Flatten across timesteps and features: (B, 30 * C_final * F_final)
        x = self.lin(x)                 # (B, 1)

        #
        return x