"""
Model Name: CNN + LSTM Hybrid (Deep Scalable Version)

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels for the first Conv1d layer
    - k0: Kernel size for all Conv1d layers
    - h0: Hidden size for LSTM
    - depth: Number of stacked Conv1d + ReLU blocks (minimum 1)

Data variables:
    - B: Batch size

Architecture:
    - Permute (B, 30, 10) -> (B, 10, 30)
    - [Conv1d -> ReLU] x depth
        - Each Conv1d uses same kernel size k0
        - Channel dimension grows as: 10 -> c0 -> 2*c0 -> 4*c0 -> ... (doubling each layer)
        - Sequence length reduces by (k0 - 1) per layer
    - Permute to (B, seq_len, channels)
    - LSTM (B, seq_len, channels) -> (B, seq_len, h0)
    - Take last timestep (B, seq_len, h0) -> (B, h0)
    - Linear (B, h0) -> (B, 1)
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
        h0: int = 12,
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
        in_channels: int = 10
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
        ### Determine final sequence length and channel size after convolutions. ###
        #
        seq_len: int = 30
        for _ in range(depth):
            seq_len = seq_len - k0 + 1
        #
        final_channels: int = in_channels  # This is c0 * (2 ** (depth - 1))

        #
        ### LSTM and final linear layer. ###
        #
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=final_channels,
            hidden_size=h0,
            batch_first=True
        )
        #
        self.lin: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.permute(0, 2, 1)          # (B, 30, 10) -> (B, 10, 30)
        x = self.conv_blocks(x)         # Apply depth-wise conv + ReLU blocks
        x = x.permute(0, 2, 1)          # (B, C_final, L_final) -> (B, L_final, C_final)
        x, _ = self.lstm(x)             # (B, L_final, C_final) -> (B, L_final, h0)
        x = x[:, -1, :]                 # Take last timestep: (B, h0)
        x = self.lin(x)                 # (B, h0) -> (B, 1)

        #
        return x