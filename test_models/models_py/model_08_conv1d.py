"""
Model Name: 1D Temporal Convolution (Deep)

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - depth: Number of convolutional blocks
    - channels: list or base number of output channels per block
    - kernel_size: Kernel size for all conv layers (or list per block)
    - pool_size: Pool size for all maxpool layers (or list per block)

Data variables:
    - B: Batch size

Architecture:
    - Permute (B, 30, 10) -> (B, 10, 30)
    - [Conv1d -> ReLU -> MaxPool1d] x depth
    - Flatten
    - Linear -> (B, 1)
"""

#
### Import Modules. ###
#
from torch import nn
from torch import Tensor
from typing import Union


#
### Utility Function to Normalize Parameters Across Depth. ###
#
def _expand_param(param: int | list[int], depth: int, param_name: str) -> list[int]:
    #
    ### Expand scalar parameter to list of length `depth`. ###
    #
    if isinstance(param, int):
        #
        return [param] * depth
    #
    elif isinstance(param, list):
        #
        if len(param) != depth:
            #
            raise ValueError(f"Length of {param_name} list ({len(param)}) must match depth ({depth}).")
        #
        return param
    #
    else:
        #
        raise TypeError(f"{param_name} must be int or list of ints.")


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(
        self,
        depth: int = 3,
        channels: int | list[int] = 8,
        kernel_size: int | list[int] = 3,
        pool_size: int | list[int] = 2
    ) -> None:

        #
        super().__init__()  # type: ignore

        #
        ### Validate and expand parameters to lists of length `depth`. ###
        #
        self.depth: int = depth
        self.channels_list: list[int] = _expand_param(channels, depth, "channels")
        self.kernel_sizes: list[int] = _expand_param(kernel_size, depth, "kernel_size")
        self.pool_sizes: list[int] = _expand_param(pool_size, depth, "pool_size")

        #
        ### Build sequential convolutional blocks. ###
        #
        layers: list[nn.Module] = []
        in_channels: int = 10  # Fixed input feature dimension after permute

        #
        L: int = 30  # Initial temporal length
        #
        for i in range(self.depth):

            #
            out_channels: int = self.channels_list[i]
            k: int = self.kernel_sizes[i]
            p: int = self.pool_sizes[i]

            #
            ### Add Conv1d + ReLU + MaxPool1d block. ###
            #
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=k))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=p))

            #
            ### Update dimensions for next layer. ###
            #
            L = (L - k + 1) // p
            in_channels = out_channels

            #
            ### Safety check to avoid zero or negative sequence length. ###
            #
            if L <= 0:
                raise ValueError(
                    f"Sequence length became non-positive ({L}) after block {i}. "
                    f"Reduce depth, kernel_size, or increase input length."
                )

        #
        self.conv_blocks: nn.Sequential = nn.Sequential(*layers)
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin: nn.Linear = nn.Linear(in_features=in_channels * L, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.permute(0, 2, 1)          # (B, 30, 10) -> (B, 10, 30)
        x = self.conv_blocks(x)         # Apply stacked conv blocks
        x = self.flatten(x)             # Flatten spatial + channel dims
        x = self.lin(x)                 # Final prediction

        #
        return x