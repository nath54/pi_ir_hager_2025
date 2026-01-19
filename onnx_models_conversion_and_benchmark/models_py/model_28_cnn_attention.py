"""
Model Name: Deep CNN + Attention

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels
    - k_h: Kernel height
    - k_w: Kernel width
    - d_k: Attention dimension
    - depth: Number of stacked CNN layers

Data variables:
    - B: Batch size
    - H: Height of feature map after all convolutions
    - W: Width of feature map after all convolutions

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Stacked Conv2d+ReLU (depth times)
        - Layer 1: (B, 1, H, W) -> (B, c0, H1, W1)
        - Layer 2..depth: (B, c0, H_i, W_i) -> (B, c0, H_i+1, W_i+1)
    - Flatten spatial dims (B, c0, H', W') -> (B, c0, H'*W')
    - Permute (B, c0, H'*W') -> (B, H'*W', c0)
    - Self-Attention over spatial locations
    - GlobalAvgPool (B, H'*W', d_k) -> (B, d_k)
    - Linear (B, d_k) -> (B, 1)
"""

#
### Import Modules. ###
#
from torch import nn
from torch import Tensor
import torch
import math


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
        k_h: int = 3,
        k_w: int = 3,
        d_k: int = 12,
        depth: int = 1
    ) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.d_k: int = d_k
        self.k_h: int = k_h
        self.k_w: int = k_w

        #
        ### Create a sequential block for stacked CNN layers. ###
        #
        layers: list[nn.Module] = []

        #
        ### First layer: 1 input channel to c0 output channels. ###
        #
        layers.append(nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=(k_h, k_w)))
        layers.append(nn.ReLU())

        #
        ### Subsequent layers: c0 input channels to c0 output channels (depth-1 more times). ###
        #
        for _ in range(depth - 1):
            #
            ### Add a CNN layer. ###
            #
            layers.append(nn.Conv2d(in_channels=c0, out_channels=c0, kernel_size=(k_h, k_w)))
            #
            ### Add a ReLU activation function. ###
            #
            layers.append(nn.ReLU())

        #
        self.cnn_stack: nn.Sequential = nn.Sequential(*layers)

        #
        ### Attention components remain the same, operating on c0 features. ###
        #
        self.query: nn.Linear = nn.Linear(in_features=c0, out_features=d_k)
        #
        self.key: nn.Linear = nn.Linear(in_features=c0, out_features=d_k)
        #
        self.value: nn.Linear = nn.Linear(in_features=c0, out_features=d_k)
        #
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)
        #
        self.lin: nn.Linear = nn.Linear(in_features=d_k, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        ### (B, 30, 10) -> (B, 1, 30, 10). ###
        #
        x = x.unsqueeze(1)
        #
        ### Stacked CNN operations. ###
        #
        x = self.cnn_stack(x)
        #
        ### Flatten spatial dimensions. ###
        #
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        #
        ### Permute to (B, H'*W', C) for Attention: (B, sequence_length, features). ###
        #
        x = x.permute(0, 2, 1)
        #
        ### Self-Attention. ###
        #
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        #
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = self.softmax(scores)
        x = torch.matmul(attn_weights, V)
        #
        ### Global Average Pool over the spatial sequence dimension (dim=1). ###
        #
        x = x.mean(dim=1)
        #
        ### Final Linear layer. ###
        #
        x = self.lin(x)

        #
        return x
