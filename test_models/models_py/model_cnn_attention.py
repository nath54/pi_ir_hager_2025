"""
Model Name: CNN + Attention

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels
    - k_h: Kernel height
    - k_w: Kernel width
    - d_k: Attention dimension

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Conv2d (B, 1, 30, 10) -> (B, c0, 30-k_h+1, 10-k_w+1)
    - ReLU (B, c0, 30-k_h+1, 10-k_w+1) -> (B, c0, 30-k_h+1, 10-k_w+1)
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
    def __init__(self, c0: int = 8, k_h: int = 3, k_w: int = 3, d_k: int = 12) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.d_k: int = d_k
        #
        self.conv2d: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=(k_h, k_w))
        #
        self.relu: nn.ReLU = nn.ReLU()
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
        x = x.unsqueeze(1)
        x = self.conv2d(x)
        x = self.relu(x)
        #
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        x = x.permute(0, 2, 1)
        #
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        #
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = self.softmax(scores)
        x = torch.matmul(attn_weights, V)
        #
        x = x.mean(dim=1)
        x = self.lin(x)

        #
        return x
