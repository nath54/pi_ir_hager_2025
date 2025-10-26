"""
Model Name: Self-Attention Layer

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - d_k: Key/Value dimension

Data variables:
    - B: Batch size

Architecture:
    - Linear Q (B, 30, 10) -> (B, 30, d_k)
    - Linear K (B, 30, 10) -> (B, 30, d_k)
    - Linear V (B, 30, 10) -> (B, 30, d_k)
    - Attention scores QK^T / sqrt(d_k) -> (B, 30, 30)
    - Softmax (B, 30, 30) -> (B, 30, 30)
    - Weighted sum with V -> (B, 30, d_k)
    - GlobalAvgPool1d (B, 30, d_k) -> (B, d_k)
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
    def __init__(self, d_k: int) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.d_k: int = d_k
        #
        self.query: nn.Linear = nn.Linear(in_features=10, out_features=d_k)
        #
        self.key: nn.Linear = nn.Linear(in_features=10, out_features=d_k)
        #
        self.value: nn.Linear = nn.Linear(in_features=10, out_features=d_k)
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
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        #
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        #
        attn_weights = self.softmax(scores)
        #
        x = torch.matmul(attn_weights, V)
        #
        x = x.mean(dim=1)
        x = self.lin(x)

        #
        return x
