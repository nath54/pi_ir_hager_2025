"""
Model Name: Self-Attention Layer

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - d_k: Key/Value dimension
    - depth: Number of stacked attention layers

Data variables:
    - B: Batch size

Architecture:
    - Linear Q (B, 30, 10) -> (B, 30, d_k)
    - Linear K (B, 30, 10) -> (B, 30, d_k)
    - Linear V (B, 30, 10) -> (B, 30, d_k)
    - Attention scores QK^T / sqrt(d_k) -> (B, 30, 30)
    - Softmax (B, 30, 30) -> (B, 30, 30)
    - Weighted sum with V -> (B, 30, d_k)
    - (Optional) Layer normalization and residual connection
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
### Multi-Head Attention Layer Class. ###
#
class AttentionLayer(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(self, d_k: int, input_dim: int) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.d_k: int = d_k
        #
        self.query: nn.Linear = nn.Linear(in_features=input_dim, out_features=d_k)
        #
        self.key: nn.Linear = nn.Linear(in_features=input_dim, out_features=d_k)
        #
        self.value: nn.Linear = nn.Linear(in_features=input_dim, out_features=d_k)
        #
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)
        #
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(normalized_shape=d_k)
        #
        self.output_projection: nn.Linear = nn.Linear(in_features=d_k, out_features=d_k)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass for single attention layer. ###
        #
        residual: Tensor = x
        #
        Q: Tensor = self.query(x)
        K: Tensor = self.key(x)
        V: Tensor = self.value(x)
        #
        scores: Tensor = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        #
        attn_weights: Tensor = self.softmax(scores)
        #
        x = torch.matmul(attn_weights, V)
        #
        x = self.output_projection(x)
        #
        x = x + residual  # Residual connection
        x = self.layer_norm(x)  # Layer normalization

        #
        return x


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(self, d_k: int = 16, depth: int = 1) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.d_k: int = d_k
        #
        self.depth: int = depth
        #
        self.input_projection: nn.Linear = nn.Linear(in_features=10, out_features=d_k)
        #
        self.attention_layers: nn.ModuleList = nn.ModuleList([
            AttentionLayer(d_k=d_k, input_dim=d_k)
            for _ in range(depth)
        ])
        #
        self.global_avg_pool: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(output_size=1)
        #
        self.lin: nn.Linear = nn.Linear(in_features=d_k, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = self.input_projection(x)
        #
        for layer in self.attention_layers:
            x = layer(x)
        #
        x = x.transpose(-2, -1)  # Transpose for pooling: (B, seq_len, d_k) -> (B, d_k, seq_len)
        x = self.global_avg_pool(x)  # (B, d_k, 1)
        x = x.squeeze(-1)  # (B, d_k)
        x = self.lin(x)

        #
        return x