"""
Model Name: Lightweight Transformer (Single Layer)

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - d_model: Model dimension
    - num_heads: Number of attention heads

Data variables:
    - B: Batch size

Architecture:
    - Linear projection (B, 30, 10) -> (B, 30, d_model)
    - Multi-Head Self-Attention (B, 30, d_model) -> (B, 30, d_model)
    - Add & Norm (B, 30, d_model) -> (B, 30, d_model)
    - Feed-Forward (B, 30, d_model) -> (B, 30, d_model)
    - Add & Norm (B, 30, d_model) -> (B, 30, d_model)
    - GlobalAvgPool (B, 30, d_model) -> (B, d_model)
    - Linear (B, d_model) -> (B, 1)
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
    def __init__(self, d_model: int = 16, num_heads: int = 2) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.proj: nn.Linear = nn.Linear(in_features=10, out_features=d_model)
        #
        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        #
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        #
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=4 * d_model),
            nn.ReLU(),
            nn.Linear(in_features=4 * d_model, out_features=d_model)
        )
        #
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        #
        self.lin: nn.Linear = nn.Linear(in_features=d_model, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = self.proj(x)
        #
        x_residual = x
        x, _ = self.attention(x, x, x)
        x = self.norm1(x + x_residual)
        #
        x_residual = x
        x = self.ffn(x)
        x = self.norm2(x + x_residual)
        #
        x = x.mean(dim=1)
        x = self.lin(x)

        #
        return x
