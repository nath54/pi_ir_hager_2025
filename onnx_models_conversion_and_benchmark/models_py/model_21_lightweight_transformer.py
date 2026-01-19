"""
Model Name: Lightweight Transformer (Configurable Depth)

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - d_model: Model dimension (default: 16)
    - num_heads: Number of attention heads (default: 2)
    - depth: Number of transformer layers (default: 1)

Data variables:
    - B: Batch size

Architecture:
    - Linear projection (B, 30, 10) -> (B, 30, d_model)
    - [Transformer Block] x depth:
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
from typing import Any, Optional

#
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F, init  # type: ignore
import math


#
def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    #
    dim: int = -1
    #
    mean: Tensor = torch.mean(x, dim, keepdim=True)
    #
    centered: Tensor = x - mean
    #
    var: Tensor = torch.sum(centered * centered, dim, keepdim=True) / x.size(-1)
    #
    rvar: Tensor = 1.0 / torch.sqrt(var + eps)
    #
    normed: Tensor = (x - mean) * rvar
    #
    return normed * weight + bias


#
class LayerNorm(nn.Module):
    #
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: tuple[int, ...] | int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:

        #
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        #
        super().__init__()  # type: ignore

        #
        self.normalized_shape: tuple[int, ...] = (
            normalized_shape
            if isinstance(normalized_shape, tuple)
            else (normalized_shape,)
        )
        #
        self.eps: float = eps
        #
        self.elementwise_affine: bool = elementwise_affine

        #
        if self.elementwise_affine:
            #
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            #
            if bias:
                #
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            #
            else:
                #
                self.register_parameter("bias", None)
        #
        else:
            #
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        #
        self.reset_parameters()

    #
    def reset_parameters(self) -> None:
        #
        if self.elementwise_affine:
            #
            init.ones_(self.weight)
            #
            if self.bias is not None:  # type: ignore
                #
                init.zeros_(self.bias)

    #
    def forward(self, input: Tensor) -> Tensor:
        #
        return layer_norm(x=input, weight=self.weight, bias=self.bias, eps=self.eps)


#
### ONNX-Compatible Multi-Head Attention. ###
#
class MultiHeadAttention(nn.Module):
    """
    Custom Multi-Head Attention that is ONNX-compatible.
    Implements scaled dot-product attention without using nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        #
        super().__init__()  # type: ignore

        #
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        #
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads

        #
        ### Linear projections for q, k, v. ###
        #
        self.q_proj: nn.Linear = nn.Linear(d_model, d_model)
        self.k_proj: nn.Linear = nn.Linear(d_model, d_model)
        self.v_proj: nn.Linear = nn.Linear(d_model, d_model)

        #
        ### Output projection. ###
        #
        self.out_proj: nn.Linear = nn.Linear(d_model, d_model)

        #
        ### Scaling factor. ###
        #
        self.scale: float = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        #
        ### x shape: (B, seq_len, d_model) ###
        #
        B, seq_len, _ = x.shape

        #
        ### Linear projections. ###
        #
        q: Tensor = self.q_proj(x)  # (B, seq_len, d_model)
        k: Tensor = self.k_proj(x)  # (B, seq_len, d_model)
        v: Tensor = self.v_proj(x)  # (B, seq_len, d_model)

        #
        ### Reshape to (B, num_heads, seq_len, head_dim). ###
        #
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        #
        ### Scaled dot-product attention. ###
        #
        # q @ k^T: (B, num_heads, seq_len, head_dim) @ (B, num_heads, head_dim, seq_len)
        # -> (B, num_heads, seq_len, seq_len)
        #
        attn_scores: Tensor = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        #
        ### Softmax over the last dimension. ###
        #
        attn_weights: Tensor = F.softmax(attn_scores, dim=-1)

        #
        ### Apply attention to values. ###
        #
        # (B, num_heads, seq_len, seq_len) @ (B, num_heads, seq_len, head_dim)
        # -> (B, num_heads, seq_len, head_dim)
        #
        attn_output: Tensor = torch.matmul(attn_weights, v)

        #
        ### Concatenate heads. ###
        #
        # (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads, head_dim)
        # -> (B, seq_len, d_model)
        #
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        )

        #
        ### Output projection. ###
        #
        output: Tensor = self.out_proj(attn_output)

        #
        return output


#
### Transformer Block. ###
#
class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feed-forward layers.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int = 4) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.attention: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
        self.norm1: LayerNorm = LayerNorm(d_model)

        #
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, ffn_hidden * d_model),
            nn.ReLU(),
            nn.Linear(ffn_hidden * d_model, d_model),
        )
        self.norm2: LayerNorm = LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:

        #
        ### Self-attention with residual connection. ###
        #
        x_residual = x
        x = self.attention(x)
        x = self.norm1(x + x_residual)

        #
        ### Feed-forward with residual connection. ###
        #
        x_residual = x
        x = self.ffn(x)
        x = self.norm2(x + x_residual)

        #
        return x


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(self, d_model: int = 16, num_heads: int = 2, depth: int = 1) -> None:

        #
        super().__init__()  # type: ignore

        #
        ### Input projection. ###
        #
        self.proj: nn.Linear = nn.Linear(in_features=10, out_features=d_model)

        #
        ### Stack of transformer blocks. ###
        #
        self.transformer_blocks: nn.ModuleList = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(depth)]
        )

        #
        ### Output projection. ###
        #
        self.lin: nn.Linear = nn.Linear(in_features=d_model, out_features=1)

    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Input projection. ###
        #
        x = self.proj(x)  # (B, 30, 10) -> (B, 30, d_model)

        #
        ### Pass through transformer blocks. ###
        #
        for block in self.transformer_blocks:
            x = block(x)  # (B, 30, d_model) -> (B, 30, d_model)

        #
        ### Global average pooling. ###
        #
        x = x.mean(dim=1)  # (B, 30, d_model) -> (B, d_model)

        #
        ### Output projection. ###
        #
        x = self.lin(x)  # (B, d_model) -> (B, 1)

        #
        return x
