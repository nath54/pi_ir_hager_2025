"""
Model Name: Vision Transformer (ViT) - ONNX Compatible

Model Input: (B, 30, 10)  # Batch, sequence length (time steps), features
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - patch_size: Size of each patch (default: 5, creates 6 patches from 30 timesteps)
    - d_model: Model dimension (default: 16)
    - num_heads: Number of attention heads (default: 2)
    - depth: Number of transformer layers (default: 2)
    - mlp_ratio: MLP hidden dimension ratio (default: 4)

Scaling Guidelines:
    - Primary scaling: depth (1->2->3->4) - Most impact on capacity
    - Secondary scaling: d_model (16->32->64) - Increases model width
    - Keep num_heads as divisor of d_model (d_model % num_heads == 0)
    - Larger patch_size = fewer patches = faster but less detail
    - mlp_ratio can be increased (4->6->8) for more nonlinearity

    Example configurations:
    - Small: depth=1, d_model=16, num_heads=2  (~2K params)
    - Medium: depth=2, d_model=32, num_heads=4  (~10K params)
    - Large: depth=3, d_model=64, num_heads=8  (~50K params)

Data variables:
    - B: Batch size

Architecture:
    - Patch Embedding (B, 30, 10) -> (B, num_patches, d_model)
    - Positional Encoding (B, num_patches, d_model) -> (B, num_patches, d_model)
    - [Transformer Block] x depth:
        - Multi-Head Self-Attention (B, num_patches, d_model) -> (B, num_patches, d_model)
        - Add & Norm
        - Feed-Forward (B, num_patches, d_model) -> (B, num_patches, d_model)
        - Add & Norm
    - GlobalAvgPool (B, num_patches, d_model) -> (B, d_model)
    - Classification Head (B, d_model) -> (B, 1)
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
        self.normalized_shape: tuple[int, ...] = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape, )
        #
        self.eps: float = eps
        #
        self.elementwise_affine: bool = elementwise_affine

        #
        if self.elementwise_affine:
            #
            self.weight = nn.Parameter( torch.empty(self.normalized_shape, **factory_kwargs) )
            #
            if bias:
                #
                self.bias = nn.Parameter( torch.empty(self.normalized_shape, **factory_kwargs) )
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
        return layer_norm(
            x=input, weight=self.weight, bias=self.bias, eps=self.eps
        )


#
### ONNX-Compatible Multi-Head Attention. ###
#
class MultiHeadAttention(nn.Module):
    """
    Custom Multi-Head Attention that is ONNX-compatible.
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
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)

        #
        ### Reshape to (B, num_heads, seq_len, head_dim). ###
        #
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        #
        ### Scaled dot-product attention. ###
        #
        attn_scores: Tensor = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        #
        ### Softmax over the last dimension. ###
        #
        attn_weights: Tensor = F.softmax(attn_scores, dim=-1)

        #
        ### Apply attention to values. ###
        #
        attn_output: Tensor = torch.matmul(attn_weights, v)

        #
        ### Concatenate heads. ###
        #
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)

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

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4) -> None:
        #
        super().__init__()  # type: ignore

        #
        self.attention: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
        self.norm1: LayerNorm = LayerNorm(d_model)

        #
        mlp_hidden: int = mlp_ratio * d_model
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model)
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
### Patch Embedding Layer. ###
#
class PatchEmbedding(nn.Module):
    """
    Converts input sequences into patches and embeds them.
    For input (B, 30, 10), with patch_size=5, creates 6 patches of 50 features each.
    """

    def __init__(self, seq_len: int, input_dim: int, patch_size: int, d_model: int) -> None:
        #
        super().__init__()  # type: ignore

        #
        assert seq_len % patch_size == 0, f"seq_len {seq_len} must be divisible by patch_size {patch_size}"

        #
        self.seq_len: int = seq_len
        self.input_dim: int = input_dim
        self.patch_size: int = patch_size
        self.num_patches: int = seq_len // patch_size
        self.patch_dim: int = patch_size * input_dim

        #
        ### Linear projection from patch to d_model. ###
        #
        self.proj: nn.Linear = nn.Linear(self.patch_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        #
        ### x shape: (B, seq_len, input_dim) ###
        #
        B = x.shape[0]

        #
        ### Reshape into patches: (B, num_patches, patch_dim). ###
        #
        x = x.view(B, self.num_patches, self.patch_dim)

        #
        ### Project to d_model. ###
        #
        x = self.proj(x)  # (B, num_patches, d_model)

        #
        return x


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(
        self,
        patch_size: int = 5,
        d_model: int = 16,
        num_heads: int = 2,
        depth: int = 2,
        mlp_ratio: int = 4
    ) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.patch_size: int = patch_size
        self.d_model: int = d_model

        #
        ### Patch embedding. ###
        #
        self.patch_embed: PatchEmbedding = PatchEmbedding(
            seq_len=30,
            input_dim=10,
            patch_size=patch_size,
            d_model=d_model
        )

        #
        num_patches: int = 30 // patch_size

        #
        ### Learnable positional embeddings. ###
        #
        self.pos_embed: nn.Parameter = nn.Parameter(torch.zeros(1, num_patches, d_model))

        #
        ### Stack of transformer blocks. ###
        #
        self.transformer_blocks: nn.ModuleList = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        #
        ### Classification head. ###
        #
        self.head: nn.Linear = nn.Linear(d_model, 1)

        #
        ### Initialize positional embeddings. ###
        #
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Patch embedding. ###
        #
        x = self.patch_embed(x)  # (B, 30, 10) -> (B, num_patches, d_model)

        #
        ### Add positional embeddings. ###
        #
        x = x + self.pos_embed  # (B, num_patches, d_model)

        #
        ### Pass through transformer blocks. ###
        #
        for block in self.transformer_blocks:
            x = block(x)  # (B, num_patches, d_model)

        #
        ### Global average pooling. ###
        #
        x = x.mean(dim=1)  # (B, num_patches, d_model) -> (B, d_model)

        #
        ### Classification head. ###
        #
        x = self.head(x)  # (B, d_model) -> (B, 1)

        #
        return x