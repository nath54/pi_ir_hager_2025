#
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F, init
#
import math


#Params
pos_encode: int = 4
features_after_conv: int = 17
n_embd: int = features_after_conv*pos_encode
dropout: float = 0.0  # Must be 0.0 for ONNX export
learning_rate: float = 0.006
num_outputs: int = 6
BATCH_SIZE: int = 40



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
    r"""Applies Layer Normalization over a mini-batch of inputs."""

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
        device=None,
        dtype=None,
    ) -> None:

        #
        factory_kwargs = {"device": device, "dtype": dtype}

        #
        super().__init__()

        #
        self.normalized_shape: tuple[int, ...] = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape, )
        self.eps: float = eps
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
            if self.bias is not None:
                #
                init.zeros_(self.bias)

    #
    def forward(self, input: Tensor) -> Tensor:
        #
        return layer_norm(
            x=input, weight=self.weight, bias=self.bias, eps=self.eps
        )

    #
    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


#
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Precompute constant values
        self.half_dim = dim // 2
        self.log_base = math.log(features_after_conv + 1) / (self.half_dim - 1)

    def forward(self, time):
        device = time.device
        # Use fixed computation without device-dependent operations
        embeddings = torch.exp(torch.arange(self.half_dim, device=device) * -self.log_base)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

#
class ConvEncode(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embdim = dim
        # Use explicit sequential without ReLU in Sequential to be more explicit
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5), stride=(2))
        self.relu = nn.ReLU()
        self.linearembd = nn.Linear(1, self.embdim)
        self.posembd = SinusoidalPositionEmbeddings(self.embdim)

        # Precompute position embeddings for known size (17 features after conv)
        self.register_buffer('fixed_time_tensor', torch.arange(17))

    def forward(self, x):
        B, WT, FD1, FD2 = x.size()
        x_conv = self.conv(x)
        x_conv = self.relu(x_conv)

        # Use fixed position encoding
        time_embedded = self.posembd(self.fixed_time_tensor)
        time_embedded2 = time_embedded.unsqueeze(0).unsqueeze(1).expand(B, -1, -1, -1)

        x_emb = self.linearembd(x_conv)
        x_code = time_embedded2 + x_emb
        x_code = x_code.view(B, 1, self.embdim * 17)  # Use hardcoded 17
        return x_code


#
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Remove dropout completely for ONNX export

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B T head_size
        q = self.query(x)  # B T head_size
        # compute attention score ('affinities')
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # B T hs @ B hs T  to B T T
        wei = F.softmax(wei, dim=-1)
        # No dropout
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


#
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # Use ModuleList but access directly without iteration
        self.head0 = Head(head_size)
        self.head1 = Head(head_size)
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # Remove dropout

    def forward(self, x):
        # Explicitly compute each head to avoid dynamic loops
        h0 = self.head0(x)
        h1 = self.head1(x)
        out = torch.cat([h0, h1], dim=-1)
        out = self.proj(out)
        return out


#
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, int(n_embd/2))
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(int(n_embd/2), n_embd)
        # Remove dropout

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


#
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


#
def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = features_after_conv * pos_encode
        num_heads = 2
        time_step = num_outputs
        n_layer = 2

        # Create individual ConvEncode modules explicitly (no ModuleList iteration)
        self.conv0 = ConvEncode(pos_encode)
        self.conv1 = ConvEncode(pos_encode)
        self.conv2 = ConvEncode(pos_encode)
        self.conv3 = ConvEncode(pos_encode)
        self.conv4 = ConvEncode(pos_encode)
        self.conv5 = ConvEncode(pos_encode)

        # Create blocks explicitly
        self.block0 = Block(features_after_conv * pos_encode, n_head=num_heads)
        self.block1 = Block(features_after_conv * pos_encode, n_head=num_heads)

        self.ln_f = LayerNorm(n_embd)

        self.fc_linear = nn.Linear(hidden_size * time_step, num_outputs)
        self.fc_relu = nn.ReLU()

    def forward(self, x):
        B, WT, FD1, FD2 = x.size()

        # Explicitly slice and process each time step (no loops, no zip, no split)
        out0 = self.conv0(x[:, 0:1, :, :])
        out1 = self.conv1(x[:, 1:2, :, :])
        out2 = self.conv2(x[:, 2:3, :, :])
        out3 = self.conv3(x[:, 3:4, :, :])
        out4 = self.conv4(x[:, 4:5, :, :])
        out5 = self.conv5(x[:, 5:6, :, :])

        # Concatenate
        concatenated_ConvEncode = torch.cat([out0, out1, out2, out3, out4, out5], dim=-2)

        # Apply blocks explicitly
        x = self.block0(concatenated_ConvEncode)
        x = self.block1(x)

        LayerNorm_output = self.ln_f(x)

        # Final FC
        x = LayerNorm_output.view(B, 1, -1)
        x = self.fc_linear(x)
        x = self.fc_relu(x)

        Probabilities = x.squeeze(1)

        return Probabilities.float()