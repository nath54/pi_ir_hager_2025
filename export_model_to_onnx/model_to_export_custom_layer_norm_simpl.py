#
### Import Modules. ###
#
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F, init
#
import sys


#
### Params. ###
#
pos_encode: int = 4
features_after_conv: int = 17
n_embd: int = 68 # features_after_conv * pos_encode
num_outputs: int = 6
batch_size: int = 40



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
    rvar: float = 1.0 / torch.sqrt(var + eps)
    #
    normed: Tensor = (x - mean) * rvar
    #
    return normed * weight + bias


#
class LayerNorm(nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The variance is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`elementwise_affine` is ``True``). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    """

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

    #
    def __init__(self) -> None:
        #
        super().__init__()  # type: ignore


    #
    def forward(self, time: Tensor) -> Tensor:

        #
        device: str | torch.device = time.device
        #
        # half_dim: int = self.dim // 2  # self.dim = 4 | half_dim = 2
        #
        embeddings_f: float = 2.8903717578961645 # math.log( 18 )
        #
        embeddings: Tensor = torch.exp(torch.arange(2, device=device) * -embeddings_f)
        #
        embeddings = time[:, None] * embeddings[None, :]
        #
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        #
        return embeddings


#
class ConvEncode(nn.Module):

    #
    def __init__(self) -> None:
        #
        super().__init__()  # type: ignore
        #
        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=1 , out_channels=1 , kernel_size=(5,5),  stride=(2)),
            nn.ReLU()
        )
        #
        self.linearembd: nn.Linear = nn.Linear(in_features=1, out_features=4)
        #
        self.posembd: SinusoidalPositionEmbeddings = SinusoidalPositionEmbeddings()

    #
    def forward(self, x: Tensor) -> Tensor :
        #
        device: str | torch.device = x.device
        #
        tmp: tuple[int, ...] = x.size()
        #
        b: int = tmp[0]
        #
        # b, wt, fd1, fd2 = x.size()

        #
        x_conv: Tensor = self.conv(x)

        #
        ### Feature dimension 1 time encode each sample from 0 to 38. ###
        #
        time_tensor: Tensor = torch.arange(x_conv.shape[2], device = device)

        #
        time_embedded: Tensor = self.posembd(time_tensor)

        #
        ### Duplicate the time vector along the batch dimension. ###
        #
        time_embedded2: Tensor = time_embedded.unsqueeze(0).unsqueeze(1).expand(b, -1, -1, -1)

        #
        x_emb: Tensor = self.linearembd(x_conv)

        #
        x_code: Tensor = time_embedded2 + x_emb

        #
        ### Shape of feature after conv multiply with emb dimension then code to batch with 1 time step before attention. ###
        #
        x_code: Tensor = x_code.view(b,1, 4 * x_conv.shape[2])

        #
        return x_code


#
class Head(nn.Module) :

    #
    def __init__(self):

        #
        super().__init__()  # type: ignore

        #
        self.key: nn.Linear = nn.Linear(68, 34, bias = False)
        self.query: nn.Linear = nn.Linear(68, 34, bias = False)
        self.value: nn.Linear = nn.Linear(68, 34, bias = False)


    #
    def forward(self, x: Tensor) -> Tensor:

        #
        # _b, _t, _c = x.shape

        #
        k: Tensor = self.key(x) # (_b, _t, head_size)
        q: Tensor = self.query(x) # (_b, _t, head_size)

        #
        ### compute attention score ('affinities'). ###
        #
        wei: Tensor = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) to (B, T, T)

        #
        wei = F.softmax(wei, dim = -1)

        #
        ### perform the weighted aggregation of the values. ###
        #
        v: Tensor = self.value(x)

        #
        out: Tensor = wei @ v

        #
        return out


#
class MultiHeadAttention(nn.Module) :

    #
    def __init__(self) -> None:

        #
        super().__init__()  # type: ignore

        #
        ### head do in parallel. ###
        #
        self.heads = nn.ModuleList([
            Head(),
            Head()
        ])

        #
        ### projection layer. ###
        #
        self.proj = nn.Linear(68, 68)


    #
    def forward(self, x: Tensor) -> Tensor:

        # Process each head individually

        head_out_0 = self.heads[0](x)
        head_out_1 = self.heads[1](x)

        head_outputs = [head_out_0, head_out_1]

        out: Tensor = torch.cat(head_outputs, dim = -1) # (Batch, Time, Channel Feature dimension) = (B,T , [h1,h1,h2,h2,h3,h3,h4,h4])

        #
        return out


#
class FeedFoward(nn.Module):

    #
    def __init__(self) -> None:

        #
        super().__init__()  # type: ignore
        #
        self.net = nn.Sequential(
            nn.Linear(68, 34) ,
            nn.ReLU(),
            nn.Linear(34, 68)
        )

    #
    def forward(self, x: Tensor) -> Tensor:
        #
        return self.net(x)


#
### Transformer block ###
#
class Block(nn.Module) :

    #
    def __init__(self) -> None:

        #
        super().__init__()  # type: ignore
        #
        # n_embd = 68
        # n_head = 2
        #
        # head_size = 34
        # head_size: int = n_embd // n_head
        #
        ### self attention. ###
        #
        self.sa: MultiHeadAttention = MultiHeadAttention()

        #
        ### Feed Forward. ###
        #
        self.ffwd: FeedFoward = FeedFoward()

        #
        ### Layer Norms. ###
        #
        self.ln1: LayerNorm = LayerNorm(68)
        self.ln2: LayerNorm = LayerNorm(68)

    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### add and norm as original paper sa => add norm ff => add and norm. ###
        #
        y: Tensor = self.sa(x)

        #
        x = self.ln1(x+y)

        #
        y = self.ffwd(x)

        #
        x = self.ln2(x+y)

        #
        return x

#
class DeepArcNet(nn.Module) :

    #
    def __init__(self) :

        #
        super().__init__()  # type: ignore

        #
        # hidden_size: int = 68 # features_after_conv * pos_encode
        # num_heads: int = 2
        # time_step: int = 6 # num_outputs
        # n_layer: int = 2

        #
        self.ConvEncode = nn.ModuleList([
            ConvEncode(),
            ConvEncode(),
            ConvEncode(),
            ConvEncode(),
            ConvEncode(),
            ConvEncode()
        ])

        #
        self.conv0 = None
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.conv5 = None

        #
        ### encode layer. ###
        #
        self.blocks = nn.Sequential(
            Block(),
            Block()
        )

        #
        ### final layer. ###
        #
        self.ln_f = LayerNorm(n_embd)

        #
        self.fc =  nn.Sequential(
            nn.Linear(408, 6),
            nn.ReLU()
        )


    #
    def forward(self, x: Tensor) -> Tensor:

        #
        tmp: tuple[int, ...] = x.size() # (Batch, Window_time, Feature_dimension_1 , Feature_dimension_2)
        #
        b: int = tmp[0]
        #
        # wt, fd1, fd2 = tmp[1], tmp[2], tmp[3]

        # #
        # ### Split input tensor and apply each ConvEncode module. ###
        # #
        # split_tensors: tuple[Tensor, ...] = x.split(1, dim=1)
        # #
        # n: int = len(split_tensors) - 1
        # #
        # x0: Tensor = split_tensors[0]
        # x1: Tensor = split_tensors[min(1, n)]
        # x2: Tensor = split_tensors[min(2, n)]
        # x3: Tensor = split_tensors[min(3, n)]
        # x4: Tensor = split_tensors[min(4, n)]
        # x5: Tensor = split_tensors[min(5, n)]
        # #
        # module0: ConvEncode = self.ConvEncode[0]
        # module1: ConvEncode = self.ConvEncode[1]
        # module2: ConvEncode = self.ConvEncode[2]
        # module3: ConvEncode = self.ConvEncode[3]
        # module4: ConvEncode = self.ConvEncode[4]
        # module5: ConvEncode = self.ConvEncode[5]

        # #
        # res0: Tensor = module0(x0)
        # res1: Tensor = module1(x1)
        # res2: Tensor = module2(x2)
        # res3: Tensor = module3(x3)
        # res4: Tensor = module4(x4)
        # res5: Tensor = module5(x5)

        # #
        # output_ConvEncode: list[Tensor] = [res0, res1, res2, res3, res4, res5]

        # #
        # concatenated_ConvEncode: Tensor = torch.cat(output_ConvEncode, dim= -2) # (Batch, Window_time, Convol_feature)


        # Explicitly slice and process each time step (no loops, no zip, no split)
        out0 = self.conv0(x[:, 0:1, :, :])
        out1 = self.conv1(x[:, 1:2, :, :])
        out2 = self.conv2(x[:, 2:3, :, :])
        out3 = self.conv3(x[:, 3:4, :, :])
        out4 = self.conv4(x[:, 4:5, :, :])
        out5 = self.conv5(x[:, 5:6, :, :])

        # Concatenate
        concatenated_ConvEncode = torch.cat([out0, out1, out2, out3, out4, out5], dim=-2)

        #
        self_attention_output: Tensor = self.blocks(concatenated_ConvEncode)

        #
        layer_norm_output: Tensor = self.ln_f(self_attention_output) # (B,T,C)

        #
        classes_output: Tensor  = self.fc(layer_norm_output.view(b,1,-1)) # (B,1,C) no more time step only one class

        #
        probabilities: Tensor = classes_output.squeeze(1)

        #
        return probabilities.float()
