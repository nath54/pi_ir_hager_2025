#
### Import Modules. ###
#
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
#
import math


#
### Params. ###
#
pos_encode: int = 4
features_after_conv: int = 17
n_embd: int = 68 # features_after_conv * pos_encode
dropout: float = 0.0
learning_rate: float = 0.006
num_outputs: int = 6
batch_size: int = 40


#
class SinusoidalPositionEmbeddings(nn.Module):

    #
    def __init__(self, dim: int) -> None:
        #
        super().__init__()  # type: ignore
        #
        self.dim: int = dim


    #
    def forward(self, time: Tensor) -> Tensor:

        #
        device: str | torch.device = time.device
        #
        half_dim: int = self.dim // 2
        #
        embeddings_f: float = math.log( features_after_conv + 1  ) / (half_dim - 1)
        #
        embeddings: Tensor = torch.exp(torch.arange(half_dim, device=device) * -embeddings_f)
        embeddings = time[:, None] * embeddings[None, :]
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
        self.posembd: SinusoidalPositionEmbeddings = SinusoidalPositionEmbeddings(dim = 4)

    #
    def forward(self, x: Tensor) -> Tensor :
        #
        device: str | torch.device = x.device
        #
        b: int
        _wt: int
        _fd1: int
        _fd2: int
        b, _wt, _fd1, _fd2 = x.size()
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
    def __init__(self, head_size: int):

        #
        super().__init__()  # type: ignore

        #
        self.key: nn.Linear = nn.Linear(n_embd, head_size , bias = False )
        self.query: nn.Linear = nn.Linear(n_embd, head_size , bias = False )
        self.value: nn.Linear = nn.Linear(n_embd, head_size , bias = False )

        #
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    #
    def forward(self, x: Tensor) -> Tensor:

        #
        _b, _t, _c = x.shape

        #
        k: Tensor = self.key(x) # (_b, _t, head_size)
        q: Tensor = self.query(x) # (_b, _t, head_size)

        #
        ### compute attention score ('affinities'). ###
        #
        wei: Tensor = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) to (B, T, T)
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

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
    def __init__(self, num_heads: int, head_size: int) -> None:

        #
        super().__init__()  # type: ignore

        #
        ### head do in parallel. ###
        #
        self.heads = nn.ModuleList([
            Head(head_size),
            Head(head_size)
        ])

        #
        ### projection layer. ###
        #
        self.proj = nn.Linear(head_size*num_heads , n_embd)

        #
        self.dropout = nn.Dropout(dropout)

    #
    def forward(self, x: Tensor) -> Tensor:

        #
        out: Tensor = torch.cat( [h(x) for h in self.heads], dim = -1 ) # (Batch, Time, Channel Feature dimension) = (B,T , [h1,h1,h2,h2,h3,h3,h4,h4])

        #
        out = self.dropout(self.proj(out))

        #
        return out


#
class FeedFoward(nn.Module):

    #
    def __init__(self, n_embd: int) -> None:

        #
        super().__init__()  # type: ignore
        #
        self.net = nn.Sequential(
            nn.Linear(n_embd, int(n_embd/2)) ,
            nn.ReLU() ,
            nn.Linear(int(n_embd/2) , n_embd) ,
            nn.Dropout(dropout),
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
    def __init__(self, n_embd: int, n_head: int) -> None:

        #
        super().__init__()  # type: ignore
        #
        head_size: int = n_embd // n_head
        #
        ### self attention. ###
        #
        self.sa: MultiHeadAttention = MultiHeadAttention(n_head, head_size)

        #
        ### Feed Forward. ###
        #
        self.ffwd: FeedFoward = FeedFoward(n_embd)

        #
        ### Layer Norms. ###
        #
        self.ln1: nn.LayerNorm = nn.LayerNorm(n_embd)
        self.ln2: nn.LayerNorm = nn.LayerNorm(n_embd)

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
        hidden_size: int = 68 # features_after_conv * pos_encode
        num_heads: int = 2
        time_step: int = 6 # num_outputs
        _n_layer: int = 2

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
        ### encode layer. ###
        #
        self.blocks = nn.Sequential(
            Block(68, n_head = num_heads ),
            Block(68, n_head = num_heads )
        )

        #
        ### final layer. ###
        #
        self.ln_f = nn.LayerNorm(n_embd)

        #
        self.fc =  nn.Sequential(
            nn.Linear(hidden_size*time_step, num_outputs),
            nn.ReLU()
        )


    #
    def forward(self, x: Tensor) -> Tensor:

        #
        _b: int
        _wt: int
        _fd1: int
        _fd2: int
        #
        _b, _wt, _fd1, _fd2 = x.size() # (Batch, Window_time, Feature_dimension_1 , Feature_dimension_2)

        #
        output_ConvEncode: list[Tensor] = [ module_c(xi) for ( xi , module_c) in zip( x.split(1, dim=1), self.ConvEncode ) ]  # type: ignore

        #
        concatenated_ConvEncode: Tensor = torch.cat(output_ConvEncode, dim= -2) # (Batch, Window_time, Convol_feature)

        #
        self_attention_output: Tensor = self.blocks(concatenated_ConvEncode)

        #
        layer_norm_output: Tensor = self.ln_f(self_attention_output) # (B,T,C)

        #
        classes_output: Tensor  = self.fc(layer_norm_output.view(_b,1,-1)) # (B,1,C) no more time step only one class

        #
        probabilities: Tensor = classes_output.squeeze(1)

        #
        return probabilities.float()
