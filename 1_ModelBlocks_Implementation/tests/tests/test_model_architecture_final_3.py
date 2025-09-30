#
### Import Modules. ###
#
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


#
### Params. ###
#
pos_encode: int = 4
features_after_conv: int = 17
n_embd: int = 68 # features_after_conv * pos_encode
num_outputs: int = 6
batch_size: int = 40


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

        #
        out: Tensor = torch.cat( [h(x) for h in self.heads], dim = -1 ) # (Batch, Time, Channel Feature dimension) = (B,T , [h1,h1,h2,h2,h3,h3,h4,h4])

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
        self.ln1: nn.LayerNorm = nn.LayerNorm(68)
        self.ln2: nn.LayerNorm = nn.LayerNorm(68)

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
        ### encode layer. ###
        #
        self.blocks = nn.Sequential(
            Block(),
            Block()
        )

        #
        ### final layer. ###
        #
        self.ln_f = nn.LayerNorm(n_embd)

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

        #
        ### Split input tensor and apply each ConvEncode module. ###
        #
        split_tensors = x.split(1, dim=1)
        #
        output_ConvEncode: list[Tensor] = []
        #
        for i, xi in enumerate(split_tensors):
            #
            module_c = self.ConvEncode[i]
            #
            output_ConvEncode.append(module_c(xi))

        #
        concatenated_ConvEncode: Tensor = torch.cat(output_ConvEncode, dim= -2) # (Batch, Window_time, Convol_feature)

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
