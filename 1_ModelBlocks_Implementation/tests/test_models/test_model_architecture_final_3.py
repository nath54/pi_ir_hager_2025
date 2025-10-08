#
### Import Modules. ###
#
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
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
        track(embeddings, "SinusoidalPositionEmbeddings | embeddings 1")
        #
        embeddings = time[:, None] * embeddings[None, :]
        #
        track(embeddings, "SinusoidalPositionEmbeddings | embeddings 2")
        #
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        #
        track(embeddings, "SinusoidalPositionEmbeddings | embeddings 3")
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
        track(x_conv, "ConvEncode | x_conv")

        #
        ### Feature dimension 1 time encode each sample from 0 to 38. ###
        #
        time_tensor: Tensor = torch.arange(x_conv.shape[2], device = device)

        #
        track(time_tensor, "ConvEncode | time_tensor")

        #
        time_embedded: Tensor = self.posembd(time_tensor)

        #
        track(time_embedded, "ConvEncode | time_embedded")

        #
        ### Duplicate the time vector along the batch dimension. ###
        #
        time_embedded2: Tensor = time_embedded.unsqueeze(0).unsqueeze(1).expand(b, -1, -1, -1)

        #
        track(time_embedded2, "ConvEncode | time_embedded2")

        #
        x_emb: Tensor = self.linearembd(x_conv)

        #
        track(x_emb, "ConvEncode | x_emb")

        #
        x_code: Tensor = time_embedded2 + x_emb

        #
        track(x_code, "ConvEncode | x_code 1")

        #
        ### Shape of feature after conv multiply with emb dimension then code to batch with 1 time step before attention. ###
        #
        x_code: Tensor = x_code.view(b,1, 4 * x_conv.shape[2])

        #
        track(x_code, "ConvEncode | x_conv 2")

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
        track(k)
        track(q)

        #
        ### compute attention score ('affinities'). ###
        #
        wei: Tensor = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) to (B, T, T)

        #
        track(wei, "Head | wei 1")

        #
        wei = F.softmax(wei, dim = -1)

        #
        track(wei, "Head | wei 2")

        #
        ### perform the weighted aggregation of the values. ###
        #
        v: Tensor = self.value(x)

        #
        track(v, "Head | v")

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

        #
        track(head_out_0, "MultiHeadAttention | head_out_0")
        track(head_out_1, "MultiHeadAttention | head_out_1")

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
        self.ln1: nn.LayerNorm = nn.LayerNorm(68)
        self.ln2: nn.LayerNorm = nn.LayerNorm(68)

    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### add and norm as original paper sa => add norm ff => add and norm. ###
        #
        y: Tensor = self.sa(x)

        #
        track(y, "Block | y 1")

        #
        x = self.ln1(x+y)

        #
        track(x, "Block | x 1")

        #
        y = self.ffwd(x)

        #
        track(y, "Block | y 2")

        #
        x = self.ln2(x+y)

        #
        track(x, "Block | x 2")

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
        track(x)

        #
        tmp: tuple[int, ...] = x.size() # (Batch, Window_time, Feature_dimension_1 , Feature_dimension_2)
        #
        b: int = tmp[0]
        #
        # wt, fd1, fd2 = tmp[1], tmp[2], tmp[3]

        #
        ### Split input tensor and apply each ConvEncode module. ###
        #
        split_tensors: tuple[Tensor, ...] = x.split(1, dim=1)
        #
        n: int = len(split_tensors) - 1
        #
        x0: Tensor = split_tensors[0]
        x1: Tensor = split_tensors[min(1, n)]
        x2: Tensor = split_tensors[min(2, n)]
        x3: Tensor = split_tensors[min(3, n)]
        x4: Tensor = split_tensors[min(4, n)]
        x5: Tensor = split_tensors[min(5, n)]
        #
        track(x0, "DeepArcNet | x0")
        track(x1, "DeepArcNet | x1")
        track(x2, "DeepArcNet | x2")
        track(x3, "DeepArcNet | x3")
        track(x4, "DeepArcNet | x4")
        track(x5, "DeepArcNet | x5")
        #
        module0: ConvEncode = self.ConvEncode[0]
        module1: ConvEncode = self.ConvEncode[1]
        module2: ConvEncode = self.ConvEncode[2]
        module3: ConvEncode = self.ConvEncode[3]
        module4: ConvEncode = self.ConvEncode[4]
        module5: ConvEncode = self.ConvEncode[5]

        #
        res0: Tensor = module0(x0)
        res1: Tensor = module1(x1)
        res2: Tensor = module2(x2)
        res3: Tensor = module3(x3)
        res4: Tensor = module4(x4)
        res5: Tensor = module5(x5)

        #
        output_ConvEncode: list[Tensor] = [res0, res1, res2, res3, res4, res5]

        #
        concatenated_ConvEncode: Tensor = torch.cat(output_ConvEncode, dim= -2) # (Batch, Window_time, Convol_feature)

        #
        track(concatenated_ConvEncode, "DeepArcNet | concatenated_ConvEncode")

        #
        self_attention_output: Tensor = self.blocks(concatenated_ConvEncode)

        #
        track(self_attention_output, "DeepArcNet | self_attention_output")

        #
        layer_norm_output: Tensor = self.ln_f(self_attention_output) # (B,T,C)

        #
        track(layer_norm_output, "DeepArcNet | layer_norm_output")

        #
        classes_output: Tensor  = self.fc(layer_norm_output.view(b,1,-1)) # (B,1,C) no more time step only one class

        #
        track(classes_output, "DeepArcNet | classes_output")

        #
        probabilities: Tensor = classes_output.squeeze(1)

        #
        track(probabilities, "DeepArcNet | probabilities")

        #
        return probabilities.float()
