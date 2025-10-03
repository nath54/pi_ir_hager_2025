 #
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn import functional as F
#
import math


#Params
pos_encode: int = 4
features_after_conv: int = 17
n_embd: int = features_after_conv*pos_encode
dropout: float = 0.0
learning_rate: float = 0.006
num_outputs: int = 6
BATCH_SIZE: int = 40


#
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log( features_after_conv + 1  ) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

#
class ConvEncode(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.embdim = dim
        self.conv = nn.Sequential( nn.Conv2d(in_channels =1 , out_channels =1 , kernel_size = (5,5),  stride=(2)),
                                    nn.ReLU())
        self.linearembd = nn.Linear(1,self.embdim)
        self.posembd = SinusoidalPositionEmbeddings (self.embdim)

    def forward (self,x) :
        device = x.device
        B,WT,FD1,FD2 = x.size()
        x_conv = self.conv(x)
        time_tensor = torch.arange(x_conv.shape[2],device = device ) # Feature dimension 1 time encode each sample from 0 to 38
        time_embedded = self.posembd(time_tensor)
        time_embedded2 = time_embedded.unsqueeze(0).unsqueeze(1).expand(B, -1, -1, -1) #Duplicate the time vector along the batch dimension
        x_emb = self.linearembd(x_conv)
        x_code = time_embedded2+x_emb
        x_code = x_code.view(B,1, self.embdim*x_conv.shape[2]) #Shape of feature after conv multiply with emb dimension then code to batch with 1 time step before attention
        return x_code


#
class Head (nn.Module) :
    def __init__(self , head_size) :
        super().__init__()

        self.key = nn.Linear(n_embd, head_size , bias = False )
        self.query = nn.Linear(n_embd, head_size , bias = False )
        self.value = nn.Linear(n_embd, head_size , bias = False )

        self.dropout = nn.Dropout(dropout)

    def forward (self,x) :

        B,T,C = x.shape
        k = self.key(x) # B T head_size
        q = self.query(x) # B T head_size
        # compute attention score ('affinities')
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # B T hs @ B hs T  to B T T
        wei = F.softmax ( wei , dim = - 1 )
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        # print('head ok')
        return out


#
class MultiHeadAttention(nn.Module) :

    def __init__(self, num_heads , head_size) :
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range ( num_heads)) # head do in parallel
        self.proj = nn.Linear(head_size*num_heads , n_embd) #projection layer
        self.dropout = nn.Dropout(dropout)
    def forward ( self, x) :
        out = torch.cat( [h(x) for h in self.heads], dim = -1 ) #(Batch, Time , Channel Feature dimension ) = (B,T , [h1,h1,h2,h2,h3,h3,h4,h4])
        out = self.dropout(self.proj(out))
        return out


#
class FeedFoward ( nn.Module) :
    def __init__ ( self , n_embd) :
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, int(n_embd/2)) ,
            nn.ReLU() ,
            nn.Linear(int(n_embd/2) , n_embd) ,
            nn.Dropout(dropout),
            )
    def forward (self,x) :
        return self.net(x)


#
class Block(nn.Module) :
    #Transformer block :
        def __init__(self, n_embd , n_head) :
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention ( n_head, head_size) # self attention
            self.ffwd = FeedFoward ( n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
        def forward (self , x) :
            #add and norm as original paper sa => add norm ff => add and norm
            y = self.sa(x)
            x = self.ln1(x+y)
            y = self.ffwd(x)
            x = self.ln2(x+y)
            return x


#
def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#
class DeepArcNet(nn.Module) :
    def __init__(self) :
        super().__init__()
        hidden_size = features_after_conv*pos_encode
        num_heads = 2
        time_step = num_outputs
        n_layer= 2

        self.ConvEncode = nn.ModuleList( ConvEncode(pos_encode) for _ in range(time_step)  )

        self.blocks = nn.Sequential(*(Block(features_after_conv*pos_encode  , n_head = num_heads ) for _ in range(n_layer))) # encode layer

        self.ln_f = nn.LayerNorm(n_embd) #final layer

        self.fc =  nn.Sequential( nn.Linear(hidden_size*time_step, num_outputs),
                nn.ReLU() )



    def forward (self,x, targets = None) :

        B,WT,FD1,FD2 = x.size() # Batch, Window time , Feature dimension 1 , Feature dimension 2

        output_ConvEncode = [ module_c(xi) for ( xi , module_c) in zip(x.split(1, dim=1) , self.ConvEncode ) ]

        concatenated_ConvEncode = torch.cat(output_ConvEncode, dim= -2) # Batch , Window time , Convol feature

        selfattention_output = self.blocks(concatenated_ConvEncode)

        LayerNorm_output = self.ln_f(selfattention_output) # (B,T,C)

        Classes_output  = self.fc(LayerNorm_output.view(B,1,-1)) # (B,1,C) no more time step only one class

        Probabilities = Classes_output.squeeze(1)


        if targets is None :
            loss = None
            return  Probabilities.float()
        else :
            #batch time channel

            loss = F.mse_loss(Probabilities,targets)

        return Probabilities.float(), loss.float()
