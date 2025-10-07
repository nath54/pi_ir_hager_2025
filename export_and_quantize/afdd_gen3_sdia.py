import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn import functional as F
import random
import pickle
import glob
from torch.utils.data import Dataset  # type: ignore

import numpy as np
from numpy.typing import NDArray
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'



#Params
pos_encode = 4
features_after_conv = 17
n_embd = features_after_conv*pos_encode
dropout = 0.0
learning_rate = 0.006
num_outputs = 6
BATCH_SIZE = 40


class CustomDataset_onRAM_dmp(Dataset):
    def __init__(self,path):
        self.data_path = path
        file_list = glob.glob(self.data_path + "*.pkl")
        self.block_size = num_outputs
        self.sample_per_hc = 38
        print(file_list)
        self.data_list = []
        self.score_list = []
        all_time_min  = np.zeros(5)+2000
        all_time_max = np.zeros(5)

        for file in file_list:

            raw_data = pickle.load(open(file,'rb'))
            class_data = raw_data[1]
            data =raw_data[0]
            stacked_data = np.vstack(data,dtype='float32')

            #Find min max of current sample
            current_min_values = np.min(stacked_data, axis=0)
            current_max_values = np.max(stacked_data, axis=0)
            #Store sample information

            self.data_list.append(stacked_data)
            self.score_list.append(class_data)

            #Update min max value
            # Iterate over the zipped arrays and update array with the minimum and maximum values
            all_time_min = [min(val1, val2) for val1, val2 in zip(all_time_min, current_min_values)]
            all_time_max =[max(val1, val2) for val1, val2 in zip(all_time_max, current_max_values)]

        print('min',all_time_min)
        print('max',all_time_max)
        torch.save(torch.stack((torch.tensor(all_time_min),torch.tensor(all_time_max))) , 'norm_coeff.tch' )
        print('data size' , len ( self.data_list) , 'score size' , len ( self.score_list))

        for sample in self.data_list :
            for i in range (np.size(sample,axis = 1 ) ) :

                sample[:,i] = (( sample[:,i] - all_time_min[i] ) / (all_time_max[i] - all_time_min[i] )*2 ) -1


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx) :


        data  = self.data_list[idx]
        #Pick a random sample
        index = random.randint(0,len(data)/38 - self.block_size )
        data = np.asarray (data[self.sample_per_hc*index : self.sample_per_hc*(index + self.block_size) ])
        class_score = self.score_list[idx]
        class_score = np.asarray(class_score[index: index + self.block_size ])
        class_score = np.hstack(class_score)
        class_score = torch.tensor(class_score)
        data_sensor = torch.tensor(data)
        return data_sensor.view(num_outputs,38,5).float(), class_score.float()


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

def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

        output_ConvEncode = [ module_c(xi) for ( xi , module_c) in zip  (x.split(1, dim=1) , self.ConvEncode ) ]

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




if __name__ == "__main__":

    # trainset = CustomDataset_onRAM_dmp('/database/')
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)


    model =  DeepArcNet()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    #test space


    # for batch, label in trainloader:
    #     model(batch,label)



    # count_parameters(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # m = model.to(device)





    # NO_EPOCHS = 1000

    # PRINT_FREQUENCY = 10
    # import time
    # start_time = time.time()
    # for epoch in range(NO_EPOCHS):
    #     mean_epoch_loss = []
    #     mean_epoch_loss_val = []
    #     for batch, label in trainloader:

    #         prob,loss = model(batch.to(device),label.to(device))
    #         mean_epoch_loss.append(loss.item())
    #         # mean_epoch_loss.append(loss.item())
    #         optimizer.zero_grad(set_to_none=True)
    #         loss.backward()
    #         optimizer.step()

    #     for batch, label in trainloader:

    #         prob,loss = model(batch.to(device),label.to(device))

    #         mean_epoch_loss_val.append(loss.item())

    #     if epoch % PRINT_FREQUENCY == 0:

    #         print('---')
    #         print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss):.4f} | Val Loss {np.mean(mean_epoch_loss_val):.4f}")
    #         end_time = time.time()
    #         execution_time = end_time - start_time
    #         start_time = end_time
    #         print(f"Execution Time: {execution_time} seconds")

    # #save model
    # torch.save(model.state_dict(), 'model.pth')
#Test

# for batch, label in trainloader:

#     prob,loss = model(batch.squeeze(0),label.squeeze(0))
#     np.set_printoptions(precision=3)
#     print('label', label.numpy())
#     print('prob',prob.detach().numpy())

