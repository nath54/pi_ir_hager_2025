f"""
Model Name: Simple Linear with N hidden dim

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - N: Number of hidden dimensions
    - h_i: Array of hidden dimensions (N = len(h_i) && len(h_i) > 2)

Data variables:
    - B: Batch size

Architecture:
    - Linear (B, 30, 10) -> (B, 30, h_i[0])
    - ReLU (B, 30, h_i[0]) -> (B, 30, h_i[0])
    - Flatten (B, 30, h_i[0]) -> (B, 30 * h_i[0])
    - Linear (B, 30 * h_i[0]) -> (B, h_i[1])
    - ReLU (B, h_i[1]) -> (B, h_i[1])
    - for i in range(2, N):
        - Linear (B, h_[i-1]) -> (B, h_[i])
        - ReLU (B, h_[i])-> (B, h_[i])
    - Linear (B, h_i[N-1]) -> (B, 1)
"""

#
### Import Modules. ###
#
import torch
from torch import nn
from torch import Tensor


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(self, h_i: list[int] = [16, 32, 64, 128, 64, 32, 16]) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.N: int = len(h_i)

        #
        assert self.N > 2, f"Error: not enough hidden dimension ! ({self.N} is not > 2)"

        #
        self.lin1: nn.Linear = nn.Linear(in_features=10, out_features=h_i[0])
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        self.lin2: nn.Linear = nn.Linear(in_features=30 * h_i[0], out_features=h_i[1])
        #
        self.lins: nn.ModuleList = nn.ModuleList([
            #
            nn.Linear(in_features=h_i[i-1], out_features=h_i[i])
            #
            for i in range(2, self.N)
        ])
        #
        self.lin_N: nn.Linear = nn.Linear(in_features=h_i[self.N-1], out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = self.lin1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.lin2(x)
        x = self.relu(x)
        #
        for lin in self.lins:
            #
            x = lin(x)
            x = self.relu(x)
        #
        x = self.lin_N(x)

        #
        return x


#
if __name__ == "__main__":

    #
    m = Model()

    #
    ri: Tensor = torch.randn((1,30,10))

    #
    print(m(ri))