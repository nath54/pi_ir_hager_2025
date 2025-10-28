"""
Model Name: Global Max Pooling with Depth

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - h0: Hidden dimension
    - depth: Number of hidden layers

Data variables:
    - B: Batch size

Architecture:
    - GlobalMaxPool (B, 30, 10) -> (B, 10)
    - Linear (B, 10) -> (B, h0)
    - ReLU (B, h0) -> (B, h0)
    - [Linear (B, h0) -> (B, h0) -> ReLU (B, h0) -> (B, h0)] * (depth - 1)
    - Linear (B, h0) -> (B, 1)
"""

#
### Import Modules. ###
#
from torch import nn
from torch import Tensor


#
### Model Class. ###
#
class Model(nn.Module):

    #
    ### Init Method. ###
    #
    def __init__(self, h0: int = 16, depth: int = 2) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.global_pool: nn.AdaptiveMaxPool1d = nn.AdaptiveMaxPool1d(1)
        #
        self.lin1: nn.Linear = nn.Linear(in_features=10, out_features=h0)
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.hidden_layers: nn.Sequential = self._make_hidden_layers(h0, depth)
        #
        self.lin_final: nn.Linear = nn.Linear(in_features=h0, out_features=1)


    #
    ### Make Hidden Layers Method. ###
    #
    def _make_hidden_layers(self, h0: int, depth: int) -> nn.Sequential:

        #
        layers: list = []

        #
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_features=h0, out_features=h0))
            layers.append(nn.ReLU())

        #
        return nn.Sequential(*layers)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.permute(0, 2, 1)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.hidden_layers(x)
        x = self.lin_final(x)

        #
        return x