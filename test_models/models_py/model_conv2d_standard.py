"""
Model Name: 2D Convolution

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels
    - k_h: Kernel height
    - k_w: Kernel width
    - p: Pool size

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Conv2d (B, 1, 30, 10) -> (B, c0, 30-k_h+1, 10-k_w+1)
    - ReLU (B, c0, 30-k_h+1, 10-k_w+1) -> (B, c0, 30-k_h+1, 10-k_w+1)
    - MaxPool2d (B, c0, 30-k_h+1, 10-k_w+1) -> (B, c0, (30-k_h+1)//p, (10-k_w+1)//p)
    - Flatten (B, c0, (30-k_h+1)//p, (10-k_w+1)//p) -> (B, c0 * ((30-k_h+1)//p) * ((10-k_w+1)//p))
    - Linear (B, c0 * ((30-k_h+1)//p) * ((10-k_w+1)//p)) -> (B, 1)
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
    def __init__(self, c0: int = 8, k_h: int = 3, k_w: int = 3, p: int = 2) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.conv2d: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=(k_h, k_w))
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.maxpool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=p)
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        H_prime: int = 30 - k_h + 1
        #
        W_prime: int = 10 - k_w + 1
        #
        H_pooled: int = H_prime // p
        #
        W_pooled: int = W_prime // p
        #
        self.lin: nn.Linear = nn.Linear(in_features=c0 * H_pooled * W_pooled, out_features=1)


    #
    ### Forward Method. ###
    #
    def forward(self, x: Tensor) -> Tensor:

        #
        ### Forward pass. ###
        #
        x = x.unsqueeze(1)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.lin(x)

        #
        return x
