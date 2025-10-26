"""
Model Name: Squeeze-and-Excitation Network (SENet)

Model Input: (B, 30, 10)  # Batch, dim feature 1, dim feature 2
Model Output: (B, 1)  # Batch, Prediction Value

Models parameters:
    - c0: Number of output channels
    - k_h: Kernel height
    - k_w: Kernel width
    - reduction_ratio: Reduction ratio for SE block

Data variables:
    - B: Batch size

Architecture:
    - Reshape (B, 30, 10) -> (B, 1, 30, 10)
    - Conv2d (B, 1, 30, 10) -> (B, c0, 30-k_h+1, 10-k_w+1)
    - ReLU (B, c0, 30-k_h+1, 10-k_w+1) -> (B, c0, 30-k_h+1, 10-k_w+1)
    - Squeeze: GlobalAvgPool2d -> (B, c0, 1, 1)
    - Excitation: Linear -> ReLU -> Linear -> Sigmoid -> (B, c0, 1, 1)
    - Scale: Multiply with original features
    - Flatten (B, c0, 30-k_h+1, 10-k_w+1) -> (B, c0*(30-k_h+1)*(10-k_w+1))
    - Linear (B, c0*(30-k_h+1)*(10-k_w+1)) -> (B, 1)
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
    def __init__(self, c0: int, k_h: int, k_w: int, reduction_ratio: int) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.conv2d: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=c0, kernel_size=(k_h, k_w))
        #
        self.relu: nn.ReLU = nn.ReLU()
        #
        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        #
        self.fc1: nn.Linear = nn.Linear(in_features=c0, out_features=c0 // reduction_ratio)
        #
        self.fc2: nn.Linear = nn.Linear(in_features=c0 // reduction_ratio, out_features=c0)
        #
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()
        #
        self.flatten: nn.Flatten = nn.Flatten()
        #
        H_prime: int = 30 - k_h + 1
        #
        W_prime: int = 10 - k_w + 1
        #
        self.lin: nn.Linear = nn.Linear(in_features=c0 * H_prime * W_prime, out_features=1)


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
        #
        x_residual = x
        #
        x_se = self.global_pool(x)
        x_se = x_se.squeeze(-1).squeeze(-1)
        x_se = self.fc1(x_se)
        x_se = self.relu(x_se)
        x_se = self.fc2(x_se)
        x_se = self.sigmoid(x_se)
        x_se = x_se.unsqueeze(-1).unsqueeze(-1)
        #
        x = x_residual * x_se
        #
        x = self.flatten(x)
        x = self.lin(x)

        #
        return x
