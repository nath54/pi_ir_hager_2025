#
import torch  # type: ignore
from torch import Tensor  # type: ignore
import torch.nn as nn  # type: ignore

#
RANDOM_CONSTANT1: int = 1
RANDOM_CONSTANT2: float = 0.1

#
class Model(nn.Module):

    def __init__(self) -> None:

        #
        super().__init__()
        self.i: int = 20
        self.fc1: nn.Linear = nn.Linear(10, self.i)
        self.seq1: nn.ModuleList = nn.ModuleList(
            [
                nn.Linear(self.i, 40),
                nn.ReLU(),
                nn.Linear(40, 50),
                nn.GLU(),
                nn.Linear(50, self.i),
                nn.ReLU()
            ]
        )
        self.fc2: nn.Linear = nn.Linear(self.i, 30)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        y: Tensor = RANDOM_CONSTANT1 * torch.ones(x.shape)
        x += y * RANDOM_CONSTANT2
        x = self.fc2(x)
        return x