#
import torch  # type: ignore
from torch import Tensor  # type: ignore
import torch.nn as nn  # type: ignore

#
RANDOM_CONSTANT1: int = 1
RANDOM_CONSTANT2: float = 0.1
NB_BLOCKS: int = 10
RANDOM_CONSTANT3: int = 20


#
class Block1(nn.Module):

    #
    def __init__(self, i: int, j: int) -> None:

        #
        super().__init__()

        #
        self.lin1: nn.Linear = nn.Linear(i, j)
        self.activ: nn.GLU = nn.GLU()
        self.lin2: nn.Linear = nn.Linear(j, i)

    #
    def forward(self, x: Tensor) -> Tensor:

        #
        x = self.lin1(x)
        x = self.activ(x)
        x = self.lin2(x)

        #
        return x



#
class Model(nn.Module):

    #
    def __init__(self) -> None:
        super().__init__()
        self.i: int = 20
        self.fc1: nn.Linear = nn.Linear(10, RANDOM_CONSTANT3)
        self.seq1: nn.Sequential = nn.Sequential(
            *(Block1(RANDOM_CONSTANT3, 50) for _ in range(NB_BLOCKS))
        )
        self.fc2: nn.Linear = nn.Linear(RANDOM_CONSTANT3, 30)

    #
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        y: Tensor = RANDOM_CONSTANT1 * torch.ones(x.shape)
        x += y * RANDOM_CONSTANT2
        x = self.fc2(x)
        return x
