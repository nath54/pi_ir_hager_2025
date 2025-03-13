import torch  # type: ignore
from torch import Tensor
import torch.nn as nn  # type: ignore

#
RANDOM_CONSTANT1: int = 1
RANDOM_CONSTANT2: float = 0.1

#
class SimpleModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(10, 20)
        self.fc2: nn.Linear = nn.Linear(20, 30)

    def forward(self, x: Tensor):
        y: Tensor = RANDOM_CONSTANT1
        x = self.fc1(x)
        x += y * RANDOM_CONSTANT2
        x = self.fc2(x)
        return x