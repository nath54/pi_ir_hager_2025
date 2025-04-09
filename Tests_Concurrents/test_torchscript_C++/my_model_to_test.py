#
import torch  # type: ignore
from torch import Tensor  # type: ignore
import torch.nn as nn  # type: ignore

#
#RANDOM_CONSTANT1: int = 1
#RANDOM_CONSTANT2: float = 0.1

#
class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.i: int = 20
        self.fc1: nn.Linear = nn.Linear(10, self.i)
        self.fc2: nn.Linear = nn.Linear(self.i, 30)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        y: Tensor =  torch.ones(x.shape)
        x += y * 0.1
        x = self.fc2(x)
        return x
    
my_model = Model()
sm = torch.jit.script(my_model)
print(sm.code)
print(sm.graph)
sm.save("my_model.pt")