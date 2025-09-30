#
from torch import Tensor
from torch import nn


#
class SimpleModel(nn.Module):

    #
    def __init__(self) -> None:

        #
        super(SimpleModel, self).__init__()  # type: ignore

        #
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

        #
        self.relu = nn.ReLU()

    #
    def forward(self, x: Tensor) -> Tensor:

        #
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        #
        return x

