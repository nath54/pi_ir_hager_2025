#
from torch import Tensor
from torch import nn


#
class SimpleModel(nn.Module):

    #
    def __init__(self, in_features: int, mid_features: int, out_features: int) -> None:

        #
        super(SimpleModel, self).__init__()  # type: ignore

        #
        self.linear1 = nn.Linear(in_features, mid_features)
        self.linear2 = nn.Linear(mid_features, out_features)

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

