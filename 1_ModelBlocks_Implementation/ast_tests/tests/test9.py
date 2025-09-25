#
from torch import Tensor
from torch import nn


#
class SimpleModel(nn.Module):

    #
    def __init__(self, in_features: int = 10, mid_features: int = 40, out_features: int = 5) -> None:

        #
        super(SimpleModel, self).__init__()  # type: ignore

        #
        self.linear1 = nn.Linear(in_features=in_features, out_features=mid_features)
        self.linear2 = nn.Linear(in_features=mid_features, out_features=out_features)

        #
        self.relu = nn.ReLU()

    #
    def forward(self, x: Tensor) -> Tensor:

        #
        b: int = x.shape[0]

        #
        print(f"Batch size: {b}")

        #
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        #
        return x

