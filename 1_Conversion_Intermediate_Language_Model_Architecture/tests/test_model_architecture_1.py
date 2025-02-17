import torch  # type: ignore
import torch.nn as nn  # type: ignore

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 30)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x