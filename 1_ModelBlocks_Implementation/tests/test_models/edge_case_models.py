"""
Edge case models for testing various scenarios.
"""

import torch
import torch.nn as nn


class EmptyModel(nn.Module):
    """Model with no layers - just identity."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class SingleLayerModel(nn.Module):
    """Model with only one layer."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


class NoWeightModel(nn.Module):
    """Model with no trainable parameters."""
    
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.identity = nn.Identity()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(x)
        x = self.identity(x)
        x = self.dropout(x)
        return x


class ConditionalModel(nn.Module):
    """Model with conditional logic."""
    
    def __init__(self, use_batch_norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.use_batch_norm = use_batch_norm
    
    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        return x


class LoopModel(nn.Module):
    """Model with loops in forward pass."""
    
    def __init__(self, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(10, 10))
        self.relu = nn.ReLU()
        self.num_layers = num_layers
    
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.relu(x)
        return x


class MultiInputModel(nn.Module):
    """Model with multiple inputs."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 3)
        self.bilinear = nn.Bilinear(5, 3, 2)
    
    def forward(self, x1, x2):
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x = self.bilinear(x1, x2)
        return x


class MultiOutputModel(nn.Module):
    """Model with multiple outputs."""
    
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(10, 20)
        self.output1 = nn.Linear(20, 5)
        self.output2 = nn.Linear(20, 3)
    
    def forward(self, x):
        x = self.shared(x)
        out1 = self.output1(x)
        out2 = self.output2(x)
        return out1, out2


class CustomLayerModel(nn.Module):
    """Model with custom layer implementation."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.custom_activation = CustomActivation()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.custom_activation(x)
        return x


class CustomActivation(nn.Module):
    """Custom activation function."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.tanh(x) * 0.5


class LargeModel(nn.Module):
    """Large model for performance testing."""
    
    def __init__(self, num_layers=50):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        self.final = nn.Linear(100, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return x


if __name__ == "__main__":
    # Test all models
    models = {
        "EmptyModel": EmptyModel(),
        "SingleLayerModel": SingleLayerModel(),
        "NoWeightModel": NoWeightModel(),
        "ConditionalModel": ConditionalModel(),
        "LoopModel": LoopModel(),
        "MultiInputModel": MultiInputModel(),
        "MultiOutputModel": MultiOutputModel(),
        "CustomLayerModel": CustomLayerModel(),
        "LargeModel": LargeModel(10)  # Smaller for testing
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # Test forward pass
        try:
            if name == "MultiInputModel":
                x1 = torch.randn(2, 10)
                x2 = torch.randn(2, 5)
                output = model(x1, x2)
            else:
                x = torch.randn(2, 10)
                output = model(x)
            
            if isinstance(output, tuple):
                print(f"  Output shapes: {[o.shape for o in output]}")
            else:
                print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"  Error: {e}")
