"""
Complex model with multiple layer types for testing.
"""

import torch
import torch.nn as nn


class ComplexModel(nn.Module):
    """A complex model with multiple layer types including skip connections."""
    
    def __init__(self):
        super().__init__()
        # Initial convolutional block
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual-like block
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Additional layers
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(64)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1000)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual-like block with skip connection
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity  # Skip connection
        x = self.relu(x)
        
        # Final processing
        x = self.dropout(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    # Test the model
    model = ComplexModel()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model: {model}")
