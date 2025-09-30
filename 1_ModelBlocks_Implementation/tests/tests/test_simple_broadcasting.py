import torch
import torch.nn as nn
import numpy as np

class SimpleConvEncode(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5), stride=(2)),
            nn.ReLU()
        )
        # Linear layer will be created dynamically based on conv output shape
        self.linearembd = None
        self.posembd = SinusoidalPositionEmbeddings()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        b = x.size(0)
        
        x_conv = self.conv(x)
        print(f"x_conv shape: {x_conv.shape}")
        
        # Time encoding
        time_tensor = torch.arange(x_conv.shape[2], device=device)
        time_embedded = self.posembd(time_tensor)
        print(f"time_embedded shape: {time_embedded.shape}")
        
        # Duplicate time vector along batch dimension and match spatial dimensions
        time_embedded2 = time_embedded.unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(b, 1, -1, x_conv.shape[3], -1)
        print(f"time_embedded2 shape: {time_embedded2.shape}")
        
        # Linear embedding - create layer dynamically
        if self.linearembd is None:
            # Apply Linear to each spatial location
            self.linearembd = nn.Linear(in_features=x_conv.shape[1], out_features=4).to(device)
        
        # Reshape for Linear layer: (batch, channels, height, width) -> (batch*height*width, channels)
        batch_size, channels, height, width = x_conv.shape
        x_conv_reshaped = x_conv.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        x_emb = self.linearembd(x_conv_reshaped)
        
        # Reshape back to (batch, height, width, features)
        x_emb = x_emb.view(batch_size, height, width, -1)
        print(f"x_emb shape: {x_emb.shape}")
        
        # Add time and spatial embeddings
        x_code = time_embedded2 + x_emb
        print(f"x_code shape after addition: {x_code.shape}")
        
        # Reshape to flatten spatial dimensions
        x_code = x_code.view(b, 1, 4 * x_conv.shape[2])
        print(f"x_code final shape: {x_code.shape}")
        
        return x_code

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, d_model: int = 4):
        super().__init__()
        self.d_model = d_model

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.d_model // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def test_simple_model():
    print("=== Testing Simple ConvEncode Model ===")
    
    # Create model
    model = SimpleConvEncode()
    
    # Create input tensor similar to the complex model
    # Input: (batch=3, channels=128, height=16, width=8)
    x = torch.randn(3, 128, 16, 8)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Final output shape: {output.shape}")
    
    return output

def test_individual_operations():
    print("\n=== Testing Individual Operations ===")
    
    # Test Conv2d
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5), stride=(2))
    x_conv = torch.randn(3, 1, 16, 8)
    conv_output = conv(x_conv)
    print(f"Conv2d input: {x_conv.shape}, output: {conv_output.shape}")
    
    # Test Linear - We need to apply Linear to each spatial location
    # So for (3, 1, 6, 2), we want to apply Linear to each (6, 2) location
    # This means we need to reshape to (3*6*2, 1) and apply Linear(1, 4)
    # Then reshape back to (3, 6, 2, 4)
    print(f"Conv output shape: {conv_output.shape}")
    
    # Reshape to (batch*height*width, channels) for Linear layer
    batch_size, channels, height, width = conv_output.shape
    conv_reshaped = conv_output.permute(0, 2, 3, 1).contiguous().view(-1, channels)
    print(f"Conv reshaped for linear: {conv_reshaped.shape}")
    
    linear = nn.Linear(in_features=channels, out_features=4)
    linear_output = linear(conv_reshaped)
    print(f"Linear output: {linear_output.shape}")
    
    # Reshape back to (batch, height, width, features)
    linear_output = linear_output.view(batch_size, height, width, -1)
    print(f"Linear output reshaped: {linear_output.shape}")
    
    # Test position embeddings
    posembd = SinusoidalPositionEmbeddings()
    time_tensor = torch.arange(conv_output.shape[2])
    time_embedded = posembd(time_tensor)
    print(f"Time tensor: {time_tensor.shape}, time_embedded: {time_embedded.shape}")
    
    # Test broadcasting - need to match the exact spatial dimensions
    # time_embedded is (6, 4), we need (3, 1, 6, 2, 4) to match linear_output (3, 6, 2, 4)
    time_embedded2 = time_embedded.unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(3, 1, -1, conv_output.shape[3], -1)
    print(f"time_embedded2 shape: {time_embedded2.shape}")
    print(f"linear_output shape: {linear_output.shape}")
    
    # Test addition
    try:
        x_code = time_embedded2 + linear_output
        print(f"Addition successful: {x_code.shape}")
    except Exception as e:
        print(f"Addition failed: {e}")
        print(f"Shapes: time_embedded2 {time_embedded2.shape}, linear_output {linear_output.shape}")

if __name__ == "__main__":
    test_individual_operations()
    test_simple_model()
