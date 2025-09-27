#
### Simple test without PyTorch dependencies ###
#

class SimpleModel:
    """
    A simple model for testing the AST extraction.
    """
    
    def __init__(self, in_features: int = 10, out_features: int = 5):
        """
        Initialize the model.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        self.in_features = in_features
        self.out_features = out_features
        self.linear = None  # This would be a layer in real PyTorch
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Simple computation
        y = x * 2
        return y
