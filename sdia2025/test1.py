import torch  # type: ignore
from torchsummary import summary  # type: ignore
import torch.nn as nn  # type: ignore


# Load the model
model = torch.load("model.pth")

# Function to get the input shape
def get_input_shape(model):
    # Check if the model has a `forward` method
    if hasattr(model, 'forward'):
        # Get the signature of the forward method
        import inspect
        signature = inspect.signature(model.forward)
        # Get the parameters of the forward method
        parameters = signature.parameters
        # Assume the first parameter is the input
        first_param = next(iter(parameters.values()))
        # Check if the parameter has a default value (e.g., input shape)
        if first_param.default != inspect.Parameter.empty:
            return first_param.default
    # If no input shape is found, return None
    return None

# Function to get the output shape
def get_output_shape(model, input_shape):
    # Create a dummy input with the given shape
    dummy_input = torch.randn(input_shape)
    # Pass the dummy input through the model
    output = model(dummy_input)
    # Return the shape of the output
    return output.shape

# Get the input shape
input_shape = get_input_shape(model)


print("Input Shape:", input_shape)
if input_shape is None:
    raise RuntimeError("Input shape is None")

# Get the output shape
output_shape = get_output_shape(model, input_shape)

print("Output Shape:", output_shape)

# I can't summary because I need input shape
# summary(model)
