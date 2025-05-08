import onnx
from onnx2keras import onnx_to_keras

# Load ONNX model
onnx_model = onnx.load('exported_model.onnx')

# Call the converter (input - is the main model input name, can be different for your model)
k_model = onnx_to_keras(onnx_model, ['x'])

#
k_mode = "exported_model.keras"

