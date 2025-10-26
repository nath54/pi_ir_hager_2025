#
import pickle

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch import Tensor

#
from model_to_export_custom_layer_norm import Model


#
def export_executorch_model(model: nn.Module, example_inputs: tuple) -> None:
    """
    Export model to ONNX using TorchScript tracing.
    This avoids issues with aten.as_strided operations.
    """
    try:
        # Use torch.jit.trace for more stable ONNX export
        model.eval()
        
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_inputs)
        
        # Export using TorchScript approach (no dynamo)
        torch.onnx.export(
            traced_model,
            example_inputs,
            "exported_model_ir9.onnx",
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print("✅ Model exported successfully to exported_model_ir9.onnx")

    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()


#
if __name__ == "__main__":

    #
    model: nn.Module = Model()

    #
    model.load_state_dict(torch.load("model_to_export_weights.pth", weights_only=True))

    #
    with open("model_to_export_example_data.pkl", "rb") as f:
        example_data: list[NDArray[np.float32]] = pickle.load( f )

    #
    input_data1: Tensor = Tensor( example_data[0] ).unsqueeze(dim=0)
    input_data2: Tensor = Tensor( example_data[1] ).unsqueeze(dim=0)

    #
    # print( f"\ninput_data : {input_data}\n" )
    print( f"\ninput_data1 shape : {input_data1.shape}\n" )

    #
    print( f"\ninput_data2 : {input_data2}\n" )
    print( f"\ninput_data2 shape : {input_data2.shape}\n" )

    #
    model.eval()

    #
    with torch.no_grad():
        predictions: Tensor = model( input_data1 )

    #
    print( f"\npredictions : {predictions}\n" )
    print( f"\npredictions shape : {predictions.shape}\n" )

    #
    export_executorch_model( model, (input_data1,) )