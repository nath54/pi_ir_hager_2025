#
import pickle

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch import Tensor

#
from model_to_export_custom_layer_norm_simpl import DeepArcNet as Model


#
def export_executorch_model(model: nn.Module, example_inputs: tuple) -> None:
    """
    Export model to ONNX using TorchScript tracing.
    Optimized for ST Edge AI - no dynamic operations, no If nodes.
    """
    try:
        # Critical: set to eval mode to remove dropout and other training-specific ops
        model.eval()

        # Disable gradient computation
        with torch.no_grad():
            # First trace the model
            traced_model = torch.jit.trace(model, example_inputs)

            # Freeze the traced model to inline all parameters
            traced_model = torch.jit.freeze(traced_model)

        # Export using TorchScript approach (no dynamo)
        torch.onnx.export(
            traced_model,
            example_inputs,
            "exported_model_ir9_no_if.onnx",
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            training=torch.onnx.TrainingMode.EVAL,  # Force eval mode
            dynamic_axes=None  # Remove dynamic axes for ST Edge AI
        )

        print("✅ Model exported successfully to exported_model_ir9_no_if.onnx")

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
    model.conv0 = model.ConvEncode[0]
    model.conv1 = model.ConvEncode[1]
    model.conv2 = model.ConvEncode[2]
    model.conv3 = model.ConvEncode[3]
    model.conv4 = model.ConvEncode[4]
    model.conv5 = model.ConvEncode[5]

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