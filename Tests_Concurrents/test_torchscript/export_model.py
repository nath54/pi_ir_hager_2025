#
import numpy
#
import pickle
import numpy as np
import torch
from torch import nn
from torch import Tensor

#
from model_to_export import Model


#
def export_executorch_model(model: nn.Module, example_inputs: tuple) -> None:

    # --- Standard ExecuTorch Export ---
    try:

        # traced_script_module = torch.jit.trace(model, example_inputs)

        onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)

        onnx_program.optimize()

        onnx_program.save("exported_model.onnx")


    except Exception as e:
        print(f"Export failed: {e}")
        # Consider logging details or using specific exception handling


#
if __name__ == "__main__":

    #
    model: nn.Module = Model()

    #
    model.load_state_dict(torch.load("model_to_export_weights.pth", weights_only=True))

    #
    with open("model_to_export_example_data.pkl", "rb") as f:
        example_data: list[np.ndarray] = pickle.load( f )

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
    predictions: Tensor = model( input_data1 )

    #
    print( f"\npredictions : {predictions}\n" )
    print( f"\npredictions shape : {predictions.shape}\n" )

    #
    export_executorch_model( model, (input_data1,) )
