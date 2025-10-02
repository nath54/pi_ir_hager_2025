#
import ai_edge_torch
import numpy
#
import pickle
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch import Tensor

#
from model_to_export import Model


#
def export_executorch_model(model: nn.Module, example_inputs: tuple) -> None:

    # --- Standard ExecuTorch Export ---
    try:

        model = model.eval()

        torch_output = model(*example_inputs)

        edge_model = ai_edge_torch.convert(model, example_inputs)

        edge_output = edge_model(*example_inputs)

        if (numpy.allclose(
            torch_output.detach().numpy(),
            edge_output,
            atol=1e-5,
            rtol=1e-5,
        )):
            print("Inference result with Pytorch and TfLite was within tolerance")
        else:
            print("Something wrong with Pytorch --> TfLite")

        edge_model.export('exported_model.tflite')
        #Stocke le fichier tflite produit de 44.5 Mo

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
    predictions: Tensor = model( input_data1 )

    #
    print( f"\npredictions : {predictions}\n" )
    print( f"\npredictions shape : {predictions.shape}\n" )

    #
    export_executorch_model( model, (input_data1,) )
