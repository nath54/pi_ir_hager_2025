import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch import Tensor
import pickle
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras

#
from model_to_export import Model

if __name__ == '__main__':

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


    k_model = \
        pytorch_to_keras(model, input_data1, [input_data1.shape], verbose=True, change_ordering=False)

    for i in range(3):
        output = model(input_data1)
        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_data1)
        error = np.max(pytorch_output - keras_output)
        print('error -- ', error)  # Around zero :)