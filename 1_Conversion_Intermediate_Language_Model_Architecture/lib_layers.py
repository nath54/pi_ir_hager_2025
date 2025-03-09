#
from typing import Any, Optional
#
from dataclasses import dataclass
#
import json
#


#
class TensorShapeElt:

    #
    pass


#
class TensorShapeOpt(TensorShapeElt):

    #
    def __init__(self, shape: TensorShapeElt) -> None:

        #
        self.shape: TensorShapeElt = shape



#
class TensorShapeVar(TensorShapeElt):

    #
    def __init__(self, name: str) -> None:

        #
        self.name: str = name



#
@dataclass
class BaseLayerInfo:

    #
    def __init__(
            self,
            name: str,
            tensor_input_shapes: dict[str, TensorShapeElt],
            tensor_output_shapes: list[TensorShapeElt],
            parameters: dict[str, tuple[str, Any]],
            with_exec: dict[str, str]
    ) -> None:

        #
        self.name: str = name

        #
        self.tensor_input_shapes: dict[str, TensorShapeElt] = tensor_input_shapes

        #
        self.tensor_output_shapes: list[TensorShapeElt] = tensor_output_shapes

        #
        self.parameters: dict[str, tuple[str, Any]] = parameters

        #
        self.with_exec: dict[str, str] = with_exec



#
def load_layers_dict(filepath: str) -> dict[str, BaseLayerInfo]:

    #
    with open(filepath, "r", encoding="utf-8") as f:

        #
        data: dict[str, dict[str, dict | str]] = json.load( f )

    #
    res: dict[str, BaseLayerInfo] = {}

    #
    layer_name: str
    #
    layer_key: str
    attr: str
    value: dict | str | list
    for layer_key in data:

        #
        # print(f"DEBUG |  *** layer_detection : {layer_key}")

        #
        for attr, value in data[layer_key].items():

            #
            layer_dict: dict = data[layer_key]

            #
            if "layer_name"  not in layer_dict:
                #
                raise KeyError(f"Error: no key `layer_name` in layers dict of dict {layer_dict}")

            #
            if "tensor_inputs_shapes" in layer_dict:

                #
                if isinstance(layer_dict["tensor_inputs_shapes"], dict):

                    #
                    pass

                #
                elif isinstance(layer_dict["tensor_inputs_shapes"], list):

                    #
                    pass

                #
                elif isinstance(layer_dict["tensor_inputs_shapes"], str):

                    #
                    pass

                #
                else:

                    #
                    raise TypeError(f"Error: `tensor_inputs_shapes` is not of type list or dicts !\n\ntensor_inputs_shapes = {layer_dict["layer_dict"]}")

                # TODO
                pass

            #
            else:

                #
                raise KeyError(f"Error: no key for tensor input(s) shape(s) in layers dict of dict {layer_dict}")

            #
            layer_name = layer_dict["layer_name"]

            #
            # print(f"DEBUG |  ***   -- attr {attr} : {value}")

    #
    return res
