"""

"""

#
from typing import Any
#
from dataclasses import dataclass
#
import json
#
from . import lib_classes as lc
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
class TensorShapeVariableLengthDimsVar(TensorShapeElt):

    #
    def __init__(self, name: str) -> None:

        #
        self.name: str = name


#
class TensorShapeVar(TensorShapeElt):

    #
    def __init__(self, name: str) -> None:

        #
        self.name: str = name


#
class TensorShapeVarList(TensorShapeElt):

    #
    def __init__(self, name: str) -> None:

        #
        self.name: str = name


#
class TensorShape:

    #
    def __init__(self, dims: list[TensorShapeElt]) -> None:

        #
        self.dims: list[TensorShapeElt] = dims




#
@dataclass
class BaseLayerInfo:

    #
    def __init__(
            self,
            name: str,
            tensor_input_shapes: dict[str, TensorShape],
            tensor_output_shapes: list[TensorShape],
            parameters: dict[str, tuple[lc.VarType, lc.Expression]],
            with_exec: dict[str, str]
    ) -> None:

        #
        self.name: str = name

        #
        self.tensor_input_shapes: dict[str, TensorShape] = tensor_input_shapes

        #
        self.tensor_output_shapes: list[TensorShape] = tensor_output_shapes

        #
        self.parameters: dict[str, tuple[lc.VarType, lc.Expression]] = parameters

        #
        self.with_exec: dict[str, str] = with_exec


#
def parse_tensor_shape_str(base_txt: str) -> TensorShape:

    #
    dims: list[TensorShapeElt] = []

    #
    txt: str = base_txt.strip()

    #
    if not txt.startswith("[") or not txt.endswith("]"):

        #
        raise SyntaxError(f"Error: not a valid tensor shape syntax !\nbase_txt = {base_txt}")

    #
    txt = txt[1:-1].strip()

    #
    elts: list[str] = txt.split(",")

    #
    nb_variable_length: int = 0
    nb_opt: int = 0

    #
    elt: str
    for elt in elts:

        #
        elt = elt.strip()

        #
        if elt.startswith("**"):

            #
            dims.append( TensorShapeVarList(name=elt[2:]) )

        #
        elif elt.startswith("*"):

            #
            dims.append( TensorShapeVariableLengthDimsVar(name=elt[1:]) )

            #
            nb_variable_length += 1

        #
        elif elt.endswith("?"):

            #
            dims.append( TensorShapeOpt(TensorShapeVar(name=elt[:-1])) )

            #
            nb_opt += 1

        #
        else:

            #
            dims.append( TensorShapeVar(name=elt) )

    #
    if nb_opt + nb_variable_length > 1:

        #
        raise ValueError(f"Error : Cannot have more than one optional variable or list of variable length in tensor shape value !\nbase_txt = {base_txt}")

    #
    return TensorShape(dims=dims)



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
    for layer_key in data:

        #
        # print(f"DEBUG |  *** layer_detection : {layer_key}")

        #
        layer_dict: dict = data[layer_key]

        #
        if "layer_name"  not in layer_dict:
            #
            raise KeyError(f"Error: no key `layer_name` in layers dict of dict {layer_dict}")
        #
        layer_name = layer_dict["layer_name"]

        #
        tensor_input_shapes: dict[str, TensorShape] = {}
        #
        if "tensor_inputs_shapes" in layer_dict:

            #
            if isinstance(layer_dict["tensor_inputs_shapes"], dict):

                #
                tensor_key: str
                tensor_value: str
                for tensor_key, tensor_value in layer_dict["tensor_inputs_shapes"].items():

                    #
                    tensor_input_shapes[tensor_key] = parse_tensor_shape_str(tensor_value)

            #
            elif isinstance(layer_dict["tensor_inputs_shapes"], list):

                #
                if len(layer_dict["tensor_inputs_shapes"]) > 1:
                    #
                    raise SystemError(f"Error, {layer_dict["tensor_inputs_shapes"]} is list of length > 1, it must be a dictionnary with variables names !\nlayer_dict = {layer_dict}")

                #
                elif len(layer_dict["tensor_inputs_shapes"]) == 0:
                    #
                    raise SystemError(f"Error, {layer_dict["tensor_inputs_shapes"]} is an empty list, it must be a list with one value or a dictionnary with variables names !\nlayer_dict = {layer_dict}")

                #
                tensor_input_shapes["X"] = parse_tensor_shape_str(layer_dict["tensor_inputs_shapes"][0])

            #
            elif isinstance(layer_dict["tensor_inputs_shapes"], str):

                #
                tensor_input_shapes["X"] = parse_tensor_shape_str(layer_dict["tensor_inputs_shapes"])

            #
            else:

                #
                raise TypeError(f"Error: `tensor_inputs_shapes` is not of type str, list or dicts !\n\ntensor_inputs_shapes = {layer_dict["layer_dict"]}")

        #
        elif "tensor_input_shape" in layer_dict:

            #
            if isinstance(layer_dict["tensor_input_shape"], str):

                #
                tensor_input_shapes["X"] = parse_tensor_shape_str(layer_dict["tensor_input_shape"])

            #
            else:

                #
                raise TypeError(f"Error, value of layer_dict['tensor_input_shape'] is not str !\nlayer_dict = {layer_dict}")

        #
        else:

            #
            raise KeyError(f"Error: no key for tensor input(s) shape(s) in layers dict of dict {layer_dict}")


        #
        tensor_output_shapes: list[TensorShape] = []

        #
        if "tensor_output_shapes" in layer_dict:


            #
            if isinstance(layer_dict["tensor_output_shapes"], list):

                #
                output_shape: str
                for output_shape in layer_dict["tensor_output_shapes"]:
                    tensor_output_shapes.append( parse_tensor_shape_str(output_shape) )

            #
            else:

                #
                raise TypeError(f"Error, value of layer_dict['tensor_output_shapes'] is not list of str !\nlayer_dict = {layer_dict}")


        #
        elif "tensor_output_shape" in layer_dict:

            #
            if isinstance(layer_dict["tensor_output_shape"], str):

                #
                tensor_output_shapes.append( parse_tensor_shape_str(layer_dict["tensor_output_shape"]) )

            #
            else:

                #
                raise TypeError(f"Error, value of layer_dict['tensor_output_shape'] is not str !\nlayer_dict = {layer_dict}")

        #
        else:

            #
            raise KeyError(f"Error: no key for tensor output(s) shape(s) in layers dict of dict {layer_dict}")

        #
        with_exec: dict[str, str] = {}
        #
        if "with" in layer_dict:
            #
            with_exec = layer_dict["with"]

        #
        parameters: dict[str, tuple[lc.VarType, lc.Expression]] = {}

        #
        if "parameters" not in layer_dict:

            #
            raise KeyError(f"Error: layers dict of dict {layer_dict} doesn't have `parameters` key !")

        #
        if not isinstance(layer_dict["parameters"], dict):

            #
            raise TypeError(f"Error: layer_dict['parameters'] is not dict[str, str] !\nlayer_dict = {layer_dict}")

        #
        param_key: str
        param_value: list
        for param_key, param_value in layer_dict["parameters"].items():

            #
            if not isinstance(param_value, list):

                #
                raise TypeError(f"Error: param_value is not tuple ! (= {parameters[param_key]})\nlayer_dict = {layer_dict}")

            #
            if len(param_value) != 2:

                #
                raise TypeError(f"Error: param_value doesn't have a lenght of 2 ! (= {parameters[param_key]})\nlayer_dict = {layer_dict}")


            #
            param_type: lc.VarType
            param_default_value: lc.Expression

            #
            if "[" in param_value[0]:

                # TODO
                pass

                #
                tensor_dims: list[int | str] = []

                #
                param_type = lc.VarTypeTensor(tensor_type="", tensor_dims=tensor_dims)

            #
            else:

                #
                param_type = lc.VarType(type_name=param_value[0])

            #
            if param_value[1] is None:

                #
                param_default_value = lc.ExpressionNoDefaultArguments()

            #
            elif isinstance(param_value[1], int) or isinstance(param_value[1], float):

                #
                param_default_value = lc.ExpressionConstantNumeric(constant=param_value[1])

            #
            elif isinstance(param_value[1], str):

                #
                if param_value[1] == "None":

                    #
                    param_default_value = lc.ExpressionNone()

                #
                else:

                    #
                    param_default_value = lc.ExpressionToEvaluate(expr_to_evaluate=param_value[1])

            #
            elif isinstance(param_value[1], list):

                #
                elements: list[lc.ExpressionConstant] = []

                #
                val: Any
                for val in param_value[1]:

                    #
                    if isinstance(val, int) or isinstance(val, float):

                        #
                        elements.append( lc.ExpressionConstantNumeric(val) )

                    #
                    else:

                        #
                        raise TypeError(f"Error: List of this kind of values are not supported : {val} !")

                #
                param_default_value = lc.ExpressionConstantList(elements=elements)

            #
            else:

                #
                raise SyntaxError(f"Unknown default value, can't extract it : {param_value[1]}")


            #
            parameters[param_key] = (param_type, param_default_value)

            #
            if len(parameters[param_key]) != 2:

                #
                raise TypeError(f"Error: parameters[{param_key}] is not tuple of length 2 ! (= {parameters[param_key]})\nlayer_dict = {layer_dict}")

            #
            if not isinstance(parameters[param_key][0], lc.VarType):

                #
                raise TypeError(f"Error: parameters[{param_key}] first value of tuple is not VarType ! (= {parameters[param_key]})\nlayer_dict = {layer_dict}")

        #
        res[layer_key] = BaseLayerInfo(
            name = layer_name,
            tensor_input_shapes = tensor_input_shapes,
            tensor_output_shapes = tensor_output_shapes,
            parameters = parameters,
            with_exec = with_exec
        )

    #
    return res
