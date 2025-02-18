from typing import Any, Optional, Dict, List, Tuple

##############################################################
#######################     LAYERS     #######################
##############################################################

class Layer:
    def __init__(self, layer_var_name: str, layer_type: str, layer_parameters_kwargs: Dict[str, Any]) -> None:
        # Name of the variable used in the model block
        self.layer_var_name: str = layer_var_name
        # Type of the layer (e.g., 'Linear', 'Conv2d', etc.)
        self.layer_type: str = layer_type
        # Dictionary mapping parameter names to their values
        self.layer_parameters_kwargs: Dict[str, Any] = layer_parameters_kwargs

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"\n\t* Layer {self.layer_var_name} = {self.layer_type}({self.layer_parameters_kwargs})\n"

class LayerCondition(Layer):
    def __init__(self, layer_var_name: str, layer_conditions_blocks: Dict[str, Layer]) -> None:
        self.layer_var_name: str = layer_var_name
        self.layer_conditions_blocks: Dict[str, Layer] = layer_conditions_blocks
        # Each key is a condition (e.g., "fn(X) < 0.4") and its corresponding value is the layer block to execute

##############################################################
#################### FLOW CONTROL INSTRUCTIONS  #############
##############################################################

class FlowControlInstruction:
    def __init__(self) -> None:
        # Generic abstract class for flow control instructions
        pass

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "\n\t * [FLOW CONTROL PLACEHOLDER]\n"

class FlowControlVariableInit(FlowControlInstruction):
    def __init__(self, var_name: str, var_type: str, var_value: Optional[Any] = None) -> None:
        self.var_name: str = var_name
        self.var_type: str = var_type
        self.var_value: Optional[Any] = var_value

    def __str__(self) -> str:
        return f"\t\t * {self.var_name}: {self.var_type} = {self.var_value}\n"

class FlowControlFunctionCall(FlowControlInstruction):
    def __init__(self, output_variables: List[str], function_called: str, function_arguments: Dict[str, Any]) -> None:
        self.output_variables: List[str] = output_variables
        self.function_called: str = function_called
        self.function_arguments: Dict[str, Any] = function_arguments

    def __str__(self) -> str:
        return f"\t\t * {self.output_variables} = {self.function_called}({self.function_arguments})\n"

class FlowControlSubBlockFunctionCall(FlowControlInstruction):
    def __init__(self, output_variables: List[str], function_called: str, function_arguments: Dict[str, Any]) -> None:
        self.output_variables: List[str] = output_variables
        self.function_called: str = function_called
        self.function_arguments: Dict[str, Any] = function_arguments

    def __str__(self) -> str:
        return f"\t\t * {self.output_variables} = {self.function_called}({self.function_arguments})\n"

class FlowControlLayerPass(FlowControlInstruction):
    def __init__(self, output_variables: List[str], layer_name: str, layer_arguments: Dict[str, Any]) -> None:
        self.output_variables: List[str] = output_variables
        self.layer_name: str = layer_name
        self.layer_arguments: Dict[str, Any] = layer_arguments

    def __str__(self) -> str:
        return f"\t\t * {self.output_variables} = {self.layer_name}({self.layer_arguments})\n"

class FlowControlReturn(FlowControlInstruction):
    def __init__(self, return_variables: List[str]) -> None:
        self.return_variables: List[str] = return_variables

    def __str__(self) -> str:
        return f"\t\t * return {self.return_variables}"

# Additional flow control classes for arithmetic operations or variable manipulations can be added here as needed.

##############################################################
#######################     BLOCKS     #######################
##############################################################

class BlockFunction:
    def __init__(self, function_name: str, function_arguments: Dict[str, Tuple[str, Any]], model_block: "ModelBlock") -> None:
        self.model_block: ModelBlock = model_block  # Access to the block's variables and layers
        self.function_name: str = function_name
        # Each argument is stored as a tuple: (variable type, variable default value)
        self.function_arguments: Dict[str, Tuple[str, Any]] = function_arguments
        # List of flow control instructions composing the function
        self.function_flow_control: List[FlowControlInstruction] = []

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        instructions = "\n".join([ffc.__str__() for ffc in self.function_flow_control])
        return f"\t * Function {self.function_name} ({self.function_arguments}) : \n{instructions}"

class ModelBlock:
    def __init__(self, block_name: str) -> None:
        self.block_name: str = block_name
        # Parameters for the block: variable name -> (type, default value)
        self.block_parameters: Dict[str, Tuple[str, Any]] = {}
        # Layers defined in this block: layer name -> Layer (or ModelBlock for sub-blocks)
        self.block_layers: Dict[str, Any] = {}
        # Functions defined within this block (e.g., the forward pass)
        self.block_functions: Dict[str, BlockFunction] = {}
        # Variables defined in this block: variable name -> (type, value)
        self.block_variables: Dict[str, Tuple[str, Any]] = {}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        layers_str = "\n".join([self.block_layers[layer].__str__() for layer in self.block_layers])
        functions_str = "\n\n".join([fn_name + " = " + self.block_functions[fn_name].__str__() for fn_name in self.block_functions])
        return (
            f"\n\nModelBlock:\n"
            f"\t-block_name: {self.block_name}\n"
            f"\t-block_parameters: {self.block_parameters}\n"
            f"\t-block_layers: \n{layers_str}\n"
            f"\t-Functions:\n\n{functions_str}\n"
        )
