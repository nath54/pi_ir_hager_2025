#
from typing import Any, Optional


##############################################################
#######################     LAYERS     #######################
##############################################################


#
class Layer:
    #
    def __init__(self, layer_name: str, layer_parameters_kwargs: dict[str, Any]) -> None:
        #
        self.layer_name: str = layer_name
        self.layer_parameters_kwargs: dict[str, Any] = layer_parameters_kwargs

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return f"\n\t* Layer:\n\t\t-name: {self.layer_name}\n\t\tparameters_kwargs: {self.layer_parameters_kwargs}\n"


####################################################################
#######################     FLOW CONTROL     #######################
####################################################################


#
class FlowControlInstruction:
    #
    def __init__(self) -> None:
        # GENERIC ABSTRACT CLASS
        pass

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return "\n\t * [FLOW CONTROL PLACEHOLDER]\n"


#
class FlowControlVariableInit(FlowControlInstruction):
    #
    def __init__(self, var_name: str, var_type: str, var_value: Optional[Any] = None) -> None:
        #
        self.var_name: str = var_name
        self.var_type: str = var_type
        self.var_value: Optional[Any] = var_value

    #
    def __str__(self) -> str:
        #
        return f"\n\t * FlowControlVariableInit:\n\t\t-var_name: {self.var_name}\n\t\t-var_type: {self.var_type}\n\t\t-var_value: {self.var_value}\n"


#
class FlowControlFunctionCall(FlowControlInstruction):
    #
    def __init__(self, output_variables: list[str], function_called: str, function_arguments: dict[str, Any]) -> None:
        #
        self.output_variables: list[str] = output_variables
        self.function_called: str = function_called
        self.function_arguments: dict[str, Any] = function_arguments

    #
    def __str__(self) -> str:
        #
        return f"\n\t * FlowControlFunctionCall:\n\t\t-output_variables: {self.output_variables}\n\t\t-function_called: {self.function_called}\n\t\t-function_arguments: {self.function_arguments}\n"


#
class FlowControlLayerPass(FlowControlInstruction):
    #
    def __init__(self, output_variables: list[str], layer_name: str, layer_arguments: dict[str, Any]) -> None:
        #
        self.output_variables: list[str] = output_variables
        self.layer_name: str = layer_name
        self.layer_arguments: dict[str, Any] = layer_arguments

    #
    def __str__(self) -> str:
        #
        return f"\n\t * FlowControlLayerPass:\n\t\t-output_variables: {self.output_variables}\n\t\t-layer_name: {self.layer_name}\n\t\t-layer_arguments: {self.layer_arguments}\n"


# TODO: Create mode FlowControl classes if needed for basic operations and variables manipulation, and other stuff


##############################################################
#######################     BLOCKS     #######################
##############################################################

#
class ModelBlock:
    #
    def __init__(self, block_name: str) -> None:
        #
        self.block_name: str = block_name
        self.block_parameters: dict[str, tuple[str, Any]] = {}
        self.block_layers: dict[str, Layer] = {}
        #
        self.forward_arguments: dict[str, tuple[str, Any]] = {}
        self.forward_flow_control: list[FlowControlInstruction] = []

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return f"\nModelBlock:\n\t-block_name: {self.block_name}\n\t-block_parameters: {self.block_parameters}\n\t-block_layers: {self.block_layers}\n\t-forward_arguments: {self.forward_arguments}\n\t-forward_flow_control: {self.forward_flow_control}\n"


