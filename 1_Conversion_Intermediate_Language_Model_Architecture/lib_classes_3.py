#
from typing import Any, Optional, Dict, List, Tuple

##############################################################
#######################     LAYERS     #######################
##############################################################

#
class Layer:
    #
    def __init__(self, layer_var_name: str, layer_type: str, layer_parameters_kwargs: Dict[str, Any]) -> None:
        #
        self.layer_var_name: str = layer_var_name  # Name of the variable (e.g. 'conv1')
        self.layer_type: str = layer_type          # Type of layer (e.g. 'Conv2d')
        self.layer_parameters_kwargs: Dict[str, Any] = layer_parameters_kwargs  # Parameters for the layer

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return f"\n\t* Layer {self.layer_var_name} = {self.layer_type}({self.layer_parameters_kwargs})\n"


#
class LayerCondition(Layer):
    #
    def __init__(self, layer_var_name: str, layer_conditions_blocks: Dict[str, Layer]) -> None:
        #
        self.layer_var_name: str = layer_var_name
        self.layer_conditions_blocks: Dict[str, Layer] = layer_conditions_blocks
        # Each key is a condition (e.g. 'fn(X) < 0.4') and the corresponding value is the Layer to execute
        # This helps simplify conditional branching in the forward pass

##############################################################
##################     FLOW CONTROL     #####################
##############################################################

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
        return f"\t\t * {self.var_name}: {self.var_type} = {self.var_value}\n"


#
class FlowControlFunctionCall(FlowControlInstruction):
    #
    def __init__(self, output_variables: List[str], function_called: str, function_arguments: Dict[str, Any]) -> None:
        #
        self.output_variables: List[str] = output_variables
        self.function_called: str = function_called
        self.function_arguments: Dict[str, Any] = function_arguments  # Mapping from argument name to value

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.output_variables} = {self.function_called}({self.function_arguments})\n"


#
class FlowControlLayerPass(FlowControlInstruction):
    #
    def __init__(self, output_variables: List[str], layer_name: str, layer_arguments: Dict[str, Any]) -> None:
        #
        self.output_variables: List[str] = output_variables
        self.layer_name: str = layer_name
        self.layer_arguments: Dict[str, Any] = layer_arguments  # Mapping from argument name to value

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.output_variables} = {self.layer_name}({self.layer_arguments})\n"


#
class FlowControlReturn(FlowControlInstruction):
    #
    def __init__(self, return_variables: List[str]) -> None:
        #
        self.return_variables: List[str] = return_variables

    #
    def __str__(self) -> str:
        #
        return f"\t\t * return {self.return_variables}\n"


#
class FlowControlIf(FlowControlInstruction):
    #
    def __init__(self, condition: str, body: List["FlowControlInstruction"], orelse: List["FlowControlInstruction"]) -> None:
        #
        self.condition: str = condition
        self.body: List["FlowControlInstruction"] = body
        self.orelse: List["FlowControlInstruction"] = orelse

    #
    def __str__(self) -> str:
        #
        body_str = "".join([instr.__str__() for instr in self.body])
        orelse_str = "".join([instr.__str__() for instr in self.orelse])
        return f"\t\t * if {self.condition}:\n{body_str}\t\t else:\n{orelse_str}\n"


##############################################################
#######################     BLOCKS     #######################
##############################################################

#
class BlockFunction:
    #
    def __init__(self, function_name: str, function_arguments: Dict[str, Tuple[str, Any]], model_block: "ModelBlock") -> None:
        #
        self.model_block: ModelBlock = model_block  # To access the block's variables and layers
        self.function_name: str = function_name
        self.function_arguments: Dict[str, Tuple[str, Any]] = function_arguments  # (variable type, variable default value)
        #
        self.function_flow_control: List[FlowControlInstruction] = []  # The instructions that form the function body


#
class ModelBlock:
    #
    def __init__(self, block_name: str) -> None:
        #
        self.block_name: str = block_name
        self.block_parameters: Dict[str, Tuple[str, Any]] = {}   # (variable type, default value)
        self.block_layers: Dict[str, Layer] = {}                 # All layers defined in the __init__
        #
        self.block_functions: Dict[str, BlockFunction] = {
            # TO ADD: additional functions if needed
        }
        #
        self.block_variables: Dict[str, Tuple[str, Any]] = {}    # Variables defined in the block
        #
        self.forward_arguments: Dict[str, Tuple[str, Any]] = {}    # (variable type, default value) for forward() inputs
        self.forward_flow_control: List[FlowControlInstruction] = []  # Instructions in the forward pass

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        layers_str = "\n".join([self.block_layers[layer].__str__() for layer in self.block_layers])
        flow_str = "".join([instr.__str__() for instr in self.forward_flow_control])
        return (f"\n\nModelBlock:\n"
                f"\t-block_name: {self.block_name}\n"
                f"\t-block_parameters: {self.block_parameters}\n"
                f"\t-block_layers: \n{layers_str}\n"
                f"\t-forward_arguments: {self.forward_arguments}\n"
                f"\t-forward_flow_control: \n{flow_str}\n")
