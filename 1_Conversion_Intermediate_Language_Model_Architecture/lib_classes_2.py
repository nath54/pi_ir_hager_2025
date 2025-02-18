# First script: lib_classes.py (extended)
from typing import Any, Optional, Union, List, Dict, Tuple

# NOTE: when saying variable type, if there is an iterable or a tensor, the shape must be indicated


##############################################################
#######################     LAYERS     #######################
##############################################################


#
class Layer:
    #
    def __init__(self, layer_var_name: str, layer_type: str, layer_parameters_kwargs: dict[str, Any]) -> None:
        #
        self.layer_var_name: str = layer_var_name  # name of the variable to be called from the model block
        self.layer_type: str = layer_type
        self.layer_parameters_kwargs: dict[str, Any] = layer_parameters_kwargs  # the dict[str, Any] is for variable name -> variable value

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
    def __init__(self, layer_var_name: str, layer_conditions_blocks: dict[str, Layer]) -> None:
        #
        self.layer_var_name: str = layer_var_name
        self.layer_conditions_blocks: dict[str, Layer] = layer_conditions_blocks  # The idea is that each str will be a condition (ex: fn(X) < 0.4), and this block will do an if, elif, elif, else, and returns the execution of the block corresponding to the validated condition.
        # The idea of this layer / block, is to simplify the conditions in the forward pass
        # So there are constraints, like the fact that each of theses layers must have the same input shapes and same output shapes
        # The idea is to create the blocks when we see an if in a forward pass (so link to the layers that are used in the main block)
        # Okay, so, finally, there is no need to create completely new blocks, just sub block functions `BlockFunction`, so there is no issues of weights / layers duplications.


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
        return f"\t\t * {self.var_name}: {self.var_type} = {self.var_value}\n"


#
class FlowControlFunctionCall(FlowControlInstruction):
    #
    def __init__(self, output_variables: list[str], function_called: str, function_arguments: dict[str, Any]) -> None:
        #
        self.output_variables: list[str] = output_variables
        self.function_called: str = function_called
        self.function_arguments: dict[str, Any] = function_arguments  # the tuple[str, Any] is for (variable type, variable default value)

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.output_variables} = {self.function_called}({self.function_arguments})\n"


#
class FlowControlLayerPass(FlowControlInstruction):
    #
    def __init__(self, output_variables: list[str], layer_name: str, layer_arguments: dict[str, Any]) -> None:
        #
        self.output_variables: list[str] = output_variables
        self.layer_name: str = layer_name
        self.layer_arguments: dict[str, Any] = layer_arguments  # the tuple[str, Any] is for (variable type, variable default value)

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.output_variables} = {self.layer_name}({self.layer_arguments})\n"


#
class FlowControlReturn(FlowControlInstruction):
    #
    def __init__(self, return_variables: list[str]) -> None:
        #
        self.return_variables: list[str] = return_variables

    #
    def __str__(self) -> str:
        #
        return f"\t\t * return {self.return_variables}"


# Additional Flow Control Classes for completeness

#
class FlowControlIf(FlowControlInstruction):
    #
    def __init__(self, condition: str, true_block: List[FlowControlInstruction], false_block: Optional[List[FlowControlInstruction]] = None) -> None:
        #
        self.condition: str = condition  # String representation of the condition
        self.true_block: List[FlowControlInstruction] = true_block  # Instructions to execute if condition is True
        self.false_block: Optional[List[FlowControlInstruction]] = false_block  # Instructions to execute if condition is False

    #
    def __str__(self) -> str:
        #
        true_block_str = "\n".join([instr.__str__() for instr in self.true_block])
        if self.false_block:
            false_block_str = "\n".join([instr.__str__() for instr in self.false_block])
            return f"\t\t * if {self.condition}:\n{true_block_str}\n\t\t * else:\n{false_block_str}\n"
        return f"\t\t * if {self.condition}:\n{true_block_str}\n"


#
class FlowControlLoop(FlowControlInstruction):
    #
    def __init__(self, loop_type: str, iterator: str, iterable: str, body: List[FlowControlInstruction]) -> None:
        #
        self.loop_type: str = loop_type  # 'for' or 'while'
        self.iterator: str = iterator  # Variable used for iteration (in for loops)
        self.iterable: str = iterable  # The condition (for while) or the iterable (for for)
        self.body: List[FlowControlInstruction] = body  # Instructions to execute in the loop body

    #
    def __str__(self) -> str:
        #
        body_str = "\n".join([instr.__str__() for instr in self.body])
        if self.loop_type == "for":
            return f"\t\t * for {self.iterator} in {self.iterable}:\n{body_str}\n"
        else:  # while
            return f"\t\t * while {self.iterable}:\n{body_str}\n"


#
class FlowControlAssignment(FlowControlInstruction):
    #
    def __init__(self, target: str, expression: str) -> None:
        #
        self.target: str = target  # Left side of assignment
        self.expression: str = expression  # Right side of assignment

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.target} = {self.expression}\n"


##############################################################
#######################     BLOCKS     #######################
##############################################################


#
class BlockFunction:
    #
    def __init__(self, function_name: str, function_arguments: dict[str, tuple[str, Any]], model_block: "ModelBlock") -> None:
        #
        self.model_block: ModelBlock = model_block  # TO access the block variables / layers
        #
        self.function_name: str = function_name
        self.function_arguments: dict[str, tuple[str, Any]] = function_arguments  # the tuple[str, Any] is for (variable type, variable default value)
        #
        self.function_flow_control: list[FlowControlInstruction] = []  # To complete while analysis

    #
    def __str__(self) -> str:
        #
        args_str = ", ".join([f"{arg_name}: {arg_type}" for arg_name, (arg_type, _) in self.function_arguments.items()])
        flow_control_str = "".join([ffc.__str__() for ffc in self.function_flow_control])
        return f"\n\t-function {self.function_name}({args_str}):\n{flow_control_str}\n"


#
class ModelBlock:
    #
    def __init__(self, block_name: str) -> None:
        #
        self.block_name: str = block_name
        self.block_parameters: dict[str, tuple[str, Any]] = {}  # the tuple[str, Any] is for (variable type, variable default value)
        self.block_layers: dict[str, Layer] = {}
        #
        self.block_functions: dict[str, BlockFunction] = {
            # TO ADD: the foward pass
        }
        #
        self.block_variables: dict[str, tuple[str, Any]] = {}  # the tuple[str, Any] is for (variable type, variable value)
        #
        self.forward_arguments: dict[str, tuple[str, Any]] = {}  # the tuple[str, Any] is for (variable type, variable default value)
        self.forward_flow_control: list[FlowControlInstruction] = []

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        layers_str = "\n".join([self.block_layers[layer].__str__() for layer in self.block_layers]) if self.block_layers else "None"
        functions_str = "\n".join([self.block_functions[func].__str__() for func in self.block_functions]) if self.block_functions else "None"
        forward_flow_str = "".join([ffc.__str__() for ffc in self.forward_flow_control]) if self.forward_flow_control else "None"

        return (f"\n\nModelBlock:\n\t-block_name: {self.block_name}\n\t-block_parameters: {self.block_parameters}\n"
                f"\t-block_layers: \n{layers_str}\n\t-block_functions: \n{functions_str}\n"
                f"\t-forward_arguments: {self.forward_arguments}\n\t-forward_flow_control: \n{forward_flow_str}\n")