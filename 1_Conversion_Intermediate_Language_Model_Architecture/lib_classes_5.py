from typing import Any, Optional, Union, List, Dict, Tuple
import re

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

    #
    def __str__(self) -> str:
        #
        conditions_str = "\n".join([f"\t  * IF {cond}: {layer}" for cond, layer in self.layer_conditions_blocks.items()])
        return f"\n\t* LayerCondition {self.layer_var_name}:\n{conditions_str}\n"


#
class SequentialLayer(Layer):
    #
    def __init__(self, layer_var_name: str, layers_list: list[Layer]) -> None:
        #
        self.layer_var_name: str = layer_var_name
        self.layers_list: list[Layer] = layers_list  # List of sequential layers
        super().__init__(layer_var_name, "nn.Sequential", {})

    #
    def __str__(self) -> str:
        #
        layers_str = "\n".join([f"\t  * {i}: {layer}" for i, layer in enumerate(self.layers_list)])
        return f"\n\t* SequentialLayer {self.layer_var_name}:\n{layers_str}\n"


#
class ModuleListLayer(Layer):
    #
    def __init__(self, layer_var_name: str, layers_list: list[Layer]) -> None:
        #
        self.layer_var_name: str = layer_var_name
        self.layers_list: list[Layer] = layers_list  # List of layers in ModuleList
        super().__init__(layer_var_name, "nn.ModuleList", {})

    #
    def __str__(self) -> str:
        #
        layers_str = "\n".join([f"\t  * {i}: {layer}" for i, layer in enumerate(self.layers_list)])
        return f"\n\t* ModuleListLayer {self.layer_var_name}:\n{layers_str}\n"


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
class FlowControlSubBlockFunctionCall(FlowControlInstruction):
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


#
class FlowControlIf(FlowControlInstruction):
    #
    def __init__(self, condition: str, true_instructions: list[FlowControlInstruction], false_instructions: Optional[list[FlowControlInstruction]] = None) -> None:
        #
        self.condition: str = condition
        self.true_instructions: list[FlowControlInstruction] = true_instructions
        self.false_instructions: Optional[list[FlowControlInstruction]] = false_instructions

    #
    def __str__(self) -> str:
        #
        true_block = "\n".join([instr.__str__() for instr in self.true_instructions])
        if self.false_instructions:
            false_block = "\n".join([instr.__str__() for instr in self.false_instructions])
            return f"\t\t * IF {self.condition}:\n{true_block}\n\t\t * ELSE:\n{false_block}\n"
        return f"\t\t * IF {self.condition}:\n{true_block}\n"


#
class FlowControlForLoop(FlowControlInstruction):
    #
    def __init__(self, target: str, iterable: str, loop_body: list[FlowControlInstruction]) -> None:
        #
        self.target: str = target
        self.iterable: str = iterable
        self.loop_body: list[FlowControlInstruction] = loop_body

    #
    def __str__(self) -> str:
        #
        body = "\n".join([instr.__str__() for instr in self.loop_body])
        return f"\t\t * FOR {self.target} in {self.iterable}:\n{body}\n"


#
class FlowControlWhileLoop(FlowControlInstruction):
    #
    def __init__(self, condition: str, loop_body: list[FlowControlInstruction]) -> None:
        #
        self.condition: str = condition
        self.loop_body: list[FlowControlInstruction] = loop_body

    #
    def __str__(self) -> str:
        #
        body = "\n".join([instr.__str__() for instr in self.loop_body])
        return f"\t\t * WHILE {self.condition}:\n{body}\n"


#
class FlowControlAssignment(FlowControlInstruction):
    #
    def __init__(self, target: str, value: Any) -> None:
        #
        self.target: str = target
        self.value: Any = value

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.target} = {self.value}\n"


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
        self.return_type: Optional[Union[str, List[str]]] = None  # Store the return type(s)

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        flow_str = "\n".join([ffc.__str__() for ffc in self.function_flow_control])
        return f"\t * Function {self.function_name} ({self.function_arguments}) -> {self.return_type}: \n{flow_str}"


#
class ModelBlock:
    #
    def __init__(self, block_name: str) -> None:
        #
        self.block_name: str = block_name
        self.block_parameters: dict[str, tuple[str, Any]] = {}  # the tuple[str, Any] is for (variable type, variable default value)
        self.block_layers: dict[str, Layer] = {}
        #
        self.block_functions: dict[str, BlockFunction] = {}
        #
        self.block_variables: dict[str, tuple[str, Any]] = {}  # the tuple[str, Any] is for (variable type, variable value)
        #
        self.parent_classes: list[str] = []  # Store parent class names for inheritance relationships

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        layers_str = "\n".join([self.block_layers[layer].__str__() for layer in self.block_layers]) if self.block_layers else "None"
        functions_str = "\n\n".join([fn_name + " = " + self.block_functions[fn_name].__str__() for fn_name in self.block_functions]) if self.block_functions else "None"

        return (f"\n\nModelBlock:\n\t-block_name: {self.block_name}\n"
                f"\t-inherits from: {', '.join(self.parent_classes)}\n"
                f"\t-block_parameters: {self.block_parameters}\n"
                f"\t-block_layers: \n{layers_str}\n"
                f"\t-Functions:\n\n{functions_str}\n")


#
def extract_shape_from_str(type_hint: str) -> Optional[List[int]]:
    """
    Extract shape information from a type hint string.
    Examples:
    - "torch.Tensor[3, 224, 224]" -> [3, 224, 224]
    - "list[int][5]" -> [5]
    """
    # Use regex to find shape information within square brackets
    match = re.search(r'\[([\d, ]+)\]', type_hint)
    if match:
        shape_str = match.group(1)
        try:
            # Convert shape string to list of integers
            shape = [int(dim.strip()) for dim in shape_str.split(',')]
            return shape
        except ValueError:
            pass
    return None