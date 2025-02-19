#
from typing import Any, Optional

# NOTE: when saying variable type, if there is an iterable or a tensor, the shape must be indicated



####################################################################
#######################     EXPRESSIONS      #######################
####################################################################


class Expression:
    #
    def __init__(self) -> None:
        # ABSTRACT CLASS
        pass

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return "[PLACEHOLDER FOR EXPRESSION]"


#
class ExpressionVariable(Expression):
    #
    def __init__(self, var_name: str) -> None:
        #
        self.var_name: str = var_name

    #
    def __str__(self) -> str:
        #
        return f"{self.var_name}"


#
class ExpressionConstant(Expression):
    #
    def __init__(self, constant: str) -> None:
        #
        self.constant: str = constant

    #
    def __str__(self) -> str:
        #
        return f"{self.constant}"


# Normally, with the following FlowControlInstructions basic instructions, there is no need to have anything else than ExpressionVariable and ExpressionConstant, because we can decompose any complex instructions in a sequence of basic instructions (we may have to create temporary variables, but aside that, it is good)


####################################################################
#######################      CONDITIONS      #######################
####################################################################


class Condition:
    #
    def __init__(self) -> None:
        # ABSTRACT CLASS
        pass

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return "[PLACEHOLDER FOR CONDITION]"

    #
    def __hash__(self) -> int:
        #
        return self.__str__().__hash__()


#
class ConditionBinary(Condition):
    #
    def __init__(self, elt1: Expression | Condition, cond_operator: str, elt2: Expression | Condition) -> None:
        #
        self.elt1: Expression | Condition = elt1
        self.cond_operator: str = cond_operator  # Can be `and`, `or`, `>`, `<`, `>=`, `<=`, `!=`, `==`
        self.elt2: Expression | Condition = elt2

    #
    def __str__(self) -> str:
        #
        return f"{self.elt1} {self.cond_operator} {self.elt2}"

#
class ConditionUnary(Condition):
    #
    def __init__(self, elt: Expression | Condition, cond_operator: Optional[str] = None) -> None:
        #
        self.elt: Expression | Condition = elt
        self.cond_operator: Optional[str] = cond_operator   # Can be `not`

    #
    def __str__(self) -> str:
        #
        if self.cond_operator is None:
            #
            return f"{self.elt}"
        #
        return f"{self.cond_operator} {self.elt}"


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
    def __init__(self, layer_var_name: str, layer_conditions_blocks: dict[Condition, Layer]) -> None:
        #
        self.layer_var_name: str = layer_var_name
        self.layer_conditions_blocks: dict[Condition, Layer] = layer_conditions_blocks  # The idea is that each str will be a condition (ex: fn(X) < 0.4), and this block will do an if, elif, elif, else, and returns the execution of the block corresponding to the validated condition.
        # The idea of this layer / block, is to simplify the conditions in the forward pass
        # So there are constraints, like the fact that each of theses layers must have the same input shapes and same output shapes
        # The idea is to create the blocks when we see an if in a forward pass (so link to the layers that are used in the main block)
        # Okay, so, finally, there is no need to create completely new blocks, just sub block functions `BlockFunction`, so there is no issues of weights / layers duplications.

    #
    def __str__(self) -> str:
        #
        res: str = "\n\t* Layer Condition:\n"
        #
        for condition in self.layer_conditions_blocks:
            #
            res += f"\t\t- {condition} : {self.layer_conditions_blocks[condition]}"
        #
        return res


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
    def __init__(self, var_name: str, var_type: str, var_value: Optional[Expression] = None) -> None:
        #
        self.var_name: str = var_name
        self.var_type: str = var_type
        self.var_value: Optional[Expression] = var_value

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.var_name}: {self.var_type} = {self.var_value}\n"


#
class FlowControlVariableAssignment(FlowControlInstruction):
    #
    def __init__(self, var_name: str, var_value: Expression) -> None:
        #
        self.var_name: str = var_name
        self.var_value: Expression = var_value

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.var_name} = {self.var_value}\n"

#
class FlowControlBasicBinaryArithmetic(FlowControlInstruction):
    #
    def __init__(self, output_var_name: str, input1_var_name: str, operation: str, input2_var_name: str) -> None:
        #
        self.output_var_name: str = output_var_name
        self.input1_var_name: str = input1_var_name
        self.input2_var_name: str = input2_var_name
        self.operation: str = operation

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.output_var_name} = {self.input1_var_name} {self.operation} {self.input2_var_name}"

#
class FlowControlBasicUnaryArithmetic(FlowControlInstruction):
    #
    def __init__(self, output_var_name: str, operation: str, input_var_name: str) -> None:
        #
        self.output_var_name: str = output_var_name
        self.input_var_name: str = input_var_name
        self.operation: str = operation

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.output_var_name} = {self.operation} {self.input_var_name}"


#
class FlowControlForLoop(FlowControlInstruction):
    #
    def __init__(self, iterable_var_name: str, iterator: str, flow_control_instructions: list[FlowControlInstruction]) -> None:
        #
        self.iterable_var_name: str = iterable_var_name
        self.iterator: str = iterator
        self.flow_control_instructions: list[FlowControlInstruction] = flow_control_instructions

    #
    def __str__(self) -> str:
        #
        return f"\t\t * for {self.iterable_var_name} in {self.iterator} {{\n{"\n".join( [fci.__str__() for fci in self.flow_control_instructions] )}\n}}"


#
class FlowControlWhileLoop(FlowControlInstruction):
    #
    def __init__(self, condition: Condition, flow_control_instructions: list[FlowControlInstruction]) -> None:
        #
        self.condition: Condition = condition
        self.flow_control_instructions: list[FlowControlInstruction] = flow_control_instructions

    #
    def __str__(self) -> str:
        #
        return f"\t\t * while {self.condition} {{\n{"\n".join( [fci.__str__() for fci in self.flow_control_instructions] )}\n}}"


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
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return f"\t * Function {self.function_name} ( {self.function_arguments} ) : \n{"\n".join([ffc.__str__() for ffc in self.function_flow_control])}"


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
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return f"\n\nModelBlock:\n\t-block_name: {self.block_name}\n\t-block_parameters: {self.block_parameters}\n\t-block_layers: \n{"\n".join([self.block_layers[layer].__str__() for layer in self.block_layers])}\n\t-Functions:\n\n{"\n\n".join([fn_name + " = " + self.block_functions[fn_name].__str__() for fn_name in self.block_functions])}\n"



#########################################################################
#######################     MAIN OUTPUT CLASS     #######################
#########################################################################


#
class Language1_Model:
    #
    def __init__(self) -> None:
        """
        Output of the analysis.

        Attributes:
            - model_blocks (dict[str, lc.ModelBlock]): list of all the blocks analyzed here, indexed by their blocks id (e.g. their name)
            - main_block (str): id of the main block, given in sys.argv with `--main-block <MainBlockName>`
        """
        #
        self.model_blocks: dict[str, ModelBlock] = {}
        #
        self.main_block: str = ""
