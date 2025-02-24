#
from typing import Any, Optional

# NOTE: when saying variable type, if there is an iterable or a tensor, the shape must be indicated



####################################################################
#######################     EXPRESSIONS      #######################
####################################################################


class Expression:
    #
    def __init__(self) -> None:
        """
        Generic class for manipulating Expressions (in the sense of compilation and control flow).
        """

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
        """
        Represents a variable expression.
        (for assignments or conditions, like : A = B -> Assignment(A, ExpressionVariable(B)))

        Args:
            var_name (str): Name of the variable this expression is representing
        """
        #
        self.var_name: str = var_name

    #
    def __str__(self) -> str:
        #
        return f"{self.var_name}"


#
class ExpressionConstant(Expression):
    #
    def __init__(self, constant: Any) -> None:
        """
        Represents a constant expression.
        Abstract Class, should not be used because of type imprecisions.

        Args:
            constant (str): Constant value
        """
        #
        self.constant: Any = constant

    #
    def __str__(self) -> str:
        #
        return f"{self.constant}"


#
class ExpressionConstantNumeric(Expression):
    #
    def __init__(self, constant: int | float) -> None:
        """
        Represents a numerical constant expression.
        (for assignments or conditions, like : A = 5 -> Assignment(A, ExpressionConstant(5)))

        Args:
            constant (int | float): Constant numerical value
        """
        #
        self.constant: int | float = constant

    #
    def __str__(self) -> str:
        #
        return f"{self.constant}"


#
class ExpressionConstantString(Expression):
    #
    def __init__(self, constant: str) -> None:
        """
        Represents a constant string expression.

        Args:
            constant (str): Constant string value
        """
        #
        self.constant: str = constant

    #
    def __str__(self) -> str:
        #
        return f"{self.constant}"

#
class ExpressionConstantList(Expression):
    #
    def __init__(self, elements: list[ExpressionConstant]) -> None:
        """
        Represents a constant list expression.

        Args:
            elements (list[ExpressionConstant]): list of the values
        """
        #
        self.elements: list[ExpressionConstant] = elements

    #
    def __str__(self) -> str:
        #
        return f"[{", ".join([elt.__str__() for elt in self.elements])}]"


#
class ExpressionConstantRange(ExpressionConstantList):
    #
    def __init__(self, end_value: int | float, start_value: int | float = 0, step: int | float = 1) -> None:
        """
        Represents a constant range expression (a list).

        Args:
            end_value (int | float): End value of the range.
            start_value (int | float, optional): Start value of the range. Defaults to 0.
            step (int | float, optional): Step of the range. Defaults to 1. (Warning: the step can't be 0 because it generates an infinite constant list).
        """

        self.end_value: int | float = end_value
        self.start_value: int | float = start_value
        self.step: int | float = step

    #
    def __str__(self) -> str:
        #
        return f"Range({self.start_value}, {self.end_value}, {self.step})"


# Normally, with the following FlowControlInstructions basic instructions, there is no need to have anything else than ExpressionVariable and ExpressionConstant, because we can decompose any complex instructions in a sequence of basic instructions (we may have to create temporary variables, but aside that, it is good)

# We will constraint the models to not use other types of constants and variables, (like dictionaries or custom objects), and we can convert the tuples into lists.
# We should constraint a list to have the same typing too.


####################################################################
#######################      CONDITIONS      #######################
####################################################################


class Condition:
    #
    def __init__(self) -> None:
        """
        Generic Abstract class for representing Conditions (for ControlFlowLoopWhile or LayerCondition)
        """
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
        """
        Represents a condition on two elements with one operation. (elt1 cond_operator elt2)

        Args:
            elt1 (Expression | Condition): left side of the condition
            cond_operator (str): Operator of the condition. Can be `and`, `or`, `>`, `<`, `>=`, `<=`, `!=`, `==`
            elt2 (Expression | Condition): right side of the condition
        """
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
        """
        Represents a condition on one element with potentially an operation. (elt) or (cond_operator elt)

        Args:
            elt (Expression | Condition): Element of the condition
            cond_operator (Optional[str], optional): A potential unary condition operator. Can be `not`. Defaults to None.
        """
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
        """
        Can represents a basic Neural Network Layer (from the list `layers.json`), or a block of a model.

        Args:
            layer_var_name (str): Name of the variable in the current model block that is containing this layer.
            layer_type (str): Name of the layer or the model block.
            layer_parameters_kwargs (dict[str, Any]): Parameters of the layer.
        """
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
    def __init__(self, layer_var_name: str, layer_conditions_blocks: dict[Condition, "FlowControlSubBlockFunctionCall"]) -> None:
        """
        Represents a condition of the control flow of a function (forward or other of a model block).
        Each of the paths in the conditions will be moved inside a sub-function (aside the foward function) of the model block, and this layer will be created and added into the current model block.

        Args:
            layer_var_name (str): Name of the variable in the current model block that is containing this layer.
            layer_conditions_blocks (dict[Condition, FlowControlSubBlockFunctionCall]): Dictionnary of couples (Condition, FlowControlSubBlockFunctionCall), where intuitively, each FlowControlSubBlockFunctionCall is associated to its corresponding Condition.
        """
        #
        self.layer_var_name: str = layer_var_name
        self.layer_conditions_blocks: dict[Condition, FlowControlSubBlockFunctionCall] = layer_conditions_blocks  # The idea is that each str will be a condition (ex: fn(X) < 0.4), and this block will do an if, elif, elif, else, and returns the execution of the block corresponding to the validated condition.
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
        """
        Generic Abstract class that represents a flow control instruction.
        A function of a model block like forward() is decomposed in a list of FlowControlInstruction.
        """

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
        """
        Represents a Variable Initialisation.

        Args:
            var_name (str): Name of the variable.
            var_type (str): Type of the variable
            var_value (Optional[Expression], optional): Potential Initialisation Value. Defaults to None.
        """
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
        """
        Represents a Variable Assignement.

        Args:
            var_name (str): Name of the variable.
            var_value (Expression): Value that the variable will take.
        """

        #
        self.var_name: str = var_name
        self.var_value: Expression = var_value

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.var_name} = {self.var_value}\n"

#
class FlowControlBasicBinaryOperation(FlowControlInstruction):
    #
    def __init__(self, output_var_name: str, input1_var_name: str, operation: str, input2_var_name: str) -> None:
        """
        Represents a basic binary operation.

        Warning: The operands can be both vector / matrix / tensor or scalars ! This should be checked in the error checking.

        Args:
            output_var_name (str): Name of the variable that will contains the result of the operation.
            input1_var_name (str): Name of the left operand.
            operation (str): Name of the operation Can be `+`, `-`, `*`, `/`, `./`, `.*`, `^`, `<<`, `>>`, `%`, or more.
            input2_var_name (str): Name of the right operand.
        """

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
class FlowControlBasicUnaryOperation(FlowControlInstruction):
    #
    def __init__(self, output_var_name: str, operation: str, input_var_name: str) -> None:
        """
        Represent a basic unary operation.

        Warning: the operand can be both vector / matrix / tensor or scalars !

        Args:
            output_var_name (str): Name of the variable that will contains the result of the operation.
            operation (str): Name of the operation. Can be `-`, `inverse`, or more.
            input_var_name (str): Name of the operand.
        """
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
    def __init__(self, iterable_var_name: str, iterator: str | Expression, flow_control_instructions: list[FlowControlInstruction]) -> None:
        """
        Represents a foor loop.

        Warning: Because of python, the iterable variable and the iterator can have complex types, must check them in the error checking and maybe even restric them.

        Args:
            iterable_var_name (str): Name of the iterable variable.
            iterator (str): Value of the Iterator.
            flow_control_instructions (list[FlowControlInstruction]): The list of the flow control instructions inside the loop.
        """
        #
        self.iterable_var_name: str = iterable_var_name
        self.iterator: str | Expression = iterator
        self.flow_control_instructions: list[FlowControlInstruction] = flow_control_instructions

    #
    def __str__(self) -> str:
        #
        return f"\t\t * for {self.iterable_var_name} in {self.iterator} {{\n{"\n".join( [fci.__str__() for fci in self.flow_control_instructions] )}\n}}"


#
class FlowControlWhileLoop(FlowControlInstruction):
    #
    def __init__(self, condition: Condition, flow_control_instructions: list[FlowControlInstruction]) -> None:
        """
        Represents a while loop. (To avoid !)

        Args:
            condition (Condition): The condition to stay in the while loop.
            flow_control_instructions (list[FlowControlInstruction]): The list of the flow control instructions inside the loop.
        """
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
        """
        Represents a function call. (of a custom function (like a function of pytorch library)).

        Warning: The goal of this model extractor is not to be a complete code compiler, so we will not analyze the functions defined outside of model classes, so their usage will generate errors. To avoid that, move the functions into the model class block, and keep the variables and flow control simples.

        Args:
            output_variables (list[str]): Name of the outputs variables to get of the function call.
            function_called (str): Name of the function to call.
            function_arguments (dict[str, Any]): Keyword arguments of the function parameters to call.
        """
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
        """
        Represents a function call of a function that is defined inside a model class block.

        Args:
            output_variables (list[str]): Name of the outputs variables to get of the function call.
            function_called (str): Name of the function to call.
            function_arguments (dict[str, Any]): Keyword arguments of the function parameters to call.
        """
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
        """
        Represents a call to a layer of the model block.

        Args:
            output_variables (list[str]): Name of the outputs variables to get of the layer call.
            layer_name (str): Name of the variable that contains the layer.
            layer_arguments (dict[str, Any]): Keywords arguments of the layer parameters to call.
        """
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
        """
        Represents a return, eg. the output of the model block.

        Args:
            return_variables (list[str]): Variable to return.
        """
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
        """
        Represents a function of a model block.
        Can be the traditional forward function, or another function.

        Args:
            function_name (str): Name of the function
            function_arguments (dict[str, tuple[str, Any]]): Arguments of the function. tuple[str, Any] means the type of the variable (the 1st str part), and the optional default value of it (the 2nd Any part).
            model_block (ModelBlock): Link to the model block that contains this function.

        Attributes:
            function_flow_control (list[FlowControlInstruction]): Represents the control flow of the function. Will be completed during the AST visit.
        """

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
        """
        Represents a model block.

        Args:
            block_name (str): Name of the block.

        Attributes:
            block_parameters (dict[str, tuple[str, Any]]): The parameters of the block. Will be completed during the visit of the __init__ method in the AST. tuple[str, Any] means the type of the variable (the 1st str part), and the optional default value of it (the 2nd Any part).
            block_layers (dict[str, Layer]): The layers of the model block, indexed by the name of the variable that contains the Layer.
            block_functions (dict[str, BlockFunction]): The functions of the model block (needs at least the forward function), indexed by the name of the functions.
            block_variables (dict[str, tuple[str, Any]]): The variables of the model block, indexed by the variable names. The tuple[str, Any] is for (variable type, varible value).
        """

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
