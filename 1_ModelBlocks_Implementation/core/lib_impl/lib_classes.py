#
from typing import Any, Optional
#
import numpy as np
from numpy.typing import NDArray


# NOTE: when saying variable type, if there is an iterable or a tensor, the shape must be indicated


##############################################################
#######################     TYPES      #######################
##############################################################


#
class VarType:

    #
    def __init__(self, type_name: str) -> None:

        #
        self.type_name: str = type_name

    #
    def __repr__(self) -> str:

        #
        return self.__str__()

    #
    def __str__(self) -> str:

        #
        return f"Type({self.type_name})"


#
class VarTypeContainer(VarType):

    #
    def __init__(self, type_name: str, contained_type: VarType) -> None:

        #
        super().__init__(type_name=type_name)

        #
        self.contained_type: VarType = contained_type


#
class VarTypeTensor(VarType):

    #
    def __init__(self, tensor_type: str, tensor_dims: list[str | int]) -> None:

        #
        super().__init__(type_name = "Tensor")

        #
        self.tensor_type: str = tensor_type
        #
        self.tensor_dims: list[str | int] = tensor_dims

    #
    def __str__(self) -> str:

        #
        return f"Tensor({self.tensor_dims}, {self.tensor_type})"



####################################################################
#######################     EXPRESSIONS      #######################
####################################################################


#
class Expression:
    #
    def __init__(self) -> None:
        """
        Generic class for manipulating Expressions (in the sense of compilation and control flow).
        """

        #
        ### ABSTRACT CLASS ###
        #
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
        super().__init__()

        #
        self.var_name: str = var_name

    #
    def __str__(self) -> str:
        #
        return f"{self.var_name}"


#
class ExpressionConstant(Expression):
    #
    def __init__(self, constant: Optional[Any] = None) -> None:
        """
        Represents a constant expression.
        Abstract Class, should not be used because of type imprecisions.

        Args:
            constant (str): Constant value
        """

        #
        super().__init__()

        #
        self.constant: Any = constant

    #
    def __str__(self) -> str:
        #
        return f"{self.constant}"


#
class ExpressionConstantNumeric(ExpressionConstant):
    #
    def __init__(self, constant: int | float) -> None:
        """
        Represents a numerical constant expression.
        (for assignments or conditions, like : A = 5 -> Assignment(A, ExpressionConstant(5)))

        Args:
            constant (int | float): Constant numerical value
        """

        #
        super().__init__()

        #
        self.constant: int | float = constant

    #
    def __str__(self) -> str:
        #
        return f"{self.constant}"



#
class ExpressionConstantBoolean(ExpressionConstant):
    #
    def __init__(self, constant: bool) -> None:
        """
        Represents a numerical constant expression.
        (for assignments or conditions, like : A = 5 -> Assignment(A, ExpressionConstant(5)))

        Args:
            constant (bool): Constant boolean value
        """

        #
        super().__init__()

        #
        self.constant: bool = constant

    #
    def __str__(self) -> str:
        #
        return f"{self.constant}"


#
class ExpressionConstantString(ExpressionConstant):
    #
    def __init__(self, constant: str) -> None:
        """
        Represents a constant string expression.

        Args:
            constant (str): Constant string value
        """

        #
        super().__init__()

        #
        self.constant: str = constant

    #
    def __str__(self) -> str:
        #
        return f"{self.constant}"


#
class ExpressionConstantList(ExpressionConstant):
    #
    def __init__(self, elements: list[ExpressionConstant] = []) -> None:
        """
        Represents a constant list expression.

        Args:
            elements (list[ExpressionConstant]): list of the values
        """

        #
        super().__init__()

        #
        self.elements: list[ExpressionConstant] = elements

    #
    def __str__(self) -> str:
        #
        return f"[{", ".join([elt.__str__() for elt in self.elements])}]"


#
class ExpressionConstantRange(ExpressionConstantList):
    #
    def __init__(self, end_value: int, start_value: int = 0, step: int = 1) -> None:
        """
        Represents a constant range expression (a list).

        Args:
            end_value (int): End value of the range.
            start_value (int, optional): Start value of the range. Defaults to 0.
            step (int, optional): Step of the range. Defaults to 1. (Warning: the step can't be 0 because it generates an infinite constant list).
        """

        #
        super().__init__()

        #
        self.end_value: int = end_value
        self.start_value: int = start_value
        self.step: int = step

    #
    def __str__(self) -> str:
        #
        return f"Range({self.start_value}, {self.end_value}, {self.step})"


#
class ExpressionNone(ExpressionConstant):
    #
    def __init__(self) -> None:
        """
        _summary_
        """

        #
        super().__init__(constant=None)

        #
        pass

    #
    def __str__(self) -> str:
        #
        return "None"


#
class ExpressionNoDefaultArguments(ExpressionConstant):
    #
    def __init__(self) -> None:
        """
        _summary_
        """

        #
        super().__init__(constant=None)

        #
        pass

    #
    def __str__(self) -> str:
        #
        return "NoDefaultArgument"


#
class ExpressionToEvaluate(Expression):

    #
    def __init__(self, expr_to_evaluate: str) -> None:

        #
        super().__init__()

        #
        self.expr_to_evaluate: str = expr_to_evaluate


#
class ExpressionTuple(ExpressionConstant):
    #
    def __init__(self, elements: tuple[ExpressionConstant, ...] = ()) -> None:
        """
        Represents a constant tuple expression.

        Args:
            elements (tuple[ExpressionConstant, ...]): tuple of the values
        """

        #
        super().__init__()

        #
        self.elements: tuple[ExpressionConstant, ...] = elements

    #
    def __str__(self) -> str:
        #
        return f"({", ".join([elt.__str__() for elt in self.elements])})"


#
class ExpressionList(Expression):
    #
    def __init__(self, elements: list[Expression] = []) -> None:
        """
        Represents a list of expression.

        Args:
            elements (list[Expression]): list of the values
        """

        #
        super().__init__()

        #
        self.elements: list[Expression] = elements

    #
    def __str__(self) -> str:
        #
        return f"[{", ".join([elt.__str__() for elt in self.elements])}]"


#
class ExpressionDict(ExpressionConstant):
    #
    def __init__(self, elements: dict[ExpressionConstant, ExpressionConstant] = {}) -> None:
        """
        Represents a constant dictionary expression.

        Args:
            elements (dict[ExpressionConstant, ExpressionConstant]): dictionary of the values
        """

        #
        super().__init__()

        #
        self.elements: dict[ExpressionConstant, ExpressionConstant] = elements

    #
    def __str__(self) -> str:
        #
        return "{" + f"{", ".join([f"{k}: {v}" for k, v in self.elements.items()])}" + "}"


#
class ExpressionSet(ExpressionConstant):
    #
    def __init__(self, elements: set[ExpressionConstant] = set()) -> None:
        """
        Represents a constant set expression.

        Args:
            elements (set[ExpressionConstant]): set of the values
        """

        #
        super().__init__()

        #
        self.elements: set[ExpressionConstant] = elements

    #
    def __str__(self) -> str:
        #
        return "{" + f"{", ".join([elt.__str__() for elt in self.elements])}" + "}"


#
class ExpressionSlice1D(Expression):

    #
    def __init__(
        self,
        start: Optional[ExpressionConstant] = None,
        end: Optional[ExpressionConstant] = None,
        step: Optional[ExpressionConstant] = None
    ) -> None:

        """
        Represents a slice expression (1D).

        Args:
            start (Optional[ExpressionConstant], optional): Start of the slice. Defaults to None.
            end (Optional[ExpressionConstant], optional): End of the slice. Defaults to None.
            step (Optional[ExpressionConstant], optional): Step of the slice. Defaults to None.
        """

        #
        super().__init__()

        #
        self.start: Optional[ExpressionConstant] = start
        self.end: Optional[ExpressionConstant] = end
        self.step: Optional[ExpressionConstant] = step

    #
    def __str__(self) -> str:

        #
        start_str = str(self.start) if self.start is not None else ""
        end_str = str(self.end) if self.end is not None else ""
        step_str = str(self.step) if self.step is not None else ""

        #
        ### Handle different cases for proper slice syntax. ###
        #
        if step_str:
            #
            return f"[{start_str}:{end_str}:{step_str}]"

        #
        elif end_str:
            #
            return f"[{start_str}:{end_str}]"

        #
        elif start_str:
            #
            return f"[{start_str}:]"

        else:
            #
            return "[:]"


#
class ExpressionSliceND(Expression):

    #
    def __init__(self, slices: list[ExpressionSlice1D]) -> None:
        """
        Represents a slice expression (ND).

        Args:
            slices (list[ExpressionSlice1D]): List of the 1D slices for each dimension.
        """

        #
        super().__init__()

        #
        self.slices: list[ExpressionSlice1D] = slices

    #
    def __str__(self) -> str:
        #
        return f"[{", ".join([s.__str__() for s in self.slices])}]"


#
class ExpressionIndexAccess(Expression):
    #
    def __init__(self, variable: ExpressionVariable, index: Expression) -> None:
        """
        Represents an index access expression.

        Args:
            variable (ExpressionVariable): Variable on which we access the index.
            index (Expression): Index to access.
        """

        #
        super().__init__()

        #
        self.variable: ExpressionVariable = variable
        self.index: Expression = index

    #
    def __str__(self) -> str:
        #
        return f"{self.variable}[{self.index}]"


#
class ExpressionAttributeAccess(Expression):
    #
    def __init__(self, variable: ExpressionVariable, attribute: str) -> None:
        """
        Represents an attribute access expression.

        Args:
            variable (ExpressionVariable): Variable on which we access the attribute.
            attribute (str): Name of the attribute to access.
        """

        #
        super().__init__()

        #
        self.variable: ExpressionVariable = variable
        self.attribute: str = attribute

    #
    def __str__(self) -> str:
        #
        return f"{self.variable}.{self.attribute}"


#
class ExpressionRange(ExpressionList):
    #
    def __init__(self, end_value: int | ExpressionVariable, start_value: int | ExpressionVariable = 0, step: int | ExpressionVariable = 1) -> None:
        """
        Represents a range expression (a list).

        Args:
            end_value (int | ExpressionVariable): End value of the range.
            start_value (int, optional): Start value of the range. Defaults to 0.
            step (int, optional): Step of the range. Defaults to 1. (Warning: the step can't be 0 because it generates an infinite constant list).
        """

        #
        super().__init__()

        #
        self.end_value: int | ExpressionVariable = end_value
        self.start_value: int | ExpressionVariable = start_value
        self.step: int | ExpressionVariable = step

    #
    def __str__(self) -> str:
        #
        return f"Range({str(self.start_value)}, {str(self.end_value)}, {str(self.step)})"


# Normally, with the following FlowControlInstructions basic instructions, there is no need to have anything else than ExpressionVariable and ExpressionConstant, because we can decompose any complex instructions in a sequence of basic instructions (we may have to create temporary variables, but aside that, it is good)

# We will constraint the models to not use other types of constants and variables, (like dictionaries or custom objects), and we can convert the tuples into lists.
# We should constraint a list to have the same typing too.


#
class ExpressionBinaryOperation(Expression):
    #
    def __init__(self, left: Expression, operator: str, right: Expression) -> None:
        """
        Represents a binary operation expression.

        Args:
            left (Expression): Left operand
            operator (str): Binary operator (+, -, *, /, etc.)
            right (Expression): Right operand
        """

        #
        super().__init__()

        #
        self.left: Expression = left
        self.operator: str = operator
        self.right: Expression = right

    #
    def __str__(self) -> str:
        #
        return f"({str(self.left)} {self.operator} {str(self.right)})"


#
class ExpressionUnaryOperation(Expression):
    #
    def __init__(self, operator: str, operand: Expression) -> None:
        """
        Represents a unary operation expression.

        Args:
            operator (str): Unary operator (+, -, not, etc.)
            operand (Expression): Operand
        """

        #
        super().__init__()

        #
        self.operator: str = operator
        self.operand: Expression = operand

    #
    def __str__(self) -> str:
        #
        return f"{self.operator}({str(self.operand)})"


#
class ExpressionFunctionCall(Expression):
    #
    def __init__(self, function_name: str, function_arguments: list[Expression]) -> None:
        """
        Represents a function call expression.

        Args:
            function_name (str): Name of the function to call
            function_arguments (list[Expression]): Function arguments
        """

        #
        super().__init__()

        #
        self.function_name: str = function_name
        self.function_arguments: list[Expression] = function_arguments

    #
    def __str__(self) -> str:
        #
        args_str = ", ".join(str(arg) for arg in self.function_arguments)
        return f"{self.function_name}({args_str})"


####################################################################
#######################      CONDITIONS      #######################
####################################################################


#
class Condition:
    #
    def __init__(self) -> None:
        """
        Generic Abstract class for representing Conditions (for FlowControlWhileLoop or FlowControlCondition )
        """

        #
        ### ABSTRACT CLASS ###
        #
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
    def __init__(
        self,
        elt1: Expression | Condition,
        cond_operator: str,
        elt2: Expression | Condition
    ) -> None:

        """
        Represents a condition on two elements with one operation. (elt1 cond_operator elt2)

        Args:
            elt1 (Expression | Condition): left side of the condition
            cond_operator (str): Operator of the condition. Can be `and`, `or`, `>`, `<`, `>=`, `<=`, `!=`, `==`
            elt2 (Expression | Condition): right side of the condition
        """


        #
        super().__init__()

        #
        self.elt1: Expression | Condition = elt1

        #
        ### Can be `and`, `or`, `>`, `<`, `>=`, `<=`, `!=`, `==`. ###
        #
        self.cond_operator: str = cond_operator

        #
        self.elt2: Expression | Condition = elt2

    #
    def __str__(self) -> str:
        #
        return f"{self.elt1} {self.cond_operator} {self.elt2}"


#
class ConditionUnary(Condition):

    #
    def __init__(
        self,
        elt: Expression | Condition,
        cond_operator: Optional[str] = None
    ) -> None:

        """
        Represents a condition on one element with potentially an operation. (elt) or (cond_operator elt)

        Args:
            elt (Expression | Condition): Element of the condition
            cond_operator (Optional[str], optional): A potential unary condition operator. Can be `not`. Defaults to None.
        """

        #
        super().__init__()

        #
        self.elt: Expression | Condition = elt

        #
        ### Can be `not`. ###
        #
        self.cond_operator: Optional[str] = cond_operator

    #
    def __str__(self) -> str:
        #
        if self.cond_operator is None:
            #
            return f"{self.elt}"
        #
        return f"{self.cond_operator} {self.elt}"


#
class ConditionElse(Condition):

    #
    def __init__(self) -> None:
        """
        Represents an else condition.
        """

        #
        super().__init__()

        #
        pass

    #
    def __str__(self) -> str:
        #
        return "Else"


##############################################################
#######################     LAYERS     #######################
##############################################################


#
class Layer:

    #
    def __init__(
        self,
        layer_var_name: str,
        layer_type: str,
        layer_parameters_kwargs: dict[str, Expression]
    ) -> None:

        """
        Can represents a basic Neural Network Layer (from the list `layers.json`), or a block of a model.

        Args:
            layer_var_name (str): Name of the variable in the current model block that is containing this layer.
            layer_type (str): Name of the layer or the model block.
            layer_parameters_kwargs (dict[str, Expression]): Parameters of the layer.
        """

        #
        ### name of the variable to be called from the model block. ###
        #
        self.layer_var_name: str = layer_var_name

        #
        ### name of the layer or the model block. ###
        #
        self.layer_type: str = layer_type

        #
        ### the dict[str, Expression] is for variable name -> variable value. ###
        #
        self.layer_parameters_kwargs: dict[str, Expression] = layer_parameters_kwargs

        #
        ### Will contain all the weights of the layer to init & forward. ###
        #
        self.layer_weights: dict[str, NDArray[np.float32]] = {}

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return f"* Layer {self.layer_var_name} = {self.layer_type}({self.layer_parameters_kwargs})"


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

        #
        ### GENERIC ABSTRACT CLASS ###
        #
        pass

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return "\n\t * [FLOW CONTROL PLACEHOLDER]"


#
class FlowControlVariableInit(FlowControlInstruction):

    #
    def __init__(
        self,
        var_name: str,
        var_type: VarType,
        var_value: Optional[Expression] = None
    ) -> None:

        """
        Represents a Variable Initialisation.

        Args:
            var_name (str): Name of the variable.
            var_type (str): Type of the variable
            var_value (Optional[Expression], optional): Potential Initialisation Value. Defaults to None.
        """

        #
        super().__init__()

        #
        self.var_name: str = var_name
        self.var_type: VarType = var_type
        self.var_value: Optional[Expression] = var_value

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.var_name}: {self.var_type} = {self.var_value}"


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
        super().__init__()

        #
        self.var_name: str = var_name
        self.var_value: Expression = var_value

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.var_name} = {self.var_value}"


#
class FlowControlBasicBinaryOperation(FlowControlInstruction):

    #
    def __init__(
        self,
        output_var_name: str,
        input1_var_name: str,
        operation: str,
        input2_var_name: str
    ) -> None:

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
        super().__init__()

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
    def __init__(
        self,
        output_var_name: str,
        operation: str,
        input_var_name: str
    ) -> None:

        """
        Represent a basic unary operation.

        Warning: the operand can be both vector / matrix / tensor or scalars !

        Args:
            output_var_name (str): Name of the variable that will contains the result of the operation.
            operation (str): Name of the operation. Can be `-`, `inverse`, or more.
            input_var_name (str): Name of the operand.
        """

        #
        super().__init__()

        #
        self.output_var_name: str = output_var_name
        self.input_var_name: str = input_var_name
        self.operation: str = operation

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.output_var_name} = {self.operation} {self.input_var_name}"


#
class FlowControlForEachLoop(FlowControlInstruction):

    #
    def __init__(
        self,
        iterable_var_name: str,
        iterator: str | Expression,
        flow_control_instructions: list[FlowControlInstruction]
    ) -> None:

        """
        Represents a foor loop.

        Warning: Because of python, the iterable variable and the iterator can have complex types, must check them in the error checking and maybe even restric them.

        Args:
            iterable_var_name (str): Name of the iterable variable.
            iterator (str): Value of the Iterator.
            flow_control_instructions (list[FlowControlInstruction]): The list of the flow control instructions inside the loop.
        """

        #
        super().__init__()

        #
        self.iterable_var_name: str = iterable_var_name
        self.iterator: str | Expression = iterator
        self.flow_control_instructions: list[FlowControlInstruction] = flow_control_instructions

    #
    def __str__(self) -> str:
        #
        return f"\t\t * for {self.iterable_var_name} in {self.iterator} {{\n{"\n".join( [fci.__str__() for fci in self.flow_control_instructions] )}\n}}"


#
class FlowControlForRange(FlowControlInstruction):

    #
    def __init__(
        self,
        iterable_var_name: str,
        for_range: ExpressionConstantRange,
        flow_control_instructions: list[FlowControlInstruction]
    ) -> None:

        """
        Represents a foor loop.

        Warning: Because of python, the iterable variable and the iterator can have complex types, must check them in the error checking and maybe even restric them.

        Args:
            iterable_var_name (str): Name of the iterable variable.
            iterator (str): Value of the Iterator.
            flow_control_instructions (list[FlowControlInstruction]): The list of the flow control instructions inside the loop.
        """

        #
        super().__init__()

        #
        self.iterable_var_name: str = iterable_var_name
        self.for_range: ExpressionConstantRange = for_range
        self.flow_control_instructions: list[FlowControlInstruction] = flow_control_instructions

    #
    def __str__(self) -> str:
        #
        return f"\t\t * for {self.iterable_var_name} in {self.iterator} {{\n{"\n".join( [fci.__str__() for fci in self.flow_control_instructions] )}\n}}"


#
### TODO: add other types of for to have better representation, easier code extraction, and easier code interpretation??? (If not, just remove this comment) ###
#


#
class FlowControlWhileLoop(FlowControlInstruction):

    #
    def __init__(
        self,
        condition: Condition,
        flow_control_instructions: list[FlowControlInstruction]
    ) -> None:

        """
        Represents a while loop. (To avoid !)

        Args:
            condition (Condition): The condition to stay in the while loop.
            flow_control_instructions (list[FlowControlInstruction]): The list of the flow control instructions inside the loop.
        """

        #
        super().__init__()

        #
        self.condition: Condition = condition

        #
        self.flow_control_instructions: list[FlowControlInstruction] = flow_control_instructions

    #
    def __str__(self) -> str:
        #
        return f"\t\t * while {self.condition} {{\n{"\n".join( [fci.__str__() for fci in self.flow_control_instructions] )}\n}}"


#
class FlowControlFunctionCall(FlowControlInstruction):

    #
    def __init__(
        self,
        output_variables: list[str],
        function_called: str,
        function_arguments: dict[str, Expression]
    ) -> None:

        """
        Represents a function call. (of a custom function (like a function of pytorch library)).

        Warning: The goal of this model extractor is not to be a complete code compiler, so we will not analyze the functions defined outside of model classes, so their usage will generate errors. To avoid that, move the functions into the model class block, and keep the variables and flow control simples.

        Args:
            output_variables (list[str]): Name of the outputs variables to get of the function call.
            function_called (str): Name of the function to call.
            function_arguments (dict[str, Any]): Keyword arguments of the function parameters to call.
        """

        #
        super().__init__()

        #
        self.output_variables: list[str] = output_variables
        self.function_called: str = function_called

        #
        ### the tuple[str, Any] is for (variable type, variable default value). ###
        #
        self.function_arguments: dict[str, Expression] = function_arguments

    #
    def __str__(self) -> str:

        #
        ### Format arguments more naturally for display. ###
        #
        formatted_args: list[str] = []

        #
        ### Handle positional arguments (keys that are numeric strings). ###
        #
        positional_args: list[tuple[int, Expression]] = []
        keyword_args: list[tuple[str, Expression]] = []

        #
        key: str
        value: Expression
        #
        for key, value in self.function_arguments.items():

            #
            if key.isdigit():
                #
                positional_args.append((int(key), value))
            #
            else:
                #
                keyword_args.append((key, value))

        #
        ### Sort positional arguments by index. ###
        #
        positional_args.sort(key=lambda x: x[0])

        #
        ### Add positional arguments. ###
        #
        for _, value in positional_args:
            #
            formatted_args.append(str(value))

        #
        ### Add keyword arguments. ###
        #
        for key, value in keyword_args:
            #
            formatted_args.append(f"{key}={value}")

        #
        args_str: str = ", ".join(formatted_args)

        #
        return f"\t\t * {self.output_variables} = {self.function_called}({args_str})"


#
class FlowControlSubBlockFunctionCall(FlowControlInstruction):

    #
    def __init__(
        self,
        output_variables: list[str],
        function_called: str,
        function_arguments: dict[str, Expression]
    ) -> None:

        """
        Represents a function call of a function that is defined inside a model class block.

        Args:
            output_variables (list[str]): Name of the outputs variables to get of the function call.
            function_called (str): Name of the function to call.
            function_arguments (dict[str, Any]): Keyword arguments of the function parameters to call.
        """

        #
        super().__init__()

        #
        self.output_variables: list[str] = output_variables

        #
        self.function_called: str = function_called

        #
        ### the tuple[str, Any] is for (variable type, variable default value). ###
        #
        self.function_arguments: dict[str, Expression] = function_arguments

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.output_variables} = {self.function_called}({self.function_arguments})"


#
class FlowControlLayerPass(FlowControlInstruction):

    #
    def __init__(
        self,
        output_variables: list[str],
        layer_name: str,
        layer_arguments: dict[str, Expression]
    ) -> None:

        """
        Represents a call to a layer of the model block.

        Args:
            output_variables (list[str]): Name of the outputs variables to get of the layer call.
            layer_name (str): Name of the variable that contains the layer.
            layer_arguments (dict[str, Any]): Keywords arguments of the layer parameters to call.
        """

        #
        super().__init__()

        #
        self.output_variables: list[str] = output_variables
        self.layer_name: str = layer_name

        #
        ### the tuple[str, Any] is for (variable type, variable default value). ###
        #
        self.layer_arguments: dict[str, Expression] = layer_arguments

    #
    def __str__(self) -> str:
        #
        return f"\t\t * {self.output_variables} = {self.layer_name}({self.layer_arguments})"


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
        super().__init__()

        #
        self.return_variables: list[str] = return_variables

    #
    def __str__(self) -> str:
        #
        return f"\t\t * return {self.return_variables}"


#
class FlowControlCondition(FlowControlInstruction):

    #
    def __init__(
        self,
        conditions_fn_call: dict[Condition, FlowControlSubBlockFunctionCall]
    ) -> None:

        """
        Represents a condition of the control flow of a function (forward or other of a model block).
        Each of the paths in the conditions will be moved inside a sub-function (aside the foward function) of the model block, and this layer will be created and added into the current model block.

        Args:
            conditions_fn_call (dict[Condition, FlowControlSubBlockFunctionCall]): Dictionnary of couples (Condition, FlowControlSubBlockFunctionCall), where intuitively, each FlowControlSubBlockFunctionCall is associated to its corresponding Condition.
        """

        #
        super().__init__()

        #
        self.conditions_fn_call: dict[Condition, FlowControlSubBlockFunctionCall] = conditions_fn_call

    #
    def __str__(self) -> str:

        #
        return f"\t\t * Conditions: {self.conditions_fn_call}"


##############################################################
#######################     BLOCKS     #######################
##############################################################


#
class BlockFunction:

    #
    def __init__(
        self,
        function_name: str,
        function_arguments: dict[str, tuple[VarType, Expression]],
        model_block: "ModelBlock",
        complex_default_argument_values_instructions_to_do_before_real_function_flow: Optional[dict[str, list[FlowControlInstruction]]] = None,
    ) -> None:

        """
        Represents a function of a model block.
        Can be the traditional forward function, or another function.

        Args:
            function_name (str): Name of the function
            function_arguments (dict[str, tuple[VarType, Expression]]): Arguments of the function. tuple[str, Any] means the type of the variable (the 1st str part), and the optional default value of it (the 2nd Any part).
            model_block (ModelBlock): Link to the model block that contains this function.

        Attributes:
            function_flow_control (list[FlowControlInstruction]): Represents the control flow of the function. Will be completed during the AST visit.
        """

        #
        ### To access the block variables / layers. ###
        #
        self.model_block: ModelBlock = model_block

        #
        self.function_name: str = function_name

        #
        ### the tuple[str, Any] is for (variable type, variable default value). ###
        #
        self.function_arguments: dict[str, tuple[VarType, Expression]] = function_arguments

        #
        ### To complete while analysis. ###
        #
        self.function_flow_control: list[FlowControlInstruction] = []

        #
        ### To complete before real function flow, and before the default argument values instructions. ###
        #
        self.complex_default_argument_values_instructions_to_do_before_real_function_flow: Optional[dict[str, list[FlowControlInstruction]]] = complex_default_argument_values_instructions_to_do_before_real_function_flow

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:
        #
        return f"\t * Function {self.function_name} ( {self.function_arguments} ) :\n{"\n".join([ffc.__str__() for ffc in self.function_flow_control])}"


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

        #
        ### the tuple[str, Any] is for (variable type, variable default value). ###
        #
        self.block_parameters: dict[str, tuple[VarType, Expression]] = {}

        #
        ### indicate the weights of this model block, with its dimensions; indexed by the variable name of the weights. ###
        #
        self.block_weights: dict[str, list[int | str]] = {}

        #
        self.block_layers: dict[str, Layer] = {}

        #
        self.block_functions: dict[str, BlockFunction] = {
            #
            ### TO ADD: the foward pass (during the code analysis). ###
            #
        }

        #
        ### the tuple[str, Any] is for (variable type, variable value). ###
        #
        self.block_variables: dict[str, tuple[VarType, Expression]] = {}

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __getattr__(self, name: str) -> Any:
        """
        Handle attribute access for layer names.
        This method should not be called directly - layers should be accessed
        through the execution context, not through the ModelBlock object.

        Args:
            name (str): The attribute name being accessed

        Raises:
            AttributeError: Always raises an error as this method should not be used
        """

        #
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'. Layer access should be through execution context.")

    #
    def __str__(self) -> str:
        #
        return f"\n\nModelBlock {self.block_name}:\n\t-block_parameters: {self.block_parameters}\n\t-block_layers:\n\t\t{"\n\t\t".join([self.block_layers[layer].__str__() for layer in self.block_layers])}\n\n\t-Functions:\n\n{"\n\n".join([self.block_functions[fn_name].__str__() for fn_name in self.block_functions])}\n"



#########################################################################
#######################     MAIN OUTPUT CLASS     #######################
#########################################################################


#
class Language_Model:
    #
    def __init__(self) -> None:
        """
        Output of the analysis.

        Attributes:
            - model_blocks (dict[str, lc.ModelBlock]): list of all the blocks analyzed here, indexed by their blocks id (e.g. their name)
            - main_block (str): id of the main block, given in sys.argv with `--main-block <MainBlockName>`
            - global constants (dict[str, tuple[str, Any]])
        """

        #
        self.model_blocks: dict[str, ModelBlock] = {}
        #
        self.main_block: str = ""
        #
        self.main_layer: Optional[Layer] = None
        #
        self.global_constants: dict[str, tuple[VarType, Expression]] = {}

    #
    def export_to_cpp(self) -> str:

        #
        res: str = ""

        #
        return res

    #
    def __repr__(self) -> str:
        #
        return self.__str__()

    #
    def __str__(self) -> str:

        #
        text: str = "\n\n" + "-" * 26 + "\n -- Model Architecture --\n" + "-" * 26 + "\n\n"

        #
        text += "Global constants:\n"

        #
        constant_name: str
        #
        for constant_name in self.global_constants:
            #
            text += f"  - {constant_name} = {self.global_constants[constant_name]}\n"

        #
        text += "\nModel blocks:\n"

        #
        for _block_name, block in self.model_blocks.items():
            #
            text += block.__str__()

        #
        return text

