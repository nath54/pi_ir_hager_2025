#
### Import Modules. ###
#
from typing import Any, Optional, cast
#
import numpy as np
from numpy.typing import NDArray
#
import copy

#
### Import all classes from lib_classes. ###
#
from . import lib_classes as lc


#
#########################################################################
########################## FORWARD INTERPRETER ###########################
#########################################################################
#


#
class ExecutionContext:
    """
    Manages the execution context during forward pass interpretation.
    Keeps track of variables, their types, and values.
    """

    #
    def __init__(self) -> None:
        """
        Initialize execution context.
        """

        #
        self.variables: dict[str, Any] = {}
        #
        self.variable_types: dict[str, lc.VarType] = {}

    #
    def set_variable(self, var_name: str, var_type: lc.VarType, value: Any) -> None:
        """
        Set a variable in the context.

        Args:
            var_name (str): Name of the variable
            var_type (lc.VarType): Type of the variable
            value (Any): Value to assign
        """

        #
        self.variables[var_name] = value
        self.variable_types[var_name] = var_type

    #
    def get_variable(self, var_name: str) -> Any:
        """
        Get a variable from the context.

        Args:
            var_name (str): Name of the variable

        Returns:
            Any: Value of the variable

        Raises:
            KeyError: If variable doesn't exist
        """

        #
        if var_name not in self.variables:
            #
            raise KeyError(f"Variable '{var_name}' not found in execution context")

        #
        return self.variables[var_name]

    #
    def has_variable(self, var_name: str) -> bool:
        """
        Check if variable exists in context.

        Args:
            var_name (str): Name of the variable

        Returns:
            bool: True if variable exists
        """

        #
        return var_name in self.variables

    #
    def copy(self) -> "ExecutionContext":
        """
        Create a copy of the execution context.

        Returns:
            ExecutionContext: Deep copy of current context
        """

        #
        new_context = ExecutionContext()
        new_context.variables = copy.deepcopy(self.variables)
        new_context.variable_types = copy.deepcopy(self.variable_types)

        #
        return new_context


#
class LanguageModel_ForwardInterpreter:
    """
    Forward pass interpreter for Language Models defined using lib_classes.py structures.
    Executes the model by interpreting the flow control instructions and performing computations.
    """

    #
    def __init__(self, language_model: lc.Language_Model) -> None:
        """
        Initialize the forward interpreter.

        Args:
            language_model (lc.Language_Model): The language model to interpret
        """

        #
        self.language_model: lc.Language_Model = language_model
        self.global_context: ExecutionContext = ExecutionContext()

        #
        ### Initialize global constants. ###
        #
        self._initialize_global_constants()

    #
    def _initialize_global_constants(self) -> None:
        """
        Initialize global constants in the execution context.
        """

        #
        for const_name, (const_type, const_expr) in self.language_model.global_constants.items():

            #
            ### Evaluate the expression and add it to the global context. ###
            #
            const_value = self._evaluate_expression(const_expr, self.global_context)
            #
            self.global_context.set_variable(const_name, const_type, const_value)

    #
    def forward(self, inputs: dict[str, NDArray[np.float32]], **kwargs: dict[str, Any]) -> dict[str, NDArray[np.float32]]:
        """
        Execute forward pass of the main model block.

        Args:
            inputs (dict[str, NDArray[np.float32]]): Input tensors
            **kwargs: Additional keyword arguments

        Returns:
            dict[str, NDArray[np.float32]]: Output tensors
        """

        #
        if not self.language_model.main_block:
            #
            raise ValueError("No main block specified in language model")

        #
        main_block = self.language_model.model_blocks[self.language_model.main_block]

        #
        ### Create execution context for this forward pass. ###
        #
        context = self.global_context.copy()

        #
        ### Set input variables. ###
        #
        for input_name, input_value in inputs.items():

            #
            ### Determine tensor type from shape. ###
            #
            tensor_dims: list[int | str] = list(input_value.shape)
            #
            tensor_type = str(input_value.dtype)
            #
            var_type = lc.VarTypeTensor(tensor_type, tensor_dims)
            #
            context.set_variable(input_name, var_type, input_value)

        #
        ### Set additional keyword arguments. ###
        #
        for kwarg_name, kwarg_value in kwargs.items():

            #
            ### Infer type based on value. ###
            #
            if isinstance(kwarg_value, (int, float)):
                #
                var_type = lc.VarType("numeric")
            #
            elif isinstance(kwarg_value, str):
                #
                var_type = lc.VarType("string")
            #
            elif isinstance(kwarg_value, np.ndarray):
                #
                np_kwarg_value: NDArray[np.float32] = cast(NDArray[np.float32], kwarg_value)
                #
                tensor_dims = list(np_kwarg_value.shape)
                tensor_type = str(np_kwarg_value.dtype)
                var_type = lc.VarTypeTensor(tensor_type, tensor_dims)
            #
            else:
                #
                var_type = lc.VarType("unknown")

            #
            context.set_variable(kwarg_name, var_type, kwarg_value)

        #
        ### Initialize block parameters and layers. ###
        #
        self._initialize_model_block(main_block, context)

        #
        ### Execute forward function. ###
        #
        if "forward" not in main_block.block_functions:
            #
            raise ValueError("No forward function found in main block")

        #
        forward_function = main_block.block_functions["forward"]

        #
        return self._execute_block_function(forward_function, context, inputs)

    #
    def _initialize_model_block(self, model_block: lc.ModelBlock, context: ExecutionContext) -> None:
        """
        Initialize model block parameters and layers in the execution context.

        Args:
            model_block (lc.ModelBlock): Model block to initialize
            context (ExecutionContext): Execution context
        """

        #
        ### Initialize block parameters. ###
        #
        for param_name, (param_type, param_expr) in model_block.block_parameters.items():
            #
            param_value = self._evaluate_expression(param_expr, context)
            #
            context.set_variable(param_name, param_type, param_value)

        #
        ### Initialize block variables. ###
        #
        for var_name, (var_type, var_expr) in model_block.block_variables.items():
            #
            var_value: Any = self._evaluate_expression(var_expr, context)
            #
            context.set_variable(var_name, var_type, var_value)

        #
        ### Initialize layers (load weights). ###
        #
        for layer_name, layer in model_block.block_layers.items():

            #
            ### Initialize layer weights if they exist. ###
            #
            layer_obj: dict[str, Any] = self._create_layer_instance(layer, context)
            #
            context.set_variable(layer_name, lc.VarType("lc.Layer"), layer_obj)

    #
    def _create_layer_instance(self, layer: lc.Layer, context: ExecutionContext) -> dict[str, Any]:
        """
        Create a layer instance with initialized weights.

        Args:
            layer (lc.Layer): lc.Layer definition
            context (ExecutionContext): Execution context

        Returns:
            dict[str, Any]: lc.Layer instance data
        """

        #
        layer_instance: dict[str, Any] = {
            "type": layer.layer_type,
            "parameters": {},
            "weights": layer.layer_weights.copy()
        }

        #
        ### Evaluate layer parameters. ###
        #
        for param_name, param_expr in layer.layer_parameters_kwargs.items():
            #
            layer_instance["parameters"][param_name] = self._evaluate_expression(param_expr, context)

        #
        return layer_instance

    #
    def _execute_block_function(
        self,
        block_function: lc.BlockFunction,
        context: ExecutionContext,
        function_args: Optional[dict[str, Any]] = None
    ) -> dict[str, NDArray[np.float32]]:

        """
        Execute a block function.

        Args:
            block_function (lc.BlockFunction): Function to execute
            context (ExecutionContext): Execution context
            function_args (dict[str, Any], optional): Function arguments

        Returns:
            dict[str, NDArray[np.float32]]: Function outputs
        """

        #
        ### Create local context for function execution. ###
        #
        local_context = context.copy()

        #
        ### Set function arguments. ###
        #
        if function_args:
            #
            for arg_name, arg_value in function_args.items():
                #
                if arg_name in block_function.function_arguments:
                    #
                    arg_type, _ = block_function.function_arguments[arg_name]
                    #
                    local_context.set_variable(arg_name, arg_type, arg_value)

        #
        ### Set default values for missing arguments. ###
        #
        for arg_name, (arg_type, default_expr) in block_function.function_arguments.items():
            #
            if not local_context.has_variable(arg_name):
                #
                if not isinstance(default_expr, lc.ExpressionNoDefaultArguments):
                    #
                    default_value: Any = self._evaluate_expression(default_expr, local_context)
                    #
                    local_context.set_variable(arg_name, arg_type, default_value)

        #
        ### Execute flow control instructions. ###
        #
        outputs: dict[str, NDArray[np.float32]] = {}
        #
        for instruction in block_function.function_flow_control:
            #
            _result: Any = self._execute_flow_control_instruction(instruction, local_context)
            #
            if isinstance(instruction, lc.FlowControlReturn):

                #
                ### Handle return statement. ###
                #
                for return_var in instruction.return_variables:
                    #
                    outputs[return_var] = local_context.get_variable(return_var)

                #
                break

        #
        return outputs

    #
    def _execute_flow_control_instruction(self, instruction: lc.FlowControlInstruction,
                                        context: ExecutionContext) -> Any:
        """
        Execute a single flow control instruction.

        Args:
            instruction (lc.FlowControlInstruction): Instruction to execute
            context (ExecutionContext): Execution context

        Returns:
            Any: Result of the instruction execution
        """

        #
        if isinstance(instruction, lc.FlowControlVariableInit):
            #
            return self._execute_variable_init(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlVariableAssignment):
            #
            return self._execute_variable_assignment(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlBasicBinaryOperation):
            #
            return self._execute_binary_operation(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlBasicUnaryOperation):
            #
            return self._execute_unary_operation(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlForLoop):
            #
            return self._execute_for_loop(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlWhileLoop):
            #
            return self._execute_while_loop(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlFunctionCall):
            #
            return self._execute_function_call(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlSubBlockFunctionCall):
            #
            return self._execute_sub_block_function_call(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlLayerPass):
            #
            return self._execute_layer_pass(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlReturn):
            #
            return self._execute_return(instruction, context)
        #
        elif isinstance(instruction, lc.FlowControlCondition):
            #
            return self._execute_condition(instruction, context)
        #
        else:
            #
            raise NotImplementedError(f"lc.Flow control instruction {type(instruction)} not implemented")

    #
    def _execute_variable_init(self, instruction: lc.FlowControlVariableInit, context: ExecutionContext) -> None:
        """
        Execute variable initialization.

        Args:
            instruction (lc.FlowControlVariableInit): Variable initialization instruction
            context (ExecutionContext): Execution context
        """

        #
        if instruction.var_value is not None:
            #
            value = self._evaluate_expression(instruction.var_value, context)
        #
        else:
            #
            value = self._get_default_value_for_type(instruction.var_type)

        #
        context.set_variable(instruction.var_name, instruction.var_type, value)

    #
    def _execute_variable_assignment(self, instruction: lc.FlowControlVariableAssignment, context: ExecutionContext) -> None:
        """
        Execute variable assignment.

        Args:
            instruction (lc.FlowControlVariableAssignment): Variable assignment instruction
            context (ExecutionContext): Execution context
        """

        #
        value = self._evaluate_expression(instruction.var_value, context)

        #
        ### Get existing type if variable exists, otherwise infer type. ###
        #
        if context.has_variable(instruction.var_name):
            #
            var_type = context.variable_types[instruction.var_name]
        #
        else:
            #
            var_type = self._infer_type_from_value(value)

        #
        context.set_variable(instruction.var_name, var_type, value)

    #
    def _execute_binary_operation(self, instruction: lc.FlowControlBasicBinaryOperation, context: ExecutionContext) -> None:
        """
        Execute binary operation.

        Args:
            instruction (lc.FlowControlBasicBinaryOperation): Binary operation instruction
            context (ExecutionContext): Execution context
        """

        #
        input1 = context.get_variable(instruction.input1_var_name)
        input2 = context.get_variable(instruction.input2_var_name)

        #
        ### Perform operation based on operator. ###
        #
        if instruction.operation == "+":
            #
            result = input1 + input2
        #
        elif instruction.operation == "-":
            #
            result = input1 - input2
        #
        elif instruction.operation == "*":
            #
            result = input1 * input2
        #
        elif instruction.operation == "/":
            #
            result = input1 / input2
        #
        elif instruction.operation == ".*":
            #
            result = np.multiply(input1, input2)
        #
        elif instruction.operation == "./":
            #
            result = np.divide(input1, input2)
        #
        elif instruction.operation == "^":
            #
            result = np.power(input1, input2)
        #
        elif instruction.operation == "<<":
            #
            result = np.left_shift(input1.astype(int), input2.astype(int))
        #
        elif instruction.operation == ">>":
            #
            result = np.right_shift(input1.astype(int), input2.astype(int))
        #
        elif instruction.operation == "%":
            #
            result = input1 % input2
        #
        else:
            #
            raise NotImplementedError(f"Binary operation '{instruction.operation}' not implemented")

        #
        ### Infer result type. ###
        #
        result_type = self._infer_type_from_value(result)
        #
        context.set_variable(instruction.output_var_name, result_type, result)

    #
    def _execute_unary_operation(self, instruction: lc.FlowControlBasicUnaryOperation, context: ExecutionContext) -> None:
        """
        Execute unary operation.

        Args:
            instruction (lc.FlowControlBasicUnaryOperation): Unary operation instruction
            context (ExecutionContext): Execution context
        """

        #
        input_value = context.get_variable(instruction.input_var_name)

        #
        ### Perform operation based on operator. ###
        #
        if instruction.operation == "-":
            #
            result = -input_value
        #
        elif instruction.operation == "inverse":
            #
            result = np.reciprocal(input_value)
        #
        else:
            #
            raise NotImplementedError(f"Unary operation '{instruction.operation}' not implemented")

        #
        ### Infer result type. ###
        #
        result_type = self._infer_type_from_value(result)
        #
        context.set_variable(instruction.output_var_name, result_type, result)

    #
    def _execute_for_loop(self, instruction: lc.FlowControlForLoop, context: ExecutionContext) -> None:
        """
        Execute for loop.

        Args:
            instruction (lc.FlowControlForLoop): For loop instruction
            context (ExecutionContext): Execution context
        """

        #
        if isinstance(instruction.iterator, lc.Expression):
            #
            iterable = self._evaluate_expression(instruction.iterator, context)
        #
        else:
            #
            raise UserWarning(f"Error, not implemented yet, instruction.iterator was not of type Expression for instruction : {instruction} !")

        #
        ### Execute loop body for each iteration. ###
        #
        for item in iterable:

            #
            ### Create loop context. ###
            #
            loop_context = context.copy()

            #
            ### Set iterator variable. ###
            #
            iterator_type = self._infer_type_from_value(item)
            #
            loop_context.set_variable(instruction.iterable_var_name, iterator_type, item)

            #
            ### Execute loop body. ###
            #
            for loop_instruction in instruction.flow_control_instructions:
                #
                self._execute_flow_control_instruction(loop_instruction, loop_context)

            #
            ### Update main context with loop results (for variables that existed before loop). ###
            #
            for var_name, value in loop_context.variables.items():
                #
                if context.has_variable(var_name):
                    #
                    context.set_variable(var_name, loop_context.variable_types[var_name], value)

    #
    def _execute_while_loop(self, instruction: lc.FlowControlWhileLoop, context: ExecutionContext) -> None:
        """
        Execute while loop.

        Args:
            instruction (lc.FlowControlWhileLoop): While loop instruction
            context (ExecutionContext): Execution context
        """

        #
        while self._evaluate_condition(instruction.condition, context):

            #
            ### Execute loop body. ###
            #
            for loop_instruction in instruction.flow_control_instructions:
                #
                self._execute_flow_control_instruction(loop_instruction, context)

    #
    def _execute_function_call(self, instruction: lc.FlowControlFunctionCall, context: ExecutionContext) -> None:
        """
        Execute external function call.

        Args:
            instruction (lc.FlowControlFunctionCall): Function call instruction
            context (ExecutionContext): Execution context
        """

        #
        ### Evaluate function arguments. ###
        #
        args: dict[str, Any] = {}
        #
        for arg_name, arg_expr in instruction.function_arguments.items():
            #
            args[arg_name] = self._evaluate_expression(arg_expr, context)

        #
        ### Call the function (this would need to be extended based on available functions). ###
        #
        result: Any = self._call_external_function(instruction.function_called, args)

        #
        ### Handle multiple outputs. ###
        #
        if len(instruction.output_variables) == 1:
            #
            result_type = self._infer_type_from_value(result)
            context.set_variable(instruction.output_variables[0], result_type, result)
        #
        else:
            #
            for i, output_var in enumerate(instruction.output_variables):
                #
                output_value: Any = result[i] if isinstance(result, (list, tuple)) else result  # type: ignore
                result_type: lc.VarType = self._infer_type_from_value(output_value)
                #
                context.set_variable(output_var, result_type, output_value)

    #
    def _execute_sub_block_function_call(
        self,
        instruction: lc.FlowControlSubBlockFunctionCall,
        context: ExecutionContext
    ) -> None:

        """
        Execute sub-block function call.

        Args:
            instruction (lc.FlowControlSubBlockFunctionCall): Sub-block function call instruction
            context (ExecutionContext): Execution context
        """

        #
        ### Get the current model block. ###
        #
        current_block: Optional[lc.ModelBlock] = None
        #
        for block in self.language_model.model_blocks.values():
            #
            if instruction.function_called in block.block_functions:
                #
                current_block = block
                break

        #
        if current_block is None:
            #
            raise ValueError(f"Function '{instruction.function_called}' not found in any model block")

        #
        ### Evaluate function arguments. ###
        #
        function_args: dict[str, Any] = {}
        #
        for arg_name, arg_expr in instruction.function_arguments.items():
            #
            function_args[arg_name] = self._evaluate_expression(arg_expr, context)

        #
        ### Execute the function. ###
        #
        target_function = current_block.block_functions[instruction.function_called]
        #
        outputs = self._execute_block_function(target_function, context, function_args)

        #
        ### Set output variables. ###
        #
        for i, output_var in enumerate(instruction.output_variables):
            #
            if i < len(outputs):
                #
                output_key = list(outputs.keys())[i]
                result_type = self._infer_type_from_value(outputs[output_key])
                #
                context.set_variable(output_var, result_type, outputs[output_key])

    #
    def _execute_layer_pass(self, instruction: lc.FlowControlLayerPass, context: ExecutionContext) -> None:
        """
        Execute layer pass (forward pass through a neural network layer).

        Args:
            instruction (lc.FlowControlLayerPass): lc.Layer pass instruction
            context (ExecutionContext): Execution context
        """

        #
        layer_instance = context.get_variable(instruction.layer_name)

        #
        ### Evaluate layer arguments. ###
        #
        layer_args: dict[str, Any] = {}
        #
        for arg_name, arg_expr in instruction.layer_arguments.items():
            #
            layer_args[arg_name] = self._evaluate_expression(arg_expr, context)

        #
        ### Execute layer forward pass. ###
        #
        result = self._execute_layer_forward(layer_instance, layer_args)

        #
        ### Handle multiple outputs. ###
        #
        if len(instruction.output_variables) == 1:
            #
            result_type = self._infer_type_from_value(result)
            context.set_variable(instruction.output_variables[0], result_type, result)
        #
        else:
            #
            for i, output_var in enumerate(instruction.output_variables):
                #
                output_value = result[i] if isinstance(result, (list, tuple)) else result
                result_type = self._infer_type_from_value(output_value)
                #
                context.set_variable(output_var, result_type, output_value)

    #
    def _execute_return(self, instruction: lc.FlowControlReturn, context: ExecutionContext) -> dict[str, Any]:
        """
        Execute return statement.

        Args:
            instruction (lc.FlowControlReturn): Return instruction
            context (ExecutionContext): Execution context

        Returns:
            dict[str, Any]: Return values
        """

        #
        return_values: dict[str, Any] = {}
        #
        for return_var in instruction.return_variables:
            #
            return_values[return_var] = context.get_variable(return_var)

        #
        return return_values

    #
    def _execute_condition(self, instruction: lc.FlowControlCondition, context: ExecutionContext) -> None:
        """
        Execute conditional statement.

        Args:
            instruction (lc.FlowControlCondition): lc.Condition instruction
            context (ExecutionContext): Execution context
        """

        #
        for condition, sub_function_call in instruction.conditions_fn_call.items():
            #
            if self._evaluate_condition(condition, context):
                #
                self._execute_flow_control_instruction(sub_function_call, context)
                break

    #
    def _evaluate_expression(self, expression: lc.Expression, context: ExecutionContext) -> Any:
        """
        Evaluate an expression to get its value.

        Args:
            expression (lc.Expression): lc.Expression to evaluate
            context (ExecutionContext): Execution context

        Returns:
            Any: Evaluated value
        """

        #
        if isinstance(expression, lc.ExpressionVariable):
            #
            return context.get_variable(expression.var_name)
        #
        elif isinstance(expression, lc.ExpressionConstantNumeric):
            #
            return expression.constant
        #
        elif isinstance(expression, lc.ExpressionConstantString):
            #
            return expression.constant
        #
        elif isinstance(expression, lc.ExpressionConstantList):
            #
            return [self._evaluate_expression(elem, context) for elem in expression.elements]
        #
        elif isinstance(expression, lc.ExpressionConstantRange):
            #
            return list(range(expression.start_value, expression.end_value, expression.step))
        #
        elif isinstance(expression, lc.ExpressionNone):
            #
            return None
        #
        elif isinstance(expression, lc.ExpressionNoDefaultArguments):
            #
            raise ValueError("No default argument provided")
        #
        elif isinstance(expression, lc.ExpressionToEvaluate):
            #
            ### This would require implementing a proper expression evaluator. ###
            ### For now, we'll raise an error. ###
            #
            raise NotImplementedError("lc.ExpressionToEvaluate not implemented")
        #
        else:
            #
            raise NotImplementedError(f"lc.Expression type {type(expression)} not implemented")

    #
    def _evaluate_condition(self, condition: lc.Condition, context: ExecutionContext) -> bool:
        """
        Evaluate a condition to get its boolean value.

        Args:
            condition (lc.Condition): lc.Condition to evaluate
            context (ExecutionContext): Execution context

        Returns:
            bool: Boolean result of condition evaluation
        """

        #
        if isinstance(condition, lc.ConditionBinary):
            #
            left = self._evaluate_expression_or_condition(condition.elt1, context)
            right = self._evaluate_expression_or_condition(condition.elt2, context)

            #
            if condition.cond_operator == "and":
                #
                return bool(left) and bool(right)
            #
            elif condition.cond_operator == "or":
                #
                return bool(left) or bool(right)
            #
            elif condition.cond_operator == ">":
                #
                return left > right
            #
            elif condition.cond_operator == "<":
                #
                return left < right
            #
            elif condition.cond_operator == ">=":
                #
                return left >= right
            #
            elif condition.cond_operator == "<=":
                #
                return left <= right
            #
            elif condition.cond_operator == "!=":
                #
                return left != right
            #
            elif condition.cond_operator == "==":
                #
                return left == right
            #
            else:
                #
                raise NotImplementedError(f"Binary condition operator '{condition.cond_operator}' not implemented")

        #
        elif isinstance(condition, lc.ConditionUnary):

            #
            value = self._evaluate_expression_or_condition(condition.elt, context)

            #
            if condition.cond_operator == "not":
                #
                return not bool(value)
            #
            elif condition.cond_operator is None:
                #
                return bool(value)
            #
            else:
                #
                raise NotImplementedError(f"Unary condition operator '{condition.cond_operator}' not implemented")

        #
        elif isinstance(condition, lc.ConditionElse):
            #
            return True

        #
        else:
            #
            raise NotImplementedError(f"lc.Condition type {type(condition)} not implemented")

    #
    def _evaluate_expression_or_condition(
        self,
        expr_or_cond: lc.Expression | lc.Condition,
        context: ExecutionContext
    ) -> Any:

        """
        Evaluate either an expression or a condition.

        Args:
            expr_or_cond (lc.Expression | lc.Condition): lc.Expression or condition to evaluate
            context (ExecutionContext): Execution context

        Returns:
            Any: Evaluated value
        """

        #
        if isinstance(expr_or_cond, lc.Expression):
            #
            return self._evaluate_expression(expr_or_cond, context)
        #
        elif isinstance(expr_or_cond, lc.Condition):  # type: ignore
            #
            return self._evaluate_condition(expr_or_cond, context)
        #
        else:
            #
            raise NotImplementedError(f"Type {type(expr_or_cond)} not supported")

    #
    def _execute_layer_forward(
        self,
        layer_instance: dict[str, Any],
        layer_args: dict[str, Any]
    ) -> NDArray[np.float32]:

        """
        Execute forward pass through a neural network layer.

        Args:
            layer_instance (dict[str, Any]): lc.Layer instance data
            layer_args (dict[str, Any]): lc.Layer arguments

        Returns:
            NDArray[np.float32]: lc.Layer output
        """

        #
        layer_type = layer_instance["type"]
        _layer_params = layer_instance["parameters"]
        layer_weights = layer_instance["weights"]

        #
        ### Get input tensor (assuming first argument is input). ###
        #
        input_tensor = list(layer_args.values())[0]

        #
        ### Execute layer based on type (this is a simplified implementation). ###
        #
        if layer_type == "Linear" or layer_type == "Dense":

            #
            ### Linear layer: y = xW + b ###
            #
            weight = layer_weights.get("weight", np.eye(input_tensor.shape[-1]))
            bias = layer_weights.get("bias", np.zeros(weight.shape[1]))

            #
            return np.dot(input_tensor, weight) + bias

        #
        elif layer_type == "ReLU":
            #
            return np.maximum(0, input_tensor)

        #
        elif layer_type == "Softmax":
            #
            exp_values = np.exp(input_tensor - np.max(input_tensor, axis=-1, keepdims=True))
            #
            return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

        #
        elif layer_type == "lc.LayerNorm":

            #
            mean = np.mean(input_tensor, axis=-1, keepdims=True)
            var = np.var(input_tensor, axis=-1, keepdims=True)
            normalized = (input_tensor - mean) / np.sqrt(var + 1e-5)

            #
            ### Apply learned parameters if available. ###
            #
            gamma = layer_weights.get("weight", np.ones(input_tensor.shape[-1]))
            beta = layer_weights.get("bias", np.zeros(input_tensor.shape[-1]))

            #
            return gamma * normalized + beta

        #
        else:
            #
            raise NotImplementedError(f"lc.Layer type '{layer_type}' not implemented")

    #
    def _call_external_function(self, function_name: str, args: dict[str, Any]) -> Any:
        """
        Call an external function (e.g., numpy, torch functions).

        Args:
            function_name (str): Name of the function to call
            args (dict[str, Any]): Function arguments

        Returns:
            Any: Function result
        """

        #
        ### This is a simplified implementation - you would extend this based on your needs. ###
        #
        if function_name == "np.matmul" or function_name == "matmul":
            #
            return np.matmul(args["a"], args["b"])
        #
        elif function_name == "np.transpose" or function_name == "transpose":
            #
            return np.transpose(list(args.values())[0])
        #
        elif function_name == "np.reshape" or function_name == "reshape":
            #
            tensor = list(args.values())[0]
            shape = list(args.values())[1]
            #
            return np.reshape(tensor, shape)  # type: ignore
        #
        elif function_name == "np.concatenate" or function_name == "concatenate":
            #
            tensors = args["tensors"] if "tensors" in args else list(args.values())[0]
            axis = args.get("axis", 0)
            #
            return np.concatenate(tensors, axis=axis)
        #
        else:
            #
            raise NotImplementedError(f"External function '{function_name}' not implemented")

    #
    def _get_default_value_for_type(self, var_type: lc.VarType) -> Any:
        """
        Get default value for a given variable type.

        Args:
            var_type (lc.VarType): Variable type

        Returns:
            Any: Default value for the type
        """

        #
        if isinstance(var_type, lc.VarTypeTensor):

            #
            ### Create zero tensor with specified dimensions. ###
            #
            dims: list[int] = []
            #
            for dim in var_type.tensor_dims:
                #
                if isinstance(dim, int):
                    #
                    dims.append(dim)
                #
                elif isinstance(dim, str):  # type: ignore
                    #
                    ### Dynamic dimension - use 1 as default. ###
                    #
                    dims.append(1)
                #
                else:
                    #
                    dims.append(1)

            #
            if var_type.tensor_type in ["float32", "float", "np.float32"]:
                #
                return np.zeros(dims, dtype=np.float32)
            #
            elif var_type.tensor_type in ["int32", "int", "np.int32"]:
                #
                return np.zeros(dims, dtype=np.int32)
            #
            elif var_type.tensor_type in ["bool", "np.bool"]:
                #
                return np.zeros(dims, dtype=bool)
            #
            else:
                #
                return np.zeros(dims, dtype=np.float32)

        #
        elif isinstance(var_type, lc.VarTypeContainer):
            #
            if var_type.type_name == "list":
                #
                return []
            #
            elif var_type.type_name == "dict":
                #
                return {}
            #
            else:
                #
                return None

        #
        elif var_type.type_name == "numeric":
            #
            return 0.0
        #
        elif var_type.type_name == "string":
            #
            return ""
        #
        elif var_type.type_name == "bool":
            #
            return False
        #
        else:
            #
            return None

    #
    def _infer_type_from_value(self, value: Any) -> lc.VarType:
        """
        Infer variable type from its value.

        Args:
            value (Any): Value to infer type from

        Returns:
            lc.VarType: Inferred variable type
        """

        #
        if isinstance(value, np.ndarray):
            #
            np_value: NDArray[np.float32] = cast(NDArray[np.float32], value)
            tensor_dims: list[int | str] = list(np_value.shape)
            tensor_type = str(np_value.dtype)
            #
            return lc.VarTypeTensor(tensor_type, tensor_dims)
        #
        elif isinstance(value, (int, float)):
            #
            return lc.VarType("numeric")
        #
        elif isinstance(value, str):
            #
            return lc.VarType("string")
        #
        elif isinstance(value, bool):
            #
            return lc.VarType("bool")
        #
        elif isinstance(value, list):
            #
            ### Infer contained type from first element if available. ###
            #
            if len(cast(list[Any], value)) > 0:
                #
                contained_type = self._infer_type_from_value(value[0])
                #
                return lc.VarTypeContainer("list", contained_type)
            #
            else:
                #
                return lc.VarTypeContainer("list", lc.VarType("unknown"))
        #
        elif isinstance(value, dict):
            #
            return lc.VarType("dict")
        #
        else:
            #
            return lc.VarType("unknown")


#
#########################################################################
########################### UTILITY FUNCTIONS ############################
#########################################################################
#


#
class ModelInterpreterUtils:
    """
    Utility functions for model interpretation and debugging.
    """

    #
    @staticmethod
    def print_execution_trace(
        interpreter: LanguageModel_ForwardInterpreter,
        inputs: dict[str, NDArray[np.float32]],
        verbose: bool = True
    ) -> dict[str, NDArray[np.float32]]:

        """
        Execute forward pass with detailed execution trace for debugging.

        Args:
            interpreter (LanguageModel_ForwardInterpreter): Forward interpreter
            inputs (dict[str, NDArray[np.float32]]): Input tensors
            verbose (bool): Whether to print detailed trace

        Returns:
            dict[str, NDArray[np.float32]]: Output tensors
        """

        #
        if verbose:
            #
            print("=" * 60)
            print("FORWARD PASS EXECUTION TRACE")
            print("=" * 60)
            #
            print(f"\nInputs:")
            #
            for input_name, input_tensor in inputs.items():
                #
                print(f"  - {input_name}: shape={input_tensor.shape}, dtype={input_tensor.dtype}")

        #
        ### Execute forward pass. ###
        #
        outputs = interpreter.forward(inputs)

        #
        if verbose:
            #
            print(f"\nOutputs:")
            #
            for output_name, output_tensor in outputs.items():
                #
                print(f"  - {output_name}: shape={output_tensor.shape}, dtype={output_tensor.dtype}")
            #
            print("=" * 60)

        #
        return outputs

    #
    @staticmethod
    def validate_model_structure(language_model: lc.Language_Model) -> list[str]:
        """
        Validate the structure of a language model and return any issues found.

        Args:
            language_model (lc.Language_Model): Language model to validate

        Returns:
            list[str]: list of validation issues (empty if no issues)
        """

        #
        issues: list[str] = []

        #
        ### Check if main block is specified. ###
        #
        if not language_model.main_block:
            #
            issues.append("No main block specified")
        #
        elif language_model.main_block not in language_model.model_blocks:
            #
            issues.append(f"Main block '{language_model.main_block}' not found in model blocks")

        #
        ### Check each model block. ###
        #
        for block_name, model_block in language_model.model_blocks.items():

            #
            ### Check if forward function exists. ###
            #
            if "forward" not in model_block.block_functions:
                #
                issues.append(f"Block '{block_name}' missing forward function")

            #
            ### Check layer references. ###
            #
            for layer_name, layer in model_block.block_layers.items():

                #
                ### Check if layer parameters reference valid variables. ###
                #
                for param_name, param_expr in layer.layer_parameters_kwargs.items():

                    #
                    if isinstance(param_expr, lc.ExpressionVariable):

                        #
                        if  param_expr.var_name not in model_block.block_parameters and \
                            param_expr.var_name not in model_block.block_variables and \
                            param_expr.var_name not in language_model.global_constants:

                            #
                            issues.append(f"lc.Layer '{layer_name}' parameter '{param_name}' references undefined variable '{param_expr.var_name}'")

            #
            ### Check function flow control instructions. ###
            #
            for func_name, block_function in model_block.block_functions.items():
                #
                for instruction in block_function.function_flow_control:
                    #
                    issues.extend(
                        ModelInterpreterUtils._validate_flow_control_instruction(
                            instruction=instruction,
                            model_block=model_block,
                            language_model=language_model,
                            context=f"{block_name}.{func_name}"
                        )
                    )

        #
        return issues

    #
    @staticmethod
    def _validate_flow_control_instruction(
        instruction: lc.FlowControlInstruction,
        model_block: lc.ModelBlock,
        language_model: lc.Language_Model,
        context: str
    ) -> list[str]:

        """
        Validate a single flow control instruction.

        Args:
            instruction (lc.FlowControlInstruction): Instruction to validate
            model_block (lc.ModelBlock): Model block containing the instruction
            language_model (lc.Language_Model): Full language model
            context (str): Context string for error reporting

        Returns:
            list[str]: list of validation issues
        """

        #
        issues: list[str] = []

        #
        ### Check variable references based on instruction type. ###
        #
        if isinstance(instruction, (lc.FlowControlVariableAssignment,
                                    lc.FlowControlBasicBinaryOperation,
                                    lc.FlowControlBasicUnaryOperation)):

            #
            ### Check if referenced variables exist (simplified check). ###
            #
            if hasattr(instruction, 'input1_var_name'):
                #
                var_name: str = cast(str, instruction.input1_var_name)  # type: ignore
                #
                if not ModelInterpreterUtils._is_valid_variable_reference(var_name, model_block, language_model):
                    #
                    issues.append(f"{context}: Variable '{var_name}' not defined")

            #
            if hasattr(instruction, 'input2_var_name'):
                #
                var_name: str = cast(str, instruction.input2_var_name)  # type: ignore
                #
                if not ModelInterpreterUtils._is_valid_variable_reference(var_name, model_block, language_model):
                    #
                    issues.append(f"{context}: Variable '{var_name}' not defined")

        #
        elif isinstance(instruction, lc.FlowControlLayerPass):
            #
            ### Check if layer exists. ###
            #
            if instruction.layer_name not in model_block.block_layers:
                #
                issues.append(f"{context}: lc.Layer '{instruction.layer_name}' not defined")

        #
        elif isinstance(instruction, lc.FlowControlSubBlockFunctionCall):
            #
            ### Check if function exists. ###
            #
            if instruction.function_called not in model_block.block_functions:
                #
                issues.append(f"{context}: Function '{instruction.function_called}' not defined")

        #
        ### Recursively check nested instructions. ###
        #
        if hasattr(instruction, 'flow_control_instructions'):
            #
            for nested_instruction in instruction.flow_control_instructions:  # type: ignore
                #
                issues.extend(ModelInterpreterUtils._validate_flow_control_instruction(nested_instruction, model_block, language_model, context))  # type: ignore

        #
        return issues

    #
    @staticmethod
    def _is_valid_variable_reference(
        var_name: str,
        model_block: lc.ModelBlock,
        language_model: lc.Language_Model
    ) -> bool:

        """
        Check if a variable reference is valid within the given context.

        Args:
            var_name (str): Variable name to check
            model_block (lc.ModelBlock): Current model block
            language_model (lc.Language_Model): Full language model

        Returns:
            bool: True if variable reference is valid
        """

        #
        return (var_name in model_block.block_parameters or
                var_name in model_block.block_variables or
                var_name in model_block.block_layers or
                var_name in language_model.global_constants)

    #
    @staticmethod
    def generate_model_summary(language_model: lc.Language_Model) -> str:
        """
        Generate a human-readable summary of the language model.

        Args:
            language_model (lc.Language_Model): Language model to summarize

        Returns:
            str: Model summary
        """

        #
        summary: list[str] = []
        summary.append("=" * 60)
        summary.append("LANGUAGE MODEL SUMMARY")
        summary.append("=" * 60)

        #
        ### Global constants. ###
        #
        if language_model.global_constants:
            #
            summary.append(f"\nGlobal Constants ({len(language_model.global_constants)}):")
            #
            for const_name, (const_type, _const_expr) in language_model.global_constants.items():
                #
                summary.append(f"  - {const_name}: {const_type}")

        #
        ### Model blocks. ###
        #
        summary.append(f"\nModel Blocks ({len(language_model.model_blocks)}):")
        #
        for block_name, model_block in language_model.model_blocks.items():
            #
            is_main = " (MAIN)" if block_name == language_model.main_block else ""
            #
            summary.append(f"  - {block_name}{is_main}")

            #
            ### Parameters. ###
            #
            if model_block.block_parameters:
                #
                summary.append(f"    Parameters: {len(model_block.block_parameters)}")

            #
            ### lc.Layers. ###
            #
            if model_block.block_layers:
                #
                summary.append(f"    lc.Layers: {len(model_block.block_layers)}")
                #
                for layer_name, layer in model_block.block_layers.items():
                    #
                    summary.append(f"      - {layer_name}: {layer.layer_type}")

            #
            ### Functions. ###
            #
            summary.append(f"    Functions: {len(model_block.block_functions)}")
            #
            for func_name in model_block.block_functions:
                #
                summary.append(f"      - {func_name}")

        #
        summary.append("=" * 60)

        #
        return "\n".join(summary)


####################################################################
##################### EXAMPLE USAGE ###############################
####################################################################


#
def example_usage():
    """
    Example of how to use the LanguageModel_ForwardInterpreter.
    """

    #
    ### Assume you have a lc.Language_Model object from your model extraction process. ###
    ### language_model = extract_model_from_code(...)  # Your existing extraction logic. ###
    #

    #
    ### For demonstration, create a simple example. ###
    #
    language_model = lc.Language_Model()
    language_model.main_block = "SimpleModel"

    #
    ### Create a simple model block. ###
    #
    simple_block = lc.ModelBlock("SimpleModel")

    #
    ### Add a simple linear layer. ###
    #
    linear_layer = lc.Layer(
        layer_var_name="linear1",
        layer_type="Linear",
        layer_parameters_kwargs={
            "in_features": lc.ExpressionConstantNumeric(10),
            "out_features": lc.ExpressionConstantNumeric(5)
        }
    )

    #
    ### Initialize weights. ###
    #
    linear_layer.layer_weights = {
        "weight": np.random.randn(10, 5).astype(np.float32),
        "bias": np.zeros(5, dtype=np.float32)
    }
    simple_block.block_layers["linear1"] = linear_layer

    #
    ### Create forward function. ###
    #
    forward_func = lc.BlockFunction(
        function_name="forward",
        function_arguments={
            "x": (lc.VarTypeTensor("float32", ["batch_size", 10]), lc.ExpressionNoDefaultArguments())
        },
        model_block=simple_block
    )

    #
    ### Add flow control instructions. ###
    #
    forward_func.function_flow_control = [
        lc.FlowControlLayerPass(
            output_variables=["output"],
            layer_name="linear1",
            layer_arguments={"input": lc.ExpressionVariable("x")}
        ),
        lc.FlowControlReturn(return_variables=["output"])
    ]

    #
    simple_block.block_functions["forward"] = forward_func
    language_model.model_blocks["SimpleModel"] = simple_block

    #
    ### Create interpreter. ###
    #
    interpreter = LanguageModel_ForwardInterpreter(language_model)

    #
    ### Validate model. ###
    #
    validation_issues = ModelInterpreterUtils.validate_model_structure(language_model)

    #
    if validation_issues:

        #
        print("Validation Issues:")
        #
        for issue in validation_issues:
            #
            print(f"  - {issue}")

        #
        return

    #
    ### Print model summary. ###
    #
    print(ModelInterpreterUtils.generate_model_summary(language_model))

    #
    ### Prepare input. ###
    #
    batch_size: int = 3
    input_tensor: NDArray[np.float32] = np.random.randn(batch_size, 10).astype(np.float32)
    #
    inputs: dict[str, NDArray[np.float32]] = {"x": input_tensor}

    #
    ### Execute forward pass with trace. ###
    #
    outputs = ModelInterpreterUtils.print_execution_trace(interpreter, inputs, verbose=True)

    #
    print(f"\nFinal output shape: {outputs['output'].shape}")
    print(f"Output sample: {outputs['output'][0]}")


#
if __name__ == "__main__":
    #
    example_usage()

