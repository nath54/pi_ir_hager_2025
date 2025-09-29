#
### Import Modules. ###
#
from tkinter import N
from typing import Any, Iterator, Optional, cast
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
    def set_variable(
        self,
        var_name: str,
        var_type: lc.VarType,
        value: Any
    ) -> None:

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
    def forward(
        self,
        inputs: dict[str, NDArray[np.float32]],
        **kwargs: dict[str, Any]
    ) -> dict[str, NDArray[np.float32]]:

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
    def _initialize_model_block(
        self,
        model_block: lc.ModelBlock,
        context: ExecutionContext
    ) -> None:

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
        ### If this layer is a BlockModuleList, also initialize its individual layers. ###
        #
        if "BlockModuleList" in layer.layer_type:

                #
                ### This is a BlockModuleList, find the corresponding ModelBlock and initialize its individual layers. ###
                #
                ### e.g., "BlockModuleList_DeepArcNet_0". ###
                #
                block_name: str = layer.layer_type

                #
                if block_name in self.language_model.model_blocks:

                    #
                    block_module: lc.ModelBlock = self.language_model.model_blocks[block_name]

                    #
                    sub_layer_name: str
                    sub_layer: lc.Layer
                    #
                    for sub_layer_name, sub_layer in block_module.block_layers.items():

                        #
                        sub_layer_obj: dict[str, Any] = self._create_layer_instance(sub_layer, context)

                        #
                        context.set_variable(sub_layer_name, lc.VarType("lc.Layer"), sub_layer_obj)

    #
    def _create_layer_instance(
        self,
        layer: lc.Layer,
        context: ExecutionContext
    ) -> Any:

        """
        Create a layer instance with initialized weights.

        Args:
            layer (lc.Layer): lc.Layer definition
            context (ExecutionContext): Execution context

        Returns:
            Any: lc.Layer instance data or callable module
        """

        #
        ### If this is a custom module (like ConvEncode), create a callable wrapper ###
        #
        if layer.layer_type in self.language_model.model_blocks:

            #
            ### This is a custom module, create a callable wrapper. ###
            #
            class ModuleWrapper:

                #
                def __init__(self,
                    layer_type: str,
                    interpreter: "LanguageModel_ForwardInterpreter",
                    context: ExecutionContext
                ) -> None:

                    self.layer_type: str = layer_type
                    self.interpreter: "LanguageModel_ForwardInterpreter" = interpreter
                    self.context: ExecutionContext = context

                #
                def __call__(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:

                    #
                    ### Get the model block for this layer type. ###
                    #
                    model_block = self.interpreter.language_model.model_blocks[self.layer_type]

                    #
                    ### Execute the forward function. ###
                    #
                    forward_function = model_block.block_functions["forward"]

                    #
                    ### Create a new context for this module call. ###
                    #
                    module_context = self.context.copy()

                    #
                    param_name: str = ""

                    #
                    ### Handle both positional and keyword arguments. ###
                    #
                    if args:

                        #
                        ### Get the parameter names from the function definition. ###
                        #
                        param_names: list[str] = list(forward_function.function_arguments.keys())

                        #
                        print(f"DEBUG | Module {self.layer_type} forward function parameters: {param_names}")

                        #
                        ### Skip 'self' parameter if it exists and use the next parameter. ###
                        #
                        if len(param_names) > 1 and param_names[0] == 'self':

                            #
                            ### Use the second parameter (after 'self'). ###
                            #
                            param_name = param_names[1]

                            #
                            print(f"DEBUG | Using parameter name: {param_name} (skipped 'self')")

                        #
                        elif len(param_names) > 0 and param_names[0] != 'self':

                            #
                            ### Use the first parameter if it's not 'self'. ###
                            #
                            param_name = param_names[0]

                            #
                            print(f"DEBUG | Using parameter name: {param_name}")

                        #
                        else:

                            #
                            ### Fallback to "x" if no suitable parameter found. ###
                            #
                            param_name = "x"

                            #
                            print(f"DEBUG | Using fallback parameter name: {param_name}")

                        #
                        ### Ensure we don't pass SelfWrapper objects - unwrap them first. ###
                        #
                        arg_value: Any = args[0]

                        #
                        if hasattr(arg_value, '__class__') and arg_value.__class__.__name__ == 'SelfWrapper':
                            #
                            raise ValueError(f"SelfWrapper object passed to ModuleWrapper: {arg_value}. This should not happen.")

                        #
                        module_context.set_variable(param_name, lc.VarType("Tensor"), arg_value)

                    #
                    elif kwargs:

                        #
                        ### Handle keyword arguments. ###
                        #
                        print(f"DEBUG | Module {self.layer_type} called with keyword arguments: {list(kwargs.keys())}")

                        #
                        ### Get the parameter names from the function definition. ###
                        #
                        param_names: list[str] = list(forward_function.function_arguments.keys())

                        #
                        print(f"DEBUG | Module {self.layer_type} forward function parameters: {param_names}")

                        #
                        ### Skip 'self' parameter if it exists. ###
                        #
                        if len(param_names) > 1 and param_names[0] == 'self':

                            #
                            ### Remove 'self' parameter. ###
                            #
                            param_names = param_names[1:]

                        #
                        ### Use the first non-self parameter name, or 'x' as fallback. ###
                        #
                        if param_names:
                            #
                            param_name = param_names[0]
                        #
                        else:
                            #
                            param_name = "x"

                        #
                        ### Get the argument value (assuming single input). ###
                        #
                        arg_value = list(kwargs.values())[0]

                        #
                        if hasattr(arg_value, '__class__') and arg_value.__class__.__name__ == 'SelfWrapper':
                            #
                            raise ValueError(f"SelfWrapper object passed to ModuleWrapper: {arg_value}. This should not happen.")

                        #
                        module_context.set_variable(param_name, lc.VarType("Tensor"), arg_value)

                    #
                    else:
                        #
                        ### No arguments provided - this shouldn't happen. ###
                        #
                        raise ValueError(f"No arguments provided to ModuleWrapper for {self.layer_type}")

                    #
                    ### Initialize the module block to ensure all necessary variables are available. ###
                    #
                    self.interpreter._initialize_model_block(model_block, module_context)

                    #
                    ### Execute the forward function. ###
                    #
                    result_dict = self.interpreter._execute_block_function(forward_function, module_context, {})

                    #
                    ### If there's only one return value, return it directly instead of the dictionary. ###
                    #
                    print(f"DEBUG | ModuleWrapper result_dict: {result_dict}")
                    print(f"DEBUG | ModuleWrapper result_dict length: {len(result_dict)}")

                    #
                    ### Only one value, we return it directly, no dictionary. ###
                    #
                    if len(result_dict) == 1:

                        #
                        result_value: Any = list(result_dict.values())[0]
                        #
                        print(f"DEBUG | ModuleWrapper single result type: {type(result_value)}")
                        print(f"DEBUG | ModuleWrapper single result: {result_value}")
                        #
                        return result_value

                    #
                    ### There are multiple return value, so we return all of them with a dictionnary. ###
                    #
                    else:

                        #
                        return result_dict

                #
                def __getitem__(self, index: str) -> Any:

                    #
                    ### Get the individual layer instance from the context. ###
                    #
                    if self.context.has_variable(str(index)):
                        #
                        return self.context.get_variable(str(index))

                    #
                    else:
                        #
                        raise IndexError(f"Layer {index} not found in context")

            #
            return ModuleWrapper(layer.layer_type, self, context)

        #
        else:

            #
            ### This is a standard layer, create dictionary representation. ###
            #
            layer_instance: dict[str, Any] = {
                "type": layer.layer_type,
                "parameters": {},
                "weights": layer.layer_weights.copy()
            }

            #
            return layer_instance

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
        ### Add 'self' to the context (represents the current model block instance) ###
        ### Create a wrapper object that provides access to layer instances. ###
        #
        class SelfWrapper:

            #
            def __init__(
                self,
                model_block: lc.ModelBlock,
                context: ExecutionContext
            ) -> None:

                #
                self.model_block: lc.ModelBlock = model_block
                self.context: ExecutionContext = context

            #
            def __getattr__(self, name: str) -> Any:

                #
                ### First try to get from the execution context (layer instances). ###
                #
                if self.context.has_variable(name):
                    #
                    return self.context.get_variable(name)

                #
                ### Then try to get from the model block. ###
                #
                elif hasattr(self.model_block, name):

                    #
                    attr: Any = getattr(self.model_block, name)

                    #
                    ### If it's a BlockModuleList, make it iterable by returning the layer instances. ###
                    #
                    if hasattr(attr, 'block_layers') and hasattr(attr, 'block_name') and 'BlockModuleList' in attr.block_name:

                        #
                        ### Return a list of the actual layer instances from the execution context. ###
                        #
                        layer_instances: list[lc.Layer] = []

                        #
                        for layer_name in sorted(attr.block_layers.keys()):

                            #
                            if self.context.has_variable(layer_name):

                                #
                                layer_instances.append(self.context.get_variable(layer_name))

                        #
                        return layer_instances

                    #
                    return attr

                #
                else:

                    #
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

            #
            def __iter__(self) -> Iterator[Any]:
                #
                ### Make SelfWrapper iterable by delegating to the model_block. ###
                #
                return iter(self.model_block)

            #
            def __deepcopy__(self, memo: Any) -> "SelfWrapper":
                #
                ### Create a new SelfWrapper with deep copies of the model_block and context. ###
                #
                new_model_block: lc.ModelBlock = copy.deepcopy(self.model_block, memo)
                new_context: ExecutionContext = copy.deepcopy(self.context, memo)

                #
                return SelfWrapper(new_model_block, new_context)

        #
        local_context.set_variable("self", lc.VarType("ModelBlock"), SelfWrapper(block_function.model_block, local_context))

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
    def _execute_flow_control_instruction(
        self, instruction: lc.FlowControlInstruction,
        context: ExecutionContext
    ) -> Any:

        """
        Execute a single flow control instruction.

        Args:
            instruction (lc.FlowControlInstruction): Instruction to execute
            context (ExecutionContext): Execution context

        Returns:
            Any: Result of the instruction execution
        """

        #
        print(f"\033[44mDEBUG | execute instruction = {instruction}\033[m")

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
    def _execute_variable_init(
        self,
        instruction: lc.FlowControlVariableInit,
        context: ExecutionContext
    ) -> None:

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
    def _execute_variable_assignment(
        self,
        instruction: lc.FlowControlVariableAssignment,
        context: ExecutionContext
    ) -> None:

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
    def _execute_binary_operation(
        self,
        instruction: lc.FlowControlBasicBinaryOperation,
        context: ExecutionContext
    ) -> None:

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
        elif instruction.operation == "@" or instruction.operation == "matmult":
            #
            ### Matrix multiplication (@ operator). ###
            #
            result = np.matmul(input1, input2)

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
    def _execute_unary_operation(
        self,
        instruction: lc.FlowControlBasicUnaryOperation,
        context: ExecutionContext
    ) -> None:

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
    def _execute_for_loop(
        self,
        instruction: lc.FlowControlForLoop,
        context: ExecutionContext
    ) -> None:

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
            ### Set iterator variable(s). ###
            # Handle tuple unpacking for cases like (var_1, var_2, ..., var_i)
            #
            if instruction.iterable_var_name.startswith("(") and instruction.iterable_var_name.endswith(")"):

                #
                ### Tuple unpacking case: "(var_1, var_2, ..., var_i)". ###
                #

                #
                ### Remove parentheses of the global tuple: "(var_1, var_2, ..., var_i)" -> "var_1, var_2, ..., var_i" ###
                #
                var_names_str: str = instruction.iterable_var_name[1:-1]

                #
                ### Get separately in a list all the variable names of the tuple. ###
                #
                var_names: list[str] = [name.strip() for name in var_names_str.split(",")]

                #
                if len(var_names) == len(item):

                    #
                    ### Unpack the tuple and set each variable. ###
                    #
                    for var_name, var_value in zip(var_names, item):

                        #
                        var_type = self._infer_type_from_value(var_value)
                        #
                        loop_context.set_variable(var_name, var_type, var_value)

                #
                else:
                    #
                    raise ValueError(f"Tuple unpacking mismatch: expected {len(var_names)} variables, got {len(item)}")

            #
            else:

                #
                ### Single variable case. ###
                #
                iterator_type = self._infer_type_from_value(item)
                #
                loop_context.set_variable(instruction.iterable_var_name, iterator_type, item)

            #
            ### Execute loop body. ###
            #
            for var_name in loop_context.variables:

                #
                var_value = loop_context.get_variable(var_name)

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
    def _execute_while_loop(
        self,
        instruction: lc.FlowControlWhileLoop,
        context: ExecutionContext
    ) -> None:

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
    def _execute_function_call(
        self,
        instruction: lc.FlowControlFunctionCall,
        context: ExecutionContext
    ) -> None:

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
            arg_value = self._evaluate_expression(arg_expr, context)

            #
            ### Keep SelfWrapper objects as-is for method call detection. ###
            ### We'll handle unwrapping in the module call logic. ###
            #
            if hasattr(arg_value, '__class__') and arg_value.__class__.__name__ == 'SelfWrapper':

                #
                print(f"DEBUG | Preserving SelfWrapper for argument {arg_name} for method call detection")

                #
                ### Don't unwrap here - let the module call logic handle it. ###
                #

            #
            args[arg_name] = arg_value

        #
        ### Check if the function is actually a PyTorch module instance ###
        #
        if context.has_variable(instruction.function_called):

            #
            ### This is a module instance, call it directly. ###
            #
            module = context.get_variable(instruction.function_called)

            #
            ### Check if it's a callable PyTorch module. ###
            #
            if callable(module):

                #
                ### If this is a ModuleWrapper, we need to pass the current context. ###
                #
                if hasattr(module, '__class__') and module.__class__.__name__ == 'ModuleWrapper':
                    #
                    ### Update the module's context to the current context before calling. ###
                    module.context = context

                #
                ### Call the module with the appropriate argument. ###
                #
                if args:

                    #
                    args_list = list(args.values())

                    #
                    ### Check if this is a method call with 'self' as first argument. ###
                    #
                    if (len(args_list) > 1 and
                        hasattr(args_list[0], '__class__') and
                        args_list[0].__class__.__name__ == 'SelfWrapper'):

                        #
                        ### This is a method call like posembd(self, time_tensor) ###
                        ### Use the second argument as the actual input ###
                        #
                        actual_arg = args_list[1]
                        #
                        result: Any = module(actual_arg)

                    #
                    else:

                        #
                        ### Regular function call. ###
                        #
                        first_arg = args_list[0]

                        #
                        ### If first_arg is a SelfWrapper, we need to unwrap it to get the actual value. ###
                        #
                        if hasattr(first_arg, '__class__') and first_arg.__class__.__name__ == 'SelfWrapper':
                            #
                            ### This shouldn't happen, but let's handle it gracefully. ###
                            #
                            raise ValueError(f"SelfWrapper object passed to module: {first_arg}. This indicates a bug in variable assignment.")

                        #
                        result: Any = module(first_arg)

                #
                else:
                    #
                    result: Any = module()

            #
            elif isinstance(module, dict) and 'type' in module:

                #
                ### This is a standard layer (dictionary representation) that should be executed as a layer pass. ###
                #
                print(f"DEBUG | Executing standard layer: {instruction.function_called}, type: {module['type']}")

                #
                ### Handle method call with 'self' as first argument. ###
                #
                if args:

                    #
                    args_list = list(args.values())

                    #
                    if (len(args_list) > 1 and
                        hasattr(args_list[0], '__class__') and
                        args_list[0].__class__.__name__ == 'SelfWrapper'):

                        #
                        ### This is a method call like linearembd(self, x_conv) ###
                        ### Use the second argument as the actual input ###
                        #
                        actual_input = args_list[1]

                    #
                    else:

                        #
                        actual_input = args_list[0]


                    #
                    ### Prepare arguments - Standard input name for layers. ###
                    #
                    layer_args = {"x": actual_input}

                    #
                    ### Execute the layer. ###
                    #
                    result = self._execute_layer_forward(module, layer_args)

                #
                else:

                    #
                    raise ValueError(f"Standard layer {instruction.function_called} called without arguments")

            #
            else:

                #
                ### Fall back to external function call. ###
                #
                result: Any = self._call_external_function(instruction.function_called, args, context)

        #
        else:

            #
            ### Call the function (this would need to be extended based on available functions). ###
            #
            result: Any = self._call_external_function(instruction.function_called, args, context)

        #
        ### Handle multiple outputs. ###
        #
        if len(instruction.output_variables) == 1:

            #
            result_type = self._infer_type_from_value(result)
            #
            context.set_variable(instruction.output_variables[0], result_type, result)

        #
        else:

            #
            for i, output_var in enumerate(instruction.output_variables):

                #
                output_value: Any = result[i] if isinstance(result, (list, tuple)) else result
                #
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
    def _execute_layer_pass(
        self,
        instruction: lc.FlowControlLayerPass,
        context: ExecutionContext
    ) -> None:

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

        #
        ### Check if this is a ModuleWrapper (custom module) or a standard layer dictionary. ###
        #
        if hasattr(layer_instance, '__class__') and layer_instance.__class__.__name__ == 'ModuleWrapper':
            #
            ### This is a custom module wrapped in ModuleWrapper, call it directly. ###
            #
            result = layer_instance(**layer_args)

        #
        else:

            #
            ### This is a standard layer dictionary, use _execute_layer_forward. ###
            #
            result = self._execute_layer_forward(layer_instance, layer_args)

        #
        ### Handle multiple outputs. ###
        #
        if len(instruction.output_variables) == 1:

            #
            result_type = self._infer_type_from_value(result)
            #
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
    def _execute_return(
        self,
        instruction: lc.FlowControlReturn,
        context: ExecutionContext
    ) -> dict[str, Any]:

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
    def _execute_condition(
        self,
        instruction: lc.FlowControlCondition,
        context: ExecutionContext
    ) -> None:

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
    def _evaluate_expression(
        self,
        expression: lc.Expression,
        context: ExecutionContext
    ) -> Any:

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
            var_name: str = expression.var_name
            var_value: Any = context.get_variable(var_name)

            #
            return var_value

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
        elif isinstance(expression, lc.ExpressionTuple):
            #
            return tuple(self._evaluate_expression(elem, context) for elem in expression.elements)

        #
        elif isinstance(expression, lc.ExpressionList):
            #
            return [self._evaluate_expression(elem, context) for elem in expression.elements]

        #
        elif isinstance(expression, lc.ExpressionDict):
            #
            return {self._evaluate_expression(k, context): self._evaluate_expression(v, context) for k, v in expression.elements.items()}

        #
        elif isinstance(expression, lc.ExpressionSet):
            #
            return {self._evaluate_expression(elem, context) for elem in expression.elements}

        #
        elif isinstance(expression, lc.ExpressionIndexAccess):

            #
            ### Handle array/tensor indexing like tensor[0] or tensor[:, :, 0] ###
            #
            variable = self._evaluate_expression(expression.variable, context)
            index = self._evaluate_expression(expression.index, context)

            #
            ### Handle different index types ###
            #
            if isinstance(index, lc.ExpressionSlice1D):

                #
                ### Handle 1D slicing ###
                #
                start = self._evaluate_expression(index.start, context) if index.start else None
                end = self._evaluate_expression(index.end, context) if index.end else None
                step = self._evaluate_expression(index.step, context) if index.step else None

                #
                ### Create slice object ###
                #
                slice_obj = slice(start, end, step)
                return variable[slice_obj]

            #
            elif isinstance(index, lc.ExpressionSliceND):

                #
                ### TOFIX: Currently np is not well referenced here, so reimport it. ###
                #
                import numpy as np

                #
                ### Handle multi-dimensional slicing ###
                #
                slices = []

                #
                for slice_1d in index.slices:

                    #
                    start = self._evaluate_expression(slice_1d.start, context) if slice_1d.start else None
                    end = self._evaluate_expression(slice_1d.end, context) if slice_1d.end else None
                    step = self._evaluate_expression(slice_1d.step, context) if slice_1d.step else None

                    #
                    ### Check if this represents a "None" slice (newaxis) ###
                    ### Look for patterns like [None:] which should be np.newaxis ###
                    #
                    slice_str = str(slice_1d)
                    #
                    if 'None' in slice_str and slice_str.startswith('[None'):
                        #
                        slices.append(np.newaxis)
                    #
                    else:
                        #
                        slices.append(slice(start, end, step))

                #
                result = variable[tuple(slices)]

                #
                return result

            #
            else:

                #
                ### Handle simple indexing ###
                #
                try:
                    #
                    return variable[index]

                #
                except IndexError as e:

                    #
                    ### Add debug information for indexing errors. ###
                    #
                    print(f"DEBUG | IndexError: variable shape: {getattr(variable, 'shape', 'no shape')}, index: {index}, index type: {type(index)}")
                    #
                    if hasattr(variable, 'ndim'):
                        #
                        print(f"DEBUG | Variable dimensions: {variable.ndim}")

                    #
                    ### Handle dimension mismatch: if we have more indices than dimensions, truncate the indices. ###
                    #
                    if hasattr(variable, 'ndim') and isinstance(index, tuple):

                        #
                        if len(index) > variable.ndim:

                            #
                            print(f"DEBUG | Truncating index from {len(index)} to {variable.ndim} dimensions")
                            #
                            truncated_index = index[:variable.ndim]

                            #
                            print(f"DEBUG | Using truncated index: {truncated_index}")

                            #
                            return variable[truncated_index]

                    #
                    raise e

        #
        elif isinstance(expression, lc.ExpressionAttributeAccess):
            #
            ### Handle attribute access like tensor.shape ###
            #
            variable = self._evaluate_expression(expression.variable, context)
            #
            return getattr(variable, expression.attribute)

        #
        elif isinstance(expression, lc.ExpressionSliceND):

            #
            ### TOFIX. ###
            #
            import numpy as np

            #
            ### Handle multi-dimensional slicing as standalone expression ###
            #
            slices = []

            #
            for slice_1d in expression.slices:

                #
                start = self._evaluate_expression(slice_1d.start, context) if slice_1d.start else None
                end = self._evaluate_expression(slice_1d.end, context) if slice_1d.end else None
                step = self._evaluate_expression(slice_1d.step, context) if slice_1d.step else None

                #
                ### Check if this represents a "None" slice (newaxis). ###
                #
                slice_str = str(slice_1d)

                #
                if 'None' in slice_str and slice_str.startswith('[None'):
                    #
                    slices.append(np.newaxis)
                #
                else:
                    #
                    slices.append(slice(start, end, step))

            #
            return tuple(slices)

        #
        elif isinstance(expression, str):
            #
            ### Handle string expressions (should be variable names) ###
            #
            return context.get_variable(expression)

        #
        else:
            #
            raise NotImplementedError(f"lc.Expression type {type(expression)} not implemented")

    #
    def _evaluate_condition(
        self,
        condition: lc.Condition,
        context: ExecutionContext
    ) -> bool:

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
        ### Import numpy for layer operations. ###
        #
        import numpy as np

        #
        ### Handle SelfWrapper objects - unwrap them to get the actual tensor. ###
        #
        if hasattr(input_tensor, '__class__') and input_tensor.__class__.__name__ == 'SelfWrapper':
            #
            ### If it's a SelfWrapper, we need to get the actual tensor value ###
            ### This shouldn't happen in normal layer execution, but let's handle it ###
            #
            raise ValueError(f"SelfWrapper object passed as input tensor: {input_tensor}. Expected actual tensor.")

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
            ### Handle multi-dimensional input by applying Linear to the last dimension. ###
            #
            original_shape = input_tensor.shape

            #
            ### Reshape to (batch*other_dims, last_dim) for matrix multiplication. ###
            #
            if len(original_shape) > 2:

                #
                ### Flatten all dimensions except the last one. ###
                #
                input_reshaped = input_tensor.reshape(-1, input_tensor.shape[-1])

            #
            else:

                #
                input_reshaped = input_tensor

            #
            ### Check dimension compatibility. ###
            #
            if input_reshaped.shape[-1] != weight.shape[0]:

                #
                print(f"DEBUG | Dimension mismatch: input has {input_reshaped.shape[-1]} features, weight expects {weight.shape[0]}")

                #
                ### Try to adjust the input to match the weight matrix. ###
                #
                if input_reshaped.shape[-1] < weight.shape[0]:

                    #
                    ### Pad with zeros. ###
                    #
                    padding = np.zeros((input_reshaped.shape[0], weight.shape[0] - input_reshaped.shape[-1]))
                    #
                    input_reshaped = np.concatenate([input_reshaped, padding], axis=1)
                    #
                    print(f"DEBUG | Padded input shape: {input_reshaped.shape}")

                #
                else:

                    #
                    ### Truncate. ###
                    #
                    input_reshaped = input_reshaped[:, :weight.shape[0]]
                    #
                    print(f"DEBUG | Truncated input shape: {input_reshaped.shape}")

            #
            ### Perform matrix multiplication. ###
            #
            result_reshaped = np.dot(input_reshaped, weight) + bias

            #
            ### Reshape back to original shape (except the last dimension). ###
            #
            if len(original_shape) > 2:

                #
                new_shape = list(original_shape[:-1]) + [result_reshaped.shape[-1]]
                #
                result = result_reshaped.reshape(new_shape)

            #
            else:
                #
                result = result_reshaped

            #
            return result

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
        elif layer_type == "Conv2d":

            #
            ### 2D Convolution layer ###
            #
            import numpy as np
            from scipy.ndimage import correlate

            #
            ### Get parameters. ###
            #
            kernel_size = _layer_params.get("kernel_size", (3, 3))
            stride = _layer_params.get("stride", 1)
            in_channels = _layer_params.get("in_channels", 1)
            out_channels = _layer_params.get("out_channels", 1)

            #
            ### Get weights and bias. ###
            #
            weight = layer_weights.get("weight", np.random.randn(out_channels, in_channels, *kernel_size))
            bias = layer_weights.get("bias", np.zeros(out_channels))

            #
            ### Input tensor shape: (batch, channels, height, width) or (batch, channels, flattened). ###
            #
            batch_size: int = input_tensor.shape[0]

            #
            ### Check if input is flattened (1D spatial dimensions). ###
            #
            if len(input_tensor.shape) == 3:  # (batch, channels, flattened)

                #
                ### Reshape back to 2D spatial dimensions. ###
                ### Assume the flattened dimension can be reshaped to a square. ###
                #
                flattened_size: int = input_tensor.shape[2]
                #
                spatial_size: int = int(np.sqrt(flattened_size))

                #
                if spatial_size * spatial_size != flattened_size:

                    #
                    ### If not a perfect square, try to find factors. ###
                    #
                    factors: list[int] = []
                    #
                    for i in range(1, int(np.sqrt(flattened_size)) + 1):
                        #
                        if flattened_size % i == 0:
                            #
                            factors.append((i, flattened_size // i))

                    #
                    if factors:

                        #
                        ### Use the largest factor. ###
                        #
                        spatial_size: int = factors[-1][0]
                        #
                        spatial_height: int = spatial_size
                        #
                        spatial_width: int = flattened_size // spatial_size

                    #
                    else:

                        #
                        spatial_height = spatial_width = spatial_size

                #
                else:

                    #
                    spatial_height = spatial_width = spatial_size

                #
                input_tensor = input_tensor.reshape(batch_size, in_channels, spatial_height, spatial_width)

            #
            ### For simplicity, using scipy's correlate function ###
            ### This is a basic implementation - in practice, you'd want a more efficient conv2d ###
            #
            output_list = []
            #
            for b in range(batch_size):

                #
                batch_output = []

                #
                for out_ch in range(out_channels):

                    #
                    ### Initialize output with the correct spatial dimensions ###
                    ### The output spatial size depends on the convolution operation ###
                    ### For now, use the same spatial size as input (assuming no padding) ###
                    #
                    spatial_shape = input_tensor[b, 0].shape
                    #
                    channel_output = np.zeros(spatial_shape)
                    #
                    for in_ch in range(in_channels):

                        #
                        ### Perform convolution (correlation in scipy) ###
                        #
                        input_slice = input_tensor[b, in_ch]
                        #
                        weight_slice = weight[out_ch, in_ch]

                        #
                        ### For 2D convolution, we need to specify axes=(0, 1) for height and width dimensions ###
                        ### But first check if we have enough dimensions ###
                        #
                        if len(input_slice.shape) == 2:
                            #
                            conv_result = correlate(input_slice, weight_slice, mode='constant', axes=(0, 1))

                        #
                        elif len(input_slice.shape) == 1:
                            #
                            conv_result = correlate(input_slice, weight_slice, mode='constant', axes=(0,))

                        #
                        else:
                            #
                            ### Fallback: let scipy determine axes automatically. ###
                            #
                            conv_result = correlate(input_slice, weight_slice, mode='constant')

                        #
                        channel_output += conv_result

                    #
                    ### Add bias. ###
                    #
                    channel_output += bias[out_ch]

                    #
                    ### Apply stride by subsampling. ###
                    #
                    if stride > 1:
                        #
                        channel_output = channel_output[::stride, ::stride]

                    #
                    batch_output.append(channel_output)

                #
                output_list.append(np.stack(batch_output, axis=0))

            #
            result = np.stack(output_list, axis=0)

            #
            return result

        #
        elif layer_type == "lc.LayerNorm" or layer_type == "LayerNorm":

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
    def _call_external_function(
        self,
        function_name: str,
        args: dict[str, Any],
        context: ExecutionContext
    ) -> Any:

        """
        Call an external function (e.g., numpy, torch functions).

        Args:
            function_name (str): Name of the function to call
            args (dict[str, Any]): Function arguments

        Returns:
            Any: Function result
        """

        #
        ### Local import for numpy functions. ###
        #
        import numpy as np

        #
        ### Get the first argument (usually the tensor/array) ###
        #
        first_arg = list(args.values())[0] if args else None

        #
        ### Handle torch. prefix functions ###
        #
        if function_name.startswith("torch."):
            #
            ### Remove "torch." prefix ###
            #
            function_name = function_name[6:]
        #
        elif function_name.startswith("F."):
            #
            ### Remove "F." prefix ###
            #
            function_name = function_name[2:]
        #
        elif function_name.startswith("np."):
            #
            ### Remove "np." prefix ###
            #
            function_name = function_name[3:]

        #
        ### Tensor/Array property functions ###
        #

        #
        if function_name == "size":
            #
            ### torch.size() or tensor.size() ###
            #
            if hasattr(first_arg, 'shape'):
                #
                return first_arg.shape
            #
            else:
                #
                return np.array(first_arg).shape

        #
        elif function_name == "shape":

            #
            ### tensor.shape ###
            #
            if hasattr(first_arg, 'shape'):
                #
                return first_arg.shape
            #
            else:
                #
                return np.array(first_arg).shape

        #
        elif function_name == "device":
            #
            ### tensor.device ###
            #
            if hasattr(first_arg, 'device'):
                #
                return str(first_arg.device)
            #
            else:
                #
                ### Default for numpy arrays. ###
                #
                return "cpu"

        #
        elif function_name == "dtype":
            #
            ### tensor.dtype ###
            #
            if hasattr(first_arg, 'dtype'):
                #
                return first_arg.dtype
            #
            else:
                #
                return np.array(first_arg).dtype

        #
        ### Type conversion functions ###
        #
        elif function_name == "float":
            #
            ### float(tensor) or float(value) ###
            #
            if hasattr(first_arg, 'item'):
                #
                return float(first_arg.item())
            #
            else:
                #
                return float(first_arg)

        #
        elif function_name == "int":
            #
            ### int(tensor) or int(value) ###
            #
            if hasattr(first_arg, 'item'):
                #
                return int(first_arg.item())
            #
            else:
                #
                return int(first_arg)

        #
        elif function_name == "str":
            #
            ### str(tensor) or str(value) ###
            #
            return str(first_arg)

        #
        elif function_name == "append":
            #
            ### List append function ###
            #
            if len(args) >= 2:
                #
                list_obj = args.get("0")
                #
                item = args.get("1")
                #
                if hasattr(list_obj, 'append'):
                    #
                    list_obj.append(item)
                    #
                    return list_obj
                #
                else:
                    #
                    ### If it's not a list, convert to list first. ###
                    #
                    if not isinstance(list_obj, list):
                        #
                        list_obj = list(list_obj)
                    #
                    list_obj.append(item)
                    #
                    return list_obj
            #
            else:
                #
                raise ValueError("append() requires at least 2 arguments")

        #
        elif function_name == "split":

            #
            ### Tensor split function ###
            #
            if len(args) >= 2:

                #
                tensor = args.get("0")
                split_size = args.get("1")
                dim = args.get("dim", 0)

                #
                if hasattr(tensor, 'split'):
                    #
                    ### PyTorch tensor ###
                    #
                    result = tensor.split(split_size, dim=dim)

                    #
                    return result

                #
                else:

                    #
                    ### NumPy array - implement split manually ###
                    #
                    import numpy as np

                    #
                    if isinstance(tensor, np.ndarray):
                        #
                        result = np.split(tensor, split_size, axis=dim)

                        #
                        return result

                    #
                    else:
                        #
                        ### Convert to numpy and split ###
                        #
                        np_tensor = np.array(tensor)
                        #
                        result = np.split(np_tensor, split_size, axis=dim)
                        #
                        return result

            #
            else:

                #
                raise ValueError("split() requires at least 2 arguments")

        #
        elif function_name == "enumerate":
            #
            ### Enumerate function ###
            #
            if len(args) >= 1:

                #
                ### Get the first argument (the iterable). ###
                #
                iterable = list(args.values())[0]
                #
                result = list(enumerate(iterable))
                #
                return result
            #
            else:
                #
                raise ValueError("enumerate() requires at least 1 argument")

        #
        ### Mathematical functions ###
        #
        elif function_name == "sin":
            #
            ### torch.sin() or np.sin(). ###
            #
            if hasattr(first_arg, 'sin'):
                #
                return first_arg.sin()
            #
            else:
                #
                return np.sin(first_arg)

        #
        elif function_name == "cos":
            #
            ### torch.cos() or np.cos() ###
            #
            if hasattr(first_arg, 'cos'):
                #
                return first_arg.cos()
            #
            else:
                #
                return np.cos(first_arg)

        #
        elif function_name == "exp":
            #
            ### torch.exp() or np.exp() ###
            #
            if hasattr(first_arg, 'exp'):
                #
                return first_arg.exp()
            #
            else:
                #
                return np.exp(first_arg)

        #
        elif function_name == "log":
            #
            ### torch.log() or np.log() ###
            #
            if hasattr(first_arg, 'log'):
                #
                return first_arg.log()
            #
            else:
                #
                return np.log(first_arg)

        #
        elif function_name == "sqrt":
            #
            ### torch.sqrt() or np.sqrt() ###
            #
            if hasattr(first_arg, 'sqrt'):
                #
                return first_arg.sqrt()
            #
            else:
                #
                return np.sqrt(first_arg)

        #
        elif function_name == "abs":
            #
            ### torch.abs() or np.abs() ###
            #
            if hasattr(first_arg, 'abs'):
                #
                return first_arg.abs()
            #
            else:
                #
                return np.abs(first_arg)

        #
        ### Array manipulation functions ###
        #
        elif function_name == "arange":
            #
            ### torch.arange() or np.arange() ###
            #
            import numpy as np
            #
            if len(args) == 1:
                #
                ### arange(end) ###
                #
                end = list(args.values())[0]
                #
                return np.arange(end)

            #
            elif len(args) >= 2:
                #
                ### Filter out keyword arguments like 'device', 'dtype', etc. ###
                ### Keep only positional arguments (numeric keys) ###
                #
                numeric_args = []
                #
                for key, value in args.items():
                    #
                    if key.isdigit():
                        #
                        numeric_args.append(value)

                #
                if len(numeric_args) == 2:
                    #
                    ### arange(start, end) ###
                    #
                    start, end = numeric_args[:2]
                    #
                    return np.arange(start, end)

                #
                elif len(numeric_args) == 3:
                    #
                    ### arange(start, end, step) ###
                    #
                    start, end, step = numeric_args[:3]
                    #
                    return np.arange(start, end, step)

                #
                elif len(numeric_args) == 1:
                    #
                    ### arange(end) with keyword args ###
                    #
                    end = numeric_args[0]
                    #
                    return np.arange(end)

                #
                else:
                    #
                    ### Default case - use first argument as end ###
                    #
                    end = list(args.values())[0]
                    #
                    return np.arange(end)

        #
        elif function_name == "cat" or function_name == "concatenate" or function_name == "torch.cat":

            #
            ### torch.cat() or np.concatenate() ###
            #
            tensors = args.get("tensors", list(args.values())[0])
            #
            axis = args.get("axis", args.get("dim", 0))

            #
            ### Handle tuple input by converting to list ###
            #
            if isinstance(tensors, tuple):
                #
                tensors = list(tensors)

            #
            if isinstance(tensors, list):
                #
                import numpy as np
                #
                result = np.concatenate(tensors, axis=axis)
                #
                return result

            #
            else:
                #
                return tensors

        #
        elif function_name == "unsqueeze":
            #
            ### torch.unsqueeze() or np.expand_dims() ###
            #
            tensor = list(args.values())[0]
            #
            dim = list(args.values())[1] if len(args) > 1 else 0

            #
            if hasattr(tensor, 'unsqueeze'):
                #
                result = tensor.unsqueeze(dim)
            #
            else:
                #
                result = np.expand_dims(tensor, axis=dim)

            #
            return result

        #
        elif function_name == "squeeze":
            #
            ### torch.squeeze() or np.squeeze() ###
            #
            tensor = list(args.values())[0]
            #
            dim = args.get("dim", None)
            #
            if hasattr(tensor, 'squeeze'):
                #
                if dim is not None:
                    #
                    return tensor.squeeze(dim)
                #
                else:
                    #
                    return tensor.squeeze()
            #
            else:
                #
                if dim is not None:
                    #
                    return np.squeeze(tensor, axis=dim)
                #
                else:
                    #
                    return np.squeeze(tensor)

        #
        elif function_name == "view":
            #
            ### torch.view() or np.reshape() ###
            #
            tensor = list(args.values())[0]
            shape = list(args.values())[1:]
            #
            import numpy as np
            #
            return np.reshape(tensor, shape)

        #
        elif function_name == "expand":
            #
            ### torch.expand() - approximate with np.broadcast_to() ###
            ### Handle -1 values which mean "keep existing dimension size" ###
            #
            tensor = list(args.values())[0]
            shape = list(args.values())[1:]

            #
            ### Replace -1 with the corresponding dimension size from the original tensor ###
            #
            final_shape = []
            #
            for i, dim_size in enumerate(shape):
                #
                if dim_size == -1:
                    #
                    if i < len(tensor.shape):
                        #
                        final_shape.append(tensor.shape[i])
                    #
                    else:
                        #
                        ### Default to 1 for new dimensions. ###
                        #
                        final_shape.append(1)

                #
                else:
                    #
                    final_shape.append(dim_size)

            #
            return np.broadcast_to(tensor, final_shape)

        #
        elif function_name == "transpose":
            #
            ### torch.transpose() or np.transpose() ###
            #
            tensor = list(args.values())[0]
            #
            if len(args) > 1:

                #
                dim0, dim1 = list(args.values())[1:3]

                #
                ### Convert negative indices to positive indices ###
                #
                if dim0 < 0:
                    #
                    dim0 = len(tensor.shape) + dim0
                #
                if dim1 < 0:
                    #
                    dim1 = len(tensor.shape) + dim1

                #
                ### Check if dimensions are valid ###
                #
                if dim0 >= len(tensor.shape) or dim1 >= len(tensor.shape) or dim0 < 0 or dim1 < 0:
                    #
                    ### Use valid dimensions ###
                    #
                    dim0 = max(0, min(dim0, len(tensor.shape) - 1))
                    dim1 = max(0, min(dim1, len(tensor.shape) - 1))

                #
                if hasattr(tensor, 'transpose'):

                    #
                    try:

                        #
                        ### For numpy arrays, transpose needs a full permutation of all axes ###
                        ### Create a permutation that swaps dim0 and dim1 ###
                        #
                        axes = list(range(len(tensor.shape)))
                        #
                        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
                        #
                        result = tensor.transpose(tuple(axes))
                        #
                        return result

                    #
                    except Exception as e:
                        #
                        ### Fall back to np.transpose ###
                        #
                        axes = list(range(len(tensor.shape)))
                        #
                        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
                        #
                        return np.transpose(tensor, tuple(axes))

                #
                else:

                    #
                    axes = list(range(len(tensor.shape)))
                    #
                    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
                    #
                    return np.transpose(tensor, tuple(axes))

            #
            else:
                #
                return np.transpose(tensor)

        #
        ### Linear algebra functions ###
        #
        elif function_name == "matmul" or function_name == "mm":
            # torch.matmul() or np.matmul()
            a = args.get("a", list(args.values())[0])
            b = args.get("b", list(args.values())[1])
            return np.matmul(a, b)
        #
        elif function_name == "bmm":
            # torch.bmm() - batch matrix multiplication
            a = args.get("a", list(args.values())[0])
            b = args.get("b", list(args.values())[1])
            return np.matmul(a, b)

        #
        ### Activation functions ###
        #
        elif function_name == "softmax":
            # torch.softmax() or F.softmax()
            tensor = list(args.values())[0]
            dim = args.get("dim", -1)
            if hasattr(tensor, 'softmax'):
                return tensor.softmax(dim=dim)
            else:
                # Manual softmax implementation
                exp_tensor = np.exp(tensor - np.max(tensor, axis=dim, keepdims=True))
                return exp_tensor / np.sum(exp_tensor, axis=dim, keepdims=True)
        #
        elif function_name == "relu":
            # torch.relu() or F.relu()
            tensor = list(args.values())[0]
            if hasattr(tensor, 'relu'):
                return tensor.relu()
            else:
                return np.maximum(0, tensor)
        #
        elif function_name == "sigmoid":
            # torch.sigmoid() or F.sigmoid()
            tensor = list(args.values())[0]
            if hasattr(tensor, 'sigmoid'):
                return tensor.sigmoid()
            else:
                return 1 / (1 + np.exp(-tensor))

        #
        ### Statistical functions ###
        #
        elif function_name == "mean":
            # torch.mean() or np.mean()
            tensor = list(args.values())[0]
            dim = args.get("dim", None)
            if hasattr(tensor, 'mean'):
                if dim is not None:
                    return tensor.mean(dim=dim)
                else:
                    return tensor.mean()
            else:
                if dim is not None:
                    return np.mean(tensor, axis=dim)
                else:
                    return np.mean(tensor)
        #
        elif function_name == "sum":
            # torch.sum() or np.sum()
            tensor = list(args.values())[0]
            dim = args.get("dim", None)
            if hasattr(tensor, 'sum'):
                if dim is not None:
                    return tensor.sum(dim=dim)
                else:
                    return tensor.sum()
            else:
                if dim is not None:
                    return np.sum(tensor, axis=dim)
                else:
                    return np.sum(tensor)
        #
        elif function_name == "max":
            # torch.max() or np.max()
            tensor = list(args.values())[0]
            dim = args.get("dim", None)
            if hasattr(tensor, 'max'):
                if dim is not None:
                    return tensor.max(dim=dim)
                else:
                    return tensor.max()
            else:
                if dim is not None:
                    return np.max(tensor, axis=dim)
                else:
                    return np.max(tensor)
        #
        elif function_name == "min":
            # torch.min() or np.min()
            tensor = list(args.values())[0]
            dim = args.get("dim", None)
            if hasattr(tensor, 'min'):
                if dim is not None:
                    return tensor.min(dim=dim)
                else:
                    return tensor.min()
            else:
                if dim is not None:
                    return np.min(tensor, axis=dim)
                else:
                    return np.min(tensor)

        #
        ### Shape manipulation functions ###
        #
        elif function_name == "reshape":
            # torch.reshape() or np.reshape()
            tensor = list(args.values())[0]
            shape = list(args.values())[1]
            if hasattr(tensor, 'reshape'):
                return tensor.reshape(shape)
            else:
                return np.reshape(tensor, shape)
        #
        elif function_name == "flatten":
            # torch.flatten() or np.flatten()
            tensor = list(args.values())[0]
            start_dim = args.get("start_dim", 0)
            end_dim = args.get("end_dim", -1)
            if hasattr(tensor, 'flatten'):
                return tensor.flatten(start_dim=start_dim, end_dim=end_dim)
            else:
                return np.flatten(tensor)

        #
        ### Indexing and slicing functions ###
        #
        elif function_name == "index_select":
            # torch.index_select()
            tensor = list(args.values())[0]
            dim = args.get("dim", 0)
            index = args.get("index", list(args.values())[1])
            if hasattr(tensor, 'index_select'):
                return tensor.index_select(dim=dim, index=index)
            else:
                return np.take(tensor, index, axis=dim)
        #
        elif function_name == "gather":
            # torch.gather()
            tensor = list(args.values())[0]
            dim = args.get("dim", 0)
            index = args.get("index", list(args.values())[1])
            if hasattr(tensor, 'gather'):
                return tensor.gather(dim=dim, index=index)
            else:
                # Manual gather implementation
                return np.take_along_axis(tensor, index, axis=dim)

        #
        ### Utility functions ###
        #
        elif function_name == "ones":
            # torch.ones() or np.ones()
            shape = list(args.values())[0]
            dtype = args.get("dtype", np.float32)
            return np.ones(shape, dtype=dtype)
        #
        elif function_name == "zeros":
            # torch.zeros() or np.zeros()
            shape = list(args.values())[0]
            dtype = args.get("dtype", np.float32)
            return np.zeros(shape, dtype=dtype)
        #
        elif function_name == "randn":
            # torch.randn() or np.random.randn()
            shape = list(args.values())[0]
            return np.random.randn(*shape).astype(np.float32)
        #
        elif function_name == "rand":
            # torch.rand() or np.random.rand()
            shape = list(args.values())[0]
            return np.random.rand(*shape).astype(np.float32)

        #
        ### Fallback for unimplemented functions ###
        #
        else:
            #
            print(f"INFORMATIONS | function_name = `{function_name}` | args = {args}")
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
                ### Track variables defined in this function ###
                #
                defined_variables = set()

                #
                ### Add function arguments to defined variables ###
                #
                defined_variables.update(block_function.function_arguments.keys())

                for instruction in block_function.function_flow_control:
                    #
                    issues.extend(
                        ModelInterpreterUtils._validate_flow_control_instruction(
                            instruction=instruction,
                            model_block=model_block,
                            language_model=language_model,
                            context=f"{block_name}.{func_name}",
                            defined_variables=defined_variables
                        )
                    )

                    ### Update defined variables based on instruction ###
                    if isinstance(instruction, lc.FlowControlVariableAssignment):
                        defined_variables.add(instruction.var_name)
                    elif isinstance(instruction, lc.FlowControlBasicBinaryOperation):
                        defined_variables.add(instruction.output_var_name)
                    elif isinstance(instruction, lc.FlowControlBasicUnaryOperation):
                        defined_variables.add(instruction.output_var_name)
                    elif isinstance(instruction, lc.FlowControlFunctionCall):
                        defined_variables.update(instruction.output_variables)
                    elif isinstance(instruction, lc.FlowControlSubBlockFunctionCall):
                        defined_variables.update(instruction.output_variables)
                    elif isinstance(instruction, lc.FlowControlLayerPass):
                        defined_variables.update(instruction.output_variables)

        #
        return issues

    #
    @staticmethod
    def _validate_flow_control_instruction(
        instruction: lc.FlowControlInstruction,
        model_block: lc.ModelBlock,
        language_model: lc.Language_Model,
        context: str,
        defined_variables: set[str] = None
    ) -> list[str]:

        """
        Validate a single flow control instruction.

        Args:
            instruction (lc.FlowControlInstruction): Instruction to validate
            model_block (lc.ModelBlock): Model block containing the instruction
            language_model (lc.Language_Model): Full language model
            context (str): Context string for error reporting
            defined_variables (set[str], optional): Set of variables defined before this instruction

        Returns:
            list[str]: list of validation issues
        """

        #
        issues: list[str] = []

        #
        if defined_variables is None:
            defined_variables = set()

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
                if (var_name not in defined_variables and
                    not ModelInterpreterUtils._is_valid_variable_reference(var_name, model_block, language_model)):
                    #
                    issues.append(f"{context}: Variable '{var_name}' not defined")

            #
            if hasattr(instruction, 'input2_var_name'):
                #
                var_name: str = cast(str, instruction.input2_var_name)  # type: ignore
                #
                if (var_name not in defined_variables and
                    not ModelInterpreterUtils._is_valid_variable_reference(var_name, model_block, language_model)):
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
        ### Check if it's a standard variable in the model structure ###
        #
        if (var_name in model_block.block_parameters or
                var_name in model_block.block_variables or
                var_name in model_block.block_layers or
            var_name in language_model.global_constants):
            return True

        #
        ### Check if it's a temporary variable created in flow control instructions ###
        #
        for function in model_block.block_functions.values():
            for instruction in function.function_flow_control:
                if isinstance(instruction, lc.FlowControlVariableAssignment):
                    if instruction.var_name == var_name:
                        return True
                elif isinstance(instruction, lc.FlowControlBasicBinaryOperation):
                    if instruction.output_var_name == var_name:
                        return True
                elif isinstance(instruction, lc.FlowControlBasicUnaryOperation):
                    if instruction.output_var_name == var_name:
                        return True
                elif isinstance(instruction, lc.FlowControlFunctionCall):
                    if var_name in instruction.output_variables:
                        return True
                elif isinstance(instruction, lc.FlowControlSubBlockFunctionCall):
                    if var_name in instruction.output_variables:
                        return True
                elif isinstance(instruction, lc.FlowControlLayerPass):
                    if var_name in instruction.output_variables:
                        return True

        #
        ### Check if it's a function argument ###
        #
        for function in model_block.block_functions.values():
            if var_name in function.function_arguments:
                return True

        #
        return False

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
    print(f"DEBUG | outputs = {outputs}")

    #
    if "output" in outputs:
        #
        print(f"\nFinal output shape: {outputs['output'].shape}")
        print(f"Output sample: {outputs['output'][0]}")


#
if __name__ == "__main__":
    #
    example_usage()

