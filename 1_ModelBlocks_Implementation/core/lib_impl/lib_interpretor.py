#
### Import Modules. ###
#
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
################### NEW HIERARCHICAL CONTEXT SYSTEM #####################
#########################################################################
#


class Symbol:
    """
    Represents a symbol in the symbol table.
    Stores variable name, type, and value.
    """

    def __init__(self, name: str, var_type: lc.VarType, value: Any) -> None:
        """
        Initialize a symbol.

        Args:
            name: Symbol name
            var_type: Variable type
            value: Variable value
        """
        self.name: str = name
        self.var_type: lc.VarType = var_type
        self.value: Any = value

    def __repr__(self) -> str:
        return f"Symbol({self.name}, {self.var_type}, {type(self.value).__name__})"


class Scope:
    """
    Represents a single scope level in the hierarchical scope system.
    Each scope maintains its own symbol table and can have a parent scope.
    """

    def __init__(self, name: str, parent: Optional["Scope"] = None) -> None:
        """
        Initialize a new scope.

        Args:
            name: Name of the scope (e.g., "global", "function_forward", "loop_0")
            parent: Parent scope for hierarchical lookups
        """
        self.name: str = name
        self.parent: Optional["Scope"] = parent
        self.symbols: dict[str, Symbol] = {}
        self.scope_level: int = 0 if parent is None else parent.scope_level + 1

    def define(self, name: str, var_type: lc.VarType, value: Any) -> None:
        """
        Define a new symbol in this scope.

        Args:
            name: Symbol name
            var_type: Variable type
            value: Variable value
        """
        self.symbols[name] = Symbol(name, var_type, value)

    def lookup_local(self, name: str) -> Optional[Symbol]:
        """
        Look up a symbol in the current scope only (no parent lookup).

        Args:
            name: Symbol name

        Returns:
            Symbol if found in current scope, None otherwise
        """
        return self.symbols.get(name)

    def lookup(self, name: str) -> Optional[Symbol]:
        """
        Look up a symbol in the current scope and parent scopes.

        Args:
            name: Symbol name

        Returns:
            Symbol if found, None otherwise
        """
        # First check current scope
        symbol = self.symbols.get(name)
        if symbol is not None:
            return symbol

        # Then check parent scope recursively
        if self.parent is not None:
            return self.parent.lookup(name)

        return None

    def update(self, name: str, value: Any) -> bool:
        """
        Update a symbol's value. Searches current and parent scopes.

        Args:
            name: Symbol name
            value: New value

        Returns:
            True if symbol was found and updated, False otherwise
        """
        # First check current scope
        if name in self.symbols:
            self.symbols[name].value = value
            return True

        # Then check parent scope recursively
        if self.parent is not None:
            return self.parent.update(name, value)

        return False

    def has_symbol(self, name: str) -> bool:
        """
        Check if symbol exists in current or parent scopes.

        Args:
            name: Symbol name

        Returns:
            True if symbol exists, False otherwise
        """
        return self.lookup(name) is not None

    def has_symbol_local(self, name: str) -> bool:
        """
        Check if symbol exists in current scope only.

        Args:
            name: Symbol name

        Returns:
            True if symbol exists in current scope, False otherwise
        """
        return name in self.symbols

    def get_all_symbols(self) -> dict[str, Symbol]:
        """
        Get all symbols accessible from this scope (including parent scopes).

        Returns:
            Dictionary of all accessible symbols
        """
        # Start with parent symbols
        if self.parent is not None:
            all_symbols = self.parent.get_all_symbols().copy()
        else:
            all_symbols = {}

        # Override with current scope symbols
        all_symbols.update(self.symbols)

        return all_symbols

    def __repr__(self) -> str:
        return f"Scope(name={self.name}, level={self.scope_level}, symbols={len(self.symbols)})"


class ExecutionContext:
    """
    Manages the execution context with hierarchical scoping.
    Uses a scope stack to maintain proper variable resolution.
    """

    def __init__(self, scope_name: str = "global", parent_scope: Optional[Scope] = None) -> None:
        """
        Initialize execution context with a new scope.

        Args:
            scope_name: Name for the initial scope
            parent_scope: Optional parent scope for hierarchical lookups
        """
        self.current_scope: Scope = Scope(scope_name, parent_scope)
        self.scope_stack: list[Scope] = [self.current_scope]

    def enter_scope(self, scope_name: str) -> "ExecutionContext":
        """
        Create a new nested scope (for functions, loops, blocks).

        Args:
            scope_name: Name for the new scope

        Returns:
            New ExecutionContext with the nested scope
        """
        new_context = ExecutionContext.__new__(ExecutionContext)
        new_context.current_scope = Scope(scope_name, self.current_scope)
        new_context.scope_stack = self.scope_stack + [new_context.current_scope]
        return new_context

    def exit_scope(self) -> Optional["ExecutionContext"]:
        """
        Exit the current scope and return to parent scope.

        Returns:
            ExecutionContext with parent scope, or None if at global scope
        """
        if len(self.scope_stack) <= 1:
            return None

        parent_context = ExecutionContext.__new__(ExecutionContext)
        parent_context.scope_stack = self.scope_stack[:-1]
        parent_context.current_scope = parent_context.scope_stack[-1]
        return parent_context

    def set_variable(self, var_name: str, var_type: lc.VarType, value: Any) -> None:
        """
        Set a variable in the current scope.
        If the variable exists in a parent scope, it updates that instead.

        Args:
            var_name: Name of the variable
            var_type: Type of the variable
            value: Value to assign
        """
        # Check if variable exists in current or parent scopes
        existing_symbol = self.current_scope.lookup(var_name)

        if existing_symbol is not None:
            # Update existing variable
            self.current_scope.update(var_name, value)
        else:
            # Define new variable in current scope
            self.current_scope.define(var_name, var_type, value)

    def set_variable_local(self, var_name: str, var_type: lc.VarType, value: Any) -> None:
        """
        Set a variable in the current scope only (forces local definition).

        Args:
            var_name: Name of the variable
            var_type: Type of the variable
            value: Value to assign
        """
        self.current_scope.define(var_name, var_type, value)

    def get_variable(self, var_name: str) -> Any:
        """
        Get a variable from the current or parent scopes.

        Args:
            var_name: Name of the variable

        Returns:
            Value of the variable

        Raises:
            KeyError: If variable doesn't exist
        """
        symbol = self.current_scope.lookup(var_name)
        if symbol is None:
            raise KeyError(f"Variable '{var_name}' not found in execution context (scope: {self.current_scope.name})")
        return symbol.value

    def has_variable(self, var_name: str) -> bool:
        """
        Check if variable exists in current or parent scopes.

        Args:
            var_name: Name of the variable

        Returns:
            True if variable exists
        """
        return self.current_scope.has_symbol(var_name)

    def has_variable_local(self, var_name: str) -> bool:
        """
        Check if variable exists in current scope only.

        Args:
            var_name: Name of the variable

        Returns:
            True if variable exists in current scope
        """
        return self.current_scope.has_symbol_local(var_name)

    def get_variable_type(self, var_name: str) -> lc.VarType:
        """
        Get the type of a variable from the current or parent scopes.

        Args:
            var_name: Name of the variable

        Returns:
            Type of the variable

        Raises:
            KeyError: If variable doesn't exist
        """
        symbol = self.current_scope.lookup(var_name)
        if symbol is None:
            raise KeyError(f"Variable '{var_name}' not found in execution context (scope: {self.current_scope.name})")
        return symbol.var_type

    @property
    def variables(self) -> dict[str, Any]:
        """
        Get all accessible variables as a dictionary (for backward compatibility).

        Returns:
            Dictionary mapping variable names to values
        """
        all_symbols = self.current_scope.get_all_symbols()
        return {name: symbol.value for name, symbol in all_symbols.items()}

    @property
    def variable_types(self) -> dict[str, lc.VarType]:
        """
        Get all accessible variable types as a dictionary (for backward compatibility).

        Returns:
            Dictionary mapping variable names to types
        """
        all_symbols = self.current_scope.get_all_symbols()
        return {name: symbol.var_type for name, symbol in all_symbols.items()}

    def copy(self) -> "ExecutionContext":
        """
        Create a shallow copy of the execution context for backward compatibility.
        This maintains the same parent scope reference but creates a new scope.

        DEPRECATED: Use enter_scope() instead for proper hierarchical scoping.

        Returns:
            New ExecutionContext with the same variables in a new scope
        """
        # Create a new scope at the same level (sibling scope)
        new_context = ExecutionContext.__new__(ExecutionContext)
        new_context.current_scope = Scope(f"{self.current_scope.name}_copy", self.current_scope.parent)
        new_context.scope_stack = self.scope_stack[:-1] + [new_context.current_scope]

        # Copy all symbols from current scope to new scope
        for name, symbol in self.current_scope.symbols.items():
            new_context.current_scope.define(name, symbol.var_type, symbol.value)

        return new_context

    def __repr__(self) -> str:
        return f"ExecutionContext(scope={self.current_scope.name}, level={self.current_scope.scope_level}, vars={len(self.current_scope.symbols)})"


#
#########################################################################
########################## FORWARD INTERPRETER ###########################
#########################################################################
#


#
class LanguageModel_ForwardInterpreter:
    """
    Forward pass interpreter for Language Models defined using lib_classes.py structures.
    Executes the model by interpreting the flow control instructions and performing computations.
    """

    #
    def __init__(self, language_model: lc.Language_Model) -> None:

        """
        Initialize the interpreter.

        Args:
            language_model (lc.Language_Model): Language model to interpret
        """

        #
        self.language_model = language_model
        #
        self.global_context: ExecutionContext = ExecutionContext("global")

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
        Execute forward pass of the model.

        Args:
            inputs (dict[str, NDArray[np.float32]]): Input data
            **kwargs: Additional keyword arguments

        Returns:
            dict[str, NDArray[np.float32]]: Model outputs
        """

        #
        ### Initialize global constants. ###
        #
        self._initialize_global_constants()

        #
        ### Create execution context for this forward pass. ###
        #
        context = self.global_context.enter_scope("forward_pass")

        #
        ### Get main model block. ###
        #
        if not self.language_model.model_blocks:
            #
            raise ValueError("No model blocks defined")

        #
        ### Use the main_block attribute if it's set, otherwise use the first block. ###
        #
        if self.language_model.main_block and self.language_model.main_block in self.language_model.model_blocks:
            main_block = self.language_model.model_blocks[self.language_model.main_block]
        else:
            main_block = list(self.language_model.model_blocks.values())[0]

        #
        ### Set input variables in the context. ###
        #
        for input_name, input_value in inputs.items():

            #
            var_type = lc.VarType("Tensor")
            #
            context.set_variable(input_name, var_type, input_value)

        #
        ### Handle additional keyword arguments as inputs. ###
        #
        for kwarg_name, kwarg_value in kwargs.items():

            #
            ### Infer type from value. ###
            #
            if isinstance(kwarg_value, (int, float)):
                #
                var_type = lc.VarType("Scalar")

            #
            elif isinstance(kwarg_value, np.ndarray):
                #
                var_type = lc.VarType("Tensor")

            #
            else:
                #
                var_type = lc.VarType("Any")

            #
            context.set_variable(kwarg_name, var_type, kwarg_value)

        #
        ### Initialize main model block. ###
        #
        self._initialize_model_block(main_block, context)

        #
        ### Execute forward function of main block. ###
        #
        forward_function = main_block.block_functions["forward"]

        #
        ### Execute the forward function. ###
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
        ### Initialize layers. ###
        #
        for layer_name, layer in model_block.block_layers.items():

            #
            ### Initialize layer weights if they exist. ###
            #
            layer_obj: dict[str, Any] = self._create_layer_instance(layer, context)
            #
            context.set_variable(layer_name, lc.VarType("lc.Layer"), layer_obj)

        #
        ### Initialize sub-layers for BlockModuleList. ###
        #
        for layer_name, layer in model_block.block_layers.items():

            #
            if 'BlockModuleList' in layer.layer_type:

                #
                ### Get the sub-model block. ###
                #
                sub_model_block = self.language_model.model_blocks[layer.layer_type]

                #
                for sub_layer_name, sub_layer in sub_model_block.block_layers.items():

                    #
                    if sub_layer_name not in context.variables:

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
        Create a layer instance (placeholder for actual layer creation).

        Args:
            layer (lc.Layer): Layer definition
            context (ExecutionContext): Execution context

        Returns:
            Any: Layer instance wrapper
        """

        #
        ### If this is a custom module (like ConvEncode), create a callable wrapper ###
        #
        if layer.layer_type in self.language_model.model_blocks:

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
                    ### Create a new scope for this module call. ###
                    #
                    module_context = self.context.enter_scope(f"module_{self.layer_type}")

                    #
                    param_name: str = ""

                    #
                    ### Prepare function arguments dictionary. ###
                    #
                    function_args: dict[str, Any] = {}

                    #
                    ### Handle both positional and keyword arguments. ###
                    #
                    if args:

                        #
                        ### Get the parameter names from the function definition. ###
                        #
                        param_names: list[str] = list(forward_function.function_arguments.keys())

                        #
                        ### Skip 'self' parameter if it exists and use the next parameter. ###
                        #
                        if len(param_names) > 1 and param_names[0] == 'self':

                            #
                            ### Use the second parameter for the input. ###
                            #
                            param_name = param_names[1]

                        #
                        elif len(param_names) > 0:

                            #
                            param_name = param_names[0]

                        #
                        else:

                            #
                            raise ValueError(f"No parameters found in forward function of {self.layer_type}")

                        #
                        ### Handle single positional argument. ###
                        #
                        if len(args) == 1:

                            #
                            arg_value = args[0]

                            #
                            ### Check if the argument is a numpy array. ###
                            #
                            if not isinstance(arg_value, np.ndarray):

                                #
                                ### If the argument is a list of numpy arrays, don't convert to numpy array. ###
                                #
                                if isinstance(arg_value, list) and all(isinstance(x, np.ndarray) for x in arg_value):

                                    #
                                    pass

                                #
                                else:

                                    #
                                    ### Convert to numpy array. ###
                                    #
                                    arg_value = np.array(arg_value, dtype=np.float32)

                            #
                            function_args[param_name] = arg_value

                        #
                        else:

                            #
                            ### Handle multiple positional arguments. ###
                            #
                            for i, arg_value in enumerate(args):

                                #
                                # Map argument index to parameter name
                                if param_names[0] == 'self':
                                    # Skip 'self' parameter, map to remaining parameters
                                    if i + 1 < len(param_names):
                                        param_name = param_names[i + 1]
                                    else:
                                        raise ValueError(f"Too many arguments for function {self.layer_type}.forward")
                                else:
                                    # No 'self' parameter, map directly
                                    if i < len(param_names):
                                        param_name = param_names[i]
                                    else:
                                        raise ValueError(f"Too many arguments for function {self.layer_type}.forward")

                                #
                                ### Check if the argument is a numpy array. ###
                                #
                                if not isinstance(arg_value, np.ndarray):

                                    #
                                    ### Convert to numpy array. ###
                                    #
                                    arg_value = np.array(arg_value, dtype=np.float32)

                                #
                                function_args[param_name] = arg_value

                    #
                    ### Handle keyword arguments. ###
                    #
                    for kwarg_name, kwarg_value in kwargs.items():

                        #
                        if not isinstance(kwarg_value, np.ndarray):

                            #
                            kwarg_value = np.array(kwarg_value, dtype=np.float32)

                        #
                        function_args[kwarg_name] = kwarg_value

                    #
                    ### Initialize the model block with the module context. ###
                    #
                    self.interpreter._initialize_model_block(model_block, module_context)

                    #
                    ### Execute the forward function with the arguments dictionary. ###
                    #
                    result_dict = self.interpreter._execute_block_function(forward_function, module_context, function_args)

                    #
                    ### Return the first output (assuming single output). ###
                    #
                    if result_dict:

                        #
                        return list(result_dict.values())[0]

                    #
                    else:

                        #
                        return None

                #
                def __getitem__(self, index: int) -> Any:

                    """
                    Allow indexing into ModuleList-like structures.
                    """

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

                def __iter__(self):
                    """
                    Make ModuleWrapper iterable like PyTorch ModuleList.
                    """

                    # Get the layers defined in the BlockModuleList block
                    if self.layer_type in self.interpreter.language_model.model_blocks:
                        model_block = self.interpreter.language_model.model_blocks[self.layer_type]
                        # Iterate over the layers in the order they are defined in the block
                        for layer_name in sorted(model_block.block_layers.keys()):
                            if self.context.has_variable(layer_name):
                                yield self.context.get_variable(layer_name)


                def __len__(self):
                    """
                    Return the length of the ModuleList.
                    """
                    # Get the layers defined in the BlockModuleList block
                    if self.layer_type in self.interpreter.language_model.model_blocks:
                        model_block = self.interpreter.language_model.model_blocks[self.layer_type]
                        # Count only the layers that are actually in the context
                        count = 0
                        for layer_name in model_block.block_layers.keys():
                            if self.context.has_variable(layer_name):
                                count += 1
                        return count
                    return 0

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
            ### Evaluate layer parameters. ###
            #
            for param_name, param_expr in layer.layer_parameters_kwargs.items():
                #
                param_value = self._evaluate_expression(param_expr, context)
                #
                ### Apply PyTorch parameter normalization ###
                #
                param_value = self._normalize_pytorch_parameter(layer.layer_type, param_name, param_value)
                #
                layer_instance["parameters"][param_name] = param_value

            #
            return layer_instance

    #
    def _normalize_pytorch_parameter(self, layer_type: str, param_name: str, param_value: Any) -> Any:

        """
        Normalize PyTorch parameter names and values to match the expected format.

        Args:
            layer_type (str): Type of the layer
            param_name (str): Parameter name
            param_value (Any): Parameter value

        Returns:
            Any: Normalized parameter value
        """

        #
        ### Handle different layer types and their parameter naming conventions. ###
        #
        if layer_type in ["Linear", "Conv2d", "BatchNorm2d"]:

            #
            ### Convert PyTorch tensors to numpy arrays. ###
            #
            if hasattr(param_value, "detach"):

                #
                return param_value.detach().cpu().numpy()

        #
        return param_value

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
        ### Create local scope for function execution. ###
        #
        local_context = context.enter_scope(f"function_{block_function.function_name}")

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
        print(f"\033[44m DEBUG | Execute instruction : {instruction}\033[m")

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
        elif isinstance(instruction, lc.FlowControlForEachLoop):
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
            raise NotImplementedError(f"Instruction type not implemented: {type(instruction)}")

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
        ### Evaluate the initialization expression. ###
        #
        value = self._evaluate_expression(instruction.init_expression, context)

        #
        ### Set the variable in the context. ###
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
        ### Evaluate the assignment expression. ###
        #
        value = self._evaluate_expression(instruction.var_value, context)

        #
        ### Get the variable type (use existing type if variable exists). ###
        #
        if context.has_variable(instruction.var_name):
            #
            var_type = context.variable_types[instruction.var_name]

        #
        else:
            #
            ### Infer type from value. ###
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
            instruction (lc.FlowControlBinaryOperation): Binary operation instruction
            context (ExecutionContext): Execution context
        """

        #
        input1 = context.get_variable(instruction.input1_var_name)
        input2 = context.get_variable(instruction.input2_var_name)

        #
        ### Perform the operation based on the operator. ###
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
        elif instruction.operation == "//":
            #
            result = input1 // input2
        #
        elif instruction.operation == "%":
            #
            result = input1 % input2
        #
        elif instruction.operation == "**":
            #
            result = input1 ** input2
        #
        elif instruction.operation == "@":
            #
            ### Matrix multiplication. ###
            #
            result = input1 @ input2
        #
        elif instruction.operation == "==":
            #
            result = input1 == input2
        #
        elif instruction.operation == "!=":
            #
            result = input1 != input2
        #
        elif instruction.operation == "<":
            #
            result = input1 < input2
        #
        elif instruction.operation == ">":
            #
            result = input1 > input2
        #
        elif instruction.operation == "<=":
            #
            result = input1 <= input2
        #
        elif instruction.operation == ">=":
            #
            result = input1 >= input2
        #
        else:
            #
            raise NotImplementedError(f"Binary operator not implemented: {instruction.operation}")

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
            instruction (lc.FlowControlUnaryOperation): Unary operation instruction
            context (ExecutionContext): Execution context
        """

        #
        input_value = context.get_variable(instruction.input_var_name)

        #
        ### Perform the operation based on the operator. ###
        #
        if instruction.operation == "-":
            #
            result = -input_value
        #
        elif instruction.operation == "+":
            #
            result = +input_value
        #
        elif instruction.operation == "not":
            #
            result = not input_value
        #
        else:
            #
            raise NotImplementedError(f"Unary operator not implemented: {instruction.operation}")

        #
        ### Infer result type. ###
        #
        result_type = self._infer_type_from_value(result)

        #
        context.set_variable(instruction.output_var_name, result_type, result)

    #
    def _execute_for_loop(
        self,
        instruction: lc.FlowControlForEachLoop,
        context: ExecutionContext
    ) -> None:

        """
        Execute for loop.

        Args:
            instruction (lc.FlowControlForEachLoop): For loop instruction
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
        for loop_iteration, item in enumerate(iterable):

            #
            ### Create loop scope. ###
            #
            loop_context = context.enter_scope(f"loop_{loop_iteration}")

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
                        loop_context.set_variable_local(var_name, var_type, var_value)

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
                loop_context.set_variable_local(instruction.iterable_var_name, iterator_type, item)

            #
            ### Execute loop body. ###
            #
            for loop_instruction in instruction.flow_control_instructions:

                #
                self._execute_flow_control_instruction(loop_instruction, loop_context)

            #
            ### Update parent context with variables that were modified in the loop. ###
            #
            for var_name in loop_context.current_scope.symbols:
                symbol = loop_context.current_scope.symbols[var_name]
                # Only update variables that existed in parent context
                if context.has_variable(var_name):
                    context.set_variable(var_name, symbol.var_type, symbol.value)

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
        iteration_count = 0
        #
        while self._evaluate_condition(instruction.condition, context):

            #
            ### Create loop scope. ###
            #
            loop_context = context.enter_scope(f"while_loop_{iteration_count}")

            #
            ### Execute loop body. ###
            #
            for loop_instruction in instruction.flow_control_instructions:

                #
                self._execute_flow_control_instruction(loop_instruction, loop_context)

            #
            ### Update parent context with variables that were modified in the loop. ###
            #
            for var_name in loop_context.current_scope.symbols:
                symbol = loop_context.current_scope.symbols[var_name]
                # Only update variables that existed in parent context
                if context.has_variable(var_name):
                    context.set_variable(var_name, symbol.var_type, symbol.value)

            #
            iteration_count += 1

    #
    def _execute_function_call(
        self,
        instruction: lc.FlowControlFunctionCall,
        context: ExecutionContext
    ) -> None:

        """
        Execute function call.

        Args:
            instruction (lc.FlowControlFunctionCall): Function call instruction
            context (ExecutionContext): Execution context
        """

        #
        ### Check if function is a module instance (layer). ###
        #
        module = None
        if context.has_variable(instruction.function_called):

            #
            ### Get the module instance. ###
            #
            module = context.get_variable(instruction.function_called)

            #
            ### Check if module is callable. ###
            #
            if callable(module):

                #
                ### If this is a ModuleWrapper, we need to pass the current context. ###
                #

                #
                ### Evaluate function arguments. ###
                #
                args: list[Any] = []

                #
                # Sort arguments by numeric key to maintain order
                sorted_args = sorted(instruction.function_arguments.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)

                #
                for arg_name, arg_expr in sorted_args:

                    #
                    arg_value = self._evaluate_expression(arg_expr, context)

                    #
                    # Skip 'self' argument as it's handled implicitly by the ModuleWrapper
                    if isinstance(arg_expr, lc.ExpressionVariable) and arg_expr.var_name == 'self':
                        continue

                    #
                    args.append(arg_value)

                #
                ### Call the module. ###
                #
                result = module(*args)

                #
                ### Store result if output variables are specified. ###
                #
                if instruction.output_variables:

                    #
                    ### Infer result type. ###
                    #
                    result_type = self._infer_type_from_value(result)

                    #
                    ### Handle single output. ###
                    #
                    if len(instruction.output_variables) == 1:

                        #
                        context.set_variable(instruction.output_variables[0], result_type, result)

                    #
                    ### Handle multiple outputs. ###
                    #
                    else:

                        #
                        if isinstance(result, tuple):

                            #
                            for output_var, output_value in zip(instruction.output_variables, result):

                                #
                                result_type = self._infer_type_from_value(output_value)
                                #
                                context.set_variable(output_var, result_type, output_value)

                        #
                        else:

                            #
                            context.set_variable(instruction.output_variables[0], result_type, result)

            #
            elif isinstance(module, dict) and 'type' in module:

                #
                ### This is a standard layer (dictionary representation) that should be executed as a layer pass. ###
                #
                # Evaluate function arguments
                args: dict[str, Any] = {}
                for arg_name, arg_expr in instruction.function_arguments.items():
                    args[arg_name] = self._evaluate_expression(arg_expr, context)

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
                ### Store result if output variables are specified. ###
                #
                if instruction.output_variables:

                    #
                    ### Infer result type. ###
                    #
                    result_type = self._infer_type_from_value(result)

                    #
                    context.set_variable(instruction.output_variables[0], result_type, result)

        #
        else:

            #
            ### External function call. ###
            #
            # Evaluate arguments for external function call
            args: dict[str, Any] = {}
            for arg_name, arg_expr in instruction.function_arguments.items():
                args[arg_name] = self._evaluate_expression(arg_expr, context)
            
            result = self._call_external_function(instruction.function_called, args, context)

            #
            ### Store result if output variables are specified. ###
            #
            if instruction.output_variables:

                #
                ### Infer result type. ###
                #
                result_type = self._infer_type_from_value(result)

                #
                context.set_variable(instruction.output_variables[0], result_type, result)

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
            if instruction.function_name in block.block_functions:

                #
                current_block = block
                #
                break

        #
        if current_block is None:
            #
            raise ValueError(f"Function '{instruction.function_name}' not found in any model block")

        #
        ### Get the function. ###
        #
        function = current_block.block_functions[instruction.function_name]

        #
        ### Evaluate function arguments. ###
        #
        args: dict[str, Any] = {}

        #
        # Get the function parameter names
        param_names = list(function.function_arguments.keys())
        
        #
        for arg_name, arg_expr in instruction.function_arguments.items():

            #
            # Map numeric argument keys to parameter names
            if arg_name.isdigit():
                arg_index = int(arg_name)
                if arg_index < len(param_names):
                    param_name = param_names[arg_index]
                    args[param_name] = self._evaluate_expression(arg_expr, context)
                else:
                    raise ValueError(f"Argument index {arg_index} out of range for function {instruction.function_name}")
            else:
                # Use the argument name directly if it's not numeric
                args[arg_name] = self._evaluate_expression(arg_expr, context)

        #
        ### Execute the function. ###
        #
        outputs = self._execute_block_function(function, context, args)

        #
        ### Store outputs in the context. ###
        #
        for output_key, output_value in outputs.items():

            #
            ### Find matching output variable. ###
            #
            for output_var in instruction.output_variables:

                #
                result_type = self._infer_type_from_value(output_value)
                #
                context.set_variable(output_var, result_type, outputs[output_key])

    #
    def _execute_layer_pass(
        self,
        instruction: lc.FlowControlLayerPass,
        context: ExecutionContext
    ) -> None:

        """
        Execute layer forward pass.

        Args:
            instruction (lc.FlowControlLayerPass): Layer pass instruction
            context (ExecutionContext): Execution context
        """

        #
        layer_instance = context.get_variable(instruction.layer_name)

        #
        ### Evaluate input arguments. ###
        #
        args: list[Any] = []

        #
        for arg_expr in instruction.layer_arguments:

            #
            arg_value = self._evaluate_expression(arg_expr, context)
            #
            args.append(arg_value)

        #
        ### Call the layer. ###
        #
        # Check if this is a ModuleWrapper (callable) or a dictionary (standard layer)
        if hasattr(layer_instance, '__call__'):
            #
            ### This is a ModuleWrapper (custom block) ###
            #
            result = layer_instance(*args)
        else:
            #
            ### This is a standard PyTorch layer (dictionary) ###
            #
            # Convert args list to dictionary format expected by _execute_layer_forward
            layer_args = {"x": args[0]} if args else {}
            result = self._execute_layer_forward(layer_instance, layer_args)

        #
        ### Store result. ###
        #
        if instruction.output_variables:

            #
            ### Infer result type. ###
            #
            result_type = self._infer_type_from_value(result)

            #
            ### Handle single output. ###
            #
            if len(instruction.output_variables) == 1:

                #
                context.set_variable(instruction.output_variables[0], result_type, result)

            #
            ### Handle multiple outputs. ###
            #
            else:

                #
                if isinstance(result, tuple):

                    #
                    for output_var, output_value in zip(instruction.output_variables, result):

                        #
                        result_type = self._infer_type_from_value(output_value)
                        #
                        context.set_variable(output_var, result_type, output_value)

                #
                else:

                    #
                    context.set_variable(instruction.output_variables[0], result_type, result)

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
        Execute conditional statement (if-elif-else).

        Args:
            instruction (lc.FlowControlCondition): Conditional instruction
            context (ExecutionContext): Execution context
        """

        #
        ### Evaluate condition. ###
        #
        condition_result = self._evaluate_condition(instruction.condition, context)

        #
        if condition_result:

            #
            ### Execute if block. ###
            #
            for if_instruction in instruction.if_flow_control_instructions:

                #
                self._execute_flow_control_instruction(if_instruction, context)

        #
        else:

            #
            ### Execute else block if it exists. ###
            #
            if instruction.else_flow_control_instructions:

                #
                for else_instruction in instruction.else_flow_control_instructions:

                    #
                    self._execute_flow_control_instruction(else_instruction, context)

    #
    def _evaluate_expression(
        self,
        expression: lc.Expression | str | int | float,
        context: ExecutionContext
    ) -> Any:

        """
        Evaluate an expression.

        Args:
            expression (lc.Expression | str | int | float): Expression to evaluate
            context (ExecutionContext): Execution context

        Returns:
            Any: Result of the expression evaluation
        """

        #
        ### Handle variable reference. ###
        #
        if isinstance(expression, str):

            #
            ### Check if it's a variable in the context. ###
            #
            if context.has_variable(expression):

                #
                return context.get_variable(expression)

            #
            ### Otherwise, return as string literal. ###
            #
            return expression

        #
        ### Handle numeric constants. ###
        #
        elif isinstance(expression, (int, float)):

            #
            return expression

        #
        ### Handle boolean constants. ###
        #
        elif isinstance(expression, bool):

            #
            return expression

        #
        ### Handle None. ###
        #
        elif expression is None:

            #
            return None

        #
        ### Handle ExpressionConstantNumeric. ###
        #
        elif isinstance(expression, lc.ExpressionConstantNumeric):

            #
            return expression.constant

        #
        ### Handle ExpressionConstantString. ###
        #
        elif isinstance(expression, lc.ExpressionConstantString):

            #
            return expression.constant

        #
        ### Handle ExpressionConstantBoolean. ###
        #
        elif isinstance(expression, lc.ExpressionConstantBoolean):

            #
            return expression.constant

        #
        ### Handle ExpressionVariable. ###
        #
        elif isinstance(expression, lc.ExpressionVariable):

            #
            return context.get_variable(expression.var_name)

        #
        ### Handle ExpressionBinaryOperation. ###
        #
        elif isinstance(expression, lc.ExpressionBinaryOperation):

            #
            left_value = self._evaluate_expression(expression.left, context)
            right_value = self._evaluate_expression(expression.right, context)

            #
            if expression.operator == "+":
                #
                return left_value + right_value
            #
            elif expression.operator == "-":
                #
                return left_value - right_value
            #
            elif expression.operator == "*":
                #
                return left_value * right_value
            #
            elif expression.operator == "/":
                #
                return left_value / right_value
            #
            elif expression.operator == "//":
                #
                return left_value // right_value
            #
            elif expression.operator == "%":
                #
                return left_value % right_value
            #
            elif expression.operator == "**":
                #
                return left_value ** right_value
            #
            elif expression.operator == "@":
                #
                return left_value @ right_value
            #
            elif expression.operator == "==":
                #
                return left_value == right_value
            #
            elif expression.operator == "!=":
                #
                return left_value != right_value
            #
            elif expression.operator == "<":
                #
                return left_value < right_value
            #
            elif expression.operator == ">":
                #
                return left_value > right_value
            #
            elif expression.operator == "<=":
                #
                return left_value <= right_value
            #
            elif expression.operator == ">=":
                #
                return left_value >= right_value
            #
            else:
                #
                raise NotImplementedError(f"Binary operator not implemented: {expression.operator}")

        #
        ### Handle ExpressionUnaryOperation. ###
        #
        elif isinstance(expression, lc.ExpressionUnaryOperation):

            #
            operand_value = self._evaluate_expression(expression.operand, context)

            #
            if expression.operator == "-":
                #
                return -operand_value
            #
            elif expression.operator == "+":
                #
                return +operand_value
            #
            elif expression.operator == "not":
                #
                return not operand_value
            #
            else:
                #
                raise NotImplementedError(f"Unary operator not implemented: {expression.operator}")

        #
        ### Handle ExpressionFunctionCall. ###
        #
        elif isinstance(expression, lc.ExpressionFunctionCall):

            #
            ### Evaluate function arguments. ###
            #
            args: list[Any] = []

            #
            for arg_expr in expression.function_arguments:

                #
                arg_value = self._evaluate_expression(arg_expr, context)
                #
                args.append(arg_value)

            #
            ### Call external function. ###
            #
            return self._call_external_function(expression.function_name, args, context)

        #
        ### Handle ExpressionSliceND. ###
        #
        elif isinstance(expression, lc.ExpressionSliceND):

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
        ### Handle ExpressionAttributeAccess. ###
        #
        elif isinstance(expression, lc.ExpressionAttributeAccess):
            #
            ### Handle attribute access like tensor.shape ###
            #
            variable = self._evaluate_expression(expression.variable, context)
            #
            return getattr(variable, expression.attribute)

        #
        ### Handle ExpressionIndexAccess. ###
        #
        elif isinstance(expression, lc.ExpressionIndexAccess):

            #
            ### Handle index access like tensor[0]. ###
            #
            variable = self._evaluate_expression(expression.variable, context)
            index = self._evaluate_expression(expression.index, context)

            #
            return variable[index]

        #
        ### Handle ExpressionSlice. ###
        #
        elif isinstance(expression, lc.ExpressionSlice1D):

            #
            ### Handle slice like tensor[start:stop:step]. ###
            #
            variable = self._evaluate_expression(expression.variable, context)

            #
            start = self._evaluate_expression(expression.start, context) if expression.start is not None else None
            stop = self._evaluate_expression(expression.stop, context) if expression.stop is not None else None
            step = self._evaluate_expression(expression.step, context) if expression.step is not None else None

            #
            return variable[start:stop:step]

        #
        ### Handle ExpressionTuple. ###
        #
        elif isinstance(expression, lc.ExpressionTuple):

            #
            ### Evaluate tuple elements. ###
            #
            elements: list[Any] = []

            #
            for element_expr in expression.elements:

                #
                element_value = self._evaluate_expression(element_expr, context)
                #
                elements.append(element_value)

            #
            return tuple(elements)

        #
        ### Handle ExpressionList. ###
        #
        elif isinstance(expression, lc.ExpressionList):

            #
            ### Evaluate list elements. ###
            #
            elements: list[Any] = []

            #
            for element_expr in expression.elements:

                #
                element_value = self._evaluate_expression(element_expr, context)
                #
                elements.append(element_value)

            #
            return elements

        #
        ### Handle ExpressionDict. ###
        #
        elif isinstance(expression, lc.ExpressionDict):

            #
            ### Evaluate dictionary elements. ###
            #
            result: dict[Any, Any] = {}

            #
            for key_expr, value_expr in expression.items.items():

                #
                key = self._evaluate_expression(key_expr, context)
                value = self._evaluate_expression(value_expr, context)
                #
                result[key] = value

            #
            return result

        #
        else:
            #
            raise NotImplementedError(f"Expression type not implemented: {type(expression)}")

    #
    def _evaluate_condition(
        self,
        condition: lc.Condition,
        context: ExecutionContext
    ) -> bool:

        """
        Evaluate a condition.

        Args:
            condition (lc.Condition): Condition to evaluate
            context (ExecutionContext): Execution context

        Returns:
            bool: Result of the condition evaluation
        """

        #
        ### Handle ConditionComparison. ###
        #
        if isinstance(condition, lc.ConditionComparison):

            #
            left_value = self._evaluate_expression(condition.left, context)
            right_value = self._evaluate_expression(condition.right, context)

            #
            if condition.operator == "==":
                #
                return left_value == right_value
            #
            elif condition.operator == "!=":
                #
                return left_value != right_value
            #
            elif condition.operator == "<":
                #
                return left_value < right_value
            #
            elif condition.operator == ">":
                #
                return left_value > right_value
            #
            elif condition.operator == "<=":
                #
                return left_value <= right_value
            #
            elif condition.operator == ">=":
                #
                return left_value >= right_value
            #
            else:
                #
                raise NotImplementedError(f"Comparison operator not implemented: {condition.operator}")

        #
        ### Handle ConditionLogical. ###
        #
        elif isinstance(condition, lc.ConditionLogical):

            #
            left_result = self._evaluate_condition(condition.left, context)
            right_result = self._evaluate_condition(condition.right, context)

            #
            if condition.operator == "and":
                #
                return left_result and right_result
            #
            elif condition.operator == "or":
                #
                return left_result or right_result
            #
            else:
                #
                raise NotImplementedError(f"Logical operator not implemented: {condition.operator}")

        #
        ### Handle ConditionUnary. ###
        #
        elif isinstance(condition, lc.ConditionUnary):

            #
            operand_result = self._evaluate_condition(condition.operand, context)

            #
            if condition.operator == "not":
                #
                return not operand_result
            #
            else:
                #
                raise NotImplementedError(f"Unary condition operator not implemented: {condition.operator}")

        #
        ### Handle ConditionExpression. ###
        #
        elif isinstance(condition, lc.ConditionExpression):

            #
            value = self._evaluate_expression(condition.expression, context)

            #
            return bool(value)

        #
        else:
            #
            raise NotImplementedError(f"Condition type not implemented: {type(condition)}")

    #
    def _evaluate_expression_or_condition(
        self,
        expr: lc.Expression | lc.Condition,
        context: ExecutionContext
    ) -> Any:

        """
        Evaluate an expression or condition.

        Args:
            expr (lc.Expression | lc.Condition): Expression or condition to evaluate
            context (ExecutionContext): Execution context

        Returns:
            Any: Result of the evaluation
        """

        #
        if isinstance(expr, lc.Condition):
            #
            return self._evaluate_condition(expr, context)
        #
        else:
            #
            return self._evaluate_expression(expr, context)

    #
    def _execute_layer_forward(
        self,
        layer_instance: dict[str, Any],
        layer_args: dict[str, Any]
    ) -> Any:

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
        ### Use global numpy import ###
        #

        #
        ### Handle SelfWrapper objects - unwrap them to get the actual tensor. ###
        #
        if hasattr(input_tensor, '__class__') and input_tensor.__class__.__name__ == 'SelfWrapper':
            #
            ### This shouldn't happen in layer forward pass, but handle gracefully. ###
            #
            raise ValueError(f"SelfWrapper object passed to layer forward pass: {input_tensor}. This indicates a bug in layer execution.")

        #
        ### Convert to numpy array if needed. ###
        #
        if not isinstance(input_tensor, np.ndarray):
            #
            input_tensor = np.array(input_tensor)

        #
        ### Execute layer based on type. ###
        #
        if layer_type == "Conv2d":
            #
            ### Convolutional layer. ###
            #
            ### Get parameters. ###
            #
            in_channels = _layer_params.get("in_channels", 1)
            out_channels = _layer_params.get("out_channels", 1)
            kernel_size = _layer_params.get("kernel_size", 3)
            stride = _layer_params.get("stride", 1)
            padding = _layer_params.get("padding", 0)
            dilation = _layer_params.get("dilation", 1)
            groups = _layer_params.get("groups", 1)
            bias = _layer_params.get("bias", True)

            #
            ### Get weights. ###
            #
            weight = layer_weights.get("weight")
            bias_weight = layer_weights.get("bias") if bias else None

            #
            ### Apply convolution. ###
            #
            ### This is a simplified implementation. ###
            ### In practice, you would use proper convolution. ###
            #
            if weight is not None:
                #
                ### For now, just return the input with some modification. ###
                ### This is a placeholder implementation. ###
                #
                result = input_tensor * 0.5  # Simplified operation
            else:
                #
                result = input_tensor

        elif layer_type == "Linear":
            #
            ### Linear layer. ###
            #
            ### Get parameters. ###
            #
            in_features = _layer_params.get("in_features", 1)
            out_features = _layer_params.get("out_features", 1)
            bias = _layer_params.get("bias", True)

            #
            ### Get weights. ###
            #
            weight = layer_weights.get("weight")
            bias_weight = layer_weights.get("bias") if bias else None

            #
            ### Apply linear transformation. ###
            #
            ### This is a simplified implementation. ###
            ### In practice, you would use proper matrix multiplication. ###
            #
            if weight is not None:
                #
                ### For now, just return the input with some modification. ###
                ### This is a placeholder implementation. ###
                #
                result = input_tensor * 0.5  # Simplified operation
            else:
                #
                result = input_tensor

        elif layer_type == "ReLU":
            #
            ### ReLU activation. ###
            #
            result = np.maximum(0, input_tensor)

        elif layer_type == "Sigmoid":
            #
            ### Sigmoid activation. ###
            #
            result = 1 / (1 + np.exp(-input_tensor))

        elif layer_type == "Tanh":
            #
            ### Tanh activation. ###
            #
            result = np.tanh(input_tensor)

        else:
            #
            ### Unknown layer type. ###
            #
            raise NotImplementedError(f"Layer type '{layer_type}' not implemented")

        #
        return result

    #
    def _call_external_function(
        self,
        function_name: str,
        args: dict[str, Any],
        context: ExecutionContext
    ) -> Any:

        """
        Call an external function (e.g., torch.cat, np.concatenate, etc.).

        Args:
            function_name (str): Name of the function
            args (dict[str, Any]): Function arguments
            context (ExecutionContext): Execution context

        Returns:
            Any: Function result
        """

        #
        ### Helper function to convert dict args to positional args ###
        #
        def get_args_list():
            """Convert dictionary arguments to positional arguments list."""
            return list(args.values())
        
        def get_arg(index: int, default=None):
            """Get argument by index from dictionary."""
            args_list = get_args_list()
            return args_list[index] if index < len(args_list) else default

        #
        ### Handle torch functions. ###
        #
        if function_name == "torch.cat" or function_name == "cat":

            #
            ### Concatenate tensors. ###
            #
            if len(args) >= 1:

                #
                tensors = get_arg(0)

                #
                ### Handle dim argument. ###
                #
                dim = get_arg(1) if len(args) > 1 else 0

                #
                ### Use numpy concatenate. ###
                #
                return np.concatenate(tensors, axis=dim)

        #
        elif function_name == "torch.stack" or function_name == "stack":

            #
            ### Stack tensors. ###
            #
            if len(args) >= 1:

                #
                tensors = get_arg(0)

                #
                ### Handle dim argument. ###
                #
                dim = get_arg(1) if len(args) > 1 else 0

                #
                ### Use numpy stack. ###
                #
                return np.stack(tensors, axis=dim)

        #
        elif function_name == "torch.transpose" or function_name == "transpose":

            #
            ### Transpose tensor. ###
            #
            if len(args) >= 3:

                #
                tensor = get_arg(0)
                dim0 = get_arg(1)
                dim1 = get_arg(2)

                #
                ### Use numpy transpose with axes swapping. ###
                #
                axes = list(range(len(tensor.shape)))
                axes[dim0], axes[dim1] = axes[dim1], axes[dim0]

                #
                return np.transpose(tensor, axes)

        #
        elif function_name == "torch.nn.functional.softmax" or function_name == "softmax" or function_name == "F.softmax":

            #
            ### Softmax function. ###
            #
            if len(args) >= 1:

                #
                tensor = get_arg(0)

                #
                ### Handle dim argument. ###
                #
                dim = get_arg(1) if len(args) > 1 else -1

                #
                ### Compute softmax. ###
                #
                exp_tensor = np.exp(tensor - np.max(tensor, axis=dim, keepdims=True))
                #
                return exp_tensor / np.sum(exp_tensor, axis=dim, keepdims=True)

        #
        elif function_name == "torch.nn.functional.relu" or function_name == "relu" or function_name == "F.relu":

            #
            ### ReLU function. ###
            #
            if len(args) >= 1:

                #
                tensor = get_arg(0)

                #
                ### Apply ReLU. ###
                #
                return np.maximum(0, tensor)

        #
        elif function_name == "torch.nn.functional.tanh" or function_name == "tanh" or function_name == "F.tanh":

            #
            ### Tanh function. ###
            #
            if len(args) >= 1:

                #
                tensor = get_arg(0)

                #
                ### Apply tanh. ###
                #
                return np.tanh(tensor)

        #
        elif function_name == "torch.nn.functional.sigmoid" or function_name == "sigmoid" or function_name == "F.sigmoid":

            #
            ### Sigmoid function. ###
            #
            if len(args) >= 1:

                #
                tensor = get_arg(0)

                #
                ### Apply sigmoid. ###
                #
                return 1 / (1 + np.exp(-tensor))

        #
        elif function_name == "torch.nn.functional.dropout" or function_name == "dropout" or function_name == "F.dropout":

            #
            ### Dropout function (inference mode - no dropout). ###
            #
            if len(args) >= 1:

                #
                tensor = get_arg(0)

                #
                ### Return tensor as-is (inference mode). ###
                #
                return tensor

        #
        elif function_name == "torch.zeros" or function_name == "zeros":

            #
            ### Create zero tensor. ###
            #
            if len(args) >= 1:

                #
                shape = get_arg(0) if isinstance(get_arg(0), tuple) else tuple(args)

                #
                return np.zeros(shape, dtype=np.float32)

        #
        elif function_name == "torch.ones" or function_name == "ones":

            #
            ### Create ones tensor. ###
            #
            if len(args) >= 1:

                #
                shape = get_arg(0) if isinstance(get_arg(0), tuple) else tuple(args)

                #
                return np.ones(shape, dtype=np.float32)

        #
        elif function_name == "torch.randn" or function_name == "randn":

            #
            ### Create random tensor. ###
            #
            if len(args) >= 1:

                #
                shape = get_arg(0) if isinstance(get_arg(0), tuple) else tuple(args)

                #
                return np.random.randn(*shape).astype(np.float32)

        #
        elif function_name == "torch.arange" or function_name == "arange":

            #
            ### torch.arange() or np.arange() ###
            #
            #
            if len(args) == 1:
                #
                ### arange(end) ###
                #
                end = get_arg(0)
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
                    start = numeric_args[0]
                    end = numeric_args[1]
                    #
                    return np.arange(start, end, dtype=np.float32)

                #
                elif len(numeric_args) >= 3:
                    #
                    ### arange(start, end, step) ###
                    #
                    start = numeric_args[0]
                    end = numeric_args[1]
                    step = numeric_args[2]
                    #
                    return np.arange(start, end, step, dtype=np.float32)

                #
                else:
                    #
                    ### Fallback to single argument ###
                    #
                    return np.arange(numeric_args[0], dtype=np.float32)

        #
        elif function_name == "torch.unsqueeze" or function_name == "unsqueeze":

            #
            ### Unsqueeze tensor. ###
            #
            if len(args) >= 2:

                #
                tensor = get_arg(0)
                dim = get_arg(1)

                #
                return np.expand_dims(tensor, axis=dim)

        #
        elif function_name == "torch.squeeze" or function_name == "squeeze":

            #
            ### Squeeze tensor. ###
            #
            if len(args) >= 1:

                #
                tensor = get_arg(0)

                #
                if len(args) >= 2:
                    #
                    dim = get_arg(1)
                    #
                    return np.squeeze(tensor, axis=dim)
                #
                else:
                    #
                    return np.squeeze(tensor)

        #
        elif function_name == "torch.reshape" or function_name == "reshape":

            #
            ### Reshape tensor. ###
            #
            if len(args) >= 2:

                #
                tensor = get_arg(0)
                shape = get_arg(1) if isinstance(get_arg(1), tuple) else tuple(args[1:])

                #
                return np.reshape(tensor, shape)

        #
        elif function_name == "expand":
            #
            ### torch.expand() - approximate with np.broadcast_to() ###
            ### Handle -1 values which mean "keep existing dimension size" ###
            #
            tensor = get_arg(0)
            shape = get_args_list()[1:]

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
        elif function_name == "torch.flatten" or function_name == "flatten":

            #
            ### Flatten tensor. ###
            #
            if len(args) >= 1:

                #
                tensor = get_arg(0)

                #
                ### Handle start_dim and end_dim. ###
                #
                start_dim = get_arg(1) if len(args) > 1 else 0
                end_dim = get_arg(2) if len(args) > 2 else -1

                #
                ### Convert negative indices. ###
                #
                if end_dim < 0:
                    #
                    end_dim = len(tensor.shape) + end_dim

                #
                ### Compute new shape. ###
                #
                new_shape = list(tensor.shape[:start_dim])
                #
                new_shape.append(-1)
                #
                new_shape.extend(list(tensor.shape[end_dim+1:]))

                #
                return np.reshape(tensor, new_shape)

        #
        elif function_name == "torch.sum" or function_name == "sum":

            #
            ### Sum tensor. ###
            #
            if len(args) >= 1:

                #
                tensor = get_arg(0)

                #
                if len(args) >= 2:
                    #
                    dim = get_arg(1)
                    #
                    keepdim = get_arg(2) if len(args) > 2 else False
                    #
                    return np.sum(tensor, axis=dim, keepdims=keepdim)
                #
                else:
                    #
                    return np.sum(tensor)

        #
        elif function_name == "torch.mean" or function_name == "mean":

            #
            ### Mean tensor. ###
            #
            if len(args) >= 1:

                #
                tensor = get_arg(0)

                #
                if len(args) >= 2:
                    #
                    dim = get_arg(1)
                    #
                    keepdim = get_arg(2) if len(args) > 2 else False
                    #
                    return np.mean(tensor, axis=dim, keepdims=keepdim)
                #
                else:
                    #
                    return np.mean(tensor)

        #
        elif function_name == "torch.max" or function_name == "max":

            #
            ### Max function - handle both tensor max and value comparison ###
            #
            if len(args) >= 1:

                #
                first_arg = get_arg(0)

                #
                if len(args) >= 2:
                    #
                    second_arg = get_arg(1)
                    
                    # Check if this is a value comparison (both args are scalars) or tensor operation
                    if (isinstance(first_arg, (int, float)) and isinstance(second_arg, (int, float))):
                        #
                        ### Value comparison ###
                        #
                        return max(first_arg, second_arg)
                    #
                    else:
                        #
                        ### Tensor operation with axis ###
                        #
                        dim = second_arg
                        keepdim = get_arg(2) if len(args) > 2 else False
                        #
                        return np.max(first_arg, axis=dim, keepdims=keepdim)
                #
                else:
                    #
                    ### Single tensor max ###
                    #
                    return np.max(first_arg)

        #
        elif function_name == "torch.min" or function_name == "min":

            #
            ### Min function - handle both tensor min and value comparison ###
            #
            if len(args) >= 1:

                #
                first_arg = get_arg(0)

                #
                if len(args) >= 2:
                    #
                    second_arg = get_arg(1)
                    
                    # Check if this is a value comparison (both args are scalars) or tensor operation
                    if (isinstance(first_arg, (int, float)) and isinstance(second_arg, (int, float))):
                        #
                        ### Value comparison ###
                        #
                        return min(first_arg, second_arg)
                    #
                    else:
                        #
                        ### Tensor operation with axis ###
                        #
                        dim = second_arg
                        keepdim = get_arg(2) if len(args) > 2 else False
                        #
                        return np.min(first_arg, axis=dim, keepdims=keepdim)
                #
                else:
                    #
                    ### Single tensor min ###
                    #
                    return np.min(first_arg)

        #
        elif function_name == "len":

            #
            ### Length function. ###
            #
            if len(args) >= 1:

                #
                obj = next(iter(args.values()))

                #
                return len(obj)


        #
        elif function_name == "size":
            #
            ### torch.size() or tensor.size() ###
            #
            tensor = next(iter(args.values()))
            #
            if hasattr(tensor, 'shape'):
                #
                ### Check if a specific dimension is requested ###
                #
                if len(args) > 1:
                    #
                    ### size(dim) - return size of specific dimension ###
                    #
                    dim_args = list(args.values())[1:]
                    #
                    if len(dim_args) == 1:
                        #
                        dim = dim_get_arg(0)
                        #
                        return tensor.shape[dim]
                    #
                #
                ### size() - return entire shape tuple ###
                #
                return tensor.shape
            #
            else:
                #
                result = np.array(tensor).shape
                return result

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
                    if isinstance(tensor, np.ndarray):
                        #
                        ### 1. Calculate the indices for the split ###
                        ### The indices will be [3, 6, 9, ...] ###
                        #
                        indices = np.arange(split_size, tensor.shape[dim], split_size)
                        # 2. Use numpy.split with the calculated indices
                        result = np.split(tensor, indices, axis=dim)
                        #
                        return result
                    #
                    else:
                        #
                        raise ValueError(f"Unsupported tensor type for split: {type(tensor)}")
            #
            else:
                #
                raise ValueError(f"split function requires at least 2 arguments, got {len(args)}")

        #
        elif function_name == "range":

            #
            ### Range function. ###
            #
            if len(args) >= 1:

                #
                start = 0
                stop = get_arg(0)
                step = 1

                #
                if len(args) >= 2:
                    #
                    start = get_arg(0)
                    stop = get_arg(1)

                #
                if len(args) >= 3:
                    #
                    step = get_arg(2)

                #
                return range(start, stop, step)

        #
        elif function_name == "enumerate":

            #
            ### Enumerate function ###
            #
            if len(args) >= 1:

                #
                ### Get the first argument (the iterable). ###
                #
                iterable = list(args)[0]
                #
                result = list(enumerate(iterable))
                #
                return result

            #
            else:
                #
                raise ValueError("enumerate() requires at least 1 argument")

        #
        elif function_name == "zip":

            #
            ### Zip function. ###
            #
            if len(args) >= 1:

                #
                return list(zip(*args))

        #
        elif function_name == "print":

            #
            ### Print function. ###
            #
            print(args)

            #
            return None

        #
        elif function_name == "type":

            #
            ### Type function. ###
            #
            if len(args) >= 1:

                #
                return type(get_arg(0))

        #
        elif function_name == "isinstance":

            #
            ### Isinstance function. ###
            #
            if len(args) >= 2:

                #
                obj = get_arg(0)
                class_or_tuple = get_arg(1)

                #
                return isinstance(obj, class_or_tuple)

        #
        elif function_name == "hasattr":

            #
            ### Hasattr function. ###
            #
            if len(args) >= 2:

                #
                obj = get_arg(0)
                name = get_arg(1)

                #
                return hasattr(obj, name)

        #
        elif function_name == "getattr":

            #
            ### Getattr function. ###
            #
            if len(args) >= 2:

                #
                obj = get_arg(0)
                name = get_arg(1)
                default = get_arg(2) if len(args) > 2 else None

                #
                return getattr(obj, name, default)

        #
        elif function_name == "setattr":

            #
            ### Setattr function. ###
            #
            if len(args) >= 3:

                #
                obj = get_arg(0)
                name = get_arg(1)
                value = get_arg(2)

                #
                setattr(obj, name, value)

                #
                return None

        #
        elif function_name == "sin":

            #
            ### Sin function. ###
            #
            if len(args) >= 1:

                #
                return np.sin(get_arg(0))

        #
        elif function_name == "cos":

            #
            ### Cos function. ###
            #
            if len(args) >= 1:

                #
                return np.cos(get_arg(0))

        #
        elif function_name == "torch.sin" or function_name == "torch.cos":

            #
            ### Torch sin/cos function. ###
            #
            if len(args) >= 1:

                #
                if "sin" in function_name:
                    #
                    return np.sin(get_arg(0))
                #
                else:
                    #
                    return np.cos(get_arg(0))

        #
        elif function_name == "torch.exp" or function_name == "exp":

            #
            ### Exp function. ###
            #
            if len(args) >= 1:

                #
                return np.exp(get_arg(0))

        #
        elif function_name == "log":

            #
            ### Log function. ###
            #
            if len(args) >= 1:

                #
                return np.log(get_arg(0))

        #
        elif function_name == "abs":

            #
            ### Abs function. ###
            #
            if len(args) >= 1:

                #
                return np.abs(get_arg(0))

        #
        elif function_name == "pow":

            #
            ### Power function. ###
            #
            if len(args) >= 2:

                #
                return np.power(get_arg(0), get_arg(1))

        #
        elif function_name == "sqrt":

            #
            ### Sqrt function. ###
            #
            if len(args) >= 1:

                #
                return np.sqrt(get_arg(0))

        #
        elif function_name == "int":

            #
            ### Int conversion. ###
            #
            if len(args) >= 1:

                #
                return int(get_arg(0))

        #
        elif function_name == "float":

            #
            ### Float conversion. ###
            #
            if len(args) >= 1:

                #
                return float(get_arg(0))

        #
        elif function_name == "str":

            #
            ### String conversion. ###
            #
            if len(args) >= 1:

                #
                return str(get_arg(0))

        #
        elif function_name == "bool":

            #
            ### Bool conversion. ###
            #
            if len(args) >= 1:

                #
                return bool(get_arg(0))

        #
        elif function_name == "list":

            #
            ### List conversion. ###
            #
            if len(args) >= 1:

                #
                return list(get_arg(0))

        #
        elif function_name == "tuple":

            #
            ### Tuple conversion. ###
            #
            if len(args) >= 1:

                #
                return tuple(get_arg(0))

        #
        elif function_name == "dict":

            #
            ### Dict creation. ###
            #
            return dict(*args)

        #
        elif function_name == "set":

            #
            ### Set creation. ###
            #
            if len(args) >= 1:

                #
                return set(get_arg(0))

        #
        elif function_name == "sorted":

            #
            ### Sorted function. ###
            #
            if len(args) >= 1:

                #
                return sorted(get_arg(0))

        #
        elif function_name == "reversed":

            #
            ### Reversed function. ###
            #
            if len(args) >= 1:

                #
                return list(reversed(get_arg(0)))

        #
        elif function_name == "all":

            #
            ### All function. ###
            #
            if len(args) >= 1:

                #
                return all(get_arg(0))

        #
        elif function_name == "any":

            #
            ### Any function. ###
            #
            if len(args) >= 1:

                #
                return any(get_arg(0))

        #
        elif function_name == "super":

            #
            ### Super function (returns a placeholder). ###
            #
            return object()

        #
        else:
            #
            raise NotImplementedError(f"External function not implemented: {function_name}")

    #
    def _get_default_value_for_type(self, var_type: lc.VarType) -> Any:

        """
        Get default value for a given type.

        Args:
            var_type (lc.VarType): Variable type

        Returns:
            Any: Default value for the type
        """

        #
        if var_type.type_name == "int":
            #
            return 0

        #
        elif var_type.type_name == "float":
            #
            return 0.0

        #
        elif var_type.type_name == "bool":
            #
            return False

        #
        elif var_type.type_name == "str":
            #
            return ""

        #
        elif var_type.type_name == "Tensor":
            #
            return np.array([], dtype=np.float32)

        #
        elif var_type.type_name == "list":
            #
            return []

        #
        elif var_type.type_name == "dict":
            #
            return {}

        #
        elif var_type.type_name == "tuple":
            #
            return ()

        #
        elif var_type.type_name == "None":
            #
            return None

        #
        else:
            #
            return None

    #
    def _infer_type_from_value(self, value: Any) -> lc.VarType:

        """
        Infer variable type from value.

        Args:
            value (Any): Value to infer type from

        Returns:
            lc.VarType: Inferred type
        """

        #
        if isinstance(value, bool):
            #
            return lc.VarType("bool")

        #
        elif isinstance(value, int):
            #
            return lc.VarType("int")

        #
        elif isinstance(value, float):
            #
            return lc.VarType("float")

        #
        elif isinstance(value, str):
            #
            return lc.VarType("str")

        #
        elif isinstance(value, np.ndarray):
            #
            return lc.VarType("Tensor")

        #
        elif isinstance(value, list):
            #
            return lc.VarType("list")

        #
        elif isinstance(value, tuple):
            #
            return lc.VarType("tuple")

        #
        elif isinstance(value, dict):
            #
            return lc.VarType("dict")

        #
        elif value is None:
            #
            return lc.VarType("None")

        #
        else:
            #
            return lc.VarType("Any")

    #
    def _validate_flow_control_instruction(
        self,
        instruction: lc.FlowControlInstruction,
        context: ExecutionContext
    ) -> bool:

        """
        Validate a flow control instruction.

        Args:
            instruction (lc.FlowControlInstruction): Instruction to validate
            context (ExecutionContext): Execution context

        Returns:
            bool: True if instruction is valid
        """

        #
        ### Validate variable initialization. ###
        #
        if isinstance(instruction, lc.FlowControlVariableInit):

            #
            ### Check if variable name is valid. ###
            #
            if not isinstance(instruction.var_name, str):
                #
                return False

            #
            ### Check if variable type is valid. ###
            #
            if not isinstance(instruction.var_type, lc.VarType):
                #
                return False

            #
            return True

        #
        ### Validate variable assignment. ###
        #
        elif isinstance(instruction, lc.FlowControlVariableAssignment):

            #
            ### Check if variable name is valid. ###
            #
            if not isinstance(instruction.var_name, str):
                #
                return False

            #
            return True

        #
        ### Validate binary operation. ###
        #
        elif isinstance(instruction, lc.FlowControlBasicBinaryOperation):

            #
            ### Check if input variables exist. ###
            #
            if not self._is_valid_variable_reference(instruction.input1_var_name, context):
                #
                return False

            #
            if not self._is_valid_variable_reference(instruction.input2_var_name, context):
                #
                return False

            #
            return True

        #
        ### Validate unary operation. ###
        #
        elif isinstance(instruction, lc.FlowControlBasicUnaryOperation):

            #
            ### Check if input variable exists. ###
            #
            if not self._is_valid_variable_reference(instruction.input_var_name, context):
                #
                return False

            #
            return True

        #
        ### Validate for loop. ###
        #
        elif isinstance(instruction, lc.FlowControlForEachLoop):

            #
            ### Check if iterator expression is valid. ###
            #
            if not isinstance(instruction.iterator, lc.Expression):
                #
                return False

            #
            return True

        #
        ### Validate while loop. ###
        #
        elif isinstance(instruction, lc.FlowControlWhileLoop):

            #
            ### Check if condition is valid. ###
            #
            if not isinstance(instruction.condition, lc.Condition):
                #
                return False

            #
            return True

        #
        ### Validate function call. ###
        #
        elif isinstance(instruction, lc.FlowControlFunctionCall):

            #
            ### Check if function name is valid. ###
            #
            if not isinstance(instruction.function_called, str):
                #
                return False

            #
            return True

        #
        ### Validate layer pass. ###
        #
        elif isinstance(instruction, lc.FlowControlLayerPass):

            #
            ### Check if layer exists. ###
            #
            if not self._is_valid_variable_reference(instruction.layer_name, context):
                #
                return False

            #
            return True

        #
        ### Validate return. ###
        #
        elif isinstance(instruction, lc.FlowControlReturn):

            #
            ### Check if return variables exist. ###
            #
            for return_var in instruction.return_variables:

                #
                if not self._is_valid_variable_reference(return_var, context):
                    #
                    return False

            #
            return True

        #
        ### Validate condition. ###
        #
        elif isinstance(instruction, lc.FlowControlCondition):

            #
            ### Check if condition is valid. ###
            #
            if not isinstance(instruction.condition, lc.Condition):
                #
                return False

            #
            return True

        #
        else:
            #
            return False

    #
    def _is_valid_variable_reference(
        self,
        var_name: str,
        context: ExecutionContext
    ) -> bool:

        """
        Check if a variable reference is valid within the given context.

        Args:
            var_name (str): Variable name
            context (ExecutionContext): Execution context

        Returns:
            bool: True if variable reference is valid
        """

        #
        ### Check if variable exists in context. ###
        #
        return context.has_variable(var_name)


#
#########################################################################
################### MODEL INTERPRETER UTILITIES #########################
#########################################################################
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

                #
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

                    #
                    ### Update defined variables based on instruction ###
                    #
                    if isinstance(instruction, lc.FlowControlVariableAssignment):
                        #
                        defined_variables.add(instruction.var_name)
                    #
                    elif isinstance(instruction, lc.FlowControlBasicBinaryOperation):
                        #
                        defined_variables.add(instruction.output_var_name)
                    #
                    elif isinstance(instruction, lc.FlowControlBasicUnaryOperation):
                        #
                        defined_variables.add(instruction.output_var_name)
                    #
                    elif isinstance(instruction, lc.FlowControlFunctionCall):
                        #
                        defined_variables.update(instruction.output_variables)
                    #
                    elif isinstance(instruction, lc.FlowControlSubBlockFunctionCall):
                        #
                        defined_variables.update(instruction.output_variables)
                    #
                    elif isinstance(instruction, lc.FlowControlLayerPass):
                        #
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
            #
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

            #
            return True

        #
        ### Check if it's a temporary variable created in flow control instructions ###
        #
        for function in model_block.block_functions.values():

            #
            for instruction in function.function_flow_control:

                #
                if isinstance(instruction, lc.FlowControlVariableAssignment):
                    #
                    if instruction.var_name == var_name:
                        #
                        return True

                #
                elif isinstance(instruction, lc.FlowControlBasicBinaryOperation):
                    #
                    if instruction.output_var_name == var_name:
                        #
                        return True

                #
                elif isinstance(instruction, lc.FlowControlBasicUnaryOperation):
                    #
                    if instruction.output_var_name == var_name:
                        #
                        return True

                #
                elif isinstance(instruction, lc.FlowControlFunctionCall):
                    #
                    if var_name in instruction.output_variables:
                        #
                        return True

                #
                elif isinstance(instruction, lc.FlowControlSubBlockFunctionCall):
                    #
                    if var_name in instruction.output_variables:
                        #
                        return True

                #
                elif isinstance(instruction, lc.FlowControlLayerPass):
                    #
                    if var_name in instruction.output_variables:
                        #
                        return True

        #
        ### Check if it's a function argument ###
        #
        for function in model_block.block_functions.values():

            #
            if var_name in function.function_arguments:
                #
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
