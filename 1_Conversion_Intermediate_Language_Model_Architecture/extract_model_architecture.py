#
import ast
import sys
import os
from typing import Optional, Dict, List, cast, Any, Callable

#
import inspect
import importlib.util
import importlib.machinery

#
import lib_classes as lc


#
def extract_expression(node: ast.AST, analyzer: "ModelAnalyzer") -> Optional[lc.Expression]:
    """
    Extracts an Expression object from an AST node, with optional reference to analyzer for global constants.

    Args:
        node (ast.AST): The AST node to analyze.
        analyzer (ModelAnalyzer, optional): The analyzer instance to access global constants.

    Returns:
        Optional[lc.Expression]: The extracted expression or None if not applicable.
    """
    #
    if isinstance(node, ast.Name):
        if analyzer and node.id in analyzer.global_constants:
            type_str, value = analyzer.global_constants[node.id]
            if type_str in ("int", "float"):
                return lc.ExpressionConstantNumeric(constant=value.constant if isinstance(value, lc.ExpressionConstant) else value)
            elif type_str == "str":
                return lc.ExpressionConstantString(constant=value.constant if isinstance(value, lc.ExpressionConstant) else value)
            elif type_str == "list":
                return lc.ExpressionConstantList(elements=value.elements if isinstance(value, lc.ExpressionConstantList) else value)
        return lc.ExpressionVariable(var_name=node.id)
    #
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return lc.ExpressionConstantNumeric(constant=node.value)
        elif isinstance(node.value, str):
            return lc.ExpressionConstantString(constant=node.value)
        elif isinstance(node.value, list):
            elements = [elt for elt in [extract_expression(ast.Constant(value=elt), analyzer) for elt in node.value] if isinstance(elt, lc.ExpressionConstant)]
            return lc.ExpressionConstantList(elements=elements)
    #
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        args = [elt.constant for elt in [extract_expression(arg, analyzer) for arg in node.args] if isinstance(elt, lc.ExpressionConstant)]
        return lc.ExpressionConstantRange(end_value=args[1] if len(args) > 1 else args[0], start_value=args[0] if len(args) > 1 else 0, step=args[2] if len(args) > 2 else 1)
    #
    return None


#
def extract_condition(node: ast.AST, analyzer: "ModelAnalyzer") -> Optional[lc.Condition]:
    """
    Extracts a Condition object from an AST node.

    Args:
        node (ast.AST): The AST node to analyze.
        analyzer (ModelAnalyzer, optional): The analyzer instance to access global constants.

    Returns:
        Optional[lc.Condition]: The extracted condition or None if not applicable.
    """
    #
    if isinstance(node, ast.Compare):
        left = extract_expression(node.left, analyzer)
        if left is None:
            return None
        ops = [op.__class__.__name__.lower() for op in node.ops]
        comparators = [comp for comp in [extract_expression(comp, analyzer) for comp in node.comparators] if comp is not None]
        if len(ops) == 1 and len(comparators) >= 1:
            return lc.ConditionBinary(elt1=left, cond_operator=ops[0], elt2=comparators[0])
    #
    elif isinstance(node, ast.BoolOp):
        values = [elt for elt in [extract_condition(val, analyzer) or extract_expression(val, analyzer) for val in node.values] if elt is not None]
        op = "and" if isinstance(node.op, ast.And) else "or"
        if len(values) < 2:
            return None
        result = values[0]
        for val in values[1:]:
            result = lc.ConditionBinary(elt1=result, cond_operator=op, elt2=val)
        return cast(lc.Condition, result)
    #
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = extract_condition(node.operand, analyzer) or extract_expression(node.operand, analyzer)
        if operand is not None:
            return lc.ConditionUnary(elt=operand, cond_operator="not")
    #
    return None


#
def process_expression(node: ast.AST, flow_control: List[lc.FlowControlInstruction], analyzer: "ModelAnalyzer") -> str:
    """
    Processes a complex expression, adding necessary flow control instructions and returning the final variable name.

    Args:
        node (ast.AST): The AST node representing the expression.
        flow_control (list[lc.FlowControlInstruction]): The flow control list to append instructions to.
        analyzer (ModelAnalyzer): The analyzer instance to access global constants.

    Returns:
        str: The variable name holding the expression's result.
    """
    #
    if isinstance(node, ast.BinOp):
        # Handle binary operations like `y * RANDOM_CONSTANT2`
        left_var = process_expression(node.left, flow_control, analyzer)
        right_var = process_expression(node.right, flow_control, analyzer)
        op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        op = op_map.get(type(node.op), None)
        if not op:
            return left_var  # Fallback if operation not supported
        result_var = f"temp_{id(node)}"
        flow_control.append(
            lc.FlowControlBasicBinaryOperation(
                output_var_name=result_var,
                input1_var_name=left_var,
                operation=op,
                input2_var_name=right_var
            )
        )
        return result_var
    #
    elif isinstance(node, ast.Name) or isinstance(node, ast.Constant):
        expr = extract_expression(node, analyzer)
        if not expr:
            return ""
        if isinstance(expr, lc.ExpressionVariable):
            return expr.var_name
        # For constants, create a temporary variable
        temp_var = f"temp_{id(node)}"
        type_str = "int" if isinstance(expr, lc.ExpressionConstantNumeric) and isinstance(expr.constant, int) else "float" if isinstance(expr, lc.ExpressionConstantNumeric) else "str" if isinstance(expr, lc.ExpressionConstantString) else "list"
        flow_control.append(
            lc.FlowControlVariableInit(
                var_name=temp_var,
                var_type=type_str,
                var_value=expr
            )
        )
        return temp_var
    #
    return ""


#
class ModelAnalyzer(ast.NodeVisitor):

    # --------------------------------------------------------- #
    # ----               INIT MODEL ANALYZER               ---- #
    # --------------------------------------------------------- #

    #
    def __init__(self) -> None:
        """
        Initializer of the ModelAnalyzer class.

        Attributes:
            model_blocks (dict[str, lc.ModelBlock]): List of all blocks analyzed, indexed by their block ID (e.g., name).
            main_block (str): ID of the main block, given in sys.argv with `--main-block <MainBlockName>`.
            current_model_visit (list[str]): Stack of current blocks being visited, access top with [-1].
            current_function_visit (str): Name of the current visited function.
            sub_block_counter (dict[str, int]): Counter for naming sub-blocks (e.g., ModuleList, Sequential).
            global_constants (dict[str, tuple[str, Any]]): Global constants defined outside classes.
        """
        #
        self.model_blocks: Dict[str, lc.ModelBlock] = {}
        self.main_block: str = ""
        self.current_model_visit: List[str] = []
        self.current_function_visit: str = ""
        self.sub_block_counter: Dict[str, int] = {}
        self.global_constants: Dict[str, tuple[str, Any]] = {}

    # --------------------------------------------------------- #
    # ----                  CLASS VISITOR                  ---- #
    # --------------------------------------------------------- #

    #
    def _is_torch_module_class(self, node: ast.ClassDef) -> bool:
        """
        Indicates if the given ClassDef node is a nn.Module subclass.

        Args:
            node (ast.ClassDef): Node to check.

        Returns:
            bool: True if subclass of nn.Module, else False.
        """
        #
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True
        return False

    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Called when the AST visitor detects a class definition.

        Args:
            node (ast.ClassDef): The class node to visit.

        Raises:
            NameError: If two classes have the same name.
        """
        #
        if not self._is_torch_module_class(node):
            return

        #
        block_name: str = node.name
        if block_name in self.model_blocks:
            raise NameError(f"ERROR: Duplicate class name detected: {block_name}")

        #
        if self.main_block == "":
            #
            if block_name in ["MainModel", "MainNet", "Model", "Net"]:
                #
                self.main_block = block_name

        #
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)
        self.current_model_visit.append(block_name)
        self.sub_block_counter[block_name] = 0

        #
        self.generic_visit(node)
        self.current_model_visit.pop()

    # --------------------------------------------------------- #
    # ----                FUNCTION VISITOR                 ---- #
    # --------------------------------------------------------- #

    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Called when the AST visitor detects a function definition.

        Args:
            node (ast.FunctionDef): The function node to visit.
        """
        #
        if not self.current_model_visit:
            return

        #
        self.current_function_visit = node.name

        #
        if node.name == "__init__":
            self._analyze_init_method(node)
        elif node.name == "forward":
            self._analyze_forward_method(node)
        else:
            self._analyse_other_method(node)

        #
        self.current_function_visit = ""
        self.generic_visit(node)

    #
    def _analyze_init_method(self, node: ast.FunctionDef) -> None:
        """
        Analyzes the __init__ method to extract parameters and layers.

        Args:
            node (ast.FunctionDef): The __init__ function node.
        """
        #
        current_block = self.model_blocks[self.current_model_visit[-1]]

        # Extract function arguments
        args = {}
        for arg in node.args.args[1:]:  # Skip 'self'
            #
            arg_name = arg.arg
            if isinstance(arg.annotation, ast.Name):
                arg_type = arg.annotation.id
            elif arg.annotation is not None:
                arg_type = ast.dump(arg.annotation)
            else:
                arg_type = "Any"
            #
            default = None
            if arg_name in node.args.defaults:
                default = extract_expression(node.args.defaults[node.args.args.index(arg) - len(node.args.defaults)], self)
            args[arg_name] = (arg_type, default)
        current_block.block_parameters = args

        # Process body for layers and variables
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Attribute):
                target = stmt.targets[0]
                if isinstance(target.value, ast.Name) and target.value.id == "self":
                    var_name = target.attr
                    if isinstance(stmt.value, ast.Call):
                        layer_type = self.get_layer_type(stmt.value.func)
                        params = {kw.arg: extract_expression(kw.value, self) or kw.value for kw in stmt.value.keywords if kw.arg is not None}
                        if layer_type in {"ModuleList", "Sequential"}:
                            self._handle_container(var_name, layer_type, stmt.value, current_block)
                        else:
                            current_block.block_layers[var_name] = lc.Layer(
                                layer_var_name=var_name,
                                layer_type=layer_type,
                                layer_parameters_kwargs=params
                            )
                    elif isinstance(stmt.value, ast.For):
                        self._handle_loop_init(var_name=var_name, for_node=stmt.value, block=current_block)

    #
    def _handle_container(self, var_name: str, container_type: str, call_node: ast.Call, block: lc.ModelBlock) -> None:
        """
        Handles nn.ModuleList and nn.Sequential by creating sub-blocks and defining their forward methods.

        Args:
            var_name (str): Name of the container variable.
            container_type (str): Type of container ("ModuleList" or "Sequential").
            call_node (ast.Call): The AST node of the container call.
            block (lc.ModelBlock): The current model block.
        """
        # Create a unique sub-block name
        sub_block_name = f"Block{container_type}_{self.current_model_visit[-1]}_{self.sub_block_counter[self.current_model_visit[-1]]}"
        self.sub_block_counter[self.current_model_visit[-1]] += 1
        sub_block = lc.ModelBlock(block_name=sub_block_name)
        self.model_blocks[sub_block_name] = sub_block

        # Initialize layers list
        layers: List[lc.Layer] = []

        # Handle arguments
        if call_node.args:
            for arg in call_node.args:
                if isinstance(arg, ast.Call):
                    layer_type = self.get_layer_type(arg.func)
                    params = {kw.arg: extract_expression(kw.value, self) or kw.value for kw in arg.keywords if kw.arg is not None}
                    layers.append(lc.Layer(f"layer_{len(layers)}", layer_type, params))
                elif isinstance(arg, ast.GeneratorExp):
                    elt = arg.elt
                    if isinstance(elt, ast.Call):
                        layer_type = self.get_layer_type(elt.func)
                        params = {kw.arg: extract_expression(kw.value, self) or kw.value for kw in elt.keywords if kw.arg is not None}
                        if arg.generators and isinstance(arg.generators[0].iter, ast.Call) and isinstance(arg.generators[0].iter.func, ast.Name) and arg.generators[0].iter.func.id == "range":
                            range_args = [expr.constant for expr in [extract_expression(a, self) for a in arg.generators[0].iter.args] if expr is not None and isinstance(expr, lc.ExpressionConstant)]
                            if len(range_args) >= 1:
                                count = range_args[0] if len(range_args) == 1 else range_args[1] - range_args[0]
                                for i in range(int(count)):
                                    layers.append(lc.Layer(f"layer_{i}", layer_type, params.copy()))

        # Define forward method
        forward_func = lc.BlockFunction(function_name="forward", function_arguments={"x": ("Any", None)}, model_block=sub_block)
        sub_block.block_functions["forward"] = forward_func
        current_input = lc.ExpressionVariable("x")
        for i, layer in enumerate(layers):
            layer_name = f"layer_{i}"
            sub_block.block_layers[layer_name] = layer
            output_var = f"out_{i}"
            forward_func.function_flow_control.append(
                lc.FlowControlLayerPass(output_variables=[output_var], layer_name=layer_name, layer_arguments={"x": current_input})
            )
            if container_type == "Sequential":
                current_input = lc.ExpressionVariable(output_var)
        if container_type == "Sequential":
            forward_func.function_flow_control.append(lc.FlowControlReturn(return_variables=[current_input.var_name]))
        elif container_type == "ModuleList":
            output_vars = [f"out_{i}" for i in range(len(layers))]
            forward_func.function_flow_control.append(lc.FlowControlReturn(return_variables=output_vars))

        # Add to parent block
        block.block_layers[var_name] = lc.Layer(
            layer_var_name=var_name,
            layer_type=container_type,
            layer_parameters_kwargs={"sub_block": sub_block_name}
        )

    #
    def _handle_loop_init(self, var_name: str, for_node: ast.For, block: lc.ModelBlock) -> None:
        """
        Handles loop-based initialization of layers (e.g., ModuleList).

        Args:
            var_name (str): Name of the variable being initialized.
            for_node (ast.For): The for loop node.
            block (lc.ModelBlock): The current model block.
        """
        #
        sub_block_name = f"BlockModuleList_{self.current_model_visit[-1]}_{self.sub_block_counter[self.current_model_visit[-1]]}"
        self.sub_block_counter[self.current_model_visit[-1]] += 1
        sub_block = lc.ModelBlock(block_name=sub_block_name)
        self.model_blocks[sub_block_name] = sub_block

        #
        if not isinstance(for_node.iter, ast.Name) or not isinstance(for_node.target, ast.Name):
            return

        # Extract loop details
        iterator = extract_expression(for_node.iter, self) or for_node.iter.id
        iterable_var = for_node.target.id
        layers: List[lc.Layer] = []
        for stmt in for_node.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                layer_type = self.get_layer_type(stmt.value.func)
                params = {kw.arg: extract_expression(kw.value, self) or kw.value for kw in stmt.value.keywords if kw.arg is not None}
                layers.append(lc.Layer(f"layer_{len(layers)}", layer_type, params))

        # Add layers to sub-block
        for i, layer in enumerate(layers):
            sub_block.block_layers[f"layer_{i}"] = layer

        # Define forward method
        forward_func = lc.BlockFunction(function_name="forward", function_arguments={"x": ("Any", None)}, model_block=sub_block)
        sub_block.block_functions["forward"] = forward_func
        output_vars = []
        for i in range(len(layers)):
            output_var = f"out_{i}"
            forward_func.function_flow_control.append(
                lc.FlowControlLayerPass(
                    output_variables=[output_var],
                    layer_name=f"layer_{i}",
                    layer_arguments={"x": lc.ExpressionVariable("x")}
                )
            )
            output_vars.append(output_var)
        forward_func.function_flow_control.append(lc.FlowControlReturn(return_variables=output_vars))

        # Add to parent block
        block.block_layers[var_name] = lc.Layer(
            layer_var_name=var_name,
            layer_type="ModuleList",
            layer_parameters_kwargs={"sub_block": sub_block_name, "iterator": iterator, "iterable_var": iterable_var}
        )

    #
    def _analyze_forward_method(self, node: ast.FunctionDef) -> None:
        """
        Analyzes the forward method to extract control flow.

        Args:
            node (ast.FunctionDef): The forward function node.
        """
        #
        current_block = self.model_blocks[self.current_model_visit[-1]]
        forward_func = lc.BlockFunction(
            function_name="forward",
            function_arguments={arg.arg: ("Any", None) for arg in node.args.args},
            model_block=current_block
        )
        current_block.block_functions["forward"] = forward_func

        #
        for stmt in node.body:
            self._process_statement(stmt, forward_func.function_flow_control)

    #
    def _process_statement(self, stmt: ast.AST, flow_control: List[lc.FlowControlInstruction]) -> None:
        """
        Processes a statement and adds it to the flow control list.

        Args:
            stmt (ast.AST): The statement to process.
            flow_control (list[lc.FlowControlInstruction]): The flow control list to append to.
        """
        #
        if isinstance(stmt, ast.Assign):
            target = stmt.targets[0].id if isinstance(stmt.targets[0], ast.Name) else None
            value = extract_expression(stmt.value, self)
            if target and value:
                flow_control.append(lc.FlowControlVariableAssignment(var_name=target, var_value=value))
            elif isinstance(stmt.value, ast.Call):
                outputs = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                func_name = self.get_layer_type(stmt.value.func)
                args = {kw.arg: extract_expression(kw.value, self) or kw.value for kw in stmt.value.keywords if kw.arg is not None}
                if func_name in self.model_blocks[self.current_model_visit[-1]].block_layers:
                    flow_control.append(lc.FlowControlLayerPass(outputs, func_name, args))
                else:
                    flow_control.append(lc.FlowControlFunctionCall(outputs, func_name, args))

        #
        elif isinstance(stmt, ast.AugAssign):
            self.visit_AugAssign(stmt)

        #
        elif isinstance(stmt, ast.For):
            if not isinstance(stmt.target, ast.Name):
                return
            iterator = extract_expression(stmt.iter, self) or (stmt.iter.id if isinstance(stmt.iter, ast.Name) else None)
            if iterator is None:
                return
            flow_control_loop: lc.FlowControlForLoop = lc.FlowControlForLoop(
                iterable_var_name=stmt.target.id,
                iterator=iterator,
                flow_control_instructions=[]
            )
            flow_control.append(flow_control_loop)
            for sub_stmt in stmt.body:
                self._process_statement(sub_stmt, flow_control_loop.flow_control_instructions)

        #
        elif isinstance(stmt, ast.While):
            condition = extract_condition(stmt.test, self)
            if condition is None:
                return
            flow_control_while: lc.FlowControlWhileLoop = lc.FlowControlWhileLoop(
                condition=condition,
                flow_control_instructions=[]
            )
            flow_control.append(flow_control_while)
            for sub_stmt in stmt.body:
                self._process_statement(sub_stmt, flow_control_while.flow_control_instructions)

        #
        elif isinstance(stmt, ast.If):
            condition = extract_condition(stmt.test, self)
            if condition is None:
                return
            sub_func_name = f"cond_{len(self.model_blocks[self.current_model_visit[-1]].block_functions)}"
            sub_func = lc.BlockFunction(sub_func_name, {"input": ("Any", None)}, self.model_blocks[self.current_model_visit[-1]])
            self.model_blocks[self.current_model_visit[-1]].block_functions[sub_func_name] = sub_func
            for sub_stmt in stmt.body:
                self._process_statement(sub_stmt, sub_func.function_flow_control)
            flow_control_subcall: lc.FlowControlSubBlockFunctionCall = lc.FlowControlSubBlockFunctionCall(
                output_variables=["output"],
                function_called=sub_func_name,
                function_arguments={"input": "x"}
            )
            flow_control.append(flow_control_subcall)
            self.model_blocks[self.current_model_visit[-1]].block_layers[sub_func_name] = lc.LayerCondition(
                layer_var_name=sub_func_name,
                layer_conditions_blocks={condition: flow_control_subcall}
            )

        #
        elif isinstance(stmt, ast.Return):
            expr: Optional[lc.Expression]
            returns: List[str] = []
            if isinstance(stmt.value, ast.Tuple):
                for val in stmt.value.elts:
                    expr = extract_expression(val, self)
                    if expr is None:
                        continue
                    elif isinstance(expr, lc.ExpressionVariable):
                        returns.append(expr.var_name)
                    elif isinstance(val, ast.Name):
                        returns.append(val.id)
            elif isinstance(stmt.value, ast.Name):
                returns.append(stmt.value.id)
            flow_control.append(lc.FlowControlReturn(return_variables=returns))

    #
    def _analyse_other_method(self, node: ast.FunctionDef) -> None:
        """
        Analyzes other methods in the class.

        Args:
            node (ast.FunctionDef): The function node.
        """
        #
        current_block = self.model_blocks[self.current_model_visit[-1]]
        func = lc.BlockFunction(
            function_name=node.name,
            function_arguments={arg.arg: ("Any", None) for arg in node.args.args},
            model_block=current_block
        )
        current_block.block_functions[node.name] = func
        for stmt in node.body:
            self._process_statement(stmt, func.function_flow_control)

    #
    def get_layer_type(self, func: ast.AST) -> str:
        """
        Extracts the layer or function type from a call.

        Args:
            func (ast.AST): The function node.

        Returns:
            str: The type or name of the function/layer.
        """
        #
        if isinstance(func, ast.Attribute):
            return func.attr
        elif isinstance(func, ast.Name):
            return func.id
        return ""

    # --------------------------------------------------------- #
    # ----                 ASSIGN VISITOR                  ---- #
    # --------------------------------------------------------- #

    #
    def visit_Assign(self, node: ast.Assign) -> None:
        """
        Handles basic assignment statements (e.g., `x = 5`).

        Args:
            node (ast.Assign): The assignment node to visit.
        """
        #
        if len(node.targets) != 1:
            return

        target = node.targets[0]
        value = extract_expression(node.value, self)

        # Global constant
        if not self.current_model_visit and not self.current_function_visit and isinstance(target, ast.Name) and value:
            type_str = "int" if isinstance(value, lc.ExpressionConstantNumeric) and isinstance(value.constant, int) else "float" if isinstance(value, lc.ExpressionConstantNumeric) else "str" if isinstance(value, lc.ExpressionConstantString) else "list"
            self.global_constants[target.id] = (type_str, value)
            return

        # Inside function
        if self.current_model_visit and self.current_function_visit:
            current_block = self.model_blocks[self.current_model_visit[-1]]
            if self.current_function_visit in current_block.block_functions:
                func = current_block.block_functions[self.current_function_visit]
                if isinstance(target, ast.Name) and value:
                    func.function_flow_control.append(
                        lc.FlowControlVariableAssignment(var_name=target.id, var_value=value)
                    )
            return

        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """
        Handles augmented assignment statements (e.g., `x += y * RANDOM_CONSTANT2`).

        Args:
            node (ast.AugAssign): The augmented assignment node to visit.
        """
        #
        target = node.target
        if not isinstance(target, ast.Name):
            return

        # Only handle inside functions
        if not (self.current_model_visit and self.current_function_visit):
            return

        current_block = self.model_blocks[self.current_model_visit[-1]]
        if self.current_function_visit not in current_block.block_functions:
            return

        func = current_block.block_functions[self.current_function_visit]
        op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        op = op_map.get(type(node.op), None)
        if not op:
            return

        # Process the right-hand side expression
        rhs_var = process_expression(node.value, func.function_flow_control, self)
        if not rhs_var:
            return

        # Add the augmented operation
        func.function_flow_control.append(
            lc.FlowControlBasicBinaryOperation(
                output_var_name=target.id,
                input1_var_name=target.id,
                operation=op,
                input2_var_name=rhs_var
            )
        )

        # No need to call generic_visit since we're handling the node fully here

    #
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """
        Handles annotated assignment statements (e.g., `x: int = 0`).

        Args:
            node (ast.AnnAssign): The annotated assignment node to visit.
        """
        #
        target = node.target
        value = extract_expression(node.value, self) if node.value else None
        annotation = node.annotation.id if isinstance(node.annotation, ast.Name) else ast.dump(node.annotation) if node.annotation else "Any"

        # Global constant
        if not self.current_model_visit and not self.current_function_visit and isinstance(target, ast.Name):
            if value:
                self.global_constants[target.id] = (annotation, value)
            return

        # Inside function
        if self.current_model_visit and self.current_function_visit and isinstance(target, ast.Name):
            current_block = self.model_blocks[self.current_model_visit[-1]]
            if self.current_function_visit in current_block.block_functions:
                func = current_block.block_functions[self.current_function_visit]
                if value:
                    func.function_flow_control.append(
                        lc.FlowControlVariableInit(var_name=target.id, var_type=annotation, var_value=value)
                    )
                else:
                    func.function_flow_control.append(
                        lc.FlowControlVariableInit(var_name=target.id, var_type=annotation)
                    )
            return

        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_AssignStmt(self, node: ast.AST) -> None:
        """
        Placeholder for AssignStmt (not a standard AST node).

        Args:
            node (ast.AST): The node to visit.
        """
        # No standard AssignStmt in Python ast
        ast.NodeVisitor.generic_visit(self, node)

    # --------------------------------------------------------- #
    # ----                 GENERIC VISITOR                 ---- #
    # --------------------------------------------------------- #

    #
    def generic_visit(self, node: ast.AST) -> None:
        """
        Generic visitor for additional node processing.

        Args:
            node (ast.AST): The node to visit.
        """
        #
        ast.NodeVisitor.generic_visit(self, node)

    # --------------------------------------------------------- #
    # ----           CLEANING & ERROR DETECTIONS           ---- #
    # --------------------------------------------------------- #

    #
    def cleaning_and_error_detections(self) -> None:
        """
        Cleans unused functions and detects errors in the model blocks.
        """
        #
        for block_name, block in list(self.model_blocks.items()):
            if "forward" not in block.block_functions:
                raise ValueError(f"ERROR: Block {block_name} has no forward method.")
            used_funcs = {"forward"}
            for func in block.block_functions.values():
                for instr in func.function_flow_control:
                    if isinstance(instr, lc.FlowControlSubBlockFunctionCall):
                        used_funcs.add(instr.function_called)
            block.block_functions = {k: v for k, v in block.block_functions.items() if k in used_funcs}


#
def extract_from_file(filepath: str, main_block_name: str = "") -> lc.Language1_Model:
    """
    Extracts a neural network model architecture from a PyTorch script file.

    Args:
        filepath (str): Path to the file to extract the architecture from.
        main_block_name (str, optional): Name of the entry point of the model. Defaults to "".

    Raises:
        FileNotFoundError: If the file is not found.

    Returns:
        lc.Language1_Model: The extracted model architecture.
    """
    #
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: file not found: `{filepath}`")

    #
    with open(filepath, "r") as source:
        tree = ast.parse(source.read())

    #
    analyzer = ModelAnalyzer()
    analyzer.visit(tree)
    analyzer.cleaning_and_error_detections()

    #
    lang1: lc.Language1_Model = lc.Language1_Model()
    lang1.main_block = main_block_name or list(analyzer.model_blocks.keys())[0]
    lang1.model_blocks = analyzer.model_blocks
    lang1.global_constants = analyzer.global_constants

    #
    return lang1


#
def list_classes(module):
    """List all the classes defined inside a module."""
    classes = [m[1] for m in inspect.getmembers(module, inspect.isclass)]
    return classes


#
def import_module_from_filepath(filepath):
    """Imports a Python module from a filepath."""
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#
def get_pytorch_main_model(model_arch: lc.Language1_Model, filepath: str) -> Callable:

    #
    if model_arch.main_block == "":
        #
        raise UserWarning("Error: No main blocks detected in the model architecture !")

    #
    net_module = import_module_from_filepath(filepath)

    #
    if not hasattr(net_module, model_arch.main_block):
        #
        raise UserWarning("Error: The given python script does not have the specified main block !")

    #
    return getattr(net_module, model_arch.main_block)


#
if __name__ == "__main__":
    #
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise UserWarning(f"Error: Usage: python {sys.argv[0]} path_to_model_script.py [--main-block=<MainBlockName>]")

    #
    path_to_file: str = sys.argv[1]
    main_block_name: str = ""
    if len(sys.argv) == 3 and sys.argv[2].startswith("--main-block"):
        main_block_name = sys.argv[2].split("=")[1] if "=" in sys.argv[2] else sys.argv[2].split()[1]

    #
    l1_model: lc.Language1_Model = extract_from_file(filepath=path_to_file, main_block_name=main_block_name)

    #
    print(l1_model)

    #
    main_model_class =  get_pytorch_main_model(model_arch=l1_model, filepath=path_to_file)
    print(main_model_class)
