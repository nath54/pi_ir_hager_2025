#
import ast
import sys
import os
from typing import Any, Optional, Dict, List, cast

#
import lib_classes as lc


#
def extract_expression(node: ast.AST) -> Optional[lc.Expression]:
    """
    Extracts an Expression object from an AST node.

    Args:
        node (ast.AST): The AST node to analyze.

    Returns:
        Union[lc.Expression, None]: The extracted expression or None if not applicable.
    """
    #
    if isinstance(node, ast.Name):
        return lc.ExpressionVariable(var_name=node.id)
    #
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return lc.ExpressionConstantNumeric(constant=node.value)
        elif isinstance(node.value, str):
            return lc.ExpressionConstantString(constant=node.value)
        elif isinstance(node.value, list):
            elements = [ elt for elt in [extract_expression(ast.Constant(value=elt)) for elt in node.value] if isinstance(elt, lc.ExpressionConstant) ]
            return lc.ExpressionConstantList(elements=elements)
    #
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        args = [ elt.constant for elt in [extract_expression(arg) for arg in node.args] if isinstance(elt, lc.ExpressionConstant) ]
        return lc.ExpressionConstantRange(end_value=args[1] if len(args) > 1 else args[0], start_value=args[0] if len(args) > 1 else 0, step=args[2] if len(args) > 2 else 1)
    #
    return None


#
def extract_condition(node: ast.AST) -> Optional[lc.Condition]:
    """
    Extracts a Condition object from an AST node.

    Args:
        node (ast.AST): The AST node to analyze.

    Returns:
        Union[lc.Condition, None]: The extracted condition or None if not applicable.
    """
    #
    if isinstance(node, ast.Compare):
        left = extract_expression(node.left)
        #
        if left is None:
            #
            return None
        #
        ops = [op.__class__.__name__.lower() for op in node.ops]
        comparators = [ comp for comp in [extract_expression(comp) for comp in node.comparators] if comp is not None]
        if len(ops) == 1:
            return lc.ConditionBinary(elt1=left, cond_operator=ops[0], elt2=comparators[0])
    #
    elif isinstance(node, ast.BoolOp):
        values = [elt for elt in [extract_condition(val) or extract_expression(val) for val in node.values] if elt is not None]
        op = "and" if isinstance(node.op, ast.And) else "or"
        result = values[0]
        #
        if len(values) < 2:
            #
            return None
        #
        for val in values[1:]:
            result = lc.ConditionBinary(elt1=result, cond_operator=op, elt2=val)
        return cast(lc.Condition, result)
    #
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = extract_condition(node.operand) or extract_expression(node.operand)
        if operand is not None:
            return lc.ConditionUnary(elt=operand, cond_operator="not")
    #
    return None


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
        """
        #
        self.model_blocks: Dict[str, lc.ModelBlock] = {}
        self.main_block: str = ""
        self.current_model_visit: List[str] = []
        self.current_function_visit: str = ""
        self.sub_block_counter: Dict[str, int] = {}

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
        current_block = self.model_blocks[self.current_model_visit[-1]]
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

            # Handle the case where arg.annotation is None or not a Name node
            if isinstance(arg.annotation, ast.Name):
                arg_type = arg.annotation.id
            elif arg.annotation is not None:
                arg_type = ast.dump(arg.annotation)  # Convert the annotation to a string representation
            else:
                arg_type = "Any"

            #
            default = None
            if arg_name in node.args.defaults:
                default = extract_expression(node.args.defaults[node.args.args.index(arg) - len(node.args.defaults)])
            args[arg_name] = (arg_type, default)
        current_block.block_parameters = args

        # Process body for layers and variables
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Attribute):
                target = stmt.targets[0]

                if isinstance(target.value, ast.Name) and target.value.id == "self":
                    var_name = target.attr
                    if isinstance(stmt.value, ast.Call):
                        # Handle layer definitions
                        layer_type = self.get_layer_type(stmt.value.func)
                        params = {kw.arg: extract_expression(kw.value) or kw.value for kw in stmt.value.keywords if kw.arg is not None}
                        if layer_type in {"ModuleList", "Sequential"}:
                            self._handle_container(var_name, layer_type, stmt.value, current_block)
                        else:
                            current_block.block_layers[var_name] = lc.Layer(
                                layer_var_name=var_name,
                                layer_type=layer_type,
                                layer_parameters_kwargs=params
                            )
                    elif isinstance(stmt.value, ast.For):
                        # Handle loop initialization (e.g., ModuleList with loop)
                        self._handle_loop_init(var_name=var_name, for_node=stmt.value, block=current_block)

    #
    def _handle_container(self, var_name: str, container_type: str, call_node: ast.Call, block: lc.ModelBlock) -> None:
        """
        Handles nn.ModuleList and nn.Sequential by creating sub-blocks.

        Args:
            var_name (str): Name of the container variable.
            container_type (str): Type of container ("ModuleList" or "Sequential").
            call_node (ast.Call): The AST node of the container call.
            block (lc.ModelBlock): The current model block.
        """
        #
        sub_block_name = f"Block{container_type}_{self.current_model_visit[-1]}_{self.sub_block_counter[self.current_model_visit[-1]]}"
        self.sub_block_counter[self.current_model_visit[-1]] += 1
        sub_block = lc.ModelBlock(block_name=sub_block_name)
        self.model_blocks[sub_block_name] = sub_block

        #
        layers: list[lc.Layer] = []
        for arg in call_node.args:
            if isinstance(arg, ast.List):
                for elt in arg.elts:
                    if isinstance(elt, ast.Call):
                        layer_type = self.get_layer_type(elt.func)
                        params = {kw.arg: extract_expression(kw.value) or kw.value for kw in elt.keywords if kw.arg is not None}
                        layers.append(lc.Layer(f"layer_{len(layers)}", layer_type, params))

        # Add layers to sub-block
        for i, layer in enumerate(layers):
            sub_block.block_layers[f"layer_{i}"] = layer

        # Add container layer to parent block
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
        iterator = extract_expression(for_node.iter) or for_node.iter.id
        iterable_var = for_node.target.id
        layers: list[lc.Layer] = []
        for stmt in for_node.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                layer_type = self.get_layer_type(stmt.value.func)
                params = {kw.arg: extract_expression(kw.value) or kw.value for kw in stmt.value.keywords if kw.arg is not None}
                layers.append(lc.Layer(f"layer_{len(layers)}", layer_type, params))

        # Add layers to sub-block
        for i, layer in enumerate(layers):
            sub_block.block_layers[f"layer_{i}"] = layer

        # Add to parent block with loop info
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
            value = extract_expression(stmt.value)
            if target and value:
                flow_control.append(lc.FlowControlVariableAssignment(var_name=target, var_value=value))
            elif isinstance(stmt.value, ast.Call):
                outputs = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                func_name = self.get_layer_type(stmt.value.func)
                args = {kw.arg: extract_expression(kw.value) or kw.value for kw in stmt.value.keywords if kw.arg is not None}
                if func_name in self.model_blocks[self.current_model_visit[-1]].block_layers:
                    flow_control.append(lc.FlowControlLayerPass(outputs, func_name, args))
                else:
                    flow_control.append(lc.FlowControlFunctionCall(outputs, func_name, args))

        #
        elif isinstance(stmt, ast.For):

            #
            if not isinstance(stmt.target, ast.Name):
                #
                return

            #
            iterator = extract_expression(stmt.iter) or (stmt.iter.id if isinstance(stmt.iter, ast.Name) else None)

            #
            if iterator is None:
                #
                return

            #
            flow_control_loop: lc.FlowControlForLoop = lc.FlowControlForLoop(
                iterable_var_name=stmt.target.id,
                iterator=iterator,
                flow_control_instructions=[]
            )

            #
            flow_control.append(flow_control_loop)
            for sub_stmt in stmt.body:
                self._process_statement(sub_stmt, flow_control_loop.flow_control_instructions)

        #
        elif isinstance(stmt, ast.While):

            #
            condition = extract_condition(stmt.test)

            #
            if condition is None:
                #
                return

            #
            flow_control_while: lc.FlowControlWhileLoop = lc.FlowControlWhileLoop(
                condition=condition,
                flow_control_instructions=[]
            )

            #
            flow_control.append(flow_control_while)

            #
            for sub_stmt in stmt.body:
                self._process_statement(sub_stmt, flow_control_while.flow_control_instructions)

        #
        elif isinstance(stmt, ast.If):
            condition = extract_condition(stmt.test)
            #
            if condition is None:
                return
            #
            sub_func_name = f"cond_{len(self.model_blocks[self.current_model_visit[-1]].block_functions)}"
            sub_func = lc.BlockFunction(sub_func_name, {"input": ("Any", None)}, self.model_blocks[self.current_model_visit[-1]])
            self.model_blocks[self.current_model_visit[-1]].block_functions[sub_func_name] = sub_func
            for sub_stmt in stmt.body:
                self._process_statement(sub_stmt, sub_func.function_flow_control)

            #
            flow_control_subcall: lc.FlowControlSubBlockFunctionCall = lc.FlowControlSubBlockFunctionCall(
                output_variables=["output"],
                function_called=sub_func_name,
                function_arguments={"input": "x"}
            )

            #
            flow_control.append(flow_control_subcall)
            self.model_blocks[self.current_model_visit[-1]].block_layers[sub_func_name] = lc.LayerCondition(
                layer_var_name=sub_func_name,
                layer_conditions_blocks={condition: flow_control_subcall}
            )

        #
        elif isinstance(stmt, ast.Return):
            #
            expr: Optional[lc.Expression]
            #
            returns: list[str] = []

            #
            if isinstance(stmt.value, ast.Tuple):
                #
                for val in stmt.value.elts:

                    #
                    expr = extract_expression(val)

                    #
                    if expr is None:
                        continue
                    elif isinstance(expr, lc.ExpressionVariable):
                        returns.append( expr.var_name )
                    #
                    elif isinstance(val, ast.Name):
                        #
                        returns.append( val.id )

            #
            elif isinstance(stmt.value, ast.Name):
                returns.append( stmt.value.id )

            #
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

            # Track used sub-functions
            used_funcs = {"forward"}
            for func in block.block_functions.values():
                for instr in func.function_flow_control:
                    if isinstance(instr, lc.FlowControlSubBlockFunctionCall):
                        used_funcs.add(instr.function_called)

            # Remove unused functions
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

    #
    return lang1


#
if __name__ == "__main__":
    #
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise UserWarning(f"Error: Usage: python {sys.argv[0]} path_to_model_script.py [--main-block <MainBlockName>]")

    #
    path_to_file: str = sys.argv[1]
    main_block_name: str = ""
    if len(sys.argv) == 3 and sys.argv[2].startswith("--main-block"):
        main_block_name = sys.argv[2].split("=")[1] if "=" in sys.argv[2] else sys.argv[2].split()[1]

    #
    l1_model: lc.Language1_Model = extract_from_file(filepath=path_to_file, main_block_name=main_block_name)

    #
    print("\n" * 2)
    for block_name, block in l1_model.model_blocks.items():
        print(block)