#
import ast
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union

#
import lib_classes_2 as lc


# Helper functions for AST analysis
def get_node_value(node: ast.AST) -> Any:
    """
    Extract value from different node types
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.BinOp):
        # Handle binary operations recursively
        left_val = get_node_value(node.left)
        right_val = get_node_value(node.right)

        if isinstance(node.op, ast.Add):
            if isinstance(left_val, str) or isinstance(right_val, str):
                return f"{left_val} + {right_val}"  # Represents string concatenation
            else:
                return left_val + right_val
        elif isinstance(node.op, ast.Sub):
            return f"{left_val} - {right_val}"
        elif isinstance(node.op, ast.Mult):
            return f"{left_val} * {right_val}"
        elif isinstance(node.op, ast.Div):
            return f"{left_val} / {right_val}"
        else:
            return f"<complex operation: {left_val} op {right_val}>"
    elif isinstance(node, ast.Attribute):
        return f"{get_node_value(node.value)}.{node.attr}"
    elif isinstance(node, ast.Call):
        func_name = get_node_value(node.func)
        args = [get_node_value(arg) for arg in node.args]
        kwargs = {kw.arg: get_node_value(kw.value) for kw in node.keywords}

        args_str = ', '.join([str(arg) for arg in args])
        kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])

        all_args = []
        if args_str:
            all_args.append(args_str)
        if kwargs_str:
            all_args.append(kwargs_str)

        return f"{func_name}({', '.join(all_args)})"
    elif isinstance(node, ast.List):
        return [get_node_value(elt) for elt in node.elts]
    elif isinstance(node, ast.Dict):
        return {get_node_value(k): get_node_value(v) for k, v in zip(node.keys, node.values)}
    else:
        return f"<unhandled node type: {type(node).__name__}>"


def get_variable_type_from_annotation(annotation: Optional[ast.AST]) -> str:
    """
    Extract type from annotation
    """
    if annotation is None:
        return "Any"

    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        return f"{get_node_value(annotation.value)}.{annotation.attr}"
    elif isinstance(annotation, ast.Subscript):
        container = get_node_value(annotation.value)
        if isinstance(annotation.slice, ast.Index):  # Python < 3.9
            params = get_node_value(annotation.slice.value)
        else:  # Python >= 3.9
            params = get_node_value(annotation.slice)

        return f"{container}[{params}]"
    elif isinstance(annotation, ast.Constant) and annotation.value is None:
        return "None"
    else:
        return f"<complex type: {type(annotation).__name__}>"


#
class ModelAnalyzer(ast.NodeVisitor):
    #
    def __init__(self) -> None:
        #
        self.model_blocks: dict[str, lc.ModelBlock] = {}
        #
        self.main_block: str = ""
        #
        self.current_model_visit: str = ""

        # Track the current function we're processing
        self.current_function: Optional[str] = None

        # Helper for resolving names
        self.imports: Dict[str, str] = {}  # Maps imported names to their original module paths

    #
    def is_torch_module_class(self, node: ast.ClassDef) -> bool:
        """
        Determine if a class inherits from torch.nn.Module
        """
        for base in node.bases:
            # Direct inheritance: class MyModel(nn.Module)
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True

            # Inheritance via variable: class MyModel(BaseModel) where BaseModel inherits from nn.Module
            if isinstance(base, ast.Name) and base.id in self.model_blocks:
                return True

        return False

    #
    def visit_Import(self, node: ast.Import) -> None:
        """
        Process import statements to track available modules
        """
        for name in node.names:
            self.imports[name.asname or name.name] = name.name
        self.generic_visit(node)

    #
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Process from-import statements to track available modules
        """
        for name in node.names:
            if node.module:
                self.imports[name.asname or name.name] = f"{node.module}.{name.name}"
            else:
                self.imports[name.asname or name.name] = name.name
        self.generic_visit(node)

    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Process class definitions to identify PyTorch modules
        """
        # Skip if not a torch.nn.Module subclass
        if not self.is_torch_module_class(node):
            self.generic_visit(node)
            return

        # Create a new model block for this class
        block_name: str = node.name
        self.current_model_visit = block_name

        if block_name in self.model_blocks:
            raise UserWarning(f"ERROR : There are two classes with the same name !!!\nBad name : {block_name}\n")

        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        # Process the class body
        self.generic_visit(node)


    #
    def process_layer_assignment(self, target: ast.AST, value: ast.AST) -> None:
        """
        Process a layer assignment statement in __init__
        """
        # We're only interested in self.layer_name = Layer(...) patterns
        if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name) or target.value.id != "self":
            return

        # Check if assignment is for a layer/module
        if not isinstance(value, ast.Call):
            return

        layer_var_name = target.attr

        # Get layer type
        if isinstance(value.func, ast.Name):
            layer_type = value.func.id
        elif isinstance(value.func, ast.Attribute):
            layer_type = f"{get_node_value(value.func.value)}.{value.func.attr}"
        else:
            layer_type = str(value.func)

        # Get parameters
        layer_parameters = {}
        for arg in value.args:
            # We'll represent positional args with numeric keys
            arg_idx = len(layer_parameters)
            layer_parameters[f"arg_{arg_idx}"] = get_node_value(arg)

        for kw in value.keywords:
            layer_parameters[kw.arg] = get_node_value(kw.value)

        # Create layer object and add to the current model block
        layer = lc.Layer(
            layer_var_name=layer_var_name,
            layer_type=layer_type,
            layer_parameters_kwargs=layer_parameters
        )

        self.model_blocks[self.current_model_visit].block_layers[layer_var_name] = layer

    #
    def process_init_body(self, node_body: List[ast.stmt]) -> None:
        """
        Process the body of an __init__ method
        """
        for item in node_body:
            # Handle simple assignments (self.layer = nn.Layer(...))
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    self.process_layer_assignment(target, item.value)

            # Handle parameter declarations with annotations (self.param: Type = value)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Attribute) and isinstance(item.target.value, ast.Name) and item.target.value.id == "self":
                param_name = item.target.attr
                param_type = get_variable_type_from_annotation(item.annotation)
                param_value = get_node_value(item.value) if item.value else None

                self.model_blocks[self.current_model_visit].block_parameters[param_name] = (param_type, param_value)

            # Handle more complex cases like loops that create layers
            elif isinstance(item, ast.For) or isinstance(item, ast.While):
                # This is a simplified approach - ideally we'd track loop variables and fully simulate execution
                self.generic_visit(item)

            # Handle conditionals that create layers
            elif isinstance(item, ast.If):
                # For conditional layer creation, we visit both branches
                self.generic_visit(item)

    #
    def process_forward_args(self, args: List[ast.arg]) -> None:
        """
        Process forward method arguments
        """
        for i, arg in enumerate(args):
            # Skip 'self' parameter
            if i == 0 and arg.arg == "self":
                continue

            arg_name = arg.arg
            arg_type = get_variable_type_from_annotation(arg.annotation) if arg.annotation else "Any"

            # Default value is None for simplicity - can't get this from the AST directly
            self.model_blocks[self.current_model_visit].forward_arguments[arg_name] = (arg_type, None)

    #
    def process_forward_body(self, node_body: List[ast.stmt]) -> None:
        """
        Process the body of a forward method
        """

        #
        flow_instr: lc.FlowControlInstruction

        #
        for item in node_body:
            # Handle variable assignment
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id

                        # If it's a layer call: x = self.layer(y)
                        if (isinstance(item.value, ast.Call) and
                            isinstance(item.value.func, ast.Attribute) and
                            isinstance(item.value.func.value, ast.Name) and
                            item.value.func.value.id == "self"):

                            layer_name = item.value.func.attr
                            args = {}

                            # Get positional args
                            for i, arg in enumerate(item.value.args):
                                args[f"arg_{i}"] = get_node_value(arg)

                            # Get keyword args
                            for kw in item.value.keywords:
                                args[kw.arg] = get_node_value(kw.value)

                            flow_instr = lc.FlowControlLayerPass(
                                output_variables=[var_name],
                                layer_name=layer_name,
                                layer_arguments=args
                            )
                            self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instr)

                        # If it's a function call: x = function(y)
                        elif isinstance(item.value, ast.Call):
                            func_name = get_node_value(item.value.func)
                            args = {}

                            # Get positional args
                            for i, arg in enumerate(item.value.args):
                                args[f"arg_{i}"] = get_node_value(arg)

                            # Get keyword args
                            for kw in item.value.keywords:
                                args[kw.arg] = get_node_value(kw.value)

                            flow_instr = lc.FlowControlFunctionCall(
                                output_variables=[var_name],
                                function_called=func_name,
                                function_arguments=args
                            )
                            self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instr)

                        # Simple variable assignment
                        else:
                            flow_instr = lc.FlowControlAssignment(
                                target=var_name,
                                expression=get_node_value(item.value)
                            )
                            self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instr)

            # Handle variable declaration with type annotation
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                var_name = item.target.id
                var_type = get_variable_type_from_annotation(item.annotation)
                var_value = get_node_value(item.value) if item.value else None

                flow_instr = lc.FlowControlVariableInit(
                    var_name=var_name,
                    var_type=var_type,
                    var_value=var_value
                )
                self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instr)

            # Handle return statements
            elif isinstance(item, ast.Return):
                if item.value is None:
                    return_vars = []
                elif isinstance(item.value, ast.Name):
                    return_vars = [item.value.id]
                elif isinstance(item.value, ast.Tuple):
                    return_vars = [get_node_value(elt) for elt in item.value.elts]
                else:
                    # For complex expressions, use a placeholder
                    return_vars = [f"<expression: {get_node_value(item.value)}>"]

                flow_instr = lc.FlowControlReturn(
                    return_variables=return_vars
                )
                self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instr)

            # Handle if statements (simplified)
            elif isinstance(item, ast.If):
                condition = get_node_value(item.test)

                # Process the true branch
                true_instructions = []
                for stmt in item.body:
                    self.process_forward_stmt(stmt, true_instructions)

                # Process the false branch if it exists
                false_instructions = None
                if item.orelse:
                    false_instructions = []
                    for stmt in item.orelse:
                        self.process_forward_stmt(stmt, false_instructions)

                flow_instr = lc.FlowControlIf(
                    condition=condition,
                    true_block=true_instructions,
                    false_block=false_instructions
                )
                self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instr)

            # Handle for loops (simplified)
            elif isinstance(item, ast.For):
                iterator = get_node_value(item.target)
                iterable = get_node_value(item.iter)

                # Process loop body
                body_instructions = []
                for stmt in item.body:
                    self.process_forward_stmt(stmt, body_instructions)

                flow_instr = lc.FlowControlLoop(
                    loop_type="for",
                    iterator=iterator,
                    iterable=iterable,
                    body=body_instructions
                )
                self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instr)

            # Handle while loops (simplified)
            elif isinstance(item, ast.While):
                condition = get_node_value(item.test)

                # Process loop body
                body_instructions = []
                for stmt in item.body:
                    self.process_forward_stmt(stmt, body_instructions)

                flow_instr = lc.FlowControlLoop(
                    loop_type="while",
                    iterator="",  # No iterator for while loops
                    iterable=condition,
                    body=body_instructions
                )
                self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instr)

    #
    def process_forward_stmt(self, stmt: ast.stmt, instructions: List[lc.FlowControlInstruction]) -> None:
        """
        Process a single statement in the forward method and add corresponding instructions
        """

        #
        flow_instr: lc.FlowControlInstruction

        # Handle variable assignment
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id

                    # If it's a layer call: x = self.layer(y)
                    if (isinstance(stmt.value, ast.Call) and
                        isinstance(stmt.value.func, ast.Attribute) and
                        isinstance(stmt.value.func.value, ast.Name) and
                        stmt.value.func.value.id == "self"):

                        layer_name = stmt.value.func.attr
                        args = {}

                        for i, arg in enumerate(stmt.value.args):
                            args[f"arg_{i}"] = get_node_value(arg)

                        for kw in stmt.value.keywords:
                            args[kw.arg] = get_node_value(kw.value)

                        flow_instr = lc.FlowControlLayerPass(
                            output_variables=[var_name],
                            layer_name=layer_name,
                            layer_arguments=args
                        )
                        instructions.append(flow_instr)

                    # If it's a function call: x = function(y)
                    elif isinstance(stmt.value, ast.Call):
                        func_name = get_node_value(stmt.value.func)
                        args = {}

                        for i, arg in enumerate(stmt.value.args):
                            args[f"arg_{i}"] = get_node_value(arg)

                        for kw in stmt.value.keywords:
                            args[kw.arg] = get_node_value(kw.value)

                        flow_instr = lc.FlowControlFunctionCall(
                            output_variables=[var_name],
                            function_called=func_name,
                            function_arguments=args
                        )

                        # If it's a function call: x = function(y)
                    elif isinstance(stmt.value, ast.Call):
                        func_name = get_node_value(stmt.value.func)
                        args = {}

                        for i, arg in enumerate(stmt.value.args):
                            args[f"arg_{i}"] = get_node_value(arg)

                        for kw in stmt.value.keywords:
                            args[kw.arg] = get_node_value(kw.value)

                        flow_instr = lc.FlowControlFunctionCall(
                            output_variables=[var_name],
                            function_called=func_name,
                            function_arguments=args
                        )
                        instructions.append(flow_instr)

                    # Simple variable assignment
                    else:
                        flow_instr = lc.FlowControlAssignment(
                            target=var_name,
                            expression=get_node_value(stmt.value)
                        )
                        instructions.append(flow_instr)

        # Handle variable declaration with type annotation
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            var_name = stmt.target.id
            var_type = get_variable_type_from_annotation(stmt.annotation)
            var_value = get_node_value(stmt.value) if stmt.value else None

            flow_instr = lc.FlowControlVariableInit(
                var_name=var_name,
                var_type=var_type,
                var_value=var_value
            )
            instructions.append(flow_instr)

        # Handle return statements
        elif isinstance(stmt, ast.Return):
            if stmt.value is None:
                return_vars = []
            elif isinstance(stmt.value, ast.Name):
                return_vars = [stmt.value.id]
            elif isinstance(stmt.value, ast.Tuple):
                return_vars = [get_node_value(elt) for elt in stmt.value.elts]
            else:
                # For complex expressions, use a placeholder
                return_vars = [f"<expression: {get_node_value(stmt.value)}>"]

            flow_instr = lc.FlowControlReturn(
                return_variables=return_vars
            )
            instructions.append(flow_instr)

        # Handle nested control flow (skip for simplicity in nested blocks)
        else:
            # This is a simplified approach - ideally we'd handle nesting properly
            pass

    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Process function definitions within a model class
        """
        # Skip if not inside a model block
        if not self.current_model_visit:
            self.generic_visit(node)
            return

        # Set current function context
        self.current_function = node.name

        # Special handling for __init__ method
        if node.name == "__init__":
            self.process_init_body(node.body)

        # Special handling for forward method
        elif node.name == "forward":
            # Process arguments first
            self.process_forward_args(node.args.args)

            # Process function body
            self.process_forward_body(node.body)

        # Other methods are stored as BlockFunctions
        else:
            # Create function arguments dict
            args_dict = {}
            for i, arg in enumerate(node.args.args):
                # Skip 'self' parameter
                if i == 0 and arg.arg == "self":
                    continue

                arg_name = arg.arg
                arg_type = get_variable_type_from_annotation(arg.annotation) if arg.annotation else "Any"

                # Get default value if available
                default_idx = i - (len(node.args.args) - len(node.args.defaults))
                default_value = None
                if default_idx >= 0:
                    default_value = get_node_value(node.args.defaults[default_idx])

                args_dict[arg_name] = (arg_type, default_value)

            # Create block function
            func = lc.BlockFunction(
                function_name=node.name,
                function_arguments=args_dict,
                model_block=self.model_blocks[self.current_model_visit]
            )

            # Add to model block
            self.model_blocks[self.current_model_visit].block_functions[node.name] = func

        # Reset function context
        self.current_function = None
        self.generic_visit(node)

    #
    def visit_Expr(self, node: ast.Expr) -> None:
        """
        Process expression statements (e.g., docstrings)
        """
        # Skip if not inside a model block
        if not self.current_model_visit:
            self.generic_visit(node)
            return

        # Handle docstrings
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # You might want to store docstrings somewhere in your model representation
            pass

        self.generic_visit(node)

    #
    def generic_visit(self, node: ast.AST) -> None:
        """
        Generic visit method that gets called for all nodes
        """
        ast.NodeVisitor.generic_visit(self, node)


#
if __name__ == "__main__":

    #
    if len(sys.argv) != 2:
        raise UserWarning(f"Error: if you use this script directly, you should use it like that :\n  python {sys.argv[0]} path_to_model_script.py")

    #
    path_to_file: str = sys.argv[1]

    #
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"Error: file not found : `{path_to_file}`")

    # Parse the source code into an AST
    with open(path_to_file, "r") as source:
        tree = ast.parse(source.read())

    # Analyze the AST
    analyzer = ModelAnalyzer()
    analyzer.visit(tree)

    # Print the discovered model blocks
    print("\n" * 2)
    for block_name, block in analyzer.model_blocks.items():
        print(f"=== Model Block: {block_name} ===")
        print(block)
        print("\n" + "=" * 80 + "\n")