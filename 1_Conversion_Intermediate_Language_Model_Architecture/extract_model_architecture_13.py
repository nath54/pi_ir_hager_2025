#
import ast
import sys
import os
from typing import List, Dict, Any, Optional, Union, Tuple
import re

#
import lib_classes_5 as lc


#
def ast_node_to_str(node: ast.AST) -> str:
    """
    Convert an AST node to its string representation.
    This helps in cases where we need to represent conditions or expressions.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{ast_node_to_str(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        return f"{ast_node_to_str(node.value)}[{ast_node_to_str(node.slice)}]"
    elif isinstance(node, ast.Index):
        return ast_node_to_str(node.value)
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Slice):
        lower = ast_node_to_str(node.lower) if node.lower else ""
        upper = ast_node_to_str(node.upper) if node.upper else ""
        step = f":{ast_node_to_str(node.step)}" if node.step else ""
        return f"{lower}:{upper}{step}"
    elif isinstance(node, ast.BinOp):
        return f"{ast_node_to_str(node.left)} {get_binop_symbol(node.op)} {ast_node_to_str(node.right)}"
    elif isinstance(node, ast.Compare):
        left = ast_node_to_str(node.left)
        comparisons = []
        for op, right in zip(node.ops, node.comparators):
            comparisons.append(f"{get_cmpop_symbol(op)} {ast_node_to_str(right)}")
        return f"{left} {' '.join(comparisons)}"
    elif isinstance(node, ast.Call):
        func = ast_node_to_str(node.func)
        args = [ast_node_to_str(arg) for arg in node.args]
        kwargs = [f"{kw.arg}={ast_node_to_str(kw.value)}" for kw in node.keywords]
        all_args = args + kwargs
        return f"{func}({', '.join(all_args)})"
    elif isinstance(node, ast.List):
        return f"[{', '.join(ast_node_to_str(elem) for elem in node.elts)}]"
    elif isinstance(node, ast.Tuple):
        return f"({', '.join(ast_node_to_str(elem) for elem in node.elts)})"
    elif isinstance(node, ast.Dict):
        keys = [ast_node_to_str(k) for k in node.keys]
        values = [ast_node_to_str(v) for v in node.values]
        return f"{{{', '.join(f'{k}: {v}' for k, v in zip(keys, values))}}}"
    else:
        return str(node)


#
def get_binop_symbol(op: ast.operator) -> str:
    """Get the symbol for a binary operation."""
    if isinstance(op, ast.Add):
        return "+"
    elif isinstance(op, ast.Sub):
        return "-"
    elif isinstance(op, ast.Mult):
        return "*"
    elif isinstance(op, ast.Div):
        return "/"
    elif isinstance(op, ast.FloorDiv):
        return "//"
    elif isinstance(op, ast.Mod):
        return "%"
    else:
        return str(op)


#
def get_cmpop_symbol(op: ast.cmpop) -> str:
    """Get the symbol for a comparison operation."""
    if isinstance(op, ast.Eq):
        return "=="
    elif isinstance(op, ast.NotEq):
        return "!="
    elif isinstance(op, ast.Lt):
        return "<"
    elif isinstance(op, ast.LtE):
        return "<="
    elif isinstance(op, ast.Gt):
        return ">"
    elif isinstance(op, ast.GtE):
        return ">="
    elif isinstance(op, ast.Is):
        return "is"
    elif isinstance(op, ast.IsNot):
        return "is not"
    elif isinstance(op, ast.In):
        return "in"
    elif isinstance(op, ast.NotIn):
        return "not in"
    else:
        return str(op)


#
def extract_type_from_annotation(annotation: Optional[ast.AST]) -> str:
    """
    Extract type information from an AST annotation node.
    This helps in determining variable types from annotations in function arguments and variable assignments.
    """
    if annotation is None:
        return "Any"

    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        return f"{ast_node_to_str(annotation.value)}.{annotation.attr}"
    elif isinstance(annotation, ast.Subscript):
        value = ast_node_to_str(annotation.value)
        slice_str = ast_node_to_str(annotation.slice)
        return f"{value}[{slice_str}]"
    elif isinstance(annotation, ast.Constant):
        return str(annotation.value)
    else:
        return ast_node_to_str(annotation)


#
def is_nn_layer_constructor(node: ast.Call) -> bool:
    """
    Check if a node represents a PyTorch nn layer constructor.
    This helps identify layer instantiations in the __init__ method.
    """
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "nn":
            return True
        if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "nn":
            return True
    return False


#
def extract_args_from_call(node: ast.Call) -> Dict[str, Any]:
    """
    Extract arguments from a function call node.
    This helps capture the parameters passed to layer constructors.
    """
    args_dict = {}

    # Handle positional arguments
    for i, arg in enumerate(node.args):
        args_dict[f"arg{i}"] = ast_node_to_str(arg)

    # Handle keyword arguments
    for keyword in node.keywords:
        args_dict[keyword.arg] = ast_node_to_str(keyword.value)

    return args_dict


#
def process_sequential_call(node: ast.Call, var_name: str) -> lc.SequentialLayer:
    """
    Process a nn.Sequential constructor call to extract the contained layers.
    This helps handle nested layer structures like Sequential.
    """
    layers = []
    for i, arg in enumerate(node.args):
        if isinstance(arg, ast.Call) and is_nn_layer_constructor(arg):
            layer_type = ast_node_to_str(arg.func)
            layer_args = extract_args_from_call(arg)
            layer = lc.Layer(f"{var_name}_{i}", layer_type, layer_args)
            layers.append(layer)

    return lc.SequentialLayer(var_name, layers)


#
def process_modulelist_call(node: ast.Call, var_name: str) -> lc.ModuleListLayer:
    """
    Process a nn.ModuleList constructor call to extract the contained layers.
    This helps handle collections of layers like ModuleList.
    """
    layers = []

    # Handle the case where ModuleList is initialized with a list
    if node.args and isinstance(node.args[0], ast.List):
        for i, elt in enumerate(node.args[0].elts):
            if isinstance(elt, ast.Call) and is_nn_layer_constructor(elt):
                layer_type = ast_node_to_str(elt.func)
                layer_args = extract_args_from_call(elt)
                layer = lc.Layer(f"{var_name}_{i}", layer_type, layer_args)
                layers.append(layer)

    return lc.ModuleListLayer(var_name, layers)


#
def process_for_loop_modulelist_init(node: ast.For, var_name: str, target_list: str, model_block: lc.ModelBlock) -> Optional[lc.ModuleListLayer]:
    """
    Process a for loop that initializes items in a ModuleList.
    This helps capture dynamically created layers in loops.
    """
    if not isinstance(node.body, list) or len(node.body) == 0:
        return None

    # Check if the loop body contains append calls to the target list
    layers = []
    for stmt in node.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "append" and
                isinstance(call.func.value, ast.Name) and call.func.value.id == target_list):

                # This is an append call to our target list
                if call.args and isinstance(call.args[0], ast.Call) and is_nn_layer_constructor(call.args[0]):
                    layer_call = call.args[0]
                    layer_type = ast_node_to_str(layer_call.func)
                    layer_args = extract_args_from_call(layer_call)

                    # Use a placeholder name since the actual name depends on loop iteration
                    layer = lc.Layer(f"{var_name}_item", layer_type, layer_args)
                    layers.append(layer)

    if layers:
        return lc.ModuleListLayer(var_name, layers)

    return None


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
        #
        self.current_function: Optional[str] = None
        #
        self.parent_classes: Dict[str, List[str]] = {}  # Map class names to their parent classes

    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Store parent class information for all classes
        self.parent_classes[node.name] = [ast_node_to_str(base) for base in node.bases]

        #
        is_module_class: bool = self.is_nn_module_class(node)

        #
        if not is_module_class:
            # Visit class body anyway for potential nested nn.Module classes
            self.generic_visit(node)
            return

        #
        block_name: str = node.name
        self.current_model_visit = block_name
        #
        if block_name in self.model_blocks:
            raise UserWarning(f"ERROR : There are two classes with the same name !!!\nBad name : {block_name}\n")

        #
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        # Set parent classes
        self.model_blocks[block_name].parent_classes = self.parent_classes[block_name]

        #
        self.generic_visit(node)

    #
    def is_nn_module_class(self, node: ast.ClassDef) -> bool:
        """
        Check if a class definition inherits from torch.nn.Module.
        This simplifies the main visit_ClassDef method by extracting the inheritance check.
        """
        for base in node.bases:
            # Check direct inheritance from nn.Module
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True

            # Check inheritance from torch.nn.Module
            if (isinstance(base, ast.Attribute) and base.attr == "Module" and
                isinstance(base.value, ast.Attribute) and base.value.attr == "nn" and
                isinstance(base.value.value, ast.Name) and base.value.value.id == "torch"):
                return True

            # Check inheritance from another class that might inherit from nn.Module
            if isinstance(base, ast.Name) and base.id in self.parent_classes:
                # Check recursively through parent hierarchy
                for parent in self.parent_classes[base.id]:
                    if "nn.Module" in parent or "torch.nn.Module" in parent:
                        return True

        return False

    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Skip if we're not in a model context
        if not self.current_model_visit:
            self.generic_visit(node)
            return

        # Set current function context
        prev_function = self.current_function
        self.current_function = node.name

        # Get the current model block
        model_block = self.model_blocks[self.current_model_visit]

        # Process based on function type
        if node.name == "__init__":
            # Extract function arguments
            args = {}
            for arg in node.args.args:
                if arg.arg != "self":  # Skip self parameter
                    arg_type = extract_type_from_annotation(arg.annotation)
                    default_value = None  # No default by default
                    args[arg.arg] = (arg_type, default_value)

            # Store parameters in the model block
            model_block.block_parameters = args

            # Analyze __init__ method to extract layer definitions
            self.analyze_init_method(node, model_block)

        elif node.name == "forward":
            # Extract function arguments
            args = {}
            for arg in node.args.args:
                if arg.arg != "self":  # Skip self parameter
                    arg_type = extract_type_from_annotation(arg.annotation)
                    default_value = None  # No default by default
                    args[arg.arg] = (arg_type, default_value)

            # Create a BlockFunction for forward
            forward_function = lc.BlockFunction("forward", args, model_block)

            # Analyze the forward method to extract flow control
            self.analyze_forward_method(node, forward_function)

            # Store the return type if available
            if node.returns:
                forward_function.return_type = extract_type_from_annotation(node.returns)

            # Add the forward function to the model block
            model_block.block_functions["forward"] = forward_function

        else:
            # Handle other functions (helper methods, etc.)
            args = {}
            for arg in node.args.args:
                if arg.arg != "self":  # Skip self parameter
                    arg_type = extract_type_from_annotation(arg.annotation)
                    default_value = None  # No default by default
                    args[arg.arg] = (arg_type, default_value)

            # Create a BlockFunction for this function
            helper_function = lc.BlockFunction(node.name, args, model_block)

            # Analyze the function body to extract flow control
            self.analyze_function_body(node, helper_function)

            # Store the return type if available
            if node.returns:
                helper_function.return_type = extract_type_from_annotation(node.returns)

            # Add the helper function to the model block
            model_block.block_functions[node.name] = helper_function

        # Restore previous function context
        self.current_function = prev_function

        # Visit function body
        self.generic_visit(node)

    #
    def analyze_init_method(self, node: ast.FunctionDef, model_block: lc.ModelBlock) -> None:
        """
        Analyze the __init__ method to extract layer definitions.
        This helps identify all the layers and their parameters defined in the model.
        """
        for item in node.body:
            # Check for layer assignments
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    # Check for layer assignments starting with self.attribute
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        attr_name = target.attr  # This is the layer variable name

                        # Check if this is a layer instantiation
                        if isinstance(item.value, ast.Call):
                            # Directly creating a layer
                            if is_nn_layer_constructor(item.value):
                                layer_type = ast_node_to_str(item.value.func)
                                layer_args = extract_args_from_call(item.value)
                                layer = lc.Layer(attr_name, layer_type, layer_args)
                                model_block.block_layers[attr_name] = layer

                            # Handle nn.Sequential special case
                            elif (isinstance(item.value.func, ast.Attribute) and
                                 isinstance(item.value.func.value, ast.Name) and
                                 item.value.func.value.id == "nn" and
                                 item.value.func.attr == "Sequential"):
                                seq_layer = process_sequential_call(item.value, attr_name)
                                model_block.block_layers[attr_name] = seq_layer

                            # Handle nn.ModuleList special case
                            elif (isinstance(item.value.func, ast.Attribute) and
                                 isinstance(item.value.func.value, ast.Name) and
                                 item.value.func.value.id == "nn" and
                                 item.value.func.attr == "ModuleList"):
                                module_list_layer = process_modulelist_call(item.value, attr_name)
                                model_block.block_layers[attr_name] = module_list_layer

                        # Store variable assignment even if it's not a layer
                        elif not isinstance(item.value, ast.Call) or not is_nn_layer_constructor(item.value):
                            var_type = "Any"  # Default type
                            var_value = ast_node_to_str(item.value)
                            model_block.block_variables[attr_name] = (var_type, var_value)

            # Check for ModuleList initialization via for loops
            elif isinstance(item, ast.For):
                # Look for iteration over a range to add layers to a ModuleList
                if isinstance(item.iter, ast.Call) and isinstance(item.iter.func, ast.Name) and item.iter.func.id == "range":
                    # Check if the body contains append to a ModuleList
                    for stmt in item.body:
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                            call = stmt.value
                            if (isinstance(call.func, ast.Attribute) and call.func.attr == "append" and
                                isinstance(call.func.value, ast.Attribute) and isinstance(call.func.value.value, ast.Name) and
                                call.func.value.value.id == "self"):

                                target_list = call.func.value.attr
                                # Check if we already have this as a ModuleList
                                if target_list in model_block.block_layers and isinstance(model_block.block_layers[target_list], lc.ModuleListLayer):
                                    # Process the for loop to extract layers being added
                                    module_list = process_for_loop_modulelist_init(item, target_list, target_list, model_block)
                                    if module_list and module_list.layers_list:
                                        # Add the extracted layers to the existing ModuleList
                                        existing_layers = model_block.block_layers[target_list].layers_list
                                        model_block.block_layers[target_list].layers_list = existing_layers + module_list.layers_list

    #
    def analyze_forward_method(self, node: ast.FunctionDef, forward_function: lc.BlockFunction) -> None:
        """
        Analyze the forward method to extract flow control instructions.
        This captures the computational flow through layers during forward pass.
        """
        self.analyze_function_body(node, forward_function)

    #
    def analyze_function_body(self, node: ast.FunctionDef, function: lc.BlockFunction) -> None:
        """
        Generic function to analyze any function body and extract flow control.
        This is used for forward and helper methods.
        """
        for item in node.body:
            flow_instruction = self.process_statement(item)
            if flow_instruction:
                function.function_flow_control.append(flow_instruction)

    #
    def process_statement(self, node: ast.AST) -> Optional[lc.FlowControlInstruction]:
        """
        Process a statement node and convert it to a FlowControlInstruction.
        This handles different types of statements like assignments, function calls, etc.
        """
        # Handle variable assignments
        if isinstance(node, ast.Assign):
            # Extract target and value
            if len(node.targets) == 1:
                target = ast_node_to_str(node.targets[0])
                value = ast_node_to_str(node.value)

                # Check if it's a layer pass
                if (isinstance(node.value, ast.Call) and
                    isinstance(node.value.func, ast.Attribute) and
                    isinstance(node.value.func.value, ast.Name) and
                    node.value.func.value.id == "self"):
                    layer_name = node.value.func.attr
                    arguments = extract_args_from_call(node.value)
                    return lc.FlowControlLayerPass([target], layer_name, arguments)

                # Check if it's a function call
                elif isinstance(node.value, ast.Call) and not is_nn_layer_constructor(node.value):
                    function_called = ast_node_to_str(node.value.func)
                    arguments = extract_args_from_call(node.value)
                    return lc.FlowControlFunctionCall([target], function_called, arguments)

                # Regular assignment
                else:
                    return lc.FlowControlAssignment(target, value)

        # Handle return statements
        elif isinstance(node, ast.Return):
            if node.value:
                # If returning a value or expression
                return_vars = [ast_node_to_str(node.value)]
                return lc.FlowControlReturn(return_vars)
            else:
                # Empty return
                return lc.FlowControlReturn([])

        # Handle if statements
        elif isinstance(node, ast.If):
            condition = ast_node_to_str(node.test)
            true_instructions = []
            false_instructions = []

            # Process true branch
            for stmt in node.body:
                instr = self.process_statement(stmt)
                if instr:
                    true_instructions.append(instr)

            # Process false branch if it exists
            if node.orelse:
                for stmt in node.orelse:
                    instr = self.process_statement(stmt)
                    if instr:
                        false_instructions.append(instr)

            return lc.FlowControlIf(condition, true_instructions, false_instructions if false_instructions else None)

        # Handle for loops
        elif isinstance(node, ast.For):
            target = ast_node_to_str(node.target)
            iterable = ast_node_to_str(node.iter)
            loop_body = []

            # Process loop body
            for stmt in node.body:
                instr = self.process_statement(stmt)
                if instr:
                    loop_body.append(instr)

            return lc.FlowControlForLoop(target, iterable, loop_body)

        # Handle while loops
        elif isinstance(node, ast.While):
            condition = ast_node_to_str(node.test)
            loop_body = []

            # Process loop body
            for stmt in node.body:
                instr = self.process_statement(stmt)
                if instr:
                    loop_body.append(instr)

            return lc.FlowControlWhileLoop(condition, loop_body)

        # Handle expression statements (like function calls without assignment)
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                # Extract function call information
                function_called = ast_node_to_str(node.value.func)
                arguments = extract_args_from_call(node.value)

                # Check if it's a call to a layer
                if (isinstance(node.value.func, ast.Attribute) and
                    isinstance(node.value.func.value, ast.Name) and
                    node.value.func.value.id == "self"):
                    return lc.FlowControlLayerPass([], node.value.func.attr, arguments)
                else:
                    return lc.FlowControlFunctionCall([], function_called, arguments)

        return None

    #
    def generic_visit(self, node: ast.AST):
        #
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

    #
    print("\n" * 2)
    for block_name, block in analyzer.model_blocks.items():
        print(block)