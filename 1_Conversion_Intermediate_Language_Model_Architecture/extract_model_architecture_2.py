#
import ast
import sys
import os
from typing import Any, Dict, List, Optional

#
import lib_classes as lc

# FOR DEBUGGING
sys.path.insert(0, "../debug/")
#
from lib_debug import debug_var  # type: ignore


#
def get_node_repr(node: ast.AST) -> str:
    """
    Return a string representation of an AST node.
    Uses ast.unparse if available.
    """
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


#
def extract_call_arguments(call_node: ast.Call) -> Dict[str, Any]:
    """
    Extracts positional and keyword arguments from a call node.
    Returns a dictionary with keys 'positional' (a list) and 'keywords' (a dict).
    """
    args_list: List[str] = [get_node_repr(arg) for arg in call_node.args]
    kwargs_dict: Dict[str, str] = {}
    for kw in call_node.keywords:
        # kw.arg might be None for **kwargs; here we simply represent it as such.
        key = kw.arg if kw.arg is not None else "**"
        kwargs_dict[key] = get_node_repr(kw.value)
    return {"positional": args_list, "keywords": kwargs_dict}


#
class ModelAnalyzer(ast.NodeVisitor):
    #
    def __init__(self) -> None:
        #
        self.model_blocks: Dict[str, lc.ModelBlock] = {}
        #
        self.main_block: str = ""
        #
        self.current_model_visit: str = ""
        # Flag to indicate the current method ('__init__' or 'forward')
        self.current_method: str = ""

    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        #
        is_module_class: bool = False

        # Check if the class inherits from torch.nn.Module
        for base in node.bases:
            if (
                isinstance(base, ast.Attribute)
                and base.attr == "Module"
                and isinstance(base.value, ast.Name)
                and base.value.id == "nn"
            ):
                is_module_class = True
                break

        #
        if not is_module_class:
            return

        #
        block_name: str = node.name
        self.current_model_visit = block_name
        #
        if block_name in self.model_blocks:
            raise UserWarning(
                f"ERROR: There are two classes with the same name !!!\nBad name: {block_name}\n"
            )

        #
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        #
        self.generic_visit(node)

    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        #
        if node.name in ("__init__", "forward"):
            self.current_method = node.name
            # Process the body of __init__ or forward using visit on child nodes
            self.generic_visit(node)
            self.current_method = ""
        else:
            self.generic_visit(node)

    #
    def visit_Assign(self, node: ast.Assign) -> Any:
        #
        # Process assignments differently based on the current method context
        if self.current_method == "__init__":
            # Handle assignments in __init__: these can be either layer definitions or parameters
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    var_name: str = target.attr
                    # If the value is a call, check if it is a layer definition (e.g., nn.Conv2d(...))
                    if isinstance(node.value, ast.Call):
                        call_node: ast.Call = node.value
                        if (
                            isinstance(call_node.func, ast.Attribute)
                            and isinstance(call_node.func.value, ast.Name)
                            and call_node.func.value.id == "nn"
                        ):
                            # Extract call details and create a Layer
                            layer_type: str = call_node.func.attr
                            call_args: Dict[str, Any] = extract_call_arguments(call_node)
                            parameters_kwargs: Dict[str, Any] = {
                                "layer_type": layer_type,
                                "call_arguments": call_args,
                            }
                            self.model_blocks[self.current_model_visit].block_layers[var_name] = lc.Layer(
                                layer_name=var_name, layer_parameters_kwargs=parameters_kwargs
                            )
                        else:
                            # Not a layer call; treat as a parameter initialization
                            var_type: str = type(node.value).__name__
                            var_value: str = get_node_repr(node.value)
                            self.model_blocks[self.current_model_visit].block_parameters[var_name] = (var_type, var_value)
                    else:
                        # Simple assignment (not a call) treated as a parameter initialization
                        var_type = type(node.value).__name__
                        var_value = get_node_repr(node.value)
                        self.model_blocks[self.current_model_visit].block_parameters[var_name] = (var_type, var_value)
        elif self.current_method == "forward":

            # Process assignments in forward: these can represent variable initializations, layer calls, or function calls.
            output_vars: List[str] = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    output_vars.append(target.id)
                elif isinstance(target, ast.Tuple):
                    # Multiple assignment: extract all variable names
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            output_vars.append(elt.id)
            #
            instruction: lc.FlowControlInstruction

            # If the right-hand side is a call, further distinguish between layer passes and other function calls.
            if isinstance(node.value, ast.Call):
                call_node = node.value
                if (
                    isinstance(call_node.func, ast.Attribute)
                    and isinstance(call_node.func.value, ast.Name)
                    and call_node.func.value.id == "self"
                ):
                    # This is a layer call (e.g., x = self.conv1(x))
                    layer_name: str = call_node.func.attr
                    call_args = extract_call_arguments(call_node)
                    instruction = lc.FlowControlLayerPass(
                        output_variables=output_vars, layer_name=layer_name, layer_arguments=call_args
                    )
                    self.model_blocks[self.current_model_visit].forward_flow_control.append(instruction)
                else:
                    # Regular function call (not a layer call)
                    function_called: str = get_node_repr(call_node.func)
                    call_args = extract_call_arguments(call_node)
                    instruction = lc.FlowControlFunctionCall(
                        output_variables=output_vars, function_called=function_called, function_arguments=call_args
                    )
                    self.model_blocks[self.current_model_visit].forward_flow_control.append(instruction)
            else:
                # Not a call; treat as variable initialization
                var_type = type(node.value).__name__
                var_value = get_node_repr(node.value)
                # For each variable initialized, record a variable initialization instruction
                for var in output_vars:
                    instruction = lc.FlowControlVariableInit(var_name=var, var_type=var_type, var_value=var_value)
                    self.model_blocks[self.current_model_visit].forward_flow_control.append(instruction)
        #
        return self.generic_visit(node)

    #
    def visit_Expr(self, node: ast.Expr) -> Any:
        #
        # In the forward method, expressions (e.g. standalone function calls) should be processed
        if self.current_method == "forward" and isinstance(node.value, ast.Call):
            call_node = node.value
            #
            instruction: lc.FlowControlInstruction
            #
            if (
                isinstance(call_node.func, ast.Attribute)
                and isinstance(call_node.func.value, ast.Name)
                and call_node.func.value.id == "self"
            ):
                # A layer pass with no explicit output assignment
                layer_name: str = call_node.func.attr
                call_args: Dict[str, Any] = extract_call_arguments(call_node)
                instruction = lc.FlowControlLayerPass(output_variables=[], layer_name=layer_name, layer_arguments=call_args)
                self.model_blocks[self.current_model_visit].forward_flow_control.append(instruction)
            else:
                # A generic function call with no output
                function_called: str = get_node_repr(call_node.func)
                call_args = extract_call_arguments(call_node)
                instruction = lc.FlowControlFunctionCall(output_variables=[], function_called=function_called, function_arguments=call_args)
                self.model_blocks[self.current_model_visit].forward_flow_control.append(instruction)
        #
        return self.generic_visit(node)

    #
    def generic_visit(self, node: ast.AST) -> Any:
        #
        # Additional checks for other node types can be implemented here if needed
        return super().generic_visit(node)


#
if __name__ == "__main__":
    #
    if len(sys.argv) != 2:
        raise UserWarning(
            f"Error: if you use this script directly, you should use it like this:\n  python {sys.argv[0]} path_to_model_script.py"
        )

    #
    path_to_file: str = sys.argv[1]

    #
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"Error: file not found: `{path_to_file}`")

    # Parse the source code into an AST
    with open(path_to_file, "r") as source:
        tree = ast.parse(source.read())

    # Analyze the AST
    analyzer = ModelAnalyzer()
    analyzer.visit(tree)

    #
    print("\n" * 2)
    print(analyzer.model_blocks)
