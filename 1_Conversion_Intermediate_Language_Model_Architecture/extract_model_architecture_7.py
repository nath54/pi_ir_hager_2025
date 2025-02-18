#
import ast
import sys
import os
from typing import Any, Optional

#
import lib_classes as lc


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
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        #
        # Check if the class is a torch.nn.Module subclass
        if not self._is_torch_nn_module_subclass(node):
            return

        # Get the block name which is the class name
        block_name: str = node.name
        self.current_model_visit = block_name

        # Check for name duplication
        if block_name in self.model_blocks:
            raise UserWarning(f"ERROR : There are two classes with the same name !!!\nBad name : {block_name}\n")

        # Init the ModelBlock
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        # Continue to visit the body of the ClassDef
        self.generic_visit(node)

    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        #
        # Analyze the __init__ method
        if node.name == "__init__":
            self._analyze_init_method(node)

        # Analyze the forward method
        elif node.name == "forward":
            self._analyze_forward_method(node)

        # Continue to visit function def content
        self.generic_visit(node)

    #
    def _analyze_init_method(self, node: ast.FunctionDef) -> None:
        #
        # Iterate over each instruction in the __init__ method body
        for item in node.body:
            #
            # Analyze layer definition assignments
            if isinstance(item, ast.Assign):
                self._analyze_layer_definition(item)

    #
    def _analyze_forward_method(self, node: ast.FunctionDef) -> None:
        #
        # Iterate over each instruction in the forward method body
        for item in node.body:
            #
            # Analyze each type of instruction to extract flow control
            if isinstance(item, ast.Assign):
                self._analyze_forward_assign(item)
            elif isinstance(item, ast.Return):
                self._analyze_forward_return(item)
            # TODO: Add other flow control instructions if needed (If, For, While, etc.)

    #
    def _analyze_layer_definition(self, assign_node: ast.Assign) -> None:
        #
        # Check if it is an assignment to a `self.layer_name`
        if not (isinstance(assign_node.targets[0], ast.Attribute) and isinstance(assign_node.targets[0].value, ast.Name) and assign_node.targets[0].value.id == "self"):
            return

        # Get the layer variable name (e.g., conv1 in self.conv1 = nn.Conv2d(...))
        layer_var_name: str = assign_node.targets[0].attr

        # Check if the assigned value is a call (e.g., nn.Conv2d(...))
        if not isinstance(assign_node.value, ast.Call):
            return

        # Get the function call info
        call_node: ast.Call = assign_node.value

        # Extract layer type (e.g., Conv2d) and ensure it is from torch.nn
        layer_type: Optional[str] = self._extract_layer_type(call_node)
        if layer_type is None:
            return

        # Extract layer parameters
        layer_parameters_kwargs: dict[str, Any] = self._extract_layer_parameters(call_node)

        # Create Layer object and add it to the current ModelBlock
        layer = lc.Layer(layer_var_name=layer_var_name, layer_type=layer_type, layer_parameters_kwargs=layer_parameters_kwargs)
        self.model_blocks[self.current_model_visit].block_layers[layer_var_name] = layer

    #
    def _analyze_forward_assign(self, assign_node: ast.Assign) -> None:
        #
        # Extract output variable names (can be multiple if unpacking)
        output_variables: list[str] = self._extract_output_variables(assign_node.targets)

        # Check if the assigned value is a function call (layer pass or function call)
        if isinstance(assign_node.value, ast.Call):
            self._analyze_forward_call(assign_node.value, output_variables)
        # TODO: Handle other types of assignments in forward if needed (e.g., arithmetic operations, variable manipulations)

    #
    def _analyze_forward_call(self, call_node: ast.Call, output_variables: list[str]) -> None:
        #
        # Extract function or layer name called
        function_called_name: Optional[str] = self._extract_function_call_name(call_node.func)
        if function_called_name is None:
            return

        # Extract function arguments
        function_arguments: dict[str, Any] = self._extract_function_arguments(call_node)

        #
        flow_instruction: lc.FlowControlInstruction

        # Check if the function called is a layer defined in __init__
        if function_called_name in self.model_blocks[self.current_model_visit].block_layers:
            # Create FlowControlLayerPass instruction
            flow_instruction = lc.FlowControlLayerPass(
                output_variables=output_variables,
                layer_name=function_called_name,
                layer_arguments=function_arguments,
            )
        else:
            # Create FlowControlFunctionCall instruction
            flow_instruction = lc.FlowControlFunctionCall(
                output_variables=output_variables,
                function_called=function_called_name,
                function_arguments=function_arguments,
            )

        # Add the flow instruction to the current ModelBlock's forward flow control
        self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instruction)

    #
    def _analyze_forward_return(self, return_node: ast.Return) -> None:
        #
        # Extract returned variables
        return_variables: list[str] = self._extract_return_variables(return_node.value)

        # Create FlowControlReturn instruction
        flow_instruction = lc.FlowControlReturn(return_variables=return_variables)

        # Add the flow instruction to the current ModelBlock's forward flow control
        self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_instruction)

    #
    def _is_torch_nn_module_subclass(self, class_node: ast.ClassDef) -> bool:
        #
        # Check if the class inherits from torch.nn.Module
        for base in class_node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True
        return False

    #
    def _extract_layer_type(self, call_node: ast.Call) -> Optional[str]:
        #
        # Extract layer type from ast.Call node, ensure it is from torch.nn
        if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == "nn":
            return call_node.func.attr
        return None

    #
    def _extract_layer_parameters(self, call_node: ast.Call) -> dict[str, Any]:
        #
        # Extract layer parameters from ast.Call node arguments (keywords arguments)
        params: dict[str, Any] = {}
        for keyword in call_node.keywords:
            params[keyword.arg] = self._get_node_value(keyword.value) # Get value of the keyword
        return params

    #
    def _extract_function_call_name(self, func_node: ast.Call | ast.Attribute | ast.Name) -> Optional[str]:
        #
        # Extract function name from different types of function call nodes
        if isinstance(func_node, ast.Name):
            return func_node.id # for functions like F.relu, but actually should be Attribute
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr # for layer calls like self.conv1(x) or F.relu (after import torch.nn.functional as F)
        return None # Not supported function call type

    #
    def _extract_function_arguments(self, call_node: ast.Call) -> dict[str, Any]:
        #
        # Extract function arguments from ast.Call node arguments (positional and keywords)
        args: dict[str, Any] = {}

        # Handle positional arguments (not named in function definition, so give index arg_0, arg_1, ...)
        for index, arg_value in enumerate(call_node.args):
            args[f"arg_{index}"] = self._get_node_value(arg_value)

        # Handle keyword arguments
        for keyword in call_node.keywords:
            args[keyword.arg] = self._get_node_value(keyword.value) # Get value of the keyword

        return args

    #
    def _extract_output_variables(self, targets_nodes: list[ast.AST]) -> list[str]:
        #
        # Extract output variable names from assignment targets (can be multiple if unpacking)
        output_vars: list[str] = []
        for target_node in targets_nodes:
            if isinstance(target_node, ast.Name):
                output_vars.append(target_node.id) # simple variable like x = ...
            elif isinstance(target_node, ast.Tuple) or isinstance(target_node, ast.List):
                for elt in target_node.elts:
                    if isinstance(elt, ast.Name):
                        output_vars.append(elt.id) # For tuple/list unpacking like x, y = ... or [x, y] = ...
            # TODO: Handle other types of assignment targets if needed
        return output_vars

    #
    def _extract_return_variables(self, return_value_node: ast.expr) -> list[str]:
        #
        # Extract returned variables from ast.Return node value
        return_vars: list[str] = []
        if isinstance(return_value_node, ast.Name):
            return_vars.append(return_value_node.id) # return x
        elif isinstance(return_value_node, ast.Tuple) or isinstance(return_value_node, ast.List):
            for elt in return_value_node.elts:
                if isinstance(elt, ast.Name):
                    return_vars.append(elt.id) # return (x, y) or return [x, y]
        # TODO: Handle other types of return values if needed (e.g., function calls, constants)
        return return_vars

    #
    def _get_node_value(self, node: ast.expr) -> Any:
        #
        # Try to get the value of a node if it's a constant, string or name, else return the node type
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_value(node.value)}.{node.attr}" # for cases like torch.relu or F.relu
        elif isinstance(node, ast.BinOp): # For binary operations like a + 1, capture the operation
            return f"BinaryOp: {type(node.op).__name__}" # Just return the type of operation for now
        # TODO: Handle other relevant node types to extract their values or representation
        return type(node) # If we can't extract a simple value, return the node type


    #
    def generic_visit(self, node: ast.AST) -> None:
        #
        # Generic visitor, can be extended to check for other node types if needed
        ast.NodeVisitor.generic_visit(self, node)


#
if __name__ == "__main__":
    #
    # Check command line arguments
    if len(sys.argv) != 2:
        raise UserWarning(f"Error: if you use this script directly, you should use it like that :\n python {sys.argv[0]} path_to_model_script.py")

    # Get path to the model script from command line arguments
    path_to_file: str = sys.argv[1]

    # Check if the file exists
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"Error: file not found : `{path_to_file}`")

    # Parse the source code into an AST
    with open(path_to_file, "r") as source:
        tree = ast.parse(source.read())

    # Analyze the AST
    analyzer = ModelAnalyzer()
    analyzer.visit(tree)

    # Print the extracted model blocks information
    print("\n" * 2)
    for block_name, model_block in analyzer.model_blocks.items():
        print(model_block)
