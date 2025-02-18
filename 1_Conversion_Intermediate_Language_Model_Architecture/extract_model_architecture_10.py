import ast
import sys
import os
from typing import Any, Dict, Optional, List

# Import the classes defined in lib_classes.py
import lib_classes_4 as lc

#######################################################################
#######################     MODEL ANALYZER     #######################
#######################################################################

class ModelAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        # Dictionary to hold all the analyzed model blocks (class name -> ModelBlock)
        self.model_blocks: Dict[str, lc.ModelBlock] = {}
        # Name of the main model block (if needed)
        self.main_block: str = ""
        # Keeps track of the current model (class) being visited
        self.current_model_visit: str = ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Check if the class inherits from torch.nn.Module
        if not self._is_nn_module_class(node):
            return

        block_name: str = node.name
        self.current_model_visit = block_name

        if block_name in self.model_blocks:
            raise UserWarning(f"ERROR: Duplicate class name detected: {block_name}")

        # Create a new ModelBlock for this class
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        # Continue visiting child nodes (methods, etc.)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "__init__":
            # Process layer initializations
            self._process_init_function(node)
        elif node.name == "forward":
            # Process the forward pass instructions
            self._process_forward_function(node)
        else:
            self.generic_visit(node)

    def _process_init_function(self, node: ast.FunctionDef) -> None:
        # Loop through each statement in the __init__ method
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                self._process_init_assignment(stmt)
            elif isinstance(stmt, ast.For):
                self._process_init_for_loop(stmt)
            # Other statement types can be handled as needed
        self.generic_visit(node)

    def _process_init_assignment(self, node: ast.Assign) -> None:
        # Process assignments like: self.layer = nn.Linear(...)
        if len(node.targets) != 1:
            return  # Skip multiple assignments

        target = node.targets[0]
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"):
            return  # Not an attribute of self

        layer_var_name = target.attr

        if isinstance(node.value, ast.Call):
            if self._is_nn_module_call(node.value):
                layer_type = self._get_call_func_name(node.value)
                layer_parameters = self._process_call_arguments(node.value)

                # Handle nn.Sequential and nn.ModuleList as sub-blocks
                if layer_type in ("Sequential", "ModuleList"):
                    sub_block = self._process_sequential_or_modulelist(node.value, layer_var_name)
                    self.model_blocks[self.current_model_visit].block_layers[layer_var_name] = sub_block
                else:
                    # Normal layer definition
                    layer = lc.Layer(
                        layer_var_name=layer_var_name,
                        layer_type=layer_type,
                        layer_parameters_kwargs=layer_parameters
                    )
                    self.model_blocks[self.current_model_visit].block_layers[layer_var_name] = layer
        else:
            # Other types of assignments (e.g., variable initializations) can be handled here if needed
            pass

    def _process_init_for_loop(self, node: ast.For) -> None:
        # Process a loop that may add layers, for example:
        #   for i in range(...):
        #       self.layers.append(nn.Conv2d(...))
        loop_block_name = f"loop_block_{node.lineno}"
        loop_block = lc.ModelBlock(block_name=loop_block_name)

        # Process each statement inside the for loop
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call_node = stmt.value
                if self._is_append_call(call_node):
                    target_list_name = self._get_append_target(call_node)
                    # Assume the argument to append is a call to a nn.Module
                    if call_node.args and isinstance(call_node.args[0], ast.Call):
                        inner_call = call_node.args[0]
                        if self._is_nn_module_call(inner_call):
                            inner_layer_type = self._get_call_func_name(inner_call)
                            inner_layer_parameters = self._process_call_arguments(inner_call)
                            layer_name = f"{target_list_name}_{inner_layer_type}_{stmt.lineno}"
                            layer = lc.Layer(
                                layer_var_name=layer_name,
                                layer_type=inner_layer_type,
                                layer_parameters_kwargs=inner_layer_parameters
                            )
                            loop_block.block_layers[layer_name] = layer
        # Add the loop block as a layer in the current model block
        self.model_blocks[self.current_model_visit].block_layers[loop_block_name] = loop_block

    def _process_forward_function(self, node: ast.FunctionDef) -> None:
        # Create a BlockFunction for the forward pass
        forward_fn = lc.BlockFunction(
            function_name="forward",
            function_arguments={},
            model_block=self.model_blocks[self.current_model_visit]
        )
        # Process each statement in the forward method
        for stmt in node.body:
            fc = self._process_forward_statement(stmt)
            if fc is not None:
                forward_fn.function_flow_control.append(fc)
        # Save the forward function in the current model block
        self.model_blocks[self.current_model_visit].block_functions["forward"] = forward_fn
        self.generic_visit(node)

    def _process_forward_statement(self, node: ast.stmt) -> Optional[lc.FlowControlInstruction]:
        # Process basic forward statements (assignments and returns)
        if isinstance(node, ast.Assign):
            # Assume a single target for simplicity
            if len(node.targets) != 1:
                return None
            target = node.targets[0]
            if isinstance(target, ast.Name):
                var_name = target.id
            elif isinstance(target, ast.Attribute):
                var_name = target.attr
            else:
                var_name = "unknown"

            if isinstance(node.value, ast.Call):
                if self._is_nn_module_call(node.value):
                    # Treat as a layer pass
                    layer_name = self._get_call_func_name(node.value)
                    layer_arguments = self._process_call_arguments(node.value)
                    return lc.FlowControlLayerPass(
                        output_variables=[var_name],
                        layer_name=layer_name,
                        layer_arguments=layer_arguments
                    )
                else:
                    # Treat as a generic function call
                    function_called = self._get_call_func_name(node.value)
                    function_arguments = self._process_call_arguments(node.value)
                    return lc.FlowControlFunctionCall(
                        output_variables=[var_name],
                        function_called=function_called,
                        function_arguments=function_arguments
                    )

        elif isinstance(node, ast.Return):
            # Process return statements
            ret_vars: List[str] = []
            if isinstance(node.value, ast.Name):
                ret_vars.append(node.value.id)
            elif isinstance(node.value, ast.Tuple):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Name):
                        ret_vars.append(elt.id)
            return lc.FlowControlReturn(return_variables=ret_vars)
        # Other statement types can be added here
        return None

    # -------------------------------------------------------------------
    # Helper Functions for AST Node Checks and Processing
    # -------------------------------------------------------------------
    def _is_nn_module_class(self, node: ast.ClassDef) -> bool:
        """
        Check if a class node inherits from nn.Module.
        """
        for base in node.bases:
            # Check for patterns like nn.Module
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                if isinstance(base.value, ast.Name) and base.value.id == "nn":
                    return True
        return False

    def _is_nn_module_call(self, node: ast.Call) -> bool:
        """
        Check if the call is to a torch.nn module (e.g., nn.Linear, nn.Conv2d, etc.)
        """
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "nn":
                return True
        return False

    def _get_call_func_name(self, node: ast.Call) -> str:
        """
        Extract the function name from a call node (e.g., 'Linear' from nn.Linear).
        """
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        elif isinstance(node.func, ast.Name):
            return node.func.id
        return "unknown"

    def _process_call_arguments(self, node: ast.Call) -> Dict[str, Any]:
        """
        Convert call arguments to a dictionary. Attempts to evaluate constants.
        """
        args_dict: Dict[str, Any] = {}
        # Process positional arguments
        for idx, arg in enumerate(node.args):
            try:
                value = ast.literal_eval(arg)
            except Exception:
                value = self._expr_to_str(arg)
            args_dict[f"arg{idx}"] = value

        # Process keyword arguments
        for kw in node.keywords:
            try:
                value = ast.literal_eval(kw.value)
            except Exception:
                value = self._expr_to_str(kw.value)
            args_dict[kw.arg] = value
        return args_dict

    def _expr_to_str(self, node: ast.AST) -> str:
        """
        Convert an AST expression node back to source code.
        """
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
        return "<expr>"

    def _is_append_call(self, node: ast.Call) -> bool:
        """
        Check if the call is an append operation (e.g., self.layers.append(...))
        """
        if isinstance(node.func, ast.Attribute) and node.func.attr == "append":
            return True
        return False

    def _get_append_target(self, node: ast.Call) -> str:
        """
        Extract the target list variable name from an append call.
        """
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Attribute):
                return node.func.value.attr
            elif isinstance(node.func.value, ast.Name):
                return node.func.value.id
        return "unknown_list"

    def _process_sequential_or_modulelist(self, node: ast.Call, var_name: str) -> lc.ModelBlock:
        """
        Process nn.Sequential or nn.ModuleList calls by creating a sub-block.
        """
        sub_block = lc.ModelBlock(block_name=f"{var_name}_subblock")
        # We assume that the first argument is a list or tuple of layer calls.
        if node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, (ast.List, ast.Tuple)):
                for idx, elt in enumerate(first_arg.elts):
                    if isinstance(elt, ast.Call) and self._is_nn_module_call(elt):
                        layer_type = self._get_call_func_name(elt)
                        layer_parameters = self._process_call_arguments(elt)
                        layer_name = f"{var_name}_{layer_type}_{idx}"
                        layer = lc.Layer(
                            layer_var_name=layer_name,
                            layer_type=layer_type,
                            layer_parameters_kwargs=layer_parameters
                        )
                        sub_block.block_layers[layer_name] = layer
        return sub_block

#######################################################################
###########################     MAIN     ############################
#######################################################################

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise UserWarning(
            f"Error: if using this script directly, please run:\n  python {sys.argv[0]} path_to_model_script.py"
        )

    path_to_file: str = sys.argv[1]
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"Error: file not found: `{path_to_file}`")

    # Read and parse the model source code
    with open(path_to_file, "r") as source:
        tree = ast.parse(source.read())

    # Analyze the AST to extract model architecture and flow control instructions
    analyzer = ModelAnalyzer()
    analyzer.visit(tree)

    # Output the analyzed model blocks
    print("\n" * 2)
    print(analyzer.model_blocks)
