#
import ast
import sys
import os

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
    def is_nn_module_subclass(self, node: ast.ClassDef) -> bool:
        # Check if the class inherits from torch.nn.Module
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True
        return False

    #
    def get_layer_kwargs_from_call(self, call_node: ast.Call) -> dict[str, Any]:
        # Extract keyword arguments from a layer call
        kwargs = {}
        for keyword in call_node.keywords:
            kwargs[keyword.arg] = self.get_value_from_node(keyword.value)
        return kwargs

    #
    def get_value_from_node(self, node: ast.expr) -> Any:
        # Extract value from different types of AST nodes representing values
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id # Or resolve variable value if possible in scope
        # TODO: Handle more complex value types if needed (e.g., expressions, lists, tuples)
        return "[UNK_VALUE]" # Placeholder for unhandled value types

    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Visit class definition nodes in the AST

        # Check if it is a class inheriting from nn.Module
        if not self.is_nn_module_subclass(node):
            return

        # Get the block name which is the class name
        block_name: str = node.name
        self.current_model_visit = block_name

        # Check for name collisions
        if block_name in self.model_blocks:
            raise UserWarning(f"ERROR : There are two classes with the same name !!!\nBad name : {block_name}\n")

        # Create a new ModelBlock instance and store it
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        # Continue visiting child nodes
        self.generic_visit(node)


    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Visit function definition nodes in the AST

        # Analyze the __init__ method to extract layer definitions
        if node.name == "__init__":
            self.analyze_init_method(node)

        # Analyze the forward method to track instructions flow
        elif node.name == "forward":
            self.analyze_forward_method(node)

        # Continue visiting child nodes
        self.generic_visit(node)


    #
    def analyze_init_method(self, node: ast.FunctionDef) -> None:
        # Analyze the __init__ method to extract layer definitions

        # Iterate through each statement in the __init__ method body
        for item in node.body:

            # Look for assignment statements
            if isinstance(item, ast.Assign):

                # We are interested in assignments like `self.layer_name = nn.SomeLayer(...)`
                for target in item.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        layer_var_name: str = target.attr # Get layer variable name (e.g., conv1)

                        # Check if the assigned value is a call (layer instantiation)
                        if isinstance(item.value, ast.Call):
                            call_node: ast.Call = item.value

                            # Check if the call is a module from torch.nn (e.g., nn.Conv2d, nn.Linear)
                            if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == "nn":
                                layer_type: str = call_node.func.attr # Get layer type (e.g., Conv2d)
                                layer_parameters_kwargs: dict[str, Any] = self.get_layer_kwargs_from_call(call_node) # Extract layer parameters

                                # Create a Layer object and add it to the current ModelBlock
                                layer = lc.Layer(layer_var_name=layer_var_name, layer_type=layer_type, layer_parameters_kwargs=layer_parameters_kwargs)
                                self.model_blocks[self.current_model_visit].block_layers[layer_var_name] = layer

                            # Handle nn.Sequential and nn.ModuleList
                            elif isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == "nn" and call_node.func.attr in ["Sequential", "ModuleList"]:
                                sequence_type: str = call_node.func.attr
                                sequence_layers = []

                                # Extract layers from nn.Sequential or nn.ModuleList arguments
                                for arg in call_node.args:
                                    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and isinstance(arg.func.value, ast.Name) and arg.func.value.id == "nn":
                                        seq_layer_type = arg.func.attr
                                        seq_layer_params = self.get_layer_kwargs_from_call(arg)
                                        sequence_layers.append({"type": seq_layer_type, "params": seq_layer_params})
                                    elif isinstance(arg, ast.Name):
                                        # Assume it's a pre-defined layer variable, need to resolve later if needed, or skip for now.
                                        sequence_layers.append({"type": "[REF_LAYER_VAR]", "params": {"ref_var": arg.id}}) # Placeholder for variable reference
                                    else:
                                        sequence_layers.append({"type": "[UNK_LAYER]", "params": {}}) # Placeholder for unhandled layer type in sequence

                                # Create a BlockFunction to represent the sequence
                                seq_function = lc.BlockFunction(function_name=f"{layer_var_name}_sequence", function_arguments={}, model_block=self.model_blocks[self.current_model_visit])
                                for i, seq_layer_info in enumerate(sequence_layers):
                                    seq_layer_pass = lc.FlowControlLayerPass(
                                        output_variables=[f"output_{layer_var_name}_{i}"], # Output name can be improved
                                        layer_name=seq_layer_info["type"],
                                        layer_arguments=seq_layer_info["params"]
                                    )
                                    seq_function.function_flow_control.append(seq_layer_pass)
                                self.model_blocks[self.current_model_visit].block_functions[f"{layer_var_name}_sequence"] = seq_function
                                # For now, not adding the sequence as a Layer, but as a BlockFunction. Can be reconsidered.


    #
    def analyze_forward_method(self, node: ast.FunctionDef) -> None:
        # Analyze the forward method to track control flow instructions

        # Get the forward BlockFunction of the current model
        forward_function_block: lc.BlockFunction = self.model_blocks[self.current_model_visit].block_functions["forward"]

        # Iterate through each statement in the forward method body
        for item in node.body:
            instruction = self.parse_forward_instruction(item)
            if instruction:
                forward_function_block.function_flow_control.append(instruction)


    #
    def parse_forward_instruction(self, item: ast.stmt) -> Optional[lc.FlowControlInstruction]:
        # Parse a statement in the forward method and return a FlowControlInstruction

        # Handle Return statements
        if isinstance(item, ast.Return):
            return_variables = []
            if isinstance(item.value, ast.Name):
                return_variables = [item.value.id] # Simple variable return
            elif isinstance(item.value, ast.Tuple):
                return_variables = [elt.id for elt in item.value.elts if isinstance(elt, ast.Name)] # Tuple return
            else:
                return_variables = ["[UNK_RETURN_VALUE]"] # Unhandled return value type
            return lc.FlowControlReturn(return_variables=return_variables)

        # Handle Assign statements (Layer pass or Function Call)
        elif isinstance(item, ast.Assign):
            output_variables = [target.id for target in item.targets if isinstance(target, ast.Name)] # Output variable names

            if isinstance(item.value, ast.Call):
                call_node: ast.Call = item.value

                # Check for Layer pass (e.g., x = self.conv1(x))
                if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == "self" and call_node.func.attr in self.model_blocks[self.current_model_visit].block_layers:
                    layer_name: str = call_node.func.attr
                    layer_arguments: dict[str, Any] = {} # TODO: Extract arguments passed to layer in forward, if any. For now empty.
                    return lc.FlowControlLayerPass(output_variables=output_variables, layer_name=layer_name, layer_arguments=layer_arguments)

                # Check for Function Call (e.g., x = some_function(x)) -  For now consider all other calls as generic function calls
                else:
                    function_called: str = "[UNK_FUNCTION_CALL]"
                    if isinstance(call_node.func, ast.Name):
                        function_called = call_node.func.id
                    elif isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name):
                        function_called = f"{call_node.func.value.id}.{call_node.func.attr}" # e.g., torch.relu

                    function_arguments: dict[str, Any] = {} # TODO: Extract function call arguments if needed
                    return lc.FlowControlFunctionCall(output_variables=output_variables, function_called=function_called, function_arguments=function_arguments)


        # Handle Expr statements (e.g., standalone function calls without assignment) - Not yet handled, can be added if needed.
        elif isinstance(item, ast.Expr):
            pass # For now ignore standalone expressions

        return None # If the statement type is not handled, return None


    #
    def generic_visit(self, node: ast.AST):
        # Generic visit function, can be extended if needed to handle other AST node types
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

    # Print the extracted model blocks
    print("\n" * 2)
    for block_name, model_block in analyzer.model_blocks.items():
        print(model_block)