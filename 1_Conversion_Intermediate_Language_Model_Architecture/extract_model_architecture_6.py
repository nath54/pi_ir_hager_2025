#
import ast
import sys
import os

#
import lib_classes_3 as lc
from typing import Any, Dict, List, Tuple

#
def get_full_attr_name(node: ast.AST) -> str:
    """
    Recursively extracts the full attribute name from an AST node.
    For example, for `nn.Conv2d` it returns 'nn.Conv2d', and for
    `torch.nn.Conv2d` it returns 'torch.nn.Conv2d'.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{get_full_attr_name(node.value)}.{node.attr}"
    else:
        return ""


#
def extract_call_arguments(call_node: ast.Call) -> Dict[str, Any]:
    """
    Extracts the arguments of a call node into a dictionary.
    Positional arguments are stored under the key 'args' as a list.
    Keyword arguments are stored with their names.
    """
    args_list = [ast.unparse(arg) for arg in call_node.args]  # Requires Python 3.9+
    kwargs: Dict[str, Any] = {}
    if args_list:
        kwargs["args"] = args_list
    for kw in call_node.keywords:
        kwargs[kw.arg] = ast.unparse(kw.value)
    return kwargs


#
class ModelAnalyzer(ast.NodeVisitor):
    #
    def __init__(self) -> None:
        #
        self.model_blocks: Dict[str, lc.ModelBlock] = {}  # Mapping: model name -> ModelBlock
        self.main_block: str = ""
        self.current_model_visit: str = ""

    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        #
        is_module_class: bool = False

        # Check if the class inherits from torch.nn.Module
        for base in node.bases:
            # Use a helper to extract the base name and test if it's a Module from nn.
            base_name: str = get_full_attr_name(base)
            if base_name in ["nn.Module", "torch.nn.Module"]:
                is_module_class = True
                break

        #
        if not is_module_class:
            return

        #
        block_name: str = node.name
        self.current_model_visit = block_name
        if block_name in self.model_blocks:
            raise UserWarning(f"ERROR : Duplicate class name detected: {block_name}")
        #
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        #
        self.generic_visit(node)

    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        #
        if node.name == "__init__":
            self.process_init_function(node)
        elif node.name == "forward":
            self.process_forward_function(node)

        #
        self.generic_visit(node)

    #
    def process_init_function(self, node: ast.FunctionDef) -> None:
        """
        Process the __init__ method to extract layer definitions.
        """
        for stmt in node.body:
            # Process only simple assignments
            if isinstance(stmt, ast.Assign) and self.is_layer_definition(stmt):
                self.handle_layer_definition(stmt)
            # Optionally, other assignments (like parameters) could be processed here

    #
    def process_forward_function(self, node: ast.FunctionDef) -> None:
        """
        Process the forward method to extract the flow control instructions.
        """
        # Extract forward arguments (skip 'self')
        forward_args: Dict[str, Tuple[str, Any]] = {}
        for arg in node.args.args[1:]:
            arg_name: str = arg.arg
            arg_type: str = "Any"
            if arg.annotation is not None:
                arg_type = ast.unparse(arg.annotation)
            forward_args[arg_name] = (arg_type, None)
        self.model_blocks[self.current_model_visit].forward_arguments = forward_args

        #
        # Process each statement in the forward method and accumulate instructions.
        for stmt in node.body:
            instr_list: List[lc.FlowControlInstruction] = self.process_forward_statement(stmt)
            self.model_blocks[self.current_model_visit].forward_flow_control.extend(instr_list)

    #
    def process_forward_statement(self, stmt: ast.AST) -> List[lc.FlowControlInstruction]:
        """
        Process a single statement in the forward method and return a list of flow control instructions.
        """
        instructions: List[lc.FlowControlInstruction] = []
        if isinstance(stmt, ast.Assign):
            instructions.extend(self.handle_forward_assign(stmt))
        elif isinstance(stmt, ast.Expr):
            # Expression statement, e.g., a function call without assignment
            if isinstance(stmt.value, ast.Call):
                instr = self.handle_forward_call(stmt.value, [])
                instructions.append(instr)
        elif isinstance(stmt, ast.Return):
            instructions.append(self.handle_return(stmt))
        elif isinstance(stmt, ast.If):
            instructions.append(self.handle_if(stmt))
        else:
            # Other types of statements can be added here
            pass

        return instructions

    #
    def is_layer_definition(self, stmt: ast.Assign) -> bool:
        """
        Checks if an assignment statement is a layer definition of the form:
            self.<layer_name> = nn.<LayerType>(...)
        """
        if len(stmt.targets) != 1:
            return False
        target = stmt.targets[0]
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"):
            return False
        if not isinstance(stmt.value, ast.Call):
            return False
        func_fullname: str = get_full_attr_name(stmt.value.func)
        return func_fullname.startswith("nn.") or func_fullname.startswith("torch.nn.")

    #
    def handle_layer_definition(self, stmt: ast.Assign) -> None:
        """
        Extracts layer information from an assignment and stores it in the ModelBlock.
        """
        target: ast.Attribute = stmt.targets[0]  # We assume one target due to previous check
        layer_var_name: str = target.attr

        # Extract the layer type (e.g., 'Conv2d')
        func_fullname: str = get_full_attr_name(stmt.value.func)
        layer_type: str = func_fullname.split(".")[-1]

        # Extract parameters from both positional and keyword arguments
        parameters: Dict[str, Any] = extract_call_arguments(stmt.value)

        #
        # Create a new Layer and add it to the current model block
        new_layer = lc.Layer(layer_var_name=layer_var_name, layer_type=layer_type, layer_parameters_kwargs=parameters)
        self.model_blocks[self.current_model_visit].block_layers[layer_var_name] = new_layer

    #
    def handle_forward_assign(self, stmt: ast.Assign) -> List[lc.FlowControlInstruction]:
        """
        Processes an assignment in the forward pass.
        It distinguishes between a layer call and a generic function call or variable initialization.
        """
        instructions: List[lc.FlowControlInstruction] = []

        # Extract output variable names from the assignment targets
        output_vars: List[str] = []
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                output_vars.append(target.id)
            else:
                output_vars.append(ast.unparse(target))

        #
        if isinstance(stmt.value, ast.Call):
            # Determine if the call is to a layer (i.e., self.<layer_name>(...))
            if self.is_layer_pass_call(stmt.value):
                instr = self.handle_forward_layer_pass(stmt.value, output_vars)
                instructions.append(instr)
            else:
                instr = self.handle_forward_call(stmt.value, output_vars)
                instructions.append(instr)
        else:
            # Non-call assignment: treat it as a variable initialization
            var_value: str = ast.unparse(stmt.value)
            instr = lc.FlowControlVariableInit(var_name=output_vars[0], var_type="Any", var_value=var_value)
            instructions.append(instr)

        return instructions

    #
    def is_layer_pass_call(self, call_node: ast.Call) -> bool:
        """
        Checks if the call is a layer call, i.e., of the form self.<layer_name>(...)
        and the <layer_name> exists in the block_layers.
        """
        if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name):
            if call_node.func.value.id == "self":
                layer_name: str = call_node.func.attr
                return layer_name in self.model_blocks[self.current_model_visit].block_layers
        return False

    #
    def handle_forward_call(self, call_node: ast.Call, output_vars: List[str]) -> lc.FlowControlFunctionCall:
        """
        Handles a generic function call (non-layer) in the forward pass.
        """
        function_called: str = ast.unparse(call_node.func)
        function_arguments: Dict[str, Any] = extract_call_arguments(call_node)
        return lc.FlowControlFunctionCall(output_variables=output_vars, function_called=function_called,
                                            function_arguments=function_arguments)

    #
    def handle_forward_layer_pass(self, call_node: ast.Call, output_vars: List[str]) -> lc.FlowControlLayerPass:
        """
        Handles a call to a layer in the forward pass (e.g., x = self.conv1(x)).
        """
        layer_name: str = call_node.func.attr  # Assumed safe due to previous check
        layer_arguments: Dict[str, Any] = extract_call_arguments(call_node)
        return lc.FlowControlLayerPass(output_variables=output_vars, layer_name=layer_name,
                                       layer_arguments=layer_arguments)

    #
    def handle_return(self, stmt: ast.Return) -> lc.FlowControlReturn:
        """
        Processes a return statement in the forward method.
        """
        return_vars: List[str] = []
        if stmt.value is None:
            return_vars.append("None")
        elif isinstance(stmt.value, ast.Tuple):
            for elt in stmt.value.elts:
                if isinstance(elt, ast.Name):
                    return_vars.append(elt.id)
                else:
                    return_vars.append(ast.unparse(elt))
        else:
            if isinstance(stmt.value, ast.Name):
                return_vars.append(stmt.value.id)
            else:
                return_vars.append(ast.unparse(stmt.value))
        return lc.FlowControlReturn(return_variables=return_vars)

    #
    def handle_if(self, stmt: ast.If) -> lc.FlowControlIf:
        """
        Processes an if statement in the forward pass.
        It recursively processes the body and the else parts.
        """
        # Extract the condition as a string
        condition_str: str = ast.unparse(stmt.test)

        # Process instructions in the 'if' body
        body_instructions: List[lc.FlowControlInstruction] = []
        for sub_stmt in stmt.body:
            body_instructions.extend(self.process_forward_statement(sub_stmt))

        # Process instructions in the 'else' body (if present)
        orelse_instructions: List[lc.FlowControlInstruction] = []
        for sub_stmt in stmt.orelse:
            orelse_instructions.extend(self.process_forward_statement(sub_stmt))

        return lc.FlowControlIf(condition=condition_str, body=body_instructions, orelse=orelse_instructions)


#
if __name__ == "__main__":

    #
    if len(sys.argv) != 2:
        raise UserWarning(f"Usage error: python {sys.argv[0]} path_to_model_script.py")

    #
    path_to_file: str = sys.argv[1]

    #
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"Error: file not found: `{path_to_file}`")

    #
    # Parse the source code into an AST
    with open(path_to_file, "r") as source:
        tree = ast.parse(source.read())

    #
    # Analyze the AST to extract model architecture and flow control instructions
    analyzer = ModelAnalyzer()
    analyzer.visit(tree)

    #
    # Print the extracted model blocks in a pretty format
    print("\n" * 2)
    print(analyzer.model_blocks)
