#
from typing import Any
#
import ast
import json
#
import sys
import os

#
def debug_var(var: Any, txt_sup: str = "") -> None:
    #
    n: int = 0
    #
    for attr in dir(var):
        if not attr.startswith("__") and attr not in ["denominator", "imag", "numerator", "real"]:
            n += 1
            debug_var( getattr(var, attr), txt_sup=f"{txt_sup}{'.' if txt_sup else ''}{attr}")
    #
    if isinstance(var, list):
        for i in range(len(var)):
            debug_var( var[i], txt_sup=f"{txt_sup}[{i}]")
    #
    if isinstance(var, dict):
        for key in var:
            debug_var( var[key], txt_sup=f"{txt_sup}['{key}']")
    #
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str) or isinstance(var, list) or isinstance(var, dict):
        print(f"DEBUG VAR | {txt_sup} = {var}")


class LayerInfo:
    def __init__(self, name, var_name, shape=None, datatype=None):
        self.name = name
        self.var_name = var_name
        self.shape = shape
        self.datatype = datatype

    def to_dict(self):
        return {
            "type": "LayerInfo",
            "name": self.name,
            "var_name": self.var_name,
            "shape": self.shape,
            "datatype": self.datatype,
            "data": "path_toward_binary_file"  # Placeholder for actual data
        }

class LayerOperation:
    def __init__(self, name, tensor_input, tensor_output):
        self.name = name
        self.tensor_input = tensor_input
        self.tensor_output = tensor_output

    def to_dict(self):
        return {
            "type": "LayerOperation",
            "name": self.name,
            "tensor_input": self.tensor_input,
            "tensor_output": self.tensor_output
        }

class BlockInfo:
    def __init__(self, name, input_shape, input_datatype, output_shape, output_datatype):
        self.name = name
        self.input_shape = input_shape
        self.input_datatype = input_datatype
        self.output_shape = output_shape
        self.output_datatype = output_datatype
        self.layers = []
        self.tensors_info = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_tensor_info(self, tensor_info):
        self.tensors_info.append(tensor_info)

    def to_dict(self):
        return {
            "type": "Block",
            "name": self.name,
            "input_shape": self.input_shape,
            "input_datatype": self.input_datatype,
            "output_shape": self.output_shape,
            "output_datatype": self.output_datatype,
            "layers": [ti.to_dict() for ti in self.tensors_info],
            "control_flow": [layer.to_dict() for layer in self.layers],
        }

class ModelInfo:
    def __init__(self, name, input_shape, input_datatype, output_shape, output_datatype):
        self.name = name
        self.input_shape = input_shape
        self.input_datatype = input_datatype
        self.output_shape = output_shape
        self.output_datatype = output_datatype
        self.blocks = []

    def add_block(self, block):
        self.blocks.append(block)

    def to_dict(self):
        return {
            "type": "Model",
            "name": self.name,
            "input_shape": self.input_shape,
            "input_datatype": self.input_datatype,
            "output_shape": self.output_shape,
            "output_datatype": self.output_datatype,
            "blocks": [block.to_dict() for block in self.blocks]
        }

class ModelAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.models = []
        self.current_model = None
        self.current_block = None
        self.tensors = {}

    def visit_ClassDef(self, node):
        # Check if the class inherits from torch.nn.Module
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                if isinstance(base.value, ast.Name) and base.value.id == "nn":
                    model_name = node.name
                    self.current_model = ModelInfo(model_name, None, None, None, None)
                    self.models.append(self.current_model)
                    # Initialize a block for the model
                    self.current_block = BlockInfo("Block1", None, None, None, None)
                    self.current_model.add_block(self.current_block)
                    break

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "__init__":
            # Analyze the __init__ method to extract layer definitions
            for item in node.body:
                if isinstance(item, ast.Assign):
                    if isinstance(item.value, ast.Call):
                        if isinstance(item.value.func, ast.Attribute):
                            if isinstance(item.value.func.value, ast.Name) and item.value.func.value.id == "nn":
                                # Handle both ast.Name and ast.Attribute targets
                                target = item.targets[0]
                                #
                                if hasattr(item.value.func, "attr"):
                                    layer_name = item.value.func.attr
                                    #
                                    var_name = item.targets[0].attr
                                    #
                                elif isinstance(target, ast.Name):
                                    layer_name = target.id
                                elif isinstance(target, ast.Attribute):
                                    layer_name = ast.unparse(target)  # Convert to string representation
                                else:
                                    continue  # Skip if the target is neither Name nor Attribute

                                #
                                layer_info = LayerInfo(layer_name, var_name)
                                #
                                self.tensors[layer_name] = layer_info
                                #
                                self.current_block.add_tensor_info( layer_info )

        elif node.name == "forward":
            # Analyze the forward method to track tensor flow
            for item in node.body:
                if isinstance(item, ast.Assign):
                    if isinstance(item.value, ast.Call):
                        if isinstance(item.value.func, ast.Attribute):
                            if isinstance(item.value.func.value, ast.Name) and item.value.func.value.id == "self":
                                layer_name = item.value.func.attr
                                tensor_input = ast.unparse(item.value.args[0])
                                # Handle both ast.Name and ast.Attribute targets
                                target = item.targets[0]
                                if isinstance(target, ast.Name):
                                    tensor_output = target.id
                                elif isinstance(target, ast.Attribute):
                                    tensor_output = ast.unparse(target)  # Convert to string representation
                                else:
                                    continue  # Skip if the target is neither Name nor Attribute

                                self.current_block.add_layer(LayerOperation(layer_name, tensor_input, tensor_output))

        self.generic_visit(node)


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

    # Convert the model information to JSON
    models_json = [model.to_dict() for model in analyzer.models]
    print(json.dumps(models_json, indent=4))