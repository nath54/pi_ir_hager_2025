import ast
import json
import sys
import os

class NodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []

    def generic_visit(self, node):
        node_dict = self.node_to_dict(node)
        self.nodes.append(node_dict)
        super().generic_visit(node)

    def node_to_dict(self, node):
        """Convert an AST node into a dictionary."""
        node_dict = {
            "type": type(node).__name__,
            "fields": {field: self.value_to_dict(value) for field, value in ast.iter_fields(node)}
        }
        return node_dict

    def value_to_dict(self, value):
        """Convert a value to a dictionary if it's a node, otherwise return the value."""
        if isinstance(value, ast.AST):
            return self.node_to_dict(value)
        elif isinstance(value, list):
            return [self.value_to_dict(item) for item in value]
        else:
            return value

def parse_and_save_ast(source_code, output_file):
    tree = ast.parse(source_code)
    visitor = NodeVisitor()
    visitor.visit(tree)

    with open(output_file, 'w') as f:
        json.dump(visitor.nodes, f, indent=4)

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
        source_code = source.read()

    output_file = 'ast_output.json'
    parse_and_save_ast(source_code, output_file)

