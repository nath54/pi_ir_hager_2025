#
import ast
import sys
import os

#
import lib_classes as lc

# FOR DEBUGGING
sys.path.insert(0, "../debug/")
#
from lib_debug import debug_var  # type: ignore

#



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
        is_module_class: bool = False

        # Check if the class inherits from torch.nn.Module
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
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
            raise UserWarning(f"ERROR : There are two classes with the same name !!!\nBad name : {block_name}\n")

        #
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        #
        self.generic_visit(node)


    #
    def visit_FunctionDef(self, node):
        if node.name == "__init__":

            # Analyze the __init__ method to extract layer definitions
            for item in node.body:

                # TODO
                pass

        #
        elif node.name == "forward":

            # Analyze the forward method to track instructions flow
            for item in node.body:

                # TODO
                pass

        #
        self.generic_visit(node)

    #
    def generic_visit(self, node: ast.AST):
        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
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
    print(analyzer.model_blocks)
