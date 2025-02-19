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

    # --------------------------------------------------------- #
    # ----               INIT MODEL ANALYZER               ---- #
    # --------------------------------------------------------- #

    #
    def __init__(self) -> None:
        #
        self.model_blocks: dict[str, lc.ModelBlock] = {}
        #
        self.main_block: str = ""
        #
        self.current_model_visit: list[str] = []  # Acess with [-1], a stack formation to manage the sub-blocks correctly and with elegance
        self.current_function_visit: str = ""


    # --------------------------------------------------------- #
    # ----                  CLASS VISITOR                  ---- #
    # --------------------------------------------------------- #

    #
    def _is_torch_module_class(self, node: ast.ClassDef) -> bool:
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True
        return False

    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:

        # Check if the class inherits from torch.nn.Module
        if not self._is_torch_module_class(node):
            return

        #
        block_name: str = node.name
        #
        if block_name in self.model_blocks:
            raise UserWarning(f"ERROR : There are two classes with the same name !!!\nBad name : {block_name}\n")

        #
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)
        self.current_model_visit.append(block_name)

        #
        self.generic_visit(node)


    # --------------------------------------------------------- #
    # ----                FUNCTION VISITOR                 ---- #
    # --------------------------------------------------------- #

    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        #
        if node.name == "__init__":
            self._analyze_init_method(node)
        #
        elif node.name == "forward":
            self._analyze_forward_method(node)
        #
        else:
            self._analyse_other_method(node)
        #
        self.generic_visit(node)

    #
    def _analyze_init_method(self, node: ast.FunctionDef) -> None:
        #
        self.current_function_visit = "__init__"
        #
        # TODO: Get the arguments of the class
        #
        # TODO: Get the layers definitions of the class
        # TODO: Get the variables definitions of the class
        # TODO: Analyze deeply the ModuleList and Sequential and create sub blocks for theses correctly
        pass

    #
    def _analyze_forward_method(self, node: ast.FunctionDef) -> None:
        #
        self.current_function_visit = "forward"
        #
        # TODO
        pass

    #
    def _analyse_other_method(self, node: ast.FunctionDef) -> None:
        #
        self.current_function_visit = node.name
        #
        # TODO
        pass

    #
    def get_layer_type(self, func: ast.Expr) -> str:
        if isinstance(func, ast.Attribute):
            return func.attr
        elif isinstance(func, ast.Name):
            return func.id
        return ""

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
