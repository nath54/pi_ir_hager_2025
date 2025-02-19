#
import ast
import sys
import os

#
import lib_classes as lc

# FOR DEBUGGING
# The debug_var displays all the attributes of a variable
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
        """
        Initialiser of the method ModelAnalyzer

        Attributes:
            - model_blocks (dict[str, lc.ModelBlock]): list of all the blocks analyzed here, indexed by their blocks id (e.g. their name)
            - main_block (str): id of the main block, given in sys.argv with `--main-block <MainBlockName>`
            - current_model_visit (list[str]): stack of all the current blocks we are working on currently, access to the top one with [-1]
            - current_function_visit (str): name of the current visited function if we are
        """
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
        """
        Indicates if the given ClassDef node is a nn.Module subClass or not.

        Args:
            node (ast.ClassDef): node to tell if subClass of nn.Module

        Returns:
            bool: True if subClass of nn.Module, else False
        """

        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True
        return False

    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Called when the ast visitor detects a definition of a class.

        Args:
            node (ast.ClassDef): The class node to visit

        Raises:
            NameError: In case there are two classes with the same name, we return an error
        """

        # Check if the class inherits from torch.nn.Module
        if not self._is_torch_module_class(node):
            return

        #
        block_name: str = node.name
        #
        if block_name in self.model_blocks:
            raise NameError(f"ERROR : There are two classes with the same name !!!\nBad name : {block_name}\n")

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
        """
        Called when the ast visitor detects a definition of a function.

        Args:
            node (ast.FunctionDef): _description_
        """

        # Check if we are currently visiting a class or not
        if not self.current_model_visit:
            return  # Ignore all the standalone functions

        # Check if we are visiting the Class Initialisation function, where there are the layers definition
        if node.name == "__init__":
            self._analyze_init_method(node)

        # Check if we are visiting the Class Forward method, were there is the main control flow of the model block
        elif node.name == "forward":
            self._analyze_forward_method(node)

        # Check if we are visiting another method of the class, we will store them temporaly, and clean thoses that aren't used in the model.
        else:
            self._analyse_other_method(node)

        # To continue the AST visit
        self.generic_visit(node)

    #
    def _analyze_init_method(self, node: ast.FunctionDef) -> None:
        """
        _summary_

        Args:
            node (ast.FunctionDef): _description_
        """

        #
        self.current_function_visit = "__init__"
        #
        # TODO: Get the argument of the current class definition
        # TODO: to put here: self.model_blocks[block_name].block_parameters
        pass
        #
        # TODO: Get the layers definitions of the class
        # TODO: Get the variables definitions of the class
        # TODO: Analyze deeply the ModuleList and Sequential and create sub blocks for theses correctly (their blocks names will be `BlockModuleList_{Parent_Block_Name}_{n°}` and `BlocSequential_{Parent_Block_Name}_{n°}`)
        pass

    #
    def _analyze_forward_method(self, node: ast.FunctionDef) -> None:
        """
        _summary_

        Args:
            node (ast.FunctionDef): _description_
        """

        #
        self.current_function_visit = "forward"
        #
        # TODO: Get the arguments of the forward method
        pass

        # TODO: Detect (create sub methods for each one):
        # TODO:   - the variables initializations (lc.FlowControlVariableInit)
        # TODO:   - the variables assignments (lc.FlowControlVariableAssignment)
        # TODO:   - the basic arithmetics (ex: A = A + B), or decompose the chain of arithmetics in multiple basic arithmetic operations (ex: A = (A + B) - C -> TMP1 = A + B; A = TMP1 - C)
        # TODO:   - the functions calls (lc.FlowControlFunctionCall or lc.FlowControlSubBlockFunctionCall)
        # TODO:   - the layers calls (lc.FlowControlLayerPass)
        # TODO:   - the returns (lc.FlowControlReturn)
        # TODO:   - the conditions blocks -> Create sub-blocks for each condition branch, add a layer to the model block layers definitions of type lc.LayerCondition, and then add a flow instruction of type lc.FlowControlLayerPass
        # TODO:   - the loops (lc.FlowControlForLoop or lc.FlowControlWhileLoop)

    #
    def _analyse_other_method(self, node: ast.FunctionDef) -> None:
        """
        _summary_

        Args:
            node (ast.FunctionDef): _description_
        """

        #
        self.current_function_visit = node.name
        #
        # TODO
        pass

    #
    def get_layer_type(self, func: ast.Expr) -> str:
        """
        _summary_

        Args:
            func (ast.Expr): _description_

        Returns:
            str: _description_
        """

        if isinstance(func, ast.Attribute):
            return func.attr
        elif isinstance(func, ast.Name):
            return func.id
        return ""


    # --------------------------------------------------------- #
    # ----                 GENERIC VISITOR                 ---- #
    # --------------------------------------------------------- #

    #
    def generic_visit(self, node: ast.AST) -> None:
        """
        _summary_

        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        #
        ast.NodeVisitor.generic_visit(self, node)


    # --------------------------------------------------------- #
    # ----            BASIC & EASY OPTIMISATIONS           ---- #
    # --------------------------------------------------------- #

    #
    def basic_optimisations(self) -> None:
        """
        _summary_
        """

        #
        # TODO: Search for basic optimisations like checking useless control flow instructions or optimisations of temporaries variables (if there is a way to reduce their numbers by checking the variables dependancy graph for instance)
        # TODO: Clarify and explain what is done in this method
        # TODO: Decompose the task in multiple sub methods for better code
        #
        pass


    # --------------------------------------------------------- #
    # ----                    CLEANING                     ---- #
    # --------------------------------------------------------- #


    #
    def cleaning(self) -> None:
        """
        _summary_
        """

        #
        # TODO: Cleaning unused functions in blocks (need a control flow analysis)
        # TODO: Clarify and explain what is done in this method
        # TODO: Decompose the task in multiple sub methods for better code
        #
        pass


#
if __name__ == "__main__":

    #
    if len(sys.argv) != 2:
        raise UserWarning(f"Error: if you use this script directly, you should use it like that :\n  python {sys.argv[0]} path_to_model_script.py")

    #
    # TODO: add the support for the main model block argument that specifies which block is the main model block
    pass

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
