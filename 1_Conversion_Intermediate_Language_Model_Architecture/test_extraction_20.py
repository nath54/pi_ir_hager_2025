#
from typing import Any
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
def custom_visit_elt(ast_node: ast.AST) -> Any:
    """
    _summary_

    Args:
        ast_node (ast.AST): _description_

    Returns:
        Any: _description_
    """

    # Check for Call, Condition, Lambda, ...
    # Low level elements
    pass

    #
    return None


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
        print(f"\n\n----\n @@@ ClassDef | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        block_name: str = node.name
        #
        if block_name in self.model_blocks:
            raise NameError(f"ERROR : There are two classes with the same name !!!\nBad name : {block_name}\n")

        #
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        #
        self.current_model_visit.append(block_name)

        #
        self.generic_visit(node)

        #
        self.current_model_visit.pop(-1)


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

        #
        print(f"\n\n----\n @@@ FunctionDef | inside_class: {self.current_model_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

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
        pass

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

        #
        if isinstance(func, ast.Attribute):
            return func.attr

        #
        elif isinstance(func, ast.Name):
            return func.id

        #
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
        pass

        #
        ast.NodeVisitor.generic_visit(self, node)


    # --------------------------------------------------------- #
    # ----                 ASSIGN VISITOR                  ---- #
    # --------------------------------------------------------- #

    #
    def visit_Assign(self, node: ast.AST) -> None:
        """
        Basic assignment.

        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ Assign | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_AugAssign(self, node: ast.AST) -> None:
        """
        Augmented assignment like `x += y`.

        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ AugAssign | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_AnnAssign(self, node: ast.AST) -> None:
        """
        Annoted Assignment like `x: int = 0`.

        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ AssAssign | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_AssignStmt(self, node: ast.AST) -> None:
        """
        Annoted Assignment like `x: int = 0`.

        Args:
            node (ast.AST): _description_
        """

        #
        print(f"\n\n----\n @@@ AssignStmt | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        ast.NodeVisitor.generic_visit(self, node)


    # --------------------------------------------------------- #
    # ----                   EXPR VISITOR                  ---- #
    # --------------------------------------------------------- #

    #
    def visit_Expr(self, node: ast.AST) -> None:
        """


        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ Expr | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_NamedExpr(self, node: ast.AST) -> None:
        """


        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ Expr | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)


    # --------------------------------------------------------- #
    # ----                       VISITOR                   ---- #
    # --------------------------------------------------------- #

    #
    def visit_For(self, node: ast.AST) -> None:
        """


        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ For | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_While(self, node: ast.AST) -> None:
        """


        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ While | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_Return(self, node: ast.AST) -> None:
        """


        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ Return | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_Match(self, node: ast.AST) -> None:
        """


        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ Match | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_If(self, node: ast.AST) -> None:
        """


        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ If | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_Break(self, node: ast.AST) -> None:
        """


        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ Break | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_Continue(self, node: ast.AST) -> None:
        """


        Args:
            node (ast.AST): _description_
        """

        #
        # TODO: is there other things to check here ? (for instance function calls, or arithmetic operations / variable manipulation and other ?, maybe create other visit_... methods ?)
        pass

        #
        print(f"\n\n----\n @@@ Continue | inside_class: {self.current_model_visit} | inside_func: {self.current_function_visit} | visitor : {node}\n\n----\n")
        #
        debug_var(var=node, txt_sup="\t")

        #
        ast.NodeVisitor.generic_visit(self, node)




    # --------------------------------------------------------- #
    # ----           CLEANING & ERROR DETECTIONS           ---- #
    # --------------------------------------------------------- #


    #
    def cleaning_and_error_detections(self) -> None:
        """
        _summary_
        """

        #
        # TODO: Cleaning unused functions in blocks (need a control flow analysis)
        # TODO: While cleaning, check for basic errors (like a block without forward method, and things like that)
        # TODO: Clarify and explain what is done in this method
        # TODO: Decompose the task in multiple sub methods for better code
        pass

        #
        pass


#
def extract_from_file(filepath: str, main_block_name: str = "") -> lc.Language1_Model:
    """
    Extracts a neural network model architecture from models that are written with Pytorch library from a python script file.

    Note: All the dependecies of the model must be in this single script file, the imports are not followed.

    Args:
        filepath (str): Path to file to extract the architecture of the model from.
        main_block_name (str, optional): Indicates the name of the entry point of the model. Defaults to "".

    Raises:
        FileNotFoundError: If the file is not found

    Returns:
        lc.Language1_Model: The architecture of the model extracted
    """

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
    lang1: lc.Language1_Model = lc.Language1_Model()
    lang1.main_block = analyzer.main_block
    lang1.model_blocks = analyzer.model_blocks

    #
    return lang1



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
    l1_model: lc.Language1_Model = extract_from_file(filepath=path_to_file)

    #
    print("\n" * 2)
    print(l1_model.model_blocks)
