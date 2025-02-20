#
from typing import Optional, Any
#
import ast
import sys
import os

#
import lib_classes as lc

#
class ModelAnalyzer(ast.NodeVisitor):

    # --------------------------------------------------------- #
    # ----               INIT MODEL ANALYZER               ---- #
    # --------------------------------------------------------- #

    #
    def __init__(self) -> None:
        """
        Initializer of the method ModelAnalyzer

        Attributes:
            - model_blocks (dict[str, lc.ModelBlock]): List of all the blocks analyzed here, indexed by their blocks id (e.g., their name)
            - main_block (str): Id of the main block, given in sys.argv with `--main-block <MainBlockName>`
            - current_model_visit (list[str]): Stack of all the current blocks we are working on currently, access to the top one with [-1]
            - current_function_visit (str): Name of the current visited function if we are
        """
        #
        self.model_blocks: dict[str, lc.ModelBlock] = {}
        #
        self.main_block: str = ""
        #
        self.current_model_visit: list[str] = []  # Access with [-1], a stack formation to manage the sub-blocks correctly and with elegance
        self.current_function_visit: str = ""

    # --------------------------------------------------------- #
    # ----                  CLASS VISITOR                  ---- #
    # --------------------------------------------------------- #

    #
    def _is_torch_module_class(self, node: ast.ClassDef) -> bool:
        """
        Indicates if the given ClassDef node is a nn.Module subClass or not.

        Args:
            node (ast.ClassDef): Node to tell if subClass of nn.Module

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

        # Pop the current model visit stack
        self.current_model_visit.pop()

    # --------------------------------------------------------- #
    # ----                FUNCTION VISITOR                 ---- #
    # --------------------------------------------------------- #

    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Called when the ast visitor detects a definition of a function.

        Args:
            node (ast.FunctionDef): Function definition node
        """

        # Check if we are currently visiting a class or not
        if not self.current_model_visit:
            return  # Ignore all the standalone functions

        # Check if we are visiting the Class Initialization function, where there are the layers definition
        if node.name == "__init__":
            self._analyze_init_method(node)

        # Check if we are visiting the Class Forward method, where there is the main control flow of the model block
        elif node.name == "forward":
            self._analyze_forward_method(node)

        # Check if we are visiting another method of the class, we will store them temporarily, and clean those that aren't used in the model.
        else:
            self._analyse_other_method(node)

        # To continue the AST visit
        self.generic_visit(node)

    #
    def _analyze_init_method(self, node: ast.FunctionDef) -> None:
        """
        Analyzes the __init__ method to extract layer definitions.

        Args:
            node (ast.FunctionDef): The __init__ method node
        """

        #
        self.current_function_visit = "__init__"

        # Get the arguments of the current class definition
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = "Any"  # Default type
            arg_default = None
            if arg.annotation:
                arg_type = self._get_type_from_annotation(arg.annotation)
            self.model_blocks[self.current_model_visit[-1]].block_parameters[arg_name] = (arg_type, arg_default)

        # Get the layers definitions of the class
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                self._analyze_layer_definition(stmt)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self._analyze_sequential_or_modulelist(stmt.value)

    #
    def _analyze_forward_method(self, node: ast.FunctionDef) -> None:
        """
        Analyzes the forward method to extract control flow instructions.

        Args:
            node (ast.FunctionDef): The forward method node
        """

        #
        self.current_function_visit = "forward"

        # Get the arguments of the forward method
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = "Any"  # Default type
            arg_default = None
            if arg.annotation:
                arg_type = self._get_type_from_annotation(arg.annotation)
            self.model_blocks[self.current_model_visit[-1]].block_variables[arg_name] = (arg_type, arg_default)

        # Analyze the body of the forward method
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                self._analyze_assignment(stmt)
            elif isinstance(stmt, ast.If):
                self._analyze_if_statement(stmt)
            elif isinstance(stmt, ast.For):
                self._analyze_for_loop(stmt)
            elif isinstance(stmt, ast.While):
                self._analyze_while_loop(stmt)
            elif isinstance(stmt, ast.Return):
                self._analyze_return_statement(stmt)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self._analyze_function_call(stmt.value)

    #
    def _analyse_other_method(self, node: ast.FunctionDef) -> None:
        """
        Analyzes other methods in the model block.

        Args:
            node (ast.FunctionDef): The method node
        """

        #
        self.current_function_visit = node.name

        # Get the arguments of the method
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = "Any"  # Default type
            arg_default = None
            if arg.annotation:
                arg_type = self._get_type_from_annotation(arg.annotation)
            self.model_blocks[self.current_model_visit[-1]].block_variables[arg_name] = (arg_type, arg_default)

        # Analyze the body of the method
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                self._analyze_assignment(stmt)
            elif isinstance(stmt, ast.If):
                self._analyze_if_statement(stmt)
            elif isinstance(stmt, ast.For):
                self._analyze_for_loop(stmt)
            elif isinstance(stmt, ast.While):
                self._analyze_while_loop(stmt)
            elif isinstance(stmt, ast.Return):
                self._analyze_return_statement(stmt)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self._analyze_function_call(stmt.value)

    #
    def _analyze_layer_definition(self, stmt: ast.Assign) -> None:
        """
        Analyzes a layer definition in the __init__ method.

        Args:
            stmt (ast.Assign): The assignment statement
        """

        for target in stmt.targets:
            if isinstance(target, ast.Name):
                layer_name = target.id
                layer_type = self.get_layer_type(stmt.value)
                layer_params = self._get_layer_params(stmt.value)
                layer = lc.Layer(layer_var_name=layer_name, layer_type=layer_type, layer_parameters_kwargs=layer_params)
                self.model_blocks[self.current_model_visit[-1]].block_layers[layer_name] = layer

    #
    def _analyze_sequential_or_modulelist(self, call: ast.Call) -> None:
        """
        Analyzes nn.Sequential or nn.ModuleList and creates sub-blocks for them.

        Args:
            call (ast.Call): The call expression
        """

        if isinstance(call.func, ast.Attribute) and call.func.attr in ["Sequential", "ModuleList"]:
            block_name = self.current_model_visit[-1]
            sub_block_name = f"{block_name}_{call.func.attr}_{len(self.model_blocks)}"
            self.model_blocks[sub_block_name] = lc.ModelBlock(block_name=sub_block_name)
            self.current_model_visit.append(sub_block_name)

            for arg in call.args:
                if isinstance(arg, ast.List):
                    for elem in arg.elts:
                        if isinstance(elem, ast.Call):
                            layer_name = self.get_layer_type(elem)
                            layer_params = self._get_layer_params(elem)
                            layer = lc.Layer(layer_var_name=layer_name, layer_type=layer_name, layer_parameters_kwargs=layer_params)
                            self.model_blocks[sub_block_name].block_layers[layer_name] = layer

            self.current_model_visit.pop()

    #
    def _analyze_assignment(self, stmt: ast.Assign) -> None:
        """
        Analyzes an assignment statement in the forward method.

        Args:
            stmt (ast.Assign): The assignment statement
        """

        for target in stmt.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                var_value = self._get_expression_from_node(stmt.value)
                instruction = lc.FlowControlVariableAssignment(var_name=var_name, var_value=var_value)
                self.model_blocks[self.current_model_visit[-1]].block_functions["forward"].function_flow_control.append(instruction)

    #
    def _analyze_if_statement(self, stmt: ast.If) -> None:
        """
        Analyzes an if statement in the forward method.

        Args:
            stmt (ast.If): The if statement
        """

        condition = self._get_condition_from_node(stmt.test)
        true_branch = [self._get_instruction_from_node(node) for node in stmt.body]
        false_branch = [self._get_instruction_from_node(node) for node in stmt.orelse]
        layer_name = f"condition_{len(self.model_blocks[self.current_model_visit[-1]].block_layers)}"
        layer = lc.LayerCondition(layer_var_name=layer_name, layer_conditions_blocks={condition: lc.FlowControlSubBlockFunctionCall(output_variables=[], function_called="", function_arguments={})})
        self.model_blocks[self.current_model_visit[-1]].block_layers[layer_name] = layer
        instruction = lc.FlowControlLayerPass(output_variables=[], layer_name=layer_name, layer_arguments={})
        self.model_blocks[self.current_model_visit[-1]].block_functions["forward"].function_flow_control.append(instruction)

    #
    def _analyze_for_loop(self, stmt: ast.For) -> None:
        """
        Analyzes a for loop in the forward method.

        Args:
            stmt (ast.For): The for loop statement
        """

        iterable = self._get_expression_from_node(stmt.iter)
        iterator = stmt.target.id
        loop_body = [self._get_instruction_from_node(node) for node in stmt.body]
        instruction = lc.FlowControlForLoop(iterable_var_name=iterable.var_name, iterator=iterator, flow_control_instructions=loop_body)
        self.model_blocks[self.current_model_visit[-1]].block_functions["forward"].function_flow_control.append(instruction)

    #
    def _analyze_while_loop(self, stmt: ast.While) -> None:
        """
        Analyzes a while loop in the forward method.

        Args:
            stmt (ast.While): The while loop statement
        """

        condition = self._get_condition_from_node(stmt.test)
        loop_body = [self._get_instruction_from_node(node) for node in stmt.body]
        instruction = lc.FlowControlWhileLoop(condition=condition, flow_control_instructions=loop_body)
        self.model_blocks[self.current_model_visit[-1]].block_functions["forward"].function_flow_control.append(instruction)

    #
    def _analyze_return_statement(self, stmt: ast.Return) -> None:
        """
        Analyzes a return statement in the forward method.

        Args:
            stmt (ast.Return): The return statement
        """

        return_vars = [self._get_expression_from_node(var).var_name for var in stmt.value.elts]
        instruction = lc.FlowControlReturn(return_variables=return_vars)
        self.model_blocks[self.current_model_visit[-1]].block_functions["forward"].function_flow_control.append(instruction)

    #
    def _analyze_function_call(self, call: ast.Call) -> None:
        """
        Analyzes a function call in the forward method.

        Args:
            call (ast.Call): The function call expression
        """

        function_name = self.get_layer_type(call.func)
        output_vars = [target.id for target in call.args]
        function_args = {kw.arg: self._get_expression_from_node(kw.value) for kw in call.keywords}
        instruction = lc.FlowControlFunctionCall(output_variables=output_vars, function_called=function_name, function_arguments=function_args)
        self.model_blocks[self.current_model_visit[-1]].block_functions["forward"].function_flow_control.append(instruction)

    #
    def _get_expression_from_node(self, node: ast.AST) -> lc.Expression:
        """
        Gets an expression from an AST node.

        Args:
            node (ast.AST): The AST node

        Returns:
            lc.Expression: The expression
        """

        if isinstance(node, ast.Name):
            return lc.ExpressionVariable(var_name=node.id)
        elif isinstance(node, ast.Constant):
            return lc.ExpressionConstantNumeric(constant=node.value)
        elif isinstance(node, ast.List):
            return lc.ExpressionConstantList(elements=[self._get_expression_from_node(elt) for elt in node.elts])
        elif isinstance(node, ast.Call):
            return lc.ExpressionConstantString(constant=self.get_layer_type(node))
        else:
            return lc.ExpressionConstant(constant=node)

    #
    def _get_condition_from_node(self, node: ast.AST) -> lc.Condition:
        """
        Gets a condition from an AST node.

        Args:
            node (ast.AST): The AST node

        Returns:
            lc.Condition: The condition
        """

        if isinstance(node, ast.Compare):
            left = self._get_expression_from_node(node.left)
            right = self._get_expression_from_node(node.comparators[0])
            operator = self._get_operator_from_node(node.ops[0])
            return lc.ConditionBinary(elt1=left, cond_operator=operator, elt2=right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._get_expression_from_node(node.operand)
            operator = self._get_operator_from_node(node.op)
            return lc.ConditionUnary(elt=operand, cond_operator=operator)
        else:
            return lc.Condition()

    #
    def _get_operator_from_node(self, node: ast.operator) -> str:
        """
        Gets an operator from an AST operator node.

        Args:
            node (ast.operator): The AST operator node

        Returns:
            str: The operator
        """

        if isinstance(node, ast.Add):
            return "+"
        elif isinstance(node, ast.Sub):
            return "-"
        elif isinstance(node, ast.Mult):
            return "*"
        elif isinstance(node, ast.Div):
            return "/"
        elif isinstance(node, ast.Mod):
            return "%"
        elif isinstance(node, ast.Pow):
            return "^"
        elif isinstance(node, ast.LShift):
            return "<<"
        elif isinstance(node, ast.RShift):
            return ">>"
        elif isinstance(node, ast.BitXor):
            return "^"
        elif isinstance(node, ast.BitAnd):
            return "&"
        elif isinstance(node, ast.BitOr):
            return "|"
        elif isinstance(node, ast.Eq):
            return "=="
        elif isinstance(node, ast.NotEq):
            return "!="
        elif isinstance(node, ast.Lt):
            return "<"
        elif isinstance(node, ast.LtE):
            return "<="
        elif isinstance(node, ast.Gt):
            return ">"
        elif isinstance(node, ast.GtE):
            return ">="
        elif isinstance(node, ast.Is):
            return "is"
        elif isinstance(node, ast.IsNot):
            return "is not"
        elif isinstance(node, ast.In):
            return "in"
        elif isinstance(node, ast.NotIn):
            return "not in"
        elif isinstance(node, ast.Not):
            return "not"
        elif isinstance(node, ast.UAdd):
            return "+"
        elif isinstance(node, ast.USub):
            return "-"
        else:
            return ""

    #
    def _get_instruction_from_node(self, node: ast.AST) -> lc.FlowControlInstruction:
        """
        Gets a flow control instruction from an AST node.

        Args:
            node (ast.AST): The AST node

        Returns:
            lc.FlowControlInstruction: The flow control instruction
        """

        if isinstance(node, ast.Assign):
            return self._analyze_assignment(node)
        elif isinstance(node, ast.If):
            return self._analyze_if_statement(node)
        elif isinstance(node, ast.For):
            return self._analyze_for_loop(node)
        elif isinstance(node, ast.While):
            return self._analyze_while_loop(node)
        elif isinstance(node, ast.Return):
            return self._analyze_return_statement(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return self._analyze_function_call(node.value)
        else:
            return lc.FlowControlInstruction()

    #
    def _get_type_from_annotation(self, annotation: ast.expr) -> str:
        """
        Gets the type from an annotation node.

        Args:
            annotation (ast.expr): The annotation node

        Returns:
            str: The type
        """

        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            return self._get_type_from_annotation(annotation.value)
        else:
            return "Any"

    #
    def _get_layer_params(self, call: ast.Call) -> dict[str, Any]:
        """
        Gets the layer parameters from a call node.

        Args:
            call (ast.Call): The call node

        Returns:
            dict[str, Any]: The layer parameters
        """

        params = {}
        for kw in call.keywords:
            params[kw.arg] = self._get_expression_from_node(kw.value)
        return params

    # --------------------------------------------------------- #
    # ----           CLEANING & ERROR DETECTIONS           ---- #
    # --------------------------------------------------------- #

    #
    def cleaning_and_error_detections(self) -> None:
        """
        Cleans and detects errors in the model.
        """

        # TODO: Cleaning unused functions in blocks (need a control flow analysis)
        # TODO: While cleaning, check for basic errors (like a block without forward method, and things like that)
        # TODO: Clarify and explain what is done in this method
        # TODO: Decompose the task in multiple sub methods for better code
        pass

#
def extract_from_file(filepath: str, main_block_name: str = "") -> lc.Language1_Model:
    """
    Extracts a neural network model architecture from models that are written with Pytorch library from a python script file.

    Note: All the dependencies of the model must be in this single script file, the imports are not followed.

    Args:
        filepath (str): Path to file to extract the architecture of the model from.
        main_block_name (str, optional): Indicates the name of the entry point of the model. Defaults to "".

    Raises:
        FileNotFoundError: If the file is not found

    Returns:
        lc.Language1_Model: The architecture of the model extracted
    """

    #
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: file not found : `{filepath}`")

    # Parse the source code into an AST
    with open(filepath, "r") as source:
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
