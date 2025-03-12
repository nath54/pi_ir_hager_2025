#
import ast
import sys
import os
import json
from typing import Optional, Dict, List, cast, Any, Callable

#
import inspect
import importlib.util
import importlib.machinery

#
import lib_classes as lc
import lib_layers as ll


# --------------------------------------------------------- #
# ----        EXTRACT NAME OR ATTRIBUTE TO STR         ---- #
# --------------------------------------------------------- #

#
def extract_name_or_attribute(node: ast.AST) -> str:
    """
    _summary_

    Args:
        node (ast.Node): _description_

    Returns:
        str: _description_
    """

    #
    if isinstance(node, ast.Name):
        return node.id

    #
    elif isinstance(node, ast.Attribute):
        return extract_name_or_attribute(node.value)

    #
    return str(node)


# --------------------------------------------------------- #
# ----               EXTRACT EXPRESSION                ---- #
# --------------------------------------------------------- #

#
def extract_expression(node: ast.AST, analyzer: "ModelAnalyzer") -> Optional[lc.Expression]:
    """
    Extracts an Expression object from an AST node, with optional reference to analyzer for global constants.

    Args:
        node (ast.AST): The AST node to analyze.
        analyzer (ModelAnalyzer, optional): The analyzer instance to access global constants.

    Returns:
        Optional[lc.Expression]: The extracted expression or None if not applicable.
    """

    #
    if isinstance(node, ast.Name):
        if analyzer and node.id in analyzer.global_constants:
            type_str, value = analyzer.global_constants[node.id]
            if type_str in ("int", "float"):
                return lc.ExpressionConstantNumeric(constant=value.constant if isinstance(value, lc.ExpressionConstant) else value)
            elif type_str == "str":
                return lc.ExpressionConstantString(constant=value.constant if isinstance(value, lc.ExpressionConstant) else value)
            elif type_str == "list":
                return lc.ExpressionConstantList(elements=value.elements if isinstance(value, lc.ExpressionConstantList) else value)
        return lc.ExpressionVariable(var_name=node.id)
    #
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return lc.ExpressionConstantNumeric(constant=node.value)
        elif isinstance(node.value, str):
            return lc.ExpressionConstantString(constant=node.value)
        elif isinstance(node.value, list):
            elements = [elt for elt in [extract_expression(ast.Constant(value=elt), analyzer) for elt in node.value] if isinstance(elt, lc.ExpressionConstant)]
            return lc.ExpressionConstantList(elements=elements)
    #
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        args = [elt.constant for elt in [extract_expression(arg, analyzer) for arg in node.args] if isinstance(elt, lc.ExpressionConstant)]
        return lc.ExpressionConstantRange(end_value=args[1] if len(args) > 1 else args[0], start_value=args[0] if len(args) > 1 else 0, step=args[2] if len(args) > 2 else 1)
    #
    return None


# --------------------------------------------------------- #
# ----                EXTRACT CONDITION                ---- #
# --------------------------------------------------------- #

#
def extract_condition(node: ast.AST, analyzer: "ModelAnalyzer") -> Optional[lc.Condition]:
    """
    Extracts a Condition object from an AST node.

    Args:
        node (ast.AST): The AST node to analyze.
        analyzer (ModelAnalyzer, optional): The analyzer instance to access global constants.

    Returns:
        Optional[lc.Condition]: The extracted condition or None if not applicable.
    """

    #
    if isinstance(node, ast.Compare):
        left = extract_expression(node.left, analyzer)
        if left is None:
            return None
        ops = [op.__class__.__name__.lower() for op in node.ops]
        comparators = [comp for comp in [extract_expression(comp, analyzer) for comp in node.comparators] if comp is not None]
        if len(ops) == 1 and len(comparators) >= 1:
            return lc.ConditionBinary(elt1=left, cond_operator=ops[0], elt2=comparators[0])
    #
    elif isinstance(node, ast.BoolOp):
        values = [elt for elt in [extract_condition(val, analyzer) or extract_expression(val, analyzer) for val in node.values] if elt is not None]
        op = "and" if isinstance(node.op, ast.And) else "or"
        if len(values) < 2:
            return None
        result = values[0]
        for val in values[1:]:
            result = lc.ConditionBinary(elt1=result, cond_operator=op, elt2=val)
        return cast(lc.Condition, result)
    #
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = extract_condition(node.operand, analyzer) or extract_expression(node.operand, analyzer)
        if operand is not None:
            return lc.ConditionUnary(elt=operand, cond_operator="not")
    #
    return None


# --------------------------------------------------------- #
# ----                   EXTRACT CALL                  ---- #
# --------------------------------------------------------- #


#
def extract_call_arg_value(node: ast.AST, analyzer: "ModelAnalyzer") -> lc.Expression:
    """
    _summary_

    Args:
        node (ast.AST): _description_
        analyzer (ModelAnalyzer): _description_

    Returns:
        lc.Expression: _description_
    """

    #
    expr: Optional[lc.Expression]
    arg_value: lc.Expression
    #
    expr = extract_expression(node=node, analyzer=analyzer)
    #
    arg_value = lc.Expression()  # Error / Default value
    #
    if expr is not None:
        #
        arg_value = expr

    # TODO: other things can appen here too

    #
    return arg_value


#
def extract_call(node: ast.Call, analyzer: "ModelAnalyzer") -> tuple[str, list[lc.Expression], dict[str, lc.Expression], list[lc.FlowControlInstruction]]:
    """
    ast.Call:
        func (ast.Name | ast.Attribute)
        args (list[ast.AST])
        keywords (list[ast.keyword])

    ast.keyword:
        arg (str)
        value (ast.Node)

    Args:
        node (ast.AST): _description_
        analyzer (ModelAnalyzer): _description_

    Returns:
        tuple[str, list[lc.Expression], dict[lc.Expression]]: _description_
    """

    # Get function name
    func_name: str = extract_name_or_attribute(node=node.func)

    # Init instructions to do before
    instructions_to_do_before: list[lc.FlowControlInstruction] = []

    # init func call args
    func_call_args: list[lc.Expression] = []

    # init func call keywords
    func_call_keywords: dict[str, lc.Expression] = {}

    # Extract simple arguments
    arg: ast.AST
    for arg in node.args:
        #
        func_call_args.append( extract_call_arg_value(node=arg, analyzer=analyzer) )

    # Extract keyword arguments
    kw: ast.keyword
    for kw in node.keywords:
        #
        if kw.arg is None:
            print(f"Error, arg is None : {kw}")
            continue
        #
        keyword_name: str = kw.arg

        #
        func_call_keywords[keyword_name] = extract_call_arg_value(node=kw.value, analyzer=analyzer)

    #
    return func_name, func_call_args, func_call_keywords, instructions_to_do_before


#
def extract_layer_call(node: ast.Call, var_name: str, layer_type: str, analyzer: "ModelAnalyzer", add_arguments_todo: bool = True) -> lc.Layer:
    """
    _summary_

    Args:
        node (ast.Call): _description_
        var_name (str): _description_
        layer_type (str): _description_
        analyzer (ModelAnalyzer): _description_

    Returns:
        lc.Layer: _description_
    """

    # On recup les infos du call
    layer_name: str
    layer_call_args: list[lc.Expression]
    layer_call_keywords: dict[str, lc.Expression]
    instructions_to_do_before: list[lc.FlowControlInstruction]
    layer_name, layer_call_args, layer_call_keywords, instructions_to_do_before = extract_call(node=node, analyzer=analyzer)

    # On ajoute le layer à la liste des layers du block
    layer: lc.Layer = lc.Layer(
        layer_var_name=var_name,
        layer_type=layer_type,
        layer_parameters_kwargs={}  # TODO: to complete after complete analysis -> check layer type (Block / Base layer)
    )

    #
    # print(f"DEBUG | extract_layer_call = {node} | layer = {layer} | add_argument_todo = {add_arguments_todo}")

    #
    if add_arguments_todo:
        analyzer.layers_arguments_todo[ layer ] = ( layer_call_args, layer_call_keywords )

    #
    return layer



# --------------------------------------------------------- #
# ----               PROCESS EXPRESSION                ---- #
# --------------------------------------------------------- #

#
def process_expression(node: ast.AST, flow_control: List[lc.FlowControlInstruction], analyzer: "ModelAnalyzer") -> str:
    """
    Processes a complex expression, adding necessary flow control instructions and returning the final variable name.

    Args:
        node (ast.AST): The AST node representing the expression.
        flow_control (list[lc.FlowControlInstruction]): The flow control list to append instructions to.
        analyzer (ModelAnalyzer): The analyzer instance to access global constants.

    Returns:
        str: The variable name holding the expression's result.
    """

    #
    if isinstance(node, ast.BinOp):
        # Handle binary operations like `y * RANDOM_CONSTANT2`
        left_var = process_expression(node.left, flow_control, analyzer)
        right_var = process_expression(node.right, flow_control, analyzer)
        op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        op = op_map.get(type(node.op), None)
        if not op:
            return left_var  # Fallback if operation not supported
        result_var = f"temp_{id(node)}"
        flow_control.append(
            lc.FlowControlBasicBinaryOperation(
                output_var_name=result_var,
                input1_var_name=left_var,
                operation=op,
                input2_var_name=right_var
            )
        )
        return result_var
    #
    elif isinstance(node, ast.Name) or isinstance(node, ast.Constant):
        expr = extract_expression(node, analyzer)
        if not expr:
            return ""
        if isinstance(expr, lc.ExpressionVariable):
            return expr.var_name
        # For constants, create a temporary variable
        temp_var = f"temp_{id(node)}"
        type_str = "int" if isinstance(expr, lc.ExpressionConstantNumeric) and isinstance(expr.constant, int) else "float" if isinstance(expr, lc.ExpressionConstantNumeric) else "str" if isinstance(expr, lc.ExpressionConstantString) else "list"
        flow_control.append(
            lc.FlowControlVariableInit(
                var_name=temp_var,
                var_type=type_str,
                var_value=expr
            )
        )
        return temp_var
    #
    return ""


#
def check_expression_type(expr: lc.Expression, type: str) -> None:

    # TODO: raise error if expression isn't compatible with given type

    #
    pass


# --------------------------------------------------------- #
# ----              CLASS  MODEL ANALYZER              ---- #
# --------------------------------------------------------- #

#
class ModelAnalyzer(ast.NodeVisitor):


    # --------------------------------------------------------- #
    # ----               INIT MODEL ANALYZER               ---- #
    # --------------------------------------------------------- #

    #
    def __init__(self, layers_filepath: str = "layers.json") -> None:
        """
        Initializer of the ModelAnalyzer class.

        Attributes:
            model_blocks (dict[str, lc.ModelBlock]): List of all blocks analyzed, indexed by their block ID (e.g., name).
            main_block (str): ID of the main block, given in sys.argv with `--main-block=<MainBlockName>`.
            current_model_visit (list[str]): Stack of current blocks being visited, access top with [-1].
            current_function_visit (str): Name of the current visited function.
            sub_block_counter (dict[str, int]): Counter for naming sub-blocks (e.g., ModuleList, Sequential).
            global_constants (dict[str, tuple[str, Any]]): Global constants defined outside classes.
        """

        # List of all blocks analyzed, indexed by their block ID (e.g., name).
        self.model_blocks: Dict[str, lc.ModelBlock] = {}

        # ID of the main block, given in sys.argv with `--main-block <MainBlockName>`.
        self.main_block: str = ""

        # Stack of current blocks being visited, access top with [-1].
        self.current_model_visit: List[str] = []

        # Name of the current visited function.
        self.current_function_visit: str = ""

        # Counter for naming sub-blocks (e.g., ModuleList, Sequential).
        self.sub_block_counter: Dict[str, int] = {}

        # Global constants defined outside classes.
        self.global_constants: Dict[str, tuple[str, Any]] = {}

        # Layer / Function Call arguments first extractions
        #   layer or function call -> ( args, kwargs )
        self.layers_arguments_todo: dict[lc.Layer | lc.FlowControlFunctionCall, tuple[list[lc.Expression], dict[str, lc.Expression]]] = {}

        #
        self.layers: dict[str, ll.BaseLayerInfo] = ll.load_layers_dict(filepath=layers_filepath)


    # --------------------------------------------------------- #
    # ----                 ARGUMENTS APPLY                 ---- #
    # --------------------------------------------------------- #

    #
    def found_arg_idx(self, arg_name: str, arg_lst: list[tuple[str, tuple[str, Any]]], layer_type: str) -> int:

        #
        i: int
        for i in range(len(arg_lst)):

            #
            if arg_lst[i][0] == arg_name:

                #
                return i

        #
        raise KeyError(f"Error: Argument `{arg_name}` not found in arg_lst : {arg_lst} of layer type {layer_type} !")


    #
    def _apply_layer_argument(self, layer: lc.Layer, args: list[lc.Expression], kwargs: dict[str, lc.Expression]) -> None:
        """
        _summary_
        """

        #
        args_lst: list[tuple[str, tuple[str, lc.Expression]]]

        #
        if layer.layer_type not in self.layers:

            #
            if layer.layer_type not in self.model_blocks:

                #
                raise NotImplementedError(f"Error: Unsupported layer type : {layer.layer_type} !")

            # liste de (nom de l'argument, (type, valeur par défaut))
            args_lst = [ (arg_name, arg_type_and_default_value) for arg_name, arg_type_and_default_value in self.model_blocks[layer.layer_type].block_parameters.items() ]

        #
        else:

            #
            layer_info: ll.BaseLayerInfo = self.layers[layer.layer_type]

            # liste de (nom de l'argument, (type, valeur par défaut))
            args_lst = [ (arg_name, arg_type_and_default_value) for arg_name, arg_type_and_default_value in layer_info.parameters.items() ]

        #
        res_args: dict[str, lc.Expression] = {}

        #
        arg: str
        for arg in kwargs:

            #
            arg_idx: int = self.found_arg_idx(arg, args_lst, layer.layer_type)

            #
            check_expression_type(kwargs[arg], args_lst[arg_idx][1][0])

            #
            res_args[arg] = kwargs[arg]

            #
            args_lst.pop(arg_idx)

        #
        arg_value: lc.Expression
        for arg_value in args:

            #
            if not args_lst:

                #
                raise IndexError(f"Error: too much arguments given for layer parameters of layer type = {layer.layer_type} :\nargs = {args}\nkwargs = {kwargs}")

            #
            check_expression_type(arg_value, args_lst[0][1][0])

            #
            res_args[args_lst[0][0]] = arg_value

            #
            args_lst.pop(0)

        #
        for i in range(len(args_lst)):

            #
            if args_lst[i][1][1] is None:

                #
                raise IndexError(f"Error: argument {args_lst[i][0]} of layer ")

            #
            res_args[args_lst[i][0]] = args_lst[i][1][1]

        #
        # print(f"DEBUG args layers | {layer} -> {res_args}")

        #
        layer.layer_parameters_kwargs = res_args

    #
    def _apply_fn_call_argument(self, fn_call: lc.FlowControlFunctionCall, args: list[lc.Expression], kwargs: dict[str, lc.Expression]) -> None:
        """
        _summary_
        """

        # TODO
        pass


    #
    def _apply_layers_or_fn_call_arguments(self) -> None:
        """
        _summary_
        """

        #
        layer_or_fcall: lc.Layer | lc.FlowControlFunctionCall
        for layer_or_fcall in self.layers_arguments_todo:

            #
            # print(f"DEBUG | {layer_or_fcall}")

            #
            if isinstance(layer_or_fcall, lc.Layer):

                #
                self._apply_layer_argument(
                        layer=layer_or_fcall,
                        args=self.layers_arguments_todo[layer_or_fcall][0],
                        kwargs=self.layers_arguments_todo[layer_or_fcall][1]
                )

            #
            elif isinstance(layer_or_fcall, lc.FlowControlFunctionCall):

                #
                self._apply_fn_call_argument(
                        fn_call=layer_or_fcall,
                        args=self.layers_arguments_todo[layer_or_fcall][0],
                        kwargs=self.layers_arguments_todo[layer_or_fcall][1]
                )



    # --------------------------------------------------------- #
    # ----                  CLASS VISITOR                  ---- #
    # --------------------------------------------------------- #

    #
    def _is_torch_module_class(self, node: ast.ClassDef) -> bool:
        """
        Indicates if the given ClassDef node is a nn.Module subclass.

        Args:
            node (ast.ClassDef): Node to check.

        Returns:
            bool: True if subclass of nn.Module, else False.
        """

        # Searching for `nn.Module`
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True

        # Not foud `nn.Module`
        return False

    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Called when the AST visitor detects a class definition.

        Args:
            node (ast.ClassDef): The class node to visit.

        Raises:
            NameError: If two classes have the same name.
        """

        # We will ignore non pytorch module classes
        if not self._is_torch_module_class(node):
            return

        # Get the name of the module block class
        block_name: str = node.name
        if block_name in self.model_blocks:
            raise NameError(f"ERROR: Duplicate class name detected: {block_name}")

        # Checking if it can be a Main Block, along with basic Main block name, if --main-block wasn't used.
        if self.main_block == "":
            #
            if block_name in ["MainModel", "MainNet", "Model", "Net"]:
                #
                self.main_block = block_name

        # Adding the discovered block to the list of all blocks
        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)

        # Indicates that we are currently visiting this block
        self.current_model_visit.append(block_name)

        # Initialy, no models have direct sub-blocks for them, so init to 0
        self.sub_block_counter[block_name] = 0

        # Continue the visit
        self.generic_visit(node)

        # After the recursive visit, indicate that we are done with this block
        self.current_model_visit.pop()


    # --------------------------------------------------------- #
    # ----                FUNCTION VISITOR                 ---- #
    # --------------------------------------------------------- #

    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Called when the AST visitor detects a function definition.

        Args:
            node (ast.FunctionDef): The function node to visit.
        """

        # We will ignore standalone functions, that are not inside a Pytorch Module Block Class
        if not self.current_model_visit:
            return

        # Indicates that we are visiting this function
        self.current_function_visit = node.name

        # Different
        if node.name == "__init__":
            self._analyze_init_method(node)
        else:
            self._analyse_other_method(node)

        #
        self.current_function_visit = ""
        self.generic_visit(node)


    #
    def _analyse_other_method(self, node: ast.FunctionDef) -> None:
        """
        Analyzes other methods in the class.

        Args:
            node (ast.FunctionDef): The function node.
        """

        # Getting the current block
        current_block: lc.ModelBlock = self.model_blocks[self.current_model_visit[-1]]

        # Creating the function container
        func = lc.BlockFunction(
            function_name=node.name,
            function_arguments=self._get_node_arguments(node=node),
            model_block=current_block
        )

        # Adding the function to the current block
        current_block.block_functions[node.name] = func

        # Process all the control flow instructions of the function body
        for stmt in node.body:
            self._process_statement(stmt, func.function_flow_control)


    #
    def _analyze_init_method(self, node: ast.FunctionDef) -> None:
        """
        Analyzes the __init__ method to extract parameters and layers.

        Args:
            node (ast.FunctionDef): The __init__ function node.
        """

        #
        current_block = self.model_blocks[self.current_model_visit[-1]]

        # Extract block arguments
        current_block.block_parameters = self._get_node_arguments(node=node)

        # Process body for layers and variables
        for stmt in node.body:
            self._process_stmt_block_init(stmt=stmt, current_block=current_block)

    #
    def _process_stmt_block_init(self, stmt: ast.AST, current_block: lc.ModelBlock) -> None:
        """
        _summary_

        Args:
            stmt (ast.AST): _description_
        """

        # Assign, layer preparation
        # if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Attribute):
        if isinstance(stmt, ast.Assign) or isinstance(stmt, ast.AnnAssign):

            #
            target: ast.Attribute

            #
            if isinstance(stmt, ast.Assign):

                #
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Attribute):

                    # Not supported
                    return

                # Get the target of the assignment
                target = stmt.targets[0]

            #
            elif isinstance(stmt, ast.AnnAssign):

                #
                if not isinstance(stmt.target, ast.Attribute):

                    # Not supported
                    return

                #
                target = stmt.target

            # If it is a self target
            if isinstance(target.value, ast.Name) and target.value.id == "self":

                #
                var_name = target.attr

                # If the value is a call
                if isinstance(stmt.value, ast.Call):

                    # Get the layer type
                    layer_type = self.get_layer_type(stmt.value.func)

                    # Manages ModuleList & Sequential
                    if layer_type in {"ModuleList", "Sequential"}:
                        self._handle_container(var_name, layer_type, stmt.value, current_block)

                    # Sinon, on a notre layer
                    else:

                        # On récupère le layer
                        layer: lc.Layer = extract_layer_call(node=stmt.value, var_name=var_name, layer_type=layer_type, analyzer=self)

                        # On ajoute le layer à la liste des layers du block
                        current_block.block_layers[var_name] = layer

                #
                # elif isinstance(stmt.value, ast.For):
                #     self._handle_loop_init(var_name=var_name, for_node=stmt.value, block=current_block)


        # For the moment, we will not support all the "dynamic blocks"
        # Meaning, that all the dimensions and all the important parameters have to be fixed.

        # Same, no class variables, we will only work with global variables for now



    #
    def _get_node_arguments(self, node: ast.FunctionDef) -> dict[str, tuple[str, Any]]:
        """
        Extract a function arguments inside of a class.

        Args:
            node (ast.FunctionDef): _description_

        Returns:
            dict[str, tuple[str, Any]]: _description_
        """

        # Extract function arguments:

        # Preparing the argument dict
        args: dict[str, tuple[str, Any]] = {}

        # Preparing var types
        arg_name: str
        arg_type: str
        default: Any

        # browse the function arguments
        for arg in node.args.args:

            # Get thet argument name
            arg_name = arg.arg

            # Skip 'self'
            if arg_name == "self":
                continue

            # Checking the argument type annotation
            if isinstance(arg.annotation, ast.Name):
                arg_type = arg.annotation.id
            elif arg.annotation is not None:
                arg_type = ast.dump(arg.annotation)
            else:
                arg_type = "Any"

            # Trying to get argument default value
            default = None
            if arg_name in node.args.defaults:
                default = extract_expression(node.args.defaults[node.args.args.index(arg) - len(node.args.defaults)], self)

            # Adding the argument to the argument dict
            args[arg_name] = (arg_type, default)

        #
        return args


    #
    def _handle_container(self, var_name: str, container_type: str, call_node: ast.Call, block: lc.ModelBlock) -> None:
        """
        Handles nn.ModuleList and nn.Sequential by creating sub-blocks and defining their forward methods.

        Args:
            var_name (str): Name of the container variable.
            container_type (str): Type of container ("ModuleList" or "Sequential").
            call_node (ast.Call): The AST node of the container call.
            block (lc.ModelBlock): The current model block.
        """

        # Create a unique sub-block name
        sub_block_name = f"Block{container_type}_{self.current_model_visit[-1]}_{self.sub_block_counter[self.current_model_visit[-1]]}"
        self.sub_block_counter[self.current_model_visit[-1]] += 1

        # Creating the sub-block
        sub_block = lc.ModelBlock(block_name=sub_block_name)
        self.model_blocks[sub_block_name] = sub_block

        # Initialize layers list
        layers: List[lc.Layer] = []

        # Init loop & layer variables
        i: int
        layer: lc.Layer

        # Handle arguments
        if call_node.args:
            for arg in call_node.args:

                #
                # Call: A call expression, such as func(...)
                #
                if isinstance(arg, ast.Call):

                    # Get the layer type
                    layer_type = self.get_layer_type(arg.func)

                    # Extract the layer and add it to the layers list
                    layer = extract_layer_call(node=arg, var_name="#TODO", layer_type=layer_type, analyzer=self)
                    layers.append( layer )

                #
                # GeneratorExp: A generator expression, such as (var for var in iterable)
                #
                # GeneartorExp:
                #   elt: (ast.AST)
                #   generators: (list[ast.comprehension])
                #
                # comprehension:
                #    target: (ast.AST)
                #    iter: (ast.AST)
                #    ifs: (_)
                #    is_async: (_)
                #
                elif isinstance(arg, ast.GeneratorExp):

                    # On récupère l'element
                    elt = arg.elt

                    # Si on a un appel (on suppose que c'est un Layer, on ne suppose pas des appels à des functions custom de code qui pourraient renvoyer des layer)
                    if isinstance(elt, ast.Call):

                        # On récupère le type du layer
                        layer_type = self.get_layer_type(elt.func)

                        # On ne supporte que des ranges simples
                        if arg.generators and isinstance(arg.generators[0].iter, ast.Call) and isinstance(arg.generators[0].iter.func, ast.Name) and arg.generators[0].iter.func.id == "range":

                            # On récupère les arguments du range
                            range_args = [expr.constant for expr in [extract_expression(a, self) for a in arg.generators[0].iter.args] if expr is not None and isinstance(expr, lc.ExpressionConstant)]

                            # Si pas de problèmes au niveau des arguments
                            if len(range_args) >= 1:

                                # On décompte la taille du range
                                count = range_args[0] if len(range_args) == 1 else range_args[1] - range_args[0]

                                # Pour chaque élément du range, on va créer un layer
                                for i in range(int(count)):

                                    #
                                    layer_range: lc.Layer = extract_layer_call(node=elt, var_name=f"{var_name}[{i}]", layer_type=layer_type, analyzer=self)

                                    #
                                    layers.append(layer_range)

        # Define forward method
        forward_func = lc.BlockFunction(
                            function_name="forward",
                            function_arguments={"x": ("Any", None)},
                            model_block=sub_block
        )

        # Add the forward function to the sub-block
        sub_block.block_functions["forward"] = forward_func

        # As you can see in the forward function arguments, the input is the variable 'x'
        current_input: lc.ExpressionVariable = lc.ExpressionVariable("x")

        # For each layers
        for i, layer in enumerate(layers):

            #
            sub_block.block_layers[layer.layer_var_name] = layer

            #
            output_var = f"out_{i}"

            #
            forward_func.function_flow_control.append(
                lc.FlowControlLayerPass(output_variables=[output_var], layer_name=layer.layer_var_name, layer_arguments={"x": current_input})
            )

            #
            if container_type == "Sequential":

                #
                current_input = lc.ExpressionVariable(output_var)

        #
        if container_type == "Sequential":

            #
            forward_func.function_flow_control.append(lc.FlowControlReturn(return_variables=[current_input.var_name]))

        #
        elif container_type == "ModuleList":

            #
            output_vars = [f"out_{i}" for i in range(len(layers))]

            #
            forward_func.function_flow_control.append(lc.FlowControlReturn(return_variables=output_vars))

        # Add to parent block
        block.block_layers[var_name] = lc.Layer(
            layer_var_name=var_name,
            layer_type=container_type,
            layer_parameters_kwargs={"sub_block": sub_block_name}
        )

    #
    def _handle_loop_init(self, var_name: str, for_node: ast.For, block: lc.ModelBlock) -> None:
        """
        Handles loop-based initialization of layers (e.g., ModuleList).

        Args:
            var_name (str): Name of the variable being initialized.
            for_node (ast.For): The for loop node.
            block (lc.ModelBlock): The current model block.
        """

        #
        sub_block_name = f"BlockModuleList_{self.current_model_visit[-1]}_{self.sub_block_counter[self.current_model_visit[-1]]}"

        #
        self.sub_block_counter[self.current_model_visit[-1]] += 1

        #
        sub_block = lc.ModelBlock(block_name=sub_block_name)
        self.model_blocks[sub_block_name] = sub_block

        #
        if not isinstance(for_node.iter, ast.Name) or not isinstance(for_node.target, ast.Name):
            return

        # Extract loop details
        iterator = extract_expression(for_node.iter, self) or for_node.iter.id

        #
        iterable_var = for_node.target.id

        #
        layers: List[lc.Layer] = []

        #
        for stmt in for_node.body:

            #
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):

                #
                layer_type = self.get_layer_type(stmt.value.func)

                #
                params = {kw.arg: extract_expression(kw.value, self) or kw.value for kw in stmt.value.keywords if kw.arg is not None}
                layers.append(lc.Layer(f"layer_{len(layers)}", layer_type, params))

        # Add layers to sub-block
        for i, layer in enumerate(layers):
            sub_block.block_layers[f"layer_{i}"] = layer

        # Define forward method
        forward_func = lc.BlockFunction(function_name="forward", function_arguments={"x": ("Any", None)}, model_block=sub_block)

        # Add forward method
        sub_block.block_functions["forward"] = forward_func

        #
        output_vars = []
        for i in range(len(layers)):

            #
            output_var = f"out_{i}"

            #
            forward_func.function_flow_control.append(
                lc.FlowControlLayerPass(
                    output_variables=[output_var],
                    layer_name=f"layer_{i}",
                    layer_arguments={"x": lc.ExpressionVariable("x")}
                )
            )

            #
            output_vars.append(output_var)

        #
        forward_func.function_flow_control.append(lc.FlowControlReturn(return_variables=output_vars))

        # Add to parent block
        block.block_layers[var_name] = lc.Layer(
            layer_var_name=var_name,
            layer_type="ModuleList",
            layer_parameters_kwargs={"sub_block": sub_block_name, "iterator": iterator, "iterable_var": iterable_var}
        )


    #
    def _process_statement(self, stmt: ast.AST, flow_control: List[lc.FlowControlInstruction]) -> None:
        """
        Processes a statement and adds it to the flow control list.

        Args:
            stmt (ast.AST): The statement to process.
            flow_control (list[lc.FlowControlInstruction]): The flow control list to append to.
        """

        #
        if isinstance(stmt, ast.Assign):

            #
            target = stmt.targets[0].id if isinstance(stmt.targets[0], ast.Name) else None

            #
            value = extract_expression(stmt.value, self)

            #
            if target and value:

                #
                flow_control.append(lc.FlowControlVariableAssignment(var_name=target, var_value=value))

            #
            elif isinstance(stmt.value, ast.Call):

                #
                outputs = [t.id for t in stmt.targets if isinstance(t, ast.Name)]

                #
                func_name = self.get_layer_type(stmt.value.func)

                #
                args = {kw.arg: extract_expression(kw.value, self) or kw.value for kw in stmt.value.keywords if kw.arg is not None}

                #
                if func_name in self.model_blocks[self.current_model_visit[-1]].block_layers:

                    #
                    flow_control.append(lc.FlowControlLayerPass(outputs, func_name, args))
                else:

                    #
                    flow_control.append(lc.FlowControlFunctionCall(outputs, func_name, args))

        #
        elif isinstance(stmt, ast.AugAssign):

            #
            self.visit_AugAssign(stmt)

        #
        elif isinstance(stmt, ast.For):

            #
            if not isinstance(stmt.target, ast.Name):
                return

            #
            iterator = extract_expression(stmt.iter, self) or (stmt.iter.id if isinstance(stmt.iter, ast.Name) else None)
            if iterator is None:
                return

            #
            flow_control_loop: lc.FlowControlForLoop = lc.FlowControlForLoop(
                iterable_var_name=stmt.target.id,
                iterator=iterator,
                flow_control_instructions=[]
            )

            #
            flow_control.append(flow_control_loop)

            #
            for sub_stmt in stmt.body:

                #
                self._process_statement(sub_stmt, flow_control_loop.flow_control_instructions)

        #
        elif isinstance(stmt, ast.While):

            #
            condition = extract_condition(stmt.test, self)
            if condition is None:
                return

            #
            flow_control_while: lc.FlowControlWhileLoop = lc.FlowControlWhileLoop(
                condition=condition,
                flow_control_instructions=[]
            )

            #
            flow_control.append(flow_control_while)
            for sub_stmt in stmt.body:

                #
                self._process_statement(sub_stmt, flow_control_while.flow_control_instructions)

        #
        elif isinstance(stmt, ast.If):

            #
            condition = extract_condition(stmt.test, self)
            if condition is None:
                return

            #
            sub_func_name = f"cond_{len(self.model_blocks[self.current_model_visit[-1]].block_functions)}"

            #
            sub_func = lc.BlockFunction(sub_func_name, {"input": ("Any", None)}, self.model_blocks[self.current_model_visit[-1]])

            #
            self.model_blocks[self.current_model_visit[-1]].block_functions[sub_func_name] = sub_func

            #
            for sub_stmt in stmt.body:
                self._process_statement(sub_stmt, sub_func.function_flow_control)

            #
            flow_control_subcall: lc.FlowControlSubBlockFunctionCall = lc.FlowControlSubBlockFunctionCall(
                output_variables=["output"],
                function_called=sub_func_name,
                function_arguments={"input": "x"}
            )

            #
            flow_control.append(flow_control_subcall)

            #
            self.model_blocks[self.current_model_visit[-1]].block_layers[sub_func_name] = lc.LayerCondition(
                layer_var_name=sub_func_name,
                layer_conditions_blocks={condition: flow_control_subcall}
            )

        #
        elif isinstance(stmt, ast.Return):

            #
            expr: Optional[lc.Expression]

            #
            returns: List[str] = []

            #
            if isinstance(stmt.value, ast.Tuple):

                #
                for val in stmt.value.elts:

                    #
                    expr = extract_expression(val, self)

                    #
                    if expr is None:
                        continue

                    #
                    elif isinstance(expr, lc.ExpressionVariable):
                        returns.append(expr.var_name)

                    #
                    elif isinstance(val, ast.Name):
                        returns.append(val.id)

            #
            elif isinstance(stmt.value, ast.Name):
                returns.append(stmt.value.id)

            #
            flow_control.append(lc.FlowControlReturn(return_variables=returns))

    #
    def get_layer_type(self, func: ast.AST) -> str:
        """
        Extracts the layer or function type from a call.

        Args:
            func (ast.AST): The function node.

        Returns:
            str: The type or name of the function/layer.
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
    # ----                 ASSIGN VISITOR                  ---- #
    # --------------------------------------------------------- #

    #
    def visit_Assign(self, node: ast.Assign) -> None:
        """
        Handles basic assignment statements (e.g., `x = 5`).

        Args:
            node (ast.Assign): The assignment node to visit.
        """

        #
        if len(node.targets) != 1:
            return

        #
        target = node.targets[0]
        value = extract_expression(node.value, self)

        # Global constant
        if not self.current_model_visit and not self.current_function_visit and isinstance(target, ast.Name) and value:

            #
            type_str = "int" if isinstance(value, lc.ExpressionConstantNumeric) and isinstance(value.constant, int) else "float" if isinstance(value, lc.ExpressionConstantNumeric) else "str" if isinstance(value, lc.ExpressionConstantString) else "list"

            #
            self.global_constants[target.id] = (type_str, value)

            #
            return

        # Inside function
        if self.current_model_visit and self.current_function_visit:

            #
            current_block = self.model_blocks[self.current_model_visit[-1]]

            #
            if self.current_function_visit in current_block.block_functions:

                #
                func = current_block.block_functions[self.current_function_visit]

                #
                if isinstance(target, ast.Name) and value:

                    #
                    func.function_flow_control.append(
                        lc.FlowControlVariableAssignment(var_name=target.id, var_value=value)
                    )

            #
            return

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """
        Handles augmented assignment statements (e.g., `x += y * RANDOM_CONSTANT2`).

        Args:
            node (ast.AugAssign): The augmented assignment node to visit.
        """

        #
        target = node.target
        if not isinstance(target, ast.Name):
            return

        # Only handle inside functions
        if not (self.current_model_visit and self.current_function_visit):
            return

        current_block = self.model_blocks[self.current_model_visit[-1]]
        if self.current_function_visit not in current_block.block_functions:
            return

        func = current_block.block_functions[self.current_function_visit]
        op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        op = op_map.get(type(node.op), None)
        if not op:
            return

        # Process the right-hand side expression
        rhs_var = process_expression(node.value, func.function_flow_control, self)
        if not rhs_var:
            return

        # Add the augmented operation
        func.function_flow_control.append(
            lc.FlowControlBasicBinaryOperation(
                output_var_name=target.id,
                input1_var_name=target.id,
                operation=op,
                input2_var_name=rhs_var
            )
        )

        # No need to call generic_visit since we're handling the node fully here

    #
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """
        Handles annotated assignment statements (e.g., `x: int = 0`).

        Args:
            node (ast.AnnAssign): The annotated assignment node to visit.
        """

        #
        target = node.target
        value = extract_expression(node.value, self) if node.value else None
        annotation = node.annotation.id if isinstance(node.annotation, ast.Name) else ast.dump(node.annotation) if node.annotation else "Any"

        # Global constant
        if not self.current_model_visit and not self.current_function_visit and isinstance(target, ast.Name):

            #
            if value:
                self.global_constants[target.id] = (annotation, value)

            # There is nothing more to analyse in this ast tree path
            return

        # Inside function
        if self.current_model_visit and self.current_function_visit and isinstance(target, ast.Name):

            #
            current_block = self.model_blocks[self.current_model_visit[-1]]

            #
            if self.current_function_visit in current_block.block_functions:

                #
                func = current_block.block_functions[self.current_function_visit]

                #
                if value:

                    #
                    func.function_flow_control.append(
                        lc.FlowControlVariableInit(var_name=target.id, var_type=annotation, var_value=value)
                    )

                #
                else:

                    #
                    func.function_flow_control.append(
                        lc.FlowControlVariableInit(var_name=target.id, var_type=annotation)
                    )

            #
            return

        #
        ast.NodeVisitor.generic_visit(self, node)

    #
    def visit_AssignStmt(self, node: ast.AST) -> None:
        """
        Placeholder for AssignStmt (not a standard AST node).

        Args:
            node (ast.AST): The node to visit.
        """

        # No standard AssignStmt in Python ast
        ast.NodeVisitor.generic_visit(self, node)

    # --------------------------------------------------------- #
    # ----                 GENERIC VISITOR                 ---- #
    # --------------------------------------------------------- #

    #
    def generic_visit(self, node: ast.AST) -> None:
        """
        Generic visitor for additional node processing.

        Args:
            node (ast.AST): The node to visit.
        """

        # Generic visit
        ast.NodeVisitor.generic_visit(self, node)

    # --------------------------------------------------------- #
    # ----           CLEANING & ERROR DETECTIONS           ---- #
    # --------------------------------------------------------- #

    #
    def cleaning_and_error_detections(self) -> None:
        """
        Cleans unused functions and detects errors in the model blocks.
        """

        #
        for block_name, block in list(self.model_blocks.items()):

            # Check for forward
            if "forward" not in block.block_functions:
                raise ValueError(f"ERROR: Block {block_name} has no forward method.")

            #
            used_funcs = {"forward"}

            #
            for func in block.block_functions.values():

                #
                for instr in func.function_flow_control:

                    #
                    if isinstance(instr, lc.FlowControlSubBlockFunctionCall):
                        used_funcs.add(instr.function_called)

            #
            # block.block_functions = {k: v for k, v in block.block_functions.items() if k in used_funcs}
            #
            # For the moment, we will just warn for unused functions
            # TODO
            pass



#
def extract_from_file(filepath: str, main_block_name: str = "") -> lc.Language1_Model:
    """
    Extracts a neural network model architecture from a PyTorch script file.

    Args:
        filepath (str): Path to the file to extract the architecture from.
        main_block_name (str, optional): Name of the entry point of the model. Defaults to "".

    Raises:
        FileNotFoundError: If the file is not found.

    Returns:
        lc.Language1_Model: The extracted model architecture.
    """

    #
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: file not found: `{filepath}`")

    #
    with open(filepath, "r") as source:
        tree = ast.parse(source.read())

    #
    analyzer = ModelAnalyzer()
    analyzer.visit(tree)
    analyzer._apply_layers_or_fn_call_arguments()
    analyzer.cleaning_and_error_detections()

    #
    lang1: lc.Language1_Model = lc.Language1_Model()
    lang1.main_block = main_block_name or list(analyzer.model_blocks.keys())[0]
    lang1.model_blocks = analyzer.model_blocks
    lang1.global_constants = analyzer.global_constants

    #
    return lang1


#
def list_classes(module: object) -> list[type]:
    """
    List all the classes defined inside a module.

    Args:
        module (object): The module to inspect.

    Returns:
        List[type]: A list of class objects found in the module.
    """

    # Initialize a list to store class objects, with type hint as List[type]
    classes: list[type] = [

        # m is a tuple of precise type :  tuple[str, type] , correponding to (name, value)

        # Extract the second element of the tuple 'm', which is the class object itself
        m[1]

        # Iterate through members of the module that are classes, using inspect.getmembers with inspect.isclass filter
        for m in inspect.getmembers(module, inspect.isclass)
    ]

    # Return the list of class objects
    return classes


#
def import_module_from_filepath(filepath: str) -> object:
    """
    Imports a Python module from a filepath.

    Args:
        filepath (str): The path to the Python file to import as a module.

    Returns:
        object: The imported module object.
    """

    # Extract the module name from the filepath by removing the extension and path
    module_name: str = os.path.splitext(os.path.basename(filepath))[0]

    # Create a module specification using the module name and filepath, spec is of type ModuleSpec
    spec: Optional[importlib.machinery.ModuleSpec] = importlib.util.spec_from_file_location(module_name, filepath)

    # Check for errors
    if spec is None or spec.loader is None:
        raise ImportError(f"Error : can't load module from file : {filepath}")

    # Create a module object from the specification, module type is dynamically determined so using Any
    module: Any = importlib.util.module_from_spec(spec)

    # Execute the module code in the module object's namespace, populating the module
    spec.loader.exec_module(module)

    # Return the imported module object
    return module


#
def get_pytorch_main_model(model_arch: lc.Language1_Model, filepath: str) -> Callable:

    #
    if model_arch.main_block == "":
        #
        raise UserWarning("Error: No main blocks detected in the model architecture !")

    #
    net_module = import_module_from_filepath(filepath)

    #
    if not hasattr(net_module, model_arch.main_block):
        #
        raise UserWarning("Error: The given python script does not have the specified main block !")

    #
    return getattr(net_module, model_arch.main_block)


#
if __name__ == "__main__":

    #
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise UserWarning(f"Error: Usage: python {sys.argv[0]} path_to_model_script.py [--main-block=<MainBlockName>]")

    #
    path_to_file: str = sys.argv[1]
    main_block_name: str = ""
    if len(sys.argv) == 3 and sys.argv[2].startswith("--main-block"):
        main_block_name = sys.argv[2].split("=")[1] if "=" in sys.argv[2] else sys.argv[2].split()[1]

    #
    l1_model: lc.Language1_Model = extract_from_file(filepath=path_to_file, main_block_name=main_block_name)

    #
    print(l1_model)

    #
    main_model_class =  get_pytorch_main_model(model_arch=l1_model, filepath=path_to_file)
    print(main_model_class)
