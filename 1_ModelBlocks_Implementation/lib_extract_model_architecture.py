#
### Import Modules. ###
#
from typing import Optional, Any, Callable
#
import ast
import sys
import os

#
import inspect
import importlib.util
import importlib.machinery

#
import core.lib_impl.lib_classes as lc
import core.lib_impl.lib_layers as ll


#
### Global temporary variable counter to avoid collisions. ###
#
temp_var_counter: int = 0


#
### Function to get the next available temporary variable name. ###
#
def get_next_temp_var_name() -> str:
    """
    Get the next available temporary variable name.

    Returns:
        str: The next available temporary variable name.
    """

    #
    global temp_var_counter

    #
    temp_var_counter += 1

    #
    return f"temp_{temp_var_counter}"


#
### Extract Name or Attribute from AST Node into string. ###
#
def extract_name_or_attribute(node: ast.AST) -> str:
    """
    Recursively extracts a fully qualified name from an AST node.

    Handles simple names (e.g., 'Linear') and attributes (e.g., 'nn.Linear').

    Args:
        node (ast.AST): The node to analyze, typically ast.Name or ast.Attribute.

    Returns:
        str: The extracted name as a string.
    """

    #
    if isinstance(node, ast.Name):
        #
        return node.id

    #
    elif isinstance(node, ast.Attribute):
        #
        ### Build the full qualified name by combining value and attr. ###
        #
        value_name = extract_name_or_attribute(node.value)

        #
        return f"{value_name}.{node.attr}"

    #
    return str(node)


#
### Helper for Extract Expression from AST Node. ###
#
def extract_expression(
    node: ast.AST,
    analyzer: "ModelAnalyzer",
    instructions_to_do_before: list[lc.FlowControlInstruction]
) -> lc.Expression:

    """
    Extracts an lc.Expression from an AST node.

    Args:
        node (ast.AST): The AST node to extract an expression from.
        analyzer (ModelAnalyzer): The analyzer instance.
        instructions_to_do_before (list[lc.FlowControlInstruction]): Instructions to execute before evaluating the expression.

    Returns:
        lc.Expression: The extracted expression object.

    Raises:
        NotImplementedError: If the node type is not supported for expression extraction.
    """

    #
    if isinstance(node, ast.Name):
        #
        return lc.ExpressionVariable(var_name=node.id)

    #
    elif isinstance(node, ast.Constant):
        #
        return lc.ExpressionConstantNumeric(constant=node.value)

    #
    elif isinstance(node, ast.Dict):
        #
        return lc.ExpressionDict(elements={k: extract_expression(v, analyzer, instructions_to_do_before) for k, v in node.items})

    #
    elif isinstance(node, ast.List):
        #
        return lc.ExpressionList(elements=[extract_expression(v, analyzer, instructions_to_do_before) for v in node.elts])

    #
    elif isinstance(node, ast.Set):
        #
        return lc.ExpressionSet(elements=[extract_expression(v, analyzer, instructions_to_do_before) for v in node.elts])

    #
    elif isinstance(node, ast.Slice):
        #
        ### Handle 1D slice expressions like [start:end:step] ###
        #
        start_expr = extract_expression(node.lower, analyzer, instructions_to_do_before) if node.lower is not None else None
        end_expr = extract_expression(node.upper, analyzer, instructions_to_do_before) if node.upper is not None else None
        step_expr = extract_expression(node.step, analyzer, instructions_to_do_before) if node.step is not None else None

        #
        return lc.ExpressionSlice1D(start=start_expr, end=end_expr, step=step_expr)

    #
    elif isinstance(node, ast.Tuple):
        #
        ### Handle tuple expressions like (a, b, c) ###
        ### Multi-dimensional slices are handled in subscript context ###
        #
        return lc.ExpressionTuple(elements=[extract_expression(v, analyzer, instructions_to_do_before) for v in node.elts])

    #
    elif isinstance(node, ast.Index):
        #
        ### Handle single index expressions like [0] or [None] ###
        #
        value_expr = extract_expression(node.value, analyzer, instructions_to_do_before)

        #
        return lc.ExpressionSlice1D(start=value_expr, end=None, step=None)

    #
    elif isinstance(node, ast.BinOp):

        #
        ### Create temporary variable and intermediate instruction to store the binary operation result in instructions_to_do_before. ###
        #
        left_expr: lc.Expression = extract_expression(node.left, analyzer, instructions_to_do_before)
        right_expr: lc.Expression = extract_expression(node.right, analyzer, instructions_to_do_before)

        #
        ### If the left or right expression is not a variable, create a temporary variable and intermediate instruction to store the expression result in instructions_to_do_before. ###
        #
        if not isinstance(left_expr, lc.ExpressionVariable):
            #
            left_instruction_var_name: str = get_next_temp_var_name()
            #
            instruction_left_expr: lc.FlowControlVariableAssignment = lc.FlowControlVariableAssignment(var_name=left_instruction_var_name, var_value=left_expr)
            #
            instructions_to_do_before.append(instruction_left_expr)
            #
            left_expr = lc.ExpressionVariable(var_name=left_instruction_var_name)

        #
        if not isinstance(right_expr, lc.ExpressionVariable):
            #
            right_instruction_var_name: str = get_next_temp_var_name()
            #
            instruction_right_expr: lc.FlowControlVariableAssignment = lc.FlowControlVariableAssignment(var_name=right_instruction_var_name, var_value=right_expr)
            #
            instructions_to_do_before.append(instruction_right_expr)
            #
            right_expr = lc.ExpressionVariable(var_name=right_instruction_var_name)

        #
        ### Create a temporary variable and intermediate instruction to store the binary operation result in instructions_to_do_before. ###
        #
        temp_var_name: str = get_next_temp_var_name()

        #
        ### Convert operator to string ###
        #
        operator_str: str = node.op.__class__.__name__.lower()

        #
        if operator_str == "add":
            #
            operator_str = "+"
        #
        elif operator_str == "sub":
            #
            operator_str = "-"
        #
        elif operator_str == "mult":
            #
            operator_str = "*"
        #
        elif operator_str == "div":
            #
            operator_str = "/"
        #
        elif operator_str == "mod":
            #
            operator_str = "%"
        #
        elif operator_str == "pow":
            #
            operator_str = "^"
        #
        elif operator_str == "lshift":
            #
            operator_str = "<<"
        #
        elif operator_str == "rshift":
            #
            operator_str = ">>"

        #
        ### Add the binary operation instruction to the instructions_to_do_before list. ###
        #
        instructions_to_do_before.append(lc.FlowControlBasicBinaryOperation(
            output_var_name=temp_var_name,
            input1_var_name=left_expr.var_name,
            operation=operator_str,
            input2_var_name=right_expr.var_name
        ))

        #
        ### Return the temporary variable expression. ###
        #
        return lc.ExpressionVariable(var_name=temp_var_name)

    #
    elif isinstance(node, ast.UnaryOp):

        #
        ### Create temporary variable and intermediate instruction to store the unary operation result in instructions_to_do_before. ###
        #
        operand_expr: lc.Expression = extract_expression(node.operand, analyzer, instructions_to_do_before)

        #
        if not isinstance(operand_expr, lc.ExpressionVariable):
            #
            operand_instruction_var_name: str = get_next_temp_var_name()
            #
            instruction_operand_expr: lc.FlowControlVariableAssignment = lc.FlowControlVariableAssignment(var_name=operand_instruction_var_name, var_value=operand_expr)
            #
            instructions_to_do_before.append(instruction_operand_expr)
            #
            operand_expr = lc.ExpressionVariable(var_name=operand_instruction_var_name)

        #
        ### Create the temporary variables. ###
        #
        temp_var_name: str = get_next_temp_var_name()

        #
        ### Convert operator to string ###
        #
        operator_str: str = node.op.__class__.__name__.lower()

        #
        if operator_str == "uadd":
            #
            operator_str = "+"
        #
        elif operator_str == "usub":
            #
            operator_str = "-"
        #
        elif operator_str == "not":
            #
            operator_str = "not"
        #
        elif operator_str == "invert":
            #
            operator_str = "~"

        #
        ### Add the unary operation instruction to the instruction_to_do_before list. ####
        #
        instructions_to_do_before.append(lc.FlowControlBasicUnaryOperation(
            output_var_name=temp_var_name,
            operation=operator_str,
            input_var_name=operand_expr.var_name
        ))

        #
        ### Return the temporary variable expression. ###
        #
        return lc.ExpressionVariable(var_name=temp_var_name)

    #
    elif isinstance(node, ast.Call):

        #
        ### Check if this is a direct library call that should not be decomposed ###
        #
        func_name = extract_name_or_attribute(node.func)
        is_library_call = func_name.startswith(('nn.', 'torch.', 'F.', 'math.', 'np.'))

        #
        if is_library_call:

            #
            ### For library calls, create a simple function call expression ###
            #
            temp_var_name: str = get_next_temp_var_name()

            #
            ### Prepare arguments for the library call ###
            #
            arg_exprs: list[lc.Expression] = []

            #
            for arg in node.args:

                #
                arg_expr = extract_expression(arg, analyzer, instructions_to_do_before)
                #
                arg_exprs.append(arg_expr)

            #
            kwarg_exprs: dict[str, lc.Expression] = {}

            #
            for kw in node.keywords:

                #
                if kw.arg is not None:
                    #
                    kwarg_exprs[kw.arg] = extract_expression(kw.value, analyzer, instructions_to_do_before)

            #
            ### Create a simple function call instruction ###
            #
            ### Convert arguments to the proper format for FlowControlFunctionCall. ###
            #
            function_arguments = {}

            #
            ### Add positional arguments with proper names. ###
            #
            for i, arg in enumerate(arg_exprs):
                #
                function_arguments[str(i)] = arg

            #
            ### Add keyword arguments. ###
            #
            function_arguments.update(kwarg_exprs)

            #
            ### Create a simple function call instruction. ###
            #
            instructions_to_do_before.append(
                lc.FlowControlFunctionCall(
                    output_variables=[temp_var_name],
                    function_called=func_name,
                    function_arguments=function_arguments
                )
            )

            #
            return lc.ExpressionVariable(var_name=temp_var_name)

        #
        ### Check if this is a method call on an object variable ###
        #
        elif isinstance(node.func, ast.Attribute):
            #
            ### This is a method call like obj.method() or self.method() ###
            #
            method_name = node.func.attr
            object_expr = extract_expression(node.func.value, analyzer, instructions_to_do_before)

            #
            ### Create a temp variable for the result ###
            #
            temp_var_name: str = get_next_temp_var_name()

            #
            ### Prepare arguments with object as first argument ###
            #
            arg_exprs: list[lc.Expression] = [object_expr]  # Object as first argument
            for arg in node.args:
                arg_expr = extract_expression(arg, analyzer, instructions_to_do_before)
                arg_exprs.append(arg_expr)

            #
            ### Prepare keyword arguments ###
            #
            kwarg_exprs: dict[str, lc.Expression] = {}
            for kw in node.keywords:
                if kw.arg is not None:
                    kwarg_exprs[kw.arg] = extract_expression(kw.value, analyzer, instructions_to_do_before)

            #
            ### Create function call instruction ###
            #
            instructions_to_do_before.append(
                lc.FlowControlFunctionCall(
                    output_variables=[temp_var_name],
                    function_called=method_name,
                    function_arguments={str(i): arg for i, arg in enumerate(arg_exprs)} | kwarg_exprs
                )
            )

            #
            return lc.ExpressionVariable(var_name=temp_var_name)

        #
        ### For non-library calls, use the original complex decomposition logic ###
        #
        ### Prepare the function call arguments. ###
        #
        arg_exprs: list[lc.Expression] = []
        #
        for arg in node.args:
            #
            arg_expr = extract_expression(arg, analyzer, instructions_to_do_before)
            arg_exprs.append(arg_expr)

        #
        ### Prepare keyword argument expressions. ###
        #
        kwarg_exprs: dict[str, lc.Expression] = {}
        #
        for kw in node.keywords:
            #
            kwarg_exprs[kw.arg] = extract_expression(kw.value, analyzer, instructions_to_do_before)

        #
        ### Prepare the function being called. ###
        #
        func_expr = extract_expression(node.func, analyzer, instructions_to_do_before)

        #
        ### If the function expression is not a variable, assign it to a temp variable. ###
        #
        if not isinstance(func_expr, lc.ExpressionVariable):

            #
            func_var_name: str = get_next_temp_var_name()

            #
            instructions_to_do_before.append(
                lc.FlowControlVariableAssignment(var_name=func_var_name, var_value=func_expr)
            )

            #
            func_expr = lc.ExpressionVariable(var_name=func_var_name)


        #
        ### Prepare argument variable names for the call instruction. ###
        #
        arg_var_names: list[str] = [
            arg.var_name if isinstance(arg, lc.ExpressionVariable) else get_next_temp_var_name()
            for i, arg in enumerate(arg_exprs)
        ]

        #
        ### If any arg is not a variable, assign to temp and update arg_var_names. ###
        #
        for i, arg in enumerate(arg_exprs):

            #
            if not isinstance(arg, lc.ExpressionVariable):

                #
                temp_arg_var = get_next_temp_var_name()
                instructions_to_do_before.append(
                    lc.FlowControlVariableAssignment(var_name=temp_arg_var, var_value=arg)
                )
                arg_var_names[i] = temp_arg_var

        #
        ### Prepare keyword argument variable names. ###
        #
        kwarg_var_names: dict[str, str] = {}
        #
        k: str
        v: lc.Expression
        #
        for k, v in kwarg_exprs.items():

            #
            if isinstance(v, lc.ExpressionVariable):
                #
                kwarg_var_names[k] = v.var_name
            #
            else:
                #
                temp_kw_var = get_next_temp_var_name()
                #
                instructions_to_do_before.append(
                    lc.FlowControlVariableAssignment(var_name=temp_kw_var, var_value=v)
                )
                #
                kwarg_var_names[k] = temp_kw_var

        #
        ### Create a temp variable for the result. ###
        #
        temp_var_name: str = get_next_temp_var_name()

        #
        ### Prepare final function arguments. ###
        #
        func_args: dict[str, lc.Expression] = {}
        #
        for i, arg_var_name in enumerate(arg_var_names):
            #
            func_args[str(i)] = arg_var_name

        #
        for k, v in kwarg_var_names.items():
            #
            func_args[k] = v

        #
        ### Add the function call instruction. ###
        #
        instructions_to_do_before.append(
            lc.FlowControlFunctionCall(
                output_variables=[temp_var_name],
                function_called=func_expr.var_name,
                function_arguments=func_args
            )
        )

        #
        ### Return the temporary variable as the result of the call. ###
        #
        return lc.ExpressionVariable(var_name=temp_var_name)

    #
    elif isinstance(node, ast.Subscript):

        #
        ### Handle subscript expressions like x[0], tensor[:, :, 0], etc. ###
        #
        temp_var_name: str = get_next_temp_var_name()

        #
        ### Extract the variable being subscripted ###
        #
        value_expr: lc.Expression = extract_expression(node.value, analyzer, instructions_to_do_before)

        #
        ### Extract the slice/index ###
        #
        slice_expr: lc.Expression

        #
        if isinstance(node.slice, ast.Tuple):

            #
            ### Multi-dimensional slice like [:, :, 0] ###
            #
            slices = []

            #
            for dim in node.slice.elts:

                #
                dim_expr = extract_expression(dim, analyzer, instructions_to_do_before)

                #
                if isinstance(dim_expr, lc.ExpressionSlice1D):
                    #
                    slices.append(dim_expr)

                #
                else:
                    #
                    ### Convert single index to slice ###
                    #
                    slices.append(lc.ExpressionSlice1D(start=dim_expr, end=None, step=None))

            #
            slice_expr = lc.ExpressionSliceND(slices=slices)

        #
        else:
            #
            slice_expr = extract_expression(node.slice, analyzer, instructions_to_do_before)

        #
        ### Create an index access expression ###
        #
        index_access_expr: lc.ExpressionIndexAccess = lc.ExpressionIndexAccess(
            variable=value_expr,
            index=slice_expr
        )

        #
        ### Create a temporary variable to store the result ###
        #
        instructions_to_do_before.append(lc.FlowControlVariableAssignment(
            var_name=temp_var_name,
            var_value=index_access_expr
        ))

        #
        return lc.ExpressionVariable(var_name=temp_var_name)

    #
    elif isinstance(node, ast.Attribute):

        #
        ### Handle attribute access like self.linear1, tensor.shape, etc. ###
        #
        temp_var_name: str = get_next_temp_var_name()

        #
        ### Extract the object being accessed ###
        #
        value_expr: lc.Expression = extract_expression(node.value, analyzer, instructions_to_do_before)

        #
        ### Create an attribute access expression ###
        #
        attr_access_expr: lc.ExpressionAttributeAccess = lc.ExpressionAttributeAccess(
            variable=value_expr,
            attribute=node.attr
        )

        #
        ### Create a temporary variable to store the result ###
        #
        instructions_to_do_before.append(lc.FlowControlVariableAssignment(
            var_name=temp_var_name,
            var_value=attr_access_expr
        ))

        #
        return lc.ExpressionVariable(var_name=temp_var_name)

    #
    elif isinstance(node, ast.ListComp):

        #
        ### Handle list comprehensions like [module_c(xi) for (xi, module_c) in zip(...)] ###
        #
        temp_var_name: str = get_next_temp_var_name()

        #
        ### For complex list comprehensions, we need to create a proper loop structure ###
        ### This is a more complete implementation that handles iteration variables ###
        #

        # Create a temporary variable to store the result list
        result_list_var: str = get_next_temp_var_name()
        instructions_to_do_before.append(lc.FlowControlVariableAssignment(
            var_name=result_list_var,
            var_value=lc.ExpressionConstantList(elements=[])
        ))

        #
        ### Handle the generator/iterator part ###
        #
        if len(node.generators) == 1:

            #
            generator = node.generators[0]

            #
            ### Extract the iterable expression ###
            #
            iterable_expr: lc.Expression = extract_expression(generator.iter, analyzer, instructions_to_do_before)

            #
            ### Handle the target (iteration variables) ###
            #
            if isinstance(generator.target, ast.Tuple):

                #
                ### Multiple iteration variables like (xi, module_c) ###
                #
                target_vars = []

                #
                for target in generator.target.elts:

                    #
                    if isinstance(target, ast.Name):
                        #
                        target_vars.append(target.id)
                    #
                    else:
                        #
                        ### Complex target, create temp variable ###
                        #
                        temp_target_var = get_next_temp_var_name()
                        #
                        target_expr = extract_expression(target, analyzer, instructions_to_do_before)
                        #
                        instructions_to_do_before.append(lc.FlowControlVariableAssignment(
                            var_name=temp_target_var,
                            var_value=target_expr
                        ))
                        #
                        target_vars.append(temp_target_var)

            #
            else:

                #
                ### Single iteration variable ###
                #
                if isinstance(generator.target, ast.Name):
                    #
                    target_vars = [generator.target.id]
                #
                else:
                    #
                    ### Complex target, create temp variable ###
                    #
                    temp_target_var = get_next_temp_var_name()
                    #
                    target_expr = extract_expression(generator.target, analyzer, instructions_to_do_before)
                    #
                    instructions_to_do_before.append(lc.FlowControlVariableAssignment(
                        var_name=temp_target_var,
                        var_value=target_expr
                    ))
                    #
                    target_vars = [temp_target_var]

            #
            ### Create a for loop to iterate over the iterable ###
            #
            loop_instructions = []

            #
            ### Extract the element expression (this will now have access to iteration variables) ###
            #
            element_expr: lc.Expression = extract_expression(node.elt, analyzer, loop_instructions)

            #
            ### Append the element to the result list ###
            #
            loop_instructions.append(lc.FlowControlFunctionCall(
                output_variables=[],
                function_called="append",
                function_arguments={
                    "0": lc.ExpressionVariable(result_list_var),
                    "1": element_expr
                }
            ))

            #
            ### Create the for loop instruction ###
            #
            for_loop = lc.FlowControlForLoop(
                iterable_var_name=target_vars[0] if len(target_vars) == 1 else f"({', '.join(target_vars)})",
                iterator=iterable_expr,
                flow_control_instructions=loop_instructions
            )

            #
            instructions_to_do_before.append(for_loop)

        #
        else:
            #
            ### Multiple generators - for now, handle only the first one ###
            ### This is a simplified case ###
            #
            generator = node.generators[0]
            #
            iterable_expr: lc.Expression = extract_expression(generator.iter, analyzer, instructions_to_do_before)

            #
            if isinstance(generator.target, ast.Name):
                #
                target_var = generator.target.id
            #
            else:
                #
                target_var = get_next_temp_var_name()
                #
                target_expr = extract_expression(generator.target, analyzer, instructions_to_do_before)
                #
                instructions_to_do_before.append(lc.FlowControlVariableAssignment(
                    var_name=target_var,
                    var_value=target_expr
                ))

            #
            ### Create a simple loop ###
            #
            loop_instructions = []
            #
            element_expr: lc.Expression = extract_expression(node.elt, analyzer, loop_instructions)
            #
            loop_instructions.append(lc.FlowControlFunctionCall(
                output_variables=[],
                function_called="append",
                function_arguments={
                    "0": lc.ExpressionVariable(result_list_var),
                    "1": element_expr
                }
            ))

            #
            for_loop = lc.FlowControlForLoop(
                iterable_var_name=target_var,
                iterator=iterable_expr,
                flow_control_instructions=loop_instructions
            )

            #
            instructions_to_do_before.append(for_loop)

        #
        ### Create a temporary variable to store the final result ###
        #
        instructions_to_do_before.append(lc.FlowControlVariableAssignment(
            var_name=temp_var_name,
            var_value=lc.ExpressionVariable(result_list_var)
        ))

        #
        return lc.ExpressionVariable(var_name=temp_var_name)

    #
    raise NotImplementedError(f"Expression type {type(node)} not implemented")


#
### Extract Condition from AST Node. ###
#
def extract_condition(
    node: ast.AST,
    analyzer: "ModelAnalyzer",
    instructions_to_do_before: list[lc.FlowControlInstruction]
) -> lc.Condition:

    """
    Extract a Condition from an AST Node.

    Args:
        node (ast.AST): The AST node to extract a Condition from.
        analyzer (ModelAnalyzer): The analyzer instance.
        instructions_to_do_before (list[lc.FlowControlInstruction]): Instructions to execute before evaluating the condition.

    Returns:
        lc.Condition: The extracted condition object.

    Raises:
        NotImplementedError: If the node type is not supported for condition extraction.
    """

    #
    if isinstance(node, ast.Compare):

        #
        ### Handle comparison operations ###
        #
        left_expr: lc.Expression = extract_expression(node.left, analyzer, instructions_to_do_before)

        #
        ### Handle single comparison (most common case). ###
        #
        if len(node.ops) == 1 and len(node.comparators) == 1:

            #
            right_expr: lc.Expression = extract_expression(node.comparators[0], analyzer, instructions_to_do_before)

            #
            op: ast.cmpop = node.ops[0]

            #
            ### Convert comparison operator to string. ###
            #
            operator_str: str = op.__class__.__name__.lower()
            #
            if operator_str == "eq":        operator_str = "=="
            elif operator_str == "noteq":   operator_str = "!="
            elif operator_str == "lt":      operator_str = "<"
            elif operator_str == "lte":     operator_str = "<="
            elif operator_str == "gt":      operator_str = ">"
            elif operator_str == "gte":     operator_str = ">="
            elif operator_str == "is":      operator_str = "is"
            elif operator_str == "isnot":   operator_str = "is not"
            elif operator_str == "in":      operator_str = "in"
            elif operator_str == "notin":   operator_str = "not in"

            #
            ### Create the condition. ###
            #
            return lc.ConditionBinary(elt1=left_expr, cond_operator=operator_str, elt2=right_expr)

        #
        raise NotImplementedError()

    #
    elif isinstance(node, ast.BoolOp):

        #
        ### Handle boolean operations (and, or) ###
        #
        operator_str: str = node.op.__class__.__name__.lower()
        #
        if operator_str == "and":   operator_str = "and"
        elif operator_str == "or":  operator_str = "or"

        #
        ### Handle the first two operands. ###
        #
        left_condition: lc.Condition = extract_condition(node.values[0], analyzer, instructions_to_do_before)
        right_condition: lc.Condition = extract_condition(node.values[1], analyzer, instructions_to_do_before)

        #
        ### Create the condition. ###
        #
        result_condition: lc.Condition = lc.ConditionBinary(elt1=left_condition, cond_operator=operator_str, elt2=right_condition)

        #
        ### Handle remaining operands if any. ###
        #
        for i in range(2, len(node.values)):

            #
            next_condition: lc.Condition = extract_condition(node.values[i], analyzer, instructions_to_do_before)

            #
            result_condition = lc.ConditionBinary(elt1=result_condition, cond_operator=operator_str, elt2=next_condition)

        #
        return result_condition

    #
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        #
        ### Handle not operations ###
        #
        operand_condition: lc.Condition = extract_condition(node.operand, analyzer, instructions_to_do_before)

        #
        return lc.ConditionUnary(elt=operand_condition, cond_operator="not")

    #
    elif isinstance(node, ast.NameConstant) and node.value is True:

        #
        return lc.ConditionUnary(elt=lc.ExpressionConstantNumeric(constant=True))

    #
    elif isinstance(node, ast.NameConstant) and node.value is False:

        #
        return lc.ConditionUnary(elt=lc.ExpressionConstantNumeric(constant=False))

    #
    raise NotImplementedError(f"Condition type {type(node)} not implemented")


#
### Check if a call is a layer instantiation. ###
#
def is_layer_instantiation(node: ast.Call, analyzer: "ModelAnalyzer") -> bool:
    """
    Check if a call node represents a layer instantiation.

    Args:
        node (ast.Call): The call node to check.
        analyzer (ModelAnalyzer): The analyzer instance with layer information.

    Returns:
        bool: True if the call is a layer instantiation, False otherwise.
    """

    #
    ### Extract the function name ###
    #
    if isinstance(node.func, ast.Name):
        #
        function_name: str = node.func.id
    #
    elif isinstance(node.func, ast.Attribute):
        #
        function_name: str = node.func.attr
    #
    else:
        #
        return False

    #
    ### Check if the function name is in the layers dictionary ###
    #
    return function_name in analyzer.layers


#
### Extract function arguments from AST Call node. ###
#
def extract_function_arguments(
    node: ast.Call,
    analyzer: "ModelAnalyzer",
    instructions_to_do_before: list[lc.FlowControlInstruction]
) -> tuple[list[lc.Expression], dict[str, lc.Expression]]:

    """
    Extract positional and keyword arguments from a function call.

    Args:
        node (ast.Call): The call node to extract arguments from.
        analyzer (ModelAnalyzer): The analyzer instance.
        instructions_to_do_before (list[lc.FlowControlInstruction]): Instructions to execute before the call.

    Returns:
        tuple[list[lc.Expression], dict[str, lc.Expression]]: Positional and keyword arguments.
    """

    #
    ### Extract positional arguments ###
    #
    positional_args: list[lc.Expression] = []
    #
    for arg in node.args:
        #
        positional_args.append(extract_expression(arg, analyzer, instructions_to_do_before))

    #
    ### Extract keyword arguments ###
    #
    keyword_args: dict[str, lc.Expression] = {}
    #
    for keyword in node.keywords:
        #
        if keyword.arg is not None:
            #
            keyword_args[keyword.arg] = extract_expression(keyword.value, analyzer, instructions_to_do_before)

    #
    return positional_args, keyword_args


#
### Extract assignment target from AST node. ###
#
def extract_target(
    node: ast.AST,
    analyzer: "ModelAnalyzer",
    instructions_to_do_before: list[lc.FlowControlInstruction]
) -> str:

    """
    Extract the target variable name from an assignment target.

    Args:
        node (ast.AST): The target node (Name, Attribute, Subscript, etc.).
        analyzer (ModelAnalyzer): The analyzer instance.
        instructions_to_do_before (list[lc.FlowControlInstruction]): Instructions to execute before the assignment.

    Returns:
        str: The target variable name.
    """

    #
    if isinstance(node, ast.Name):
        #
        return node.id

    #
    elif isinstance(node, ast.Attribute):
        #
        ### For attributes like self.linear1, return the attribute name ###
        #
        return node.attr

    #
    elif isinstance(node, ast.Subscript):
        #
        ### For subscripts like x[0], return the variable name ###
        #
        if isinstance(node.value, ast.Name):
            #
            return node.value.id

        #
        else:
            #
            ### For complex subscripts, create a temporary variable ###
            #
            temp_var_name: str = get_next_temp_var_name()
            #
            instructions_to_do_before.append(lc.FlowControlVariableAssignment(
                var_name=temp_var_name,
                var_value=extract_expression(node.value, analyzer, instructions_to_do_before)
            ))
            #
            return temp_var_name

    #
    else:
        #
        ### For other complex targets, create a temporary variable ###
        #
        temp_var_name: str = get_next_temp_var_name()
        #
        instructions_to_do_before.append(lc.FlowControlVariableAssignment(
            var_name=temp_var_name,
            var_value=extract_expression(node, analyzer, instructions_to_do_before)
        ))
        #
        return temp_var_name


#
### Extract Call arg value. ###
#
def extract_call_arg_value(
    node: ast.AST,
    analyzer: "ModelAnalyzer",
    instructions_to_do_before: list[lc.FlowControlInstruction]
) -> lc.Expression:

    """
    Extracts the value of an argument node from an AST Call.

    Args:
        node (ast.AST): The AST node representing the argument value.
        analyzer (ModelAnalyzer): The analyzer instance for context and helper functions.
        instructions_to_do_before (list[lc.FlowControlInstruction]): Instructions to execute before the call.

    Returns:
        lc.Expression: The extracted expression representing the argument value.
    """

    #
    return extract_expression(node=node, analyzer=analyzer, instructions_to_do_before=instructions_to_do_before)


#
### Extract Call. ###
#
def extract_call(
    node: ast.Call,
    analyzer: "ModelAnalyzer",
    instructions_to_do_before: list[lc.FlowControlInstruction]
) -> tuple[str, list[lc.Expression], dict[str, lc.Expression]]:

    """
    ast.Call:
        func (ast.Name | ast.Attribute)
        args (list[ast.AST])
        keywords (list[ast.keyword])

    ast.keyword:
        arg (str)
        value (ast.Node)

    Extracts information from an ast.Call node.

    Args:
        node (ast.Call): The AST node representing a function or method call.
        analyzer (ModelAnalyzer): The analyzer instance for context and helper functions.
        instructions_to_do_before (list[lc.FlowControlInstruction]): Any instructions that must be executed before the call (e.g., for evaluating complex arguments).

    Returns:
        tuple:
            - func_name (str): The name of the function or method being called.
            - args (list[lc.Expression]): List of positional argument expressions.
            - kwargs (dict[str, lc.Expression]): Dictionary of keyword argument expressions.
    """

    #
    ### Extract function name ###
    #
    func_name: str = extract_name_or_attribute(node.func)

    #
    ### Init arguments ###
    #
    func_call_args: list[lc.Expression] = []
    func_call_keywords: dict[str, lc.Expression] = {}

    #
    ### Extract arguments ###
    #
    for arg in node.args:
        #
        arg_expr = extract_call_arg_value(node=arg, analyzer=analyzer, instructions_to_do_before=instructions_to_do_before)
        #
        func_call_args.append(arg_expr)

    #
    ### Extract keyword arguments ###
    #
    for kw in node.keywords:
        #
        if kw.arg is None:
            #
            continue

        #
        keyword_name: str = kw.arg
        func_call_keywords[keyword_name] = extract_call_arg_value(node=kw.value, analyzer=analyzer, instructions_to_do_before=instructions_to_do_before)

    #
    return func_name, func_call_args, func_call_keywords


#
### Extract Layer call. ###
#
def extract_layer_call(
    node: ast.Call,
    var_name: str,
    layer_type: str,
    analyzer: "ModelAnalyzer",
    instructions_to_do_before: list[lc.FlowControlInstruction],
    add_arguments_todo: bool = True,
) -> lc.Layer:

    """
    Extracts a layer call from an AST Call node and returns a Layer object.

    Args:
        node (ast.Call): The AST Call node representing the layer instantiation.
        var_name (str): The variable name to assign the layer to.
        layer_type (str): The type of the layer (e.g., "Linear", "Conv2d").
        analyzer (ModelAnalyzer): The analyzer instance for context and state.
        instructions_to_do_before (list[lc.FlowControlInstruction]): Instructions to execute before the call.
        add_arguments_todo (bool, optional): Whether to add the layer's arguments to the analyzer's arguments-to-do list. Defaults to True.

    Returns:
        lc.Layer: The constructed Layer object with the given variable name and type.
    """

    #
    ### Extract call information. ###
    #
    _layer_name: str
    layer_call_args: list[lc.Expression]
    layer_call_keywords: dict[str, lc.Expression]
    #
    _layer_name, layer_call_args, layer_call_keywords = extract_call(node=node, analyzer=analyzer, instructions_to_do_before=instructions_to_do_before)

    #
    ### Create the layer with proper parameters. ###
    #
    layer: lc.Layer = lc.Layer(
        layer_var_name=var_name,
        layer_type=layer_type,
        layer_parameters_kwargs=layer_call_keywords
    )

    #
    if add_arguments_todo:
        #
        analyzer.layers_arguments_todo[layer] = (layer_call_args, layer_call_keywords)

    #
    return layer


#
### Class to analyze a model file and extract its architecture. ###
#
class ModelAnalyzer(ast.NodeVisitor):

    #
    ### Class constructor. ###
    #
    def __init__(self, layers_filepath: str = "core/layers.json") -> None:
        """
        Initializer of the ModelAnalyzer class.

        Attributes:
            model_blocks (dict[str, lc.ModelBlock]): list of all blocks analyzed, indexed by their block ID (e.g., name).
            main_block (str): ID of the main block, given in sys.argv with `--main-block=<MainBlockName>`.
            current_model_visit (list[str]): Stack of current blocks being visited, access top with [-1].
            current_function_visit (str): Name of the current visited function.
            sub_block_counter (dict[str, int]): Counter for naming sub-blocks (e.g., ModuleList, Sequential).
            global_constants (dict[str, tuple[str, Any]]): Global constants defined outside classes.

        Args:
            layers_filepath (str): The filepath to the layers.json file.
        """

        #
        ### list of all blocks analyzed, indexed by their block ID (e.g., name). ###
        #
        self.model_blocks: dict[str, lc.ModelBlock] = {}

        #
        ### ID of the main block, given in sys.argv with `--main-block <MainBlockName>`. ###
        #
        self.main_block: str = ""

        #
        ### Stack of current blocks being visited, access top with [-1]. ###
        #
        self.current_model_visit: list[str] = []

        #
        ### Name of the current visited function. ###
        #
        self.current_function_visit: str = ""

        #
        ### Counter for naming sub-blocks (e.g., ModuleList, Sequential). ###
        #
        self.sub_block_counter: dict[str, int] = {}

        #
        ### Global instructions to execute before global variable assignements. ###
        #
        self.global_instructions_to_do_before_global_variable_assignments: list[lc.FlowControlInstruction] = []

        #
        ### Global constants defined outside classes. ###
        #
        self.global_variables: dict[str, tuple[lc.VarType, lc.Expression]] = {}

        #
        ### Layer / Function Call arguments first extractions ###
        ###   layer or function call -> ( args, kwargs ) ###
        #
        self.layers_arguments_todo: dict[lc.Layer | lc.FlowControlFunctionCall, tuple[list[lc.Expression], dict[str, lc.Expression]]] = {}

        #
        self.layers: dict[str, ll.BaseLayerInfo] = ll.load_layers_dict(filepath=layers_filepath)


    #
    ###### HELPER FUNCTIONS
    #

    #
    ### Extract layer type from a function node. ###
    #
    def get_layer_type(self, func_node: ast.AST) -> str:
        """
        Extract layer type from a function node.

        Args:
            func_node (ast.AST): The function node to extract the type from.

        Returns:
            str: The layer type name.
        """

        #
        layer_type = extract_name_or_attribute(func_node)

        #
        ### Remove nn. prefix if present to match all_layers_info. ###
        #
        if layer_type.startswith('nn.'):
            #
            ### Remove 'nn.' prefix. ###
            #
            layer_type = layer_type[3:]

        #
        return layer_type


    #
    ### Handle container. ###
    #
    def _handle_container(
        self,
        var_name: str,
        container_type: str,
        call_node: ast.Call,
        block: lc.ModelBlock
    ) -> None:

        """
        Handles nn.ModuleList and nn.Sequential by creating sub-blocks and defining their forward methods.

        Args:
            var_name (str): Name of the container variable.
            container_type (str): Type of container ("ModuleList" or "Sequential").
            call_node (ast.Call): The AST node of the container call.
            block (lc.ModelBlock): The current model block.
        """

        #
        ### Create a unique sub-block name. ###
        #
        sub_block_name: str = f"Block{container_type}_{self.current_model_visit[-1]}_{self.sub_block_counter[self.current_model_visit[-1]]}"
        self.sub_block_counter[self.current_model_visit[-1]] += 1

        #
        ### Creating the sub-block. ###
        #
        sub_block = lc.ModelBlock(block_name=sub_block_name)
        self.model_blocks[sub_block_name] = sub_block

        #
        ### Initialize layers list. ###
        #
        layers: list[lc.Layer] = []

        #
        ### Init loop & layer variables. ###
        #
        i: int
        layer: lc.Layer

        #
        ### Handle arguments. ###
        #
        if call_node.args:

            #
            for i, arg in enumerate(call_node.args):

                #
                ### Call: A call expression, such as func(...). ###
                #
                if isinstance(arg, ast.Call):

                    #
                    ### Get the layer type. ###
                    #
                    layer_type: str = self.get_layer_type(arg.func)

                    #
                    ### Extract the layer and add it to the layers list. ###
                    #
                    instructions_to_do_before: list[lc.FlowControlInstruction] = []
                    layer: lc.Layer = extract_layer_call(node=arg, var_name=f"{i}", layer_type=layer_type, analyzer=self, instructions_to_do_before=instructions_to_do_before)
                    #
                    layers.append( layer )

                #
                elif isinstance(arg, ast.List) or isinstance(arg, ast.Tuple):

                    #
                    ### Handle arguments. ###
                    #
                    for j, sub_arg in enumerate(arg.elts):

                        #
                        ### Call: A call expression, such as func(...). ###
                        #
                        if isinstance(sub_arg, ast.Call):

                            #
                            ### Get the layer type. ###
                            #
                            layer_type: str = self.get_layer_type(sub_arg.func)

                            #
                            ### Extract the layer and add it to the layers list. ###
                            #
                            instructions_to_do_before: list[lc.FlowControlInstruction] = []
                            layer: lc.Layer = extract_layer_call(node=sub_arg, var_name=f"{j}", layer_type=layer_type, analyzer=self, instructions_to_do_before=instructions_to_do_before)
                            #
                            layers.append( layer )

        #
        ### Add layers to sub_block and define forward method. ###
        #
        for i, layer in enumerate(layers):
            #
            sub_block.block_layers[layer.layer_var_name] = layer

        #
        ### Define forward method. ###
        #
        forward_func = lc.BlockFunction(
            function_name="forward",
            function_arguments={"x": (lc.VarType("Tensor"), lc.ExpressionNoDefaultArguments())},
            model_block=sub_block
        )

        #
        ### Add the forward function to the sub-block. ###
        #
        sub_block.block_functions["forward"] = forward_func

        #
        ### As you can see in the forward function arguments, the input is the variable 'x'. ###
        #
        current_input: lc.ExpressionVariable = lc.ExpressionVariable("x")

        #
        ### For each layer, add flow control instructions. ###
        #
        for i, layer in enumerate(layers):

            #
            output_var: str = f"out_{i}"

            #
            forward_func.function_flow_control.append(
                lc.FlowControlLayerPass(
                    output_variables=[output_var],
                    layer_name=layer.layer_var_name,
                    layer_arguments={"x": current_input}
                )
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
            output_vars: list[str] = [f"out_{i}" for i in range(len(layers))]

            #
            forward_func.function_flow_control.append(lc.FlowControlReturn(return_variables=output_vars))

        #
        ### Add to parent block. ###
        #
        block.block_layers[var_name] = lc.Layer(
            layer_var_name=var_name,
            layer_type=sub_block_name,
            layer_parameters_kwargs={}
        )


    #
    ### Handle module or sequential assignment. ###
    #
    def _handle_module_or_sequential_assignment(
        self,
        var_name: str,
        call_node: ast.Call,
        target: ast.AST
    ) -> None:

        """
        Handles nn.Module and nn.Sequential assignments by creating new model blocks.

        Args:
            var_name (str): Name of the variable being assigned.
            call_node (ast.Call): The AST call node for nn.Module or nn.Sequential.
            target (ast.AST): The target node of the assignment.
        """

        #
        ### Ensure we are inside a class ###
        #
        if not self.current_model_visit:
            #
            print(f"\033[1;31m - WARNING: nn.Module/nn.Sequential assignment `{var_name}` is not inside a class, skipping. - \033[m")
            #
            return

        #
        ### Get the current model block ###
        #
        current_block_name: str = self.current_model_visit[-1]
        current_block: lc.ModelBlock = self.model_blocks[current_block_name]

        #
        ### Determine the type (Module, Sequential, or ModuleList) ###
        #
        func_name = extract_name_or_attribute(call_node.func)

        #
        if func_name in ["nn.Module", "Module"]:
            #
            container_type = "Module"
        #
        elif func_name in ["nn.Sequential", "Sequential"]:
            #
            container_type = "Sequential"
        #
        elif func_name in ["nn.ModuleList", "ModuleList"]:
            #
            container_type = "ModuleList"
        #
        else:
            #
            print(f"\033[1;31m - WARNING: Unknown container type `{func_name}` for variable `{var_name}`, skipping. - \033[m")
            #
            return

        #
        ### Create a unique sub-block name ###
        #
        sub_block_name: str = f"Block{container_type}_{self.current_model_visit[-1]}_{self.sub_block_counter.get(self.current_model_visit[-1], 0)}"
        #
        if self.current_model_visit[-1] not in self.sub_block_counter:
            #
            self.sub_block_counter[self.current_model_visit[-1]] = 0

        #
        self.sub_block_counter[self.current_model_visit[-1]] += 1

        #
        ### Create the sub-block ###
        #
        sub_block = lc.ModelBlock(block_name=sub_block_name)
        #
        self.model_blocks[sub_block_name] = sub_block

        #
        ### Initialize layers list ###
        #
        layers: list[lc.Layer] = []

        #
        ### Handle arguments (layers in the container) ###
        #
        if call_node.args:

            #
            for i, arg in enumerate(call_node.args):

                #
                ### Handle the case where the argument is a List containing multiple calls. ###
                #
                if isinstance(arg, ast.List):

                    #
                    for j, list_item in enumerate(arg.elts):

                        #
                        if isinstance(list_item, ast.Call):

                            #
                            ### Get the layer type. ###
                            #
                            layer_type: str = self.get_layer_type(list_item.func)

                            #
                            ### Check if it's a known layer type. ###
                            #
                            if layer_type in self.layers:

                                #
                                ### Extract the layer and add it to the layers list. ###
                                #
                                instructions_to_do_before: list[lc.FlowControlInstruction] = []

                                #
                                layer: lc.Layer = extract_layer_call(
                                    node=list_item,
                                    var_name=f"{j}",
                                    layer_type=layer_type,
                                    analyzer=self,
                                    instructions_to_do_before=instructions_to_do_before
                                )

                                #
                                layers.append(layer)

                            #
                            else:

                                #
                                ### If it's not a known layer, treat it as a sub-module. ###
                                ### This handles cases like custom modules or other nn.Module subclasses. ###
                                #
                                sub_module_layer = lc.Layer(
                                    layer_var_name=f"{j}",
                                    layer_type=layer_type,  # This will be the model block name (e.g., "ConvEncode")
                                    layer_parameters_kwargs={}
                                )

                                #
                                layers.append(sub_module_layer)

                #
                elif isinstance(arg, ast.Call):

                    #
                    ### Get the layer type. ###
                    #
                    layer_type: str = self.get_layer_type(arg.func)

                    #
                    ### Check if it's a known layer type. ###
                    #
                    if layer_type in self.layers:

                        #
                        ### Extract the layer and add it to the layers list. ###
                        #
                        instructions_to_do_before: list[lc.FlowControlInstruction] = []

                        #
                        layer: lc.Layer = extract_layer_call(
                            node=arg,
                            var_name=f"{i}",
                            layer_type=layer_type,
                            analyzer=self,
                            instructions_to_do_before=instructions_to_do_before
                        )

                        #
                        layers.append(layer)

                    #
                    else:

                        #
                        ### If it's not a known layer, treat it as a sub-module. ###
                        ### This handles cases like custom modules or other nn.Module subclasses. ###
                        #
                        sub_module_layer = lc.Layer(
                            layer_var_name=f"{i}",
                            layer_type=layer_type,  # This will be the model block name (e.g., "ConvEncode"). ###
                            layer_parameters_kwargs={}
                        )

                        #
                        layers.append(sub_module_layer)

        #
        ### Add layers to sub_block ###
        #
        for i, layer in enumerate(layers):
            #
            sub_block.block_layers[layer.layer_var_name] = layer

        #
        ### Define forward method ###
        #
        forward_func = lc.BlockFunction(
            function_name="forward",
            function_arguments={"x": (lc.VarType("Tensor"), lc.ExpressionNoDefaultArguments())},
            model_block=sub_block
        )

        #
        ### Add the forward function to the sub-block ###
        #
        sub_block.block_functions["forward"] = forward_func

        #
        ### Create forward pass flow control ###
        #
        current_input: lc.ExpressionVariable = lc.ExpressionVariable("x")

        #
        if container_type == "Sequential":

            #
            ### For Sequential, pass output of one layer as input to the next ###
            #
            for i, layer in enumerate(layers):

                #
                output_var: str = f"out_{i}"

                #
                forward_func.function_flow_control.append(
                    lc.FlowControlLayerPass(
                        output_variables=[output_var],
                        layer_name=layer.layer_var_name,
                        layer_arguments={"x": current_input}
                    )
                )

                #
                current_input = lc.ExpressionVariable(output_var)

            #
            ### Return the final output ###
            #
            forward_func.function_flow_control.append(
                lc.FlowControlReturn(return_variables=[current_input.var_name])
            )

        #
        elif container_type == "Module":

            #
            ### For Module, process all layers and return all outputs ###
            #
            output_vars: list[str] = []

            #
            for i, layer in enumerate(layers):

                #
                output_var: str = f"out_{i}"
                output_vars.append(output_var)

                #
                forward_func.function_flow_control.append(
                    lc.FlowControlLayerPass(
                        output_variables=[output_var],
                        layer_name=layer.layer_var_name,
                        layer_arguments={"x": current_input}
                    )
                )

            #
            ### Return all outputs ###
            #
            forward_func.function_flow_control.append(
                lc.FlowControlReturn(return_variables=output_vars)
            )

        #
        elif container_type == "ModuleList":

            #
            ### For ModuleList, process all layers and return all outputs ###
            #
            output_vars: list[str] = []

            #
            for i, layer in enumerate(layers):

                #
                output_var: str = f"out_{i}"
                output_vars.append(output_var)

                #
                forward_func.function_flow_control.append(
                    lc.FlowControlLayerPass(
                        output_variables=[output_var],
                        layer_name=layer.layer_var_name,
                        layer_arguments={"x": current_input}
                    )
                )

            #
            ### Return all outputs ###
            #
            forward_func.function_flow_control.append(
                lc.FlowControlReturn(return_variables=output_vars)
            )

        #
        ### Add to parent block as a layer ###
        #
        current_block.block_layers[var_name] = lc.Layer(
            layer_var_name=var_name,
            layer_type=sub_block_name,
            layer_parameters_kwargs={}
        )


    #
    ### Handle loop initialization. ###
    #
    def _handle_loop_init(
        self,
        var_name: str,
        for_node: ast.For,
        block: lc.ModelBlock
    ) -> None:

        """
        Handles loop-based initialization of layers (e.g., ModuleList).

        Args:
            var_name (str): Name of the variable being initialized.
            for_node (ast.For): The for loop node.
            block (lc.ModelBlock): The current model block.
        """

        #
        sub_block_name: str = f"BlockModuleList_{self.current_model_visit[-1]}_{self.sub_block_counter[self.current_model_visit[-1]]}"

        #
        self.sub_block_counter[self.current_model_visit[-1]] += 1

        #
        sub_block: lc.ModelBlock = lc.ModelBlock(block_name=sub_block_name)
        #
        self.model_blocks[sub_block_name] = sub_block

        #
        if not isinstance(for_node.iter, ast.Name) or not isinstance(for_node.target, ast.Name):
            #
            return

        #
        ### Extract loop details. ###
        #
        instructions_to_do_before: list[lc.FlowControlInstruction] = []
        #
        iterator: Optional[lc.Expression] | str = extract_expression(for_node.iter, self, instructions_to_do_before) or for_node.iter.id

        #
        iterable_var: str = for_node.target.id

        #
        layers: list[lc.Layer] = []

        #
        for stmt in for_node.body:

            #
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):

                #
                layer_type: str = self.get_layer_type(stmt.value.func)

                #
                instructions_to_do_before_layer: list[lc.FlowControlInstruction] = []

                #
                layer: lc.Layer = extract_layer_call(
                    node=stmt.value,
                    var_name=f"{len(layers)}",
                    layer_type=layer_type,
                    analyzer=self,
                    instructions_to_do_before=instructions_to_do_before_layer
                )

                #
                layers.append(layer)

        #
        ### Add layers to sub-block. ###
        #
        for i, layer in enumerate(layers):
            #
            sub_block.block_layers[f"{i}"] = layer

        #
        ### Define forward method. ###
        #
        forward_func: lc.BlockFunction = lc.BlockFunction(
            function_name="forward",
            function_arguments={ "x": (lc.VarType("Tensor"), lc.ExpressionNoDefaultArguments()) },
            model_block=sub_block
        )

        #
        ### Add forward method. ###
        #
        sub_block.block_functions["forward"] = forward_func

        #
        output_vars: list[str] = []
        #
        for i in range(len(layers)):

            #
            output_var: str = f"out_{i}"

            #
            forward_func.function_flow_control.append(
                lc.FlowControlLayerPass(
                    output_variables=[output_var],
                    layer_name=layers[i].layer_var_name,
                    layer_arguments={"x": lc.ExpressionVariable("x")}
                )
            )

            #
            output_vars.append(output_var)  # type: ignore

        #
        forward_func.function_flow_control.append(lc.FlowControlReturn(return_variables=output_vars))  # type: ignore

        #
        if isinstance(iterator, str):

            #
            iterator = lc.ExpressionVariable(iterator)

        #
        ### Add to parent block. ###
        #
        block.block_layers[var_name] = lc.Layer(
            layer_var_name=var_name,
            layer_type="ModuleList",
            layer_parameters_kwargs={"sub_block": lc.ExpressionVariable(sub_block_name), "iterator": iterator, "iterable_var": lc.ExpressionVariable(iterable_var)}
        )


    #
    ### Helper function to extract function arguments. ###
    #
    def extract_function_arguments(
        self,
        node: ast.FunctionDef
    ) -> tuple[ dict[str, tuple[lc.VarType, lc.Expression]], Optional[dict[str, list[lc.FlowControlInstruction]]]]:

        """
        Extracts function arguments from a function definition node.

        Args:
            node (ast.FunctionDef): The function definition node to extract the arguments from.

        Returns:
            tuple[ dict[str, tuple[lc.VarType, lc.Expression]], Optional[dict[str, list[lc.FlowControlInstruction]]]]: The function arguments and the complex default argument values instructions to do before real function flow.
        """

        #
        ### Prepare function arguments. ###
        #
        function_arguments: dict[str, tuple[lc.VarType, lc.Expression]] = {}

        #
        ### Prepare complex default argument values instructions to do before real function flow. ###
        #
        complex_default_argument_values_instructions_to_do_before_real_function_flow: Optional[dict[str, list[lc.FlowControlInstruction]]] = {}

        #
        ### For each argument, extract the argument name, type, and default value. ###
        #
        ### Get default values from node.defaults (positional) and node.kw_defaults (keyword-only). ###
        #
        defaults = node.defaults if hasattr(node, 'defaults') else []
        kw_defaults = node.kw_defaults if hasattr(node, 'kw_defaults') else []

        #
        ### Calculate the number of positional args with defaults. ###
        #
        num_positional_with_defaults: int = len(defaults)
        num_positional_args: int = len(node.args.args)
        num_positional_without_defaults: int = num_positional_args - num_positional_with_defaults

        #
        arg: ast.arg
        #
        for i, arg in enumerate(node.args.args):

            #
            ### Extract argument name. ###
            #
            arg_name: str = arg.arg

            #
            ### Prepare instructions to do before the real function flow. ###
            #
            instructions_to_do_before: list[lc.FlowControlInstruction] = []

            #
            ### Extract argument type. ###
            #
            arg_type: lc.VarType = lc.VarType(type_name=extract_name_or_attribute(arg.annotation) if arg.annotation else "Any")

            #
            ### Extract argument default value. ###
            #
            arg_default_value: lc.Expression

            #
            ### Check if this argument has a default value. ###
            #
            if i >= num_positional_without_defaults:
                #
                ### This argument has a default value. ###
                #
                default_index = i - num_positional_without_defaults
                #
                if default_index < len(defaults):
                    #
                    arg_default_value = extract_expression(defaults[default_index], self, instructions_to_do_before)
                #
                else:
                    #
                    arg_default_value = lc.ExpressionNoDefaultArguments()
            #
            else:
                #
                ### This argument has no default value. ###
                #
                arg_default_value = lc.ExpressionNoDefaultArguments()

            #
            ### Add argument to function arguments. ###
            #
            function_arguments[arg_name] = (arg_type, arg_default_value)

            #
            ### If there are instructions to do before the real function flow, add them to the complex default argument values instructions to do before real function flow. ###
            #
            if len(instructions_to_do_before) > 0:
                #
                complex_default_argument_values_instructions_to_do_before_real_function_flow[arg_name] = instructions_to_do_before

        #
        ### If there are no complex default argument values instructions to do before real function flow, set it to None. ###
        #
        if len(complex_default_argument_values_instructions_to_do_before_real_function_flow) == 0:
            #
            complex_default_argument_values_instructions_to_do_before_real_function_flow = None

        #
        ### Return the function arguments and the complex default argument values instructions to do before real function flow. ###
        #
        return function_arguments, complex_default_argument_values_instructions_to_do_before_real_function_flow


    #
    ###### VISIT FUNCTIONS ######
    #


    #
    ### Visit a ClassDef node in the AST. ###
    #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visits a class definition node in the AST.

        Args:
            node (ast.ClassDef): The class definition node to visit.
        """

        #
        ### For testing purposes, allow any class (not just nn.Module) ###
        ### In production, you would check for nn.Module inheritance ###
        #
        # if not any(isinstance(base, ast.Attribute) and base.attr == "Module" or isinstance(base, ast.Name) and base.id == "Module" for base in node.bases):
        #     #
        #     print(f"\033[1;31m - WARNING: Class `{node.name}` does not inherit from nn.Module, skipping. - \033[m")
        #     #
        #     return

        #
        ### Extract the class name. ###
        #
        class_name: str = node.name

        #
        ### Create a new ModelBlock for the class. ###
        #
        model_block: lc.ModelBlock = lc.ModelBlock(block_name=class_name)

        #
        ### Add the new block to the model_blocks dictionary. ###
        #
        self.model_blocks[node.name] = model_block

        #
        ### Update the current model visit stack. ###
        #
        self.current_model_visit.append(node.name)

        #
        ### Visit all body nodes of the class. ###
        #
        for body_node in node.body:
            #
            self.visit(body_node)

        #
        ### Pop the current model visit stack. ###
        #
        self.current_model_visit.pop()


    #
    ### Visit a FunctionDef node in the AST. ###
    #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visits a function definition node in the AST.

        Args:
            node (ast.FunctionDef): The function definition node to visit.
        """

        #
        ### Ensure we are inside a model block. ###
        #
        if not self.current_model_visit:
            #
            print(f"\033[1;31m - WARNING: Function `{node.name}` is not inside a model block, skipping. - \033[m")
            #
            return

        #
        ### Get the current model block being visited. ###
        #
        current_block_name: str = self.current_model_visit[-1]
        current_block: lc.ModelBlock = self.model_blocks[current_block_name]

        #
        self.visit_method_function(node=node, current_block=current_block)


    #
    ### Visit a method function in a model block. ###
    #
    def visit_method_function(self, node: ast.FunctionDef, current_block: lc.ModelBlock) -> None:
        """
        Visits a method function in a model block.

        Args:
            node (ast.FunctionDef): The method function definition node to visit.
            current_block (lc.ModelBlock): The current model block being visited.
        """

        #
        ### Extract function name. ###
        #
        function_name: str = node.name

        #
        ### Extract function arguments. ###
        #
        function_arguments: dict[str, tuple[lc.VarType, lc.Expression]] = {}
        complex_default_argument_values_instructions_to_do_before_real_function_flow: Optional[dict[str, list[lc.FlowControlInstruction]]] = None
        function_arguments, complex_default_argument_values_instructions_to_do_before_real_function_flow = self.extract_function_arguments(node)

        #
        ### Create a new ModelFunction for the __init__ method. ###
        #
        block_function: lc.BlockFunction = lc.BlockFunction(
            function_name=function_name,
            function_arguments=function_arguments,
            model_block=current_block,
            complex_default_argument_values_instructions_to_do_before_real_function_flow=complex_default_argument_values_instructions_to_do_before_real_function_flow
        )

        #
        ### Add the new function to the current model block. ###
        #
        current_block.block_functions[node.name] = block_function

        #
        ### Update the current function visit. ###
        #
        self.current_function_visit = node.name

        #
        ### Visit all body nodes of the function. ###
        #
        for body_node in node.body:
            #
            self.visit(body_node)

        #
        ### Clear the current function visit. ###
        #
        self.current_function_visit = ""


    #
    ### Visit an Assign node in the AST. ###
    #
    def visit_Assign(self, node: ast.Assign) -> None:
        """
        Visits an assignment node in the AST.
        If we are not inside a block method, the assigned elements will be considered as global variables.
        If this is a mutliple target assignement, break down each assignement separately.
        Auto convert in tuples if python implicit tupling.
        Best possible variable type analysis and detection.
        Manage correctly and save at the most appropriate place the class variables, the global variables, the inside method variables, the temporary variables, the elements attributes, and etc...
        Manage properly the variable scopes and everything.

        Args:
            node (ast.Assign): The assignment node to visit.
        """


        #
        ### If multiple targets (e.g., a = b = c), handle each separately ###
        #
        for target in node.targets:

            #
            ### Prepare a list to collect instructions that must be executed before the assignment. ###
            #
            instructions_to_do_before: list[lc.FlowControlInstruction] = []

            #
            ### Extract the variable name (or handle complex targets). ###
            #
            var_name: str = extract_target(target, self, instructions_to_do_before)

            #
            ### Check if this is a nn.Module, nn.Sequential, or nn.ModuleList assignment that needs special handling ###
            #
            if isinstance(node.value, ast.Call):
                func_name = extract_name_or_attribute(node.value.func)
                # Check if this is nn.Module, nn.Sequential, or nn.ModuleList
                if func_name in ["nn.Module", "nn.Sequential", "nn.ModuleList", "Module", "Sequential", "ModuleList"]:
                    # Only handle if we're inside a class (regardless of whether we're in a function)
                    if self.current_model_visit:
                        self._handle_module_or_sequential_assignment(var_name, node.value, target)
                        continue

            #
            ### Extract the value being assigned (as an lc.Expression). ###
            #
            value_expr: lc.Expression = extract_expression(node.value, self, instructions_to_do_before)

            #
            ### Determine the scope: are we inside a class, a function, or at the global level? ###
            #
            inside_class = bool(self.current_model_visit)
            inside_function = bool(self.current_function_visit)

            #
            ### If inside a class and inside a function, this is likely an instance or local variable. ###
            #
            if inside_class and inside_function:

                #
                ### Get the current model block and function. ###
                #
                current_block_name: str = self.current_model_visit[-1]
                current_block: lc.ModelBlock = self.model_blocks[current_block_name]
                current_function: lc.BlockFunction = current_block.block_functions[self.current_function_visit]

                #
                ### Check if this is a layer assignment that should be added to block_layers ###
                #
                if isinstance(node.value, ast.Call):

                    #
                    func_name = extract_name_or_attribute(node.value.func)

                    #
                    ### Remove nn. prefix if present. ###
                    #
                    if func_name.startswith('nn.'):
                        #
                        func_name = func_name[3:]

                    #
                    ### Check if this is a known layer type. ###
                    #
                    if func_name in self.layers:

                        #
                        ### This is a layer assignment, add it to the current block. ###
                        #
                        instructions_to_do_before_layer: list[lc.FlowControlInstruction] = []
                        #
                        layer: lc.Layer = extract_layer_call(
                            node=node.value,
                            var_name=var_name,
                            layer_type=func_name,
                            analyzer=self,
                            instructions_to_do_before=instructions_to_do_before_layer
                        )

                        #
                        ### Add layer to the current block. ###
                        #
                        current_block.block_layers[var_name] = layer

                        #
                        ### Add any instructions to the current function. ###
                        #
                        for instr in instructions_to_do_before_layer:
                            current_function.function_flow_control.append(instr)

                        #
                        ### Skip the regular assignment processing. ###
                        #
                        continue

                    #
                    ### Check if this is a custom module instantiation (not a known layer but a model block). ###
                    #
                    elif func_name in self.model_blocks:

                        #
                        ### This is a custom module instantiation, add it to the current block. ###
                        #
                        custom_layer = lc.Layer(
                            layer_var_name=var_name,
                            layer_type=func_name,  # This will be the model block name (e.g., "MultiHeadAttention")
                            layer_parameters_kwargs={}
                        )

                        #
                        ### Add layer to the current block. ###
                        #
                        current_block.block_layers[var_name] = custom_layer

                        #
                        ### Skip the regular assignment processing. ###
                        #
                        continue

                #
                ### Add any instructions that must be executed before the assignment. ###
                #
                for instr in instructions_to_do_before:
                    #
                    current_function.function_flow_control.append(instr)

                #
                ### Add the assignment as a flow control instruction in the function. ###
                #
                current_function.function_flow_control.append(
                    lc.FlowControlVariableAssignment(var_name=var_name, var_value=value_expr)
                )

            #
            ### If inside a class but not inside a function, this is a class variable. ###
            #
            elif inside_class and not inside_function:

                #
                raise UserWarning("Class variables are not supported yet.")

            #
            # If not inside a class, this is a global variable
            #
            else:

                #
                ### Add any instructions that must be executed before the assignment. ###
                #
                for instr in instructions_to_do_before:
                    #
                    self.global_instructions_to_do_before_global_variable_assignments.append(instr)

                #
                ### Save as a global variable. ###
                #
                self.global_variables[var_name] = (type(value_expr).__name__, value_expr)


    #
    ### Visit an AnnAssign node in the AST. ###
    #
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """
        Visits an annotated assignment node in the AST.
        This handles cases like 'self.fc1: nn.Linear = nn.Linear(...)'

        Args:
            node (ast.AnnAssign): The annotated assignment node to visit.
        """

        #
        ### Extract the variable name ###
        #
        if isinstance(node.target, ast.Attribute):
            #
            var_name = node.target.attr
        #
        elif isinstance(node.target, ast.Name):
            #
            var_name = node.target.id
        #
        else:
            #
            return

        #
        ### Check if this is a nn.Module or nn.Sequential assignment that needs special handling ###
        #
        if isinstance(node.value, ast.Call):

            #
            func_name = extract_name_or_attribute(node.value.func)

            #
            ### Check if this is nn.Module or nn.Sequential. ###
            #
            if func_name in ["nn.Module", "nn.Sequential", "Module", "Sequential"]:
                #
                ### Only handle if we're inside a class (regardless of whether we're in a function). ###
                #
                if self.current_model_visit:
                    #
                    self._handle_module_or_sequential_assignment(var_name, node.value, node.target)
                    #
                    return

        #
        ### Check if this is a layer assignment that should be added to block_layers ###
        #
        if isinstance(node.value, ast.Call):

            #
            func_name = extract_name_or_attribute(node.value.func)

            #
            ### Check if this is a known layer type. ###
            ### Try both the full name and just the layer name (e.g., "nn.Linear" and "Linear"). ###
            #
            layer_name = func_name.split('.')[-1] if '.' in func_name else func_name

            #
            if func_name in self.layers or layer_name in self.layers:

                #
                ### Use the layer name that exists in the dictionary. ###
                #
                actual_layer_name = func_name if func_name in self.layers else layer_name

                #
                ### This is a layer assignment, add it to the current block. ###
                #
                if self.current_model_visit and self.current_function_visit:

                    #
                    current_block_name: str = self.current_model_visit[-1]
                    current_block: lc.ModelBlock = self.model_blocks[current_block_name]

                    #
                    ### Extract layer parameters. ###
                    #
                    instructions_to_do_before: list[lc.FlowControlInstruction] = []
                    #
                    layer: lc.Layer = extract_layer_call(
                        node=node.value,
                        var_name=var_name,
                        layer_type=actual_layer_name,
                        analyzer=self,
                        instructions_to_do_before=instructions_to_do_before
                    )

                    #
                    ### Add layer to the current block. ###
                    #
                    current_block.block_layers[var_name] = layer

                    #
                    ### Add any instructions to the current function. ###
                    #
                    current_function: lc.BlockFunction = current_block.block_functions[self.current_function_visit]
                    #
                    for instr in instructions_to_do_before:
                        #
                        current_function.function_flow_control.append(instr)

                    #
                    return

        #
        ### Handle as regular assignment ###
        #
        ### Convert AnnAssign to regular Assign for processing. ###
        #
        assign_node = ast.Assign(targets=[node.target], value=node.value)
        #
        self.visit_Assign(assign_node)


    #
    ### Visit a For node in the AST. ###
    #
    def visit_For(self, node: ast.For) -> None:
        """
        Visits a for loop node in the AST.

        Args:
            node (ast.For): The for loop node to visit.
        """

        #
        ### Ensure we are inside a function. ###
        #
        if not self.current_function_visit:
            #
            print(f"\033[1;31m - WARNING: For loop is not inside a function, skipping. - \033[m")
            #
            return

        #
        ### Get the current model block and function. ###
        #
        current_block_name: str = self.current_model_visit[-1]
        current_block: lc.ModelBlock = self.model_blocks[current_block_name]
        current_function: lc.BlockFunction = current_block.block_functions[self.current_function_visit]

        #
        ### Extract the iterable expression. ###
        #
        instructions_to_do_before: list[lc.FlowControlInstruction] = []
        iterable_expr: lc.Expression = extract_expression(node.iter, self, instructions_to_do_before)

        #
        ### Extract the target variable(s). ###
        #
        if isinstance(node.target, ast.Tuple):

            #
            ### Multiple iteration variables like (i, xi) ###
            #
            target_vars = []

            #
            for target in node.target.elts:

                #
                if isinstance(target, ast.Name):
                    #
                    target_vars.append(target.id)
                #
                else:
                    #
                    ### Complex target, create temp variable ###
                    #
                    temp_target_var = get_next_temp_var_name()
                    #
                    target_expr = extract_expression(target, self, instructions_to_do_before)
                    #
                    instructions_to_do_before.append(lc.FlowControlVariableAssignment(
                        var_name=temp_target_var,
                        var_value=target_expr
                    ))
                    #
                    target_vars.append(temp_target_var)
        #
        else:
            #
            ### Single iteration variable ###
            #
            if isinstance(node.target, ast.Name):
                #
                target_vars = [node.target.id]
            #
            else:
                #
                ### Complex target, create temp variable ###
                #
                temp_target_var = get_next_temp_var_name()
                #
                target_expr = extract_expression(node.target, self, instructions_to_do_before)
                #
                instructions_to_do_before.append(lc.FlowControlVariableAssignment(
                    var_name=temp_target_var,
                    var_value=target_expr
                ))
                #
                target_vars = [temp_target_var]

        #
        ### Extract loop body instructions. ###
        #
        # Store the current function's instructions
        original_instructions = current_function.function_flow_control.copy()
        # Clear the current function's instructions temporarily
        current_function.function_flow_control.clear()

        # Process the loop body
        for body_node in node.body:
            self.visit(body_node)

        # Get the loop body instructions
        loop_instructions = current_function.function_flow_control.copy()

        # Restore the original instructions
        current_function.function_flow_control = original_instructions

        #
        ### Add any instructions that must be executed before the loop. ###
        #
        for instruction in instructions_to_do_before:
            current_function.function_flow_control.append(instruction)

        #
        ### Create the for loop instruction. ###
        #
        for_loop = lc.FlowControlForLoop(
            iterable_var_name=target_vars[0] if len(target_vars) == 1 else f"({', '.join(target_vars)})",
            iterator=iterable_expr,
            flow_control_instructions=loop_instructions
        )

        #
        ### Add the for loop to the function flow control. ###
        #
        current_function.function_flow_control.append(for_loop)


    #
    ### Visit an Expr node in the AST. ###
    #
    def visit_Expr(self, node: ast.Expr) -> None:
        """
        Visits an expression node in the AST.
        This handles standalone expressions like function calls that don't assign to a variable.

        Args:
            node (ast.Expr): The expression node to visit.
        """

        #
        ### Ensure we are inside a function. ###
        #
        if not self.current_function_visit:
            #
            print(f"\033[1;31m - WARNING: Expression is not inside a function, skipping. - \033[m")
            #
            return

        #
        ### Get the current model block and function. ###
        #
        current_block_name: str = self.current_model_visit[-1]
        current_block: lc.ModelBlock = self.model_blocks[current_block_name]
        current_function: lc.BlockFunction = current_block.block_functions[self.current_function_visit]

        #
        ### Extract the expression and any instructions that must be executed before it. ###
        #
        instructions_to_do_before: list[lc.FlowControlInstruction] = []
        value_expr: lc.Expression = extract_expression(node.value, self, instructions_to_do_before)

        #
        ### Add any instructions that must be executed before the expression. ###
        ### Note: extract_expression already handles function calls and creates the appropriate ###
        ### FlowControlFunctionCall instructions, so we don't need to duplicate that logic here. ###
        #
        for instruction in instructions_to_do_before:
            #
            current_function.function_flow_control.append(instruction)


    #
    ### Visit a Return node in the AST. ###
    #
    def visit_Return(self, node: ast.Return) -> None:
        """
        Visits a return node in the AST.

        Args:
            node (ast.Return): The return node to visit.
        """

        #
        ### Ensure we are inside a function. ###
        #
        if not self.current_function_visit:
            #
            print(f"\033[1;31m - WARNING: Return statement is not inside a function, skipping. - \033[m")
            #
            return

        #
        ### Get the current model block and function. ###
        #
        current_block_name: str = self.current_model_visit[-1]
        #
        current_block: lc.ModelBlock = self.model_blocks[current_block_name]
        #
        current_function: lc.BlockFunction = current_block.block_functions[self.current_function_visit]

        #
        ### Extract return variables. ###
        #
        if node.value is not None:

            #
            ### Single return value. ###
            #
            instructions_to_do_before: list[lc.FlowControlInstruction] = []
            #
            return_expr: lc.Expression = extract_expression(node.value, self, instructions_to_do_before)

            #
            return_variables: list[lc.ExpressionVariable] = []

            #
            if isinstance(return_expr, lc.ExpressionTuple):

                #
                for rexpr in return_expr.elements:

                    #
                    if isinstance(rexpr, lc.ExpressionVariable):
                        #
                        return_variables.append(rexpr)
                    #
                    else:
                        #
                        temp_var_name: str = get_next_temp_var_name()
                        #
                        instructions_to_do_before.append(
                            lc.FlowControlVariableAssignment(
                                var_name=temp_var_name,
                                var_value=rexpr
                            )
                        )
                        #
                        return_variables.append(lc.ExpressionVariable(var_name=temp_var_name))

            #
            elif isinstance(return_expr, lc.ExpressionVariable):

                #
                return_variables.append(return_expr)

            #
            else:

                #
                temp_var_name: str = get_next_temp_var_name()
                #
                instructions_to_do_before.append(
                    lc.FlowControlVariableAssignment(
                        var_name=temp_var_name,
                        var_value=return_expr
                    )
                )
                #
                return_variables.append(lc.ExpressionVariable(var_name=temp_var_name))

            #
            ### Add any instructions that need to be done before the return. ###
            #
            for instruction in instructions_to_do_before:
                #
                current_function.function_flow_control.append(instruction)

            #
            ### Create return instruction. ###
            #
            return_instruction: lc.FlowControlReturn = lc.FlowControlReturn(
                return_variables=[rexpr.var_name for rexpr in return_variables]
            )

            #
            ### Add the return to the function flow control. ###
            #
            current_function.function_flow_control.append(return_instruction)

        #
        ### No return value. ###
        #
        else:

            #
            return_instruction: lc.FlowControlReturn = lc.FlowControlReturn(return_variables=[])
            #
            current_function.function_flow_control.append(return_instruction)


    #
    ### Generic visit method for AST nodes. ###
    #
    def generic_visit(self, node: ast.AST) -> None:
        """
        Generic visitor for additional node processing.

        Args:
            node (ast.AST): The node to visit.
        """

        #
        ### Generic visit. ###
        #
        ast.NodeVisitor.generic_visit(self, node)


    #
    ### Cleaning and error detections. ###
    #
    def cleaning_and_error_detections(self) -> None:
        """
        Cleans unused functions and detects errors in the model blocks.
        """

        #
        for block_name, block in list(self.model_blocks.items()):

            #
            ### Check for forward. ###
            #
            if "forward" not in block.block_functions:
                #
                raise ValueError(f"ERROR: Block {block_name} has no forward method.")

            #
            used_funcs: set[str] = {"forward"}

            #
            for func in block.block_functions.values():

                #
                for instr in func.function_flow_control:

                    #
                    if isinstance(instr, lc.FlowControlSubBlockFunctionCall):
                        #
                        used_funcs.add(instr.function_called)

            #
            # block.block_functions = {k: v for k, v in block.block_functions.items() if k in used_funcs}
            #
            ### For the moment, we will just warn for unused functions. ###
            #
            # TODO
            pass


#
def extract_from_file(filepath: str, main_block_name: str = "") -> lc.Language_Model:
    """
    Extracts a neural network model architecture from a PyTorch script file.

    Args:
        filepath (str): Path to the file to extract the architecture from.
        main_block_name (str, optional): Name of the entry point of the model. Defaults to "".

    Raises:
        FileNotFoundError: If the file is not found.

    Returns:
        lc.Language_Model: The extracted model architecture.
    """

    #
    if not os.path.exists(filepath):
        #
        raise FileNotFoundError(f"Error: file not found: `{filepath}`")

    #
    with open(filepath, "r") as source:
        #
        tree: ast.Module = ast.parse(source.read())

    #
    analyzer: ModelAnalyzer = ModelAnalyzer()
    analyzer.main_block = main_block_name
    analyzer.visit(tree)
    analyzer.cleaning_and_error_detections()

    #
    lang1: lc.Language_Model = lc.Language_Model()
    lang1.main_block = analyzer.main_block
    lang1.model_blocks = analyzer.model_blocks
    lang1.global_constants = analyzer.global_variables

    #
    return lang1


#
def list_classes(module: object) -> list[type]:
    """
    list all the classes defined inside a module.

    Args:
        module (object): The module to inspect.

    Returns:
        list[type]: A list of class objects found in the module.
    """

    #
    ### Initialize a list to store class objects, with type hint as list[type]. ###
    #
    classes: list[type] = [

        #
        ### m is a tuple of precise type :  tuple[str, type] , correponding to (name, value). ###
        #
        #
        ### Extract the second element of the tuple 'm', which is the class object itself. ###
        #
        m[1]

        #
        ### Iterate through members of the module that are classes, using inspect.getmembers with inspect.isclass filter. ###
        #
        for m in inspect.getmembers(module, inspect.isclass)
    ]

    #
    ### Return the list of class objects. ###
    #
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

    #
    ### Extract the module name from the filepath by removing the extension and path. ###
    #
    module_name: str = os.path.splitext(os.path.basename(filepath))[0]

    #
    ### Create a module specification using the module name and filepath, spec is of type ModuleSpec. ###
    #
    spec: Optional[importlib.machinery.ModuleSpec] = importlib.util.spec_from_file_location(module_name, filepath)

    #
    ### Check for errors. ###
    #
    if spec is None or spec.loader is None:
        #
        raise ImportError(f"Error : can't load module from file : {filepath}")

    #
    ### Create a module object from the specification, module type is dynamically determined so using Any. ###
    #
    module: Any = importlib.util.module_from_spec(spec)

    #
    ### Execute the module code in the module object's namespace, populating the module. ###
    #
    spec.loader.exec_module(module)

    #
    ### Return the imported module object. ###
    #
    return module


#
def get_pytorch_main_model(model_arch: lc.Language_Model, filepath: str) -> Callable[..., Any]:

    #
    if model_arch.main_block == "":
        #
        raise UserWarning("Error: No main blocks detected in the model architecture !")

    #
    net_module: object = import_module_from_filepath(filepath)

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
        #
        raise UserWarning(f"Error: Usage: python {sys.argv[0]} path_to_model_script.py [--main-block=<MainBlockName>]")

    #
    path_to_file: str = sys.argv[1]
    main_block_name: str = ""
    #
    if len(sys.argv) == 3 and sys.argv[2].startswith("--main-block"):
        #
        main_block_name = sys.argv[2].split("=")[1] if "=" in sys.argv[2] else sys.argv[2].split()[1]

    #
    l1_model: lc.Language_Model = extract_from_file(filepath=path_to_file, main_block_name=main_block_name)

    #
    print(l1_model)

    #
    main_model_class: Callable[..., Any] = get_pytorch_main_model(model_arch=l1_model, filepath=path_to_file)
    #
    print(main_model_class)

    #
    main_model: Any = main_model_class()

    #
    print(main_model)

    #
    # name: str
    # param: nn.Parameter
    # #
    # for name, param in main_model.named_parameters():
    #
    #     #
    #     print(f"DEBUG | name = {name} | param = {param.data.shape}")

