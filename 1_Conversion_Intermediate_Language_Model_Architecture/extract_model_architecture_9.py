#
from typing import Any, Optional
#
import ast
import sys
import os

import lib_classes as lc


class ModelAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.model_blocks: dict[str, lc.ModelBlock] = {}
        self.main_block: str = ""
        self.current_model_visit: str = ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        is_module_class = any(
            isinstance(base, ast.Attribute) and base.attr == "Module" and
            isinstance(base.value, ast.Name) and base.value.id == "nn"
            for base in node.bases
        )
        if not is_module_class:
            return

        block_name = node.name
        self.current_model_visit = block_name
        if block_name in self.model_blocks:
            raise UserWarning(f"Duplicate class name: {block_name}")

        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        current_block = self.model_blocks.get(self.current_model_visit)
        if not current_block:
            self.generic_visit(node)
            return

        if node.name == "__init__":
            for item in node.body:
                if isinstance(item, ast.Assign):
                    self._process_init_assignment(item, current_block)
        elif node.name == "forward":
            current_block.forward_arguments = {
                arg.arg: ('Any', None) for arg in node.args.args
            }
            for item in node.body:
                if isinstance(item, ast.Assign):
                    self._process_forward_assign(item, current_block)
                elif isinstance(item, ast.Return):
                    self._process_forward_return(item, current_block)

        self.generic_visit(node)

    def _process_init_assignment(self, item: ast.Assign, current_block: lc.ModelBlock) -> None:
        for target in item.targets:
            if not (isinstance(target, ast.Attribute) and
                    isinstance(target.value, ast.Name) and
                    target.value.id == "self"):
                continue

            layer_var_name = target.attr
            if not isinstance(item.value, ast.Call):
                continue

            call_node = item.value
            func = call_node.func
            if not (isinstance(func, ast.Attribute) or not (isinstance(func.value, ast.Name))):
                continue

            if func.value.id != "nn":
                continue

            layer_type = func.attr
            layer_parameters = {}
            for i, arg in enumerate(call_node.args):
                layer_parameters[f"arg{i}"] = self._node_to_variable_repr(arg)
            for kw in call_node.keywords:
                layer_parameters[kw.arg] = self._node_to_variable_repr(kw.value)

            layer = lc.Layer(layer_var_name, layer_type, layer_parameters)
            current_block.block_layers[layer_var_name] = layer

    def _process_forward_assign(self, item: ast.Assign, current_block: lc.ModelBlock) -> None:
        targets = item.targets
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            return

        output_var = targets[0].id
        value = item.value
        if isinstance(value, ast.Call):
            call_node = value
            func_name = self._get_func_name(call_node)
            if (isinstance(call_node.func, ast.Attribute) and
                    isinstance(call_node.func.value, ast.Name) and
                    call_node.func.value.id == "self"):
                layer_name = call_node.func.attr
                layer_args = self._process_call_arguments(call_node)
                current_block.forward_flow_control.append(
                    lc.FlowControlLayerPass([output_var], layer_name, layer_args)
                )
            else:
                func_args = self._process_call_arguments(call_node)
                current_block.forward_flow_control.append(
                    lc.FlowControlFunctionCall([output_var], func_name, func_args)
                )
        else:
            var_type = self._infer_variable_type(value)
            var_value = self._node_to_variable_repr(value)
            current_block.forward_flow_control.append(
                lc.FlowControlVariableInit(output_var, var_type, var_value))

    def _process_forward_return(self, item: ast.Return, current_block: lc.ModelBlock) -> None:
        return_vars = []
        if isinstance(item.value, ast.Tuple):
            return_vars = [self._node_to_variable_repr(e) for e in item.value.elts]
        elif item.value:
            return_vars = [self._node_to_variable_repr(item.value)]
        current_block.forward_flow_control.append(lc.FlowControlReturn(return_vars))

    def _get_func_name(self, call_node: ast.Call) -> str:
        parts = []
        current = call_node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _node_to_variable_repr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._node_to_variable_repr(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Call):
            args = [self._node_to_variable_repr(arg) for arg in node.args]
            kwargs = [f"{kw.arg}={self._node_to_variable_repr(kw.value)}" for kw in node.keywords]
            return f"{self._get_func_name(node)}({', '.join(args + kwargs)})"
        elif isinstance(node, ast.BinOp):
            left = self._node_to_variable_repr(node.left)
            op = self._get_operator_symbol(node.op)
            right = self._node_to_variable_repr(node.right)
            return f"({left} {op} {right})"
        return "<expression>"

    def _process_call_arguments(self, call_node: ast.Call) -> dict[str, Any]:
        args = {f"arg{i}": self._node_to_variable_repr(arg) for i, arg in enumerate(call_node.args)}
        args.update({kw.arg: self._node_to_variable_repr(kw.value) for kw in call_node.keywords})
        return args

    def _infer_variable_type(self, node: ast.AST) -> str:
        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node)
            if "torch" in func_name or "F." in func_name:
                return "Tensor"
        elif isinstance(node, (ast.List, ast.ListComp)):
            return "List"
        elif isinstance(node, (ast.Dict, ast.DictComp)):
            return "Dict"
        return "Any"

    @staticmethod
    def _get_operator_symbol(op: ast.operator) -> str:
        if isinstance(op, ast.Add):
            return "+"
        elif isinstance(op, ast.Sub):
            return "-"
        elif isinstance(op, ast.Mult):
            return "*"
        elif isinstance(op, ast.Div):
            return "/"
        return "?"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise UserWarning(f"Usage: python {sys.argv[0]} path_to_model_script.py")

    path_to_file = sys.argv[1]
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"File not found: {path_to_file}")

    with open(path_to_file, "r") as source:
        tree = ast.parse(source.read())

    analyzer = ModelAnalyzer()
    analyzer.visit(tree)
    print("\n\nAnalysis Results:")
    for block in analyzer.model_blocks.values():
        print(block)