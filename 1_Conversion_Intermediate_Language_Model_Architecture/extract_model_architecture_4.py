#
from typing import Optional, Any
#
import ast
import sys
import os
import lib_classes as lc


class ModelAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.model_blocks: dict[str, lc.ModelBlock] = {}
        self.current_model_visit: str = ""
        self.imported_nn_aliases: set[str] = {'nn'}  # Default alias for 'torch.nn'

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        is_nn_module = any(
            isinstance(base, ast.Attribute) and base.attr == "Module" and
            isinstance(base.value, ast.Name) and base.value.id in self.imported_nn_aliases
            for base in node.bases
        )
        if not is_nn_module:
            return

        block_name = node.name
        self.current_model_visit = block_name
        if block_name in self.model_blocks:
            raise ValueError(f"Duplicate model block name: {block_name}")
        self.model_blocks[block_name] = lc.ModelBlock(block_name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "__init__":
            current_block = self.model_blocks[self.current_model_visit]
            for item in node.body:
                if isinstance(item, ast.Assign):
                    self.process_init_assign(item, current_block)
        elif node.name == "forward":
            current_block = self.model_blocks[self.current_model_visit]
            for item in node.body:
                self.process_forward_item(item, current_block)
        self.generic_visit(node)

    def process_init_assign(self, item: ast.Assign, block: lc.ModelBlock) -> None:
        for target in item.targets:
            if (isinstance(target, ast.Attribute) and
                isinstance(target.value, ast.Name) and
                target.value.id == 'self' and
                isinstance(item.value, ast.Call)):
                call = item.value
                if (isinstance(call.func, ast.Attribute) and
                    isinstance(call.func.value, ast.Name) and
                    call.func.value.id in self.imported_nn_aliases):
                    layer_type = call.func.attr
                    layer_name = target.attr
                    parameters = self.process_call_arguments(call)
                    block.block_layers[layer_name] = lc.Layer(layer_name, layer_type, parameters)

    def process_forward_item(self, item: ast.AST, block: lc.ModelBlock) -> None:
        if isinstance(item, ast.Assign):
            self.process_forward_assign(item, block)
        elif isinstance(item, ast.Expr):
            self.process_forward_expr(item, block)

    def process_forward_assign(self, item: ast.Assign, block: lc.ModelBlock) -> None:
        targets = [t.id for t in item.targets if isinstance(t, ast.Name)]
        if not targets:
            return
        rhs = item.value
        if isinstance(rhs, ast.Call):
            if (isinstance(rhs.func, ast.Attribute) and
                isinstance(rhs.func.value, ast.Name) and
                rhs.func.value.id == 'self'):
                layer_name = rhs.func.attr
                args = self.process_call_arguments(rhs)
                block.forward_flow_control.append(
                    lc.FlowControlLayerPass(targets, layer_name, args)
                )
            else:
                func_name = ast.unparse(rhs.func).strip()
                args = self.process_call_arguments(rhs)
                block.forward_flow_control.append(
                    lc.FlowControlFunctionCall(targets, func_name, args)
                )
        else:
            var_name = targets[0]
            var_type, var_value = self.process_rhs_value(rhs)
            block.forward_flow_control.append(
                lc.FlowControlVariableInit(var_name, var_type, var_value)
            )

    #
    def process_forward_expr(self, item: ast.Expr, block: lc.ModelBlock) -> None:
        # TODO
        pass

    def process_rhs_value(self, rhs: ast.AST) -> tuple[str, Any]:
        try:
            value = ast.literal_eval(rhs)
            return type(value).__name__, value
        except Exception as _:
            return 'unknown', ast.unparse(rhs).strip()

    def process_call_arguments(self, call: ast.Call) -> dict[str, Any]:
        args = {}
        for idx, arg in enumerate(call.args):
            key = f'arg{idx}'
            args[key] = self.parse_arg_value(arg)
        for kw in call.keywords:
            if kw.arg is None:
                continue
            key = kw.arg
            args[key] = self.parse_arg_value(kw.value)
        return args

    def parse_arg_value(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception as _:
            return ast.unparse(node).strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_script.py>")
        sys.exit(1)
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    analyzer = ModelAnalyzer()
    analyzer.visit(tree)
    print("\nModel Blocks Analyzed:")
    for name, block in analyzer.model_blocks.items():
        print(block)