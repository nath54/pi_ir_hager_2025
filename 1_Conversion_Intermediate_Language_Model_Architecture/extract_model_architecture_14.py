#
from typing import Optional, Any, Dict, Tuple, List
#
import ast
import sys
import os
from lib_classes_6 import ModelBlock, Layer, BlockFunction, FlowControlInstruction, FlowControlVariableInit, FlowControlFunctionCall, FlowControlLayerPass, FlowControlConditional


class ModelAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.blocks: Dict[str, ModelBlock] = {}
        self.current_block: Optional[ModelBlock] = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if any(base.id == 'Module' for base in node.bases if isinstance(base, ast.Name)):
            block = ModelBlock(node.name)
            self.blocks[node.name] = block
            self.current_block = block
            self.generic_visit(node)
            self.current_block = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.current_block and node.name == '__init__':
            self._process_init(node)
        elif self.current_block and node.name == 'forward':
            self._process_forward(node)
        self.generic_visit(node)

    def _process_init(self, node: ast.FunctionDef) -> None:
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                self._process_assign(stmt)

    def _process_assign(self, stmt: ast.Assign) -> None:
        for target in stmt.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                var_name = target.attr
                if isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if isinstance(call.func, ast.Attribute) and call.func.value.id == 'nn':
                        layer_type = call.func.attr
                        args = [self._unparse(a) for a in call.args]
                        kwargs = {k.arg: self._unparse(k.value) for k in call.keywords}
                        layer = Layer(var_name, layer_type, {'args': args, **kwargs})
                        self.current_block.add_layer(layer)
                    elif isinstance(call.func, ast.Name) and call.func.id == 'Sequential':
                        self._process_sequential(var_name, call)

    def _process_sequential(self, name: str, call: ast.Call) -> None:
        sub_block = ModelBlock(name)
        for i, arg in enumerate(call.args):
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.value.id == 'nn':
                layer_type = arg.func.attr
                args = [self._unparse(a) for a in arg.args]
                kwargs = {k.arg: self._unparse(k.value) for k in arg.keywords}
                layer = Layer(str(i), layer_type, {'args': args, **kwargs})
                sub_block.add_layer(layer)
        self.current_block.add_sub_block(name, sub_block)

    def _process_forward(self, node: ast.FunctionDef) -> None:
        func = BlockFunction('forward', {}, self.current_block)
        for stmt in node.body:
            func.instructions.extend(self._process_stmt(stmt))
        self.current_block.functions['forward'] = func

    def _process_stmt(self, stmt: ast.stmt) -> List[FlowControlInstruction]:
        if isinstance(stmt, ast.Assign):
            return self._process_assign_forward(stmt)
        elif isinstance(stmt, ast.If):
            return self._process_if(stmt)
        return []

    def _process_assign_forward(self, stmt: ast.Assign) -> List[FlowControlInstruction]:
        if isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Attribute) and call.func.value.value.id == 'self':
                layer_name = call.func.value.attr
                args = {f'arg{i}': self._unparse(a) for i, a in enumerate(call.args)}
                outputs = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                return [FlowControlLayerPass(outputs, layer_name, args)]
        return []

    def _process_if(self, stmt: ast.If) -> List[FlowControlInstruction]:
        condition = self._unparse(stmt.test)
        true_branch = []
        for s in stmt.body:
            true_branch.extend(self._process_stmt(s))
        false_branch = []
        for s in stmt.orelse:
            false_branch.extend(self._process_stmt(s))
        return [FlowControlConditional(condition, true_branch, false_branch)]

    def _unparse(self, node: ast.AST) -> str:
        return ast.unparse(node)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model_analyzer.py <file>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        tree = ast.parse(f.read())

    analyzer = ModelAnalyzer()
    analyzer.visit(tree)

    for name, block in analyzer.blocks.items():
        print(block)