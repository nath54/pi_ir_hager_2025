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
        is_module_class: bool = self.is_torch_module_class(node)
        if not is_module_class:
            return

        block_name: str = node.name
        self.current_model_visit = block_name

        if block_name in self.model_blocks:
            raise UserWarning(f"ERROR : There are two classes with the same name !!!\nBad name : {block_name}\n")

        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "__init__":
            self.analyze_init_method(node)
        elif node.name == "forward":
            self.analyze_forward_method(node)
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST):
        ast.NodeVisitor.generic_visit(self, node)

    def is_torch_module_class(self, node: ast.ClassDef) -> bool:
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True
        return False

    def analyze_init_method(self, node):
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        layer_var_name = target.id
                        if isinstance(item.value, ast.Call):
                            layer_type = item.value.func.attr
                            layer_parameters_kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in item.value.keywords}
                            layer = lc.Layer(layer_var_name, layer_type, layer_parameters_kwargs)
                            self.model_blocks[self.current_model_visit].block_layers[layer_var_name] = layer

    def analyze_forward_method(self, node):
        for item in node.body:
            if isinstance(item, ast.Assign):
                output_variables = [t.id for t in item.targets if isinstance(t, ast.Name)]
                if isinstance(item.value, ast.Call):
                    function_called = item.value.func.attr
                    function_arguments = {kw.arg: ast.literal_eval(kw.value) for kw in item.value.keywords}
                    flow_control = lc.FlowControlFunctionCall(output_variables, function_called, function_arguments)
                    self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_control)
            elif isinstance(item, ast.Return):
                return_variables = [v.id for v in item.value.elts if isinstance(v, ast.Name)]
                flow_control = lc.FlowControlReturn(return_variables)
                self.model_blocks[self.current_model_visit].forward_flow_control.append(flow_control)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise UserWarning(f"Error: if you use this script directly, you should use it like that :\n  python {sys.argv[0]} path_to_model_script.py")

    path_to_file: str = sys.argv[1]

    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"Error: file not found : `{path_to_file}`")

    with open(path_to_file, "r") as source:
        tree = ast.parse(source.read())

    analyzer = ModelAnalyzer()
    analyzer.visit(tree)

    print("\n" * 2)
    print(analyzer.model_blocks)
