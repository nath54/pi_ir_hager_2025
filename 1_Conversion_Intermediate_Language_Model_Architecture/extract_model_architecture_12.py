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
        is_module_class = self.is_torch_module_class(node)
        if not is_module_class:
            return

        block_name = node.name
        self.current_model_visit = block_name

        if block_name in self.model_blocks:
            raise UserWarning(f"ERROR: There are two classes with the same name!!!\nBad name: {block_name}\n")

        self.model_blocks[block_name] = lc.ModelBlock(block_name=block_name)
        self.generic_visit(node)

    def is_torch_module_class(self, node: ast.ClassDef) -> bool:
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True
        return False

    def visit_FunctionDef(self, node):
        if node.name == "__init__":
            self.analyze_init_method(node)
        elif node.name == "forward":
            self.analyze_forward_method(node)
        self.generic_visit(node)

    def analyze_init_method(self, node):
        for item in node.body:
            if isinstance(item, ast.Assign):
                self.process_layer_definition(item)
            elif isinstance(item, ast.For):
                self.process_loop_initialization(item)

    def process_layer_definition(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                layer_var_name = target.id
                if isinstance(node.value, ast.Call):
                    layer_type = self.get_layer_type(node.value.func)
                    layer_parameters_kwargs = self.extract_kwargs(node.value.keywords)
                    layer = lc.Layer(layer_var_name, layer_type, layer_parameters_kwargs)
                    self.model_blocks[self.current_model_visit].block_layers[layer_var_name] = layer

    def get_layer_type(self, func):
        if isinstance(func, ast.Attribute):
            return func.attr
        elif isinstance(func, ast.Name):
            return func.id
        return ""

    def extract_kwargs(self, keywords):
        kwargs = {}
        for keyword in keywords:
            if isinstance(keyword.value, ast.Constant):
                kwargs[keyword.arg] = keyword.value.value
        return kwargs

    def process_loop_initialization(self, node: ast.For):
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute):
            if node.iter.func.attr in ["ModuleList", "Sequential"]:
                self.process_module_list_or_sequential(node)

    def process_module_list_or_sequential(self, node: ast.For):
        for item in node.body:
            if isinstance(item, ast.Assign):
                self.process_layer_definition(item)

    def analyze_forward_method(self, node):
        for item in node.body:
            if isinstance(item, ast.Assign):
                self.process_forward_assignment(item)
            elif isinstance(item, ast.Return):
                self.process_forward_return(item)
            elif isinstance(item, ast.For):
                self.process_loop_initialization(item)

    def process_forward_assignment(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                if isinstance(node.value, ast.Call):
                    function_called = self.get_layer_type(node.value.func)
                    function_arguments = self.extract_kwargs(node.value.keywords)
                    flow_control = lc.FlowControlLayerPass([var_name], function_called, function_arguments)
                    self.model_blocks[self.current_model_visit].block_functions["forward"].function_flow_control.append(flow_control)

    def process_forward_return(self, node: ast.Return):
        return_variables = [target.id for target in node.value.elts if isinstance(target, ast.Name)]
        flow_control = lc.FlowControlReturn(return_variables)
        self.model_blocks[self.current_model_visit].block_functions["forward"].function_flow_control.append(flow_control)

    def generic_visit(self, node: ast.AST):
        ast.NodeVisitor.generic_visit(self, node)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise UserWarning(f"Error: if you use this script directly, you should use it like that:\n  python {sys.argv[0]} path_to_model_script.py")

    path_to_file = sys.argv[1]

    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"Error: file not found: `{path_to_file}`")

    with open(path_to_file, "r") as source:
        tree = ast.parse(source.read())

    analyzer = ModelAnalyzer()
    analyzer.visit(tree)

    print("\n" * 2)
    print(analyzer.model_blocks)
