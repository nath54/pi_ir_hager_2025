from typing import Any, Optional, List, Dict


class Layer:
    def __init__(self, layer_var_name: str, layer_type: str, layer_parameters_kwargs: Dict[str, Any]) -> None:
        self.layer_var_name = layer_var_name
        self.layer_type = layer_type
        self.layer_parameters_kwargs = layer_parameters_kwargs

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        params = ', '.join([f'{k}={v}' for k, v in self.layer_parameters_kwargs.items()])
        return f"Layer({self.layer_var_name}, {self.layer_type}, {params})"


class LayerCondition(Layer):
    def __init__(self, layer_var_name: str, layer_conditions_blocks: Dict[str, 'ModelBlock']) -> None:
        super().__init__(layer_var_name, "Condition", {})
        self.layer_conditions_blocks = layer_conditions_blocks


class FlowControlInstruction:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "[FlowControlInstruction]"


class FlowControlVariableInit(FlowControlInstruction):
    def __init__(self, var_name: str, var_type: str, var_value: Any) -> None:
        super().__init__()
        self.var_name = var_name
        self.var_type = var_type
        self.var_value = var_value

    def __str__(self) -> str:
        return f"Init {self.var_name}: {self.var_type} = {self.var_value}"


class FlowControlFunctionCall(FlowControlInstruction):
    def __init__(self, output_vars: List[str], function: str, args: Dict[str, Any]) -> None:
        super().__init__()
        self.output_vars = output_vars
        self.function = function
        self.args = args

    def __str__(self) -> str:
        args_str = ', '.join([f'{k}={v}' for k, v in self.args.items()])
        return f"{', '.join(self.output_vars)} = {self.function}({args_str})"


class FlowControlLayerPass(FlowControlInstruction):
    def __init__(self, output_vars: List[str], layer_name: str, args: Dict[str, Any]) -> None:
        super().__init__()
        self.output_vars = output_vars
        self.layer_name = layer_name
        self.args = args

    def __str__(self) -> str:
        args_str = ', '.join([f'{k}={v}' for k, v in self.args.items()])
        return f"{', '.join(self.output_vars)} = {self.layer_name}({args_str})"


class FlowControlConditional(FlowControlInstruction):
    def __init__(self, condition: str, true_branch: List[FlowControlInstruction], false_branch: List[FlowControlInstruction]) -> None:
        super().__init__()
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __str__(self) -> str:
        true_str = '\n'.join([f"  {x}" for x in self.true_branch])
        false_str = '\n'.join([f"  {x}" for x in self.false_branch])
        return f"If {self.condition}:\n{true_str}\nElse:\n{false_str}"


class BlockFunction:
    def __init__(self, name: str, args: Dict[str, tuple], model_block: 'ModelBlock') -> None:
        self.name = name
        self.args = args
        self.model_block = model_block
        self.instructions: List[FlowControlInstruction] = []

    def __str__(self) -> str:
        instrs = '\n'.join([str(x) for x in self.instructions])
        return f"Function {self.name}:\n{instrs}"


class ModelBlock:
    def __init__(self, name: str) -> None:
        self.name = name
        self.layers: Dict[str, Layer] = {}
        self.sub_blocks: Dict[str, 'ModelBlock'] = {}
        self.functions: Dict[str, BlockFunction] = {}
        self.parameters: Dict[str, tuple] = {}
        self.variables: Dict[str, tuple] = {}

    def add_layer(self, layer: Layer) -> None:
        self.layers[layer.layer_var_name] = layer

    def add_sub_block(self, name: str, block: 'ModelBlock') -> None:
        self.sub_blocks[name] = block

    def __str__(self) -> str:
        layers = '\n'.join([str(l) for l in self.layers.values()])
        sub_blocks = '\n'.join([str(b) for b in self.sub_blocks.values()])
        funcs = '\n'.join([str(f) for f in self.functions.values()])
        return f"ModelBlock {self.name}:\nLayers:\n{layers}\nSub-blocks:\n{sub_blocks}\nFunctions:\n{funcs}"