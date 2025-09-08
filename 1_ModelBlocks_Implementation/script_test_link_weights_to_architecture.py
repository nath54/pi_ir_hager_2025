#
from typing import Callable, Any, Optional
#
import sys
#
import torch
from torch import nn
#
from lib_extract_model_architecture import extract_from_file, get_pytorch_main_model
import core.lib_impl.lib_classes as lc
import core.lib_impl.lib_layers as ll


#
def get_model_block_or_layer_from_named_pytorch_layer(path: str, model: lc.Language_Model, current_block: Optional[str], all_layers_info: dict[str, ll.BaseLayerInfo]) -> Optional[lc.Layer]:

    #
    crt_block_name: str = current_block if current_block is not None else model.main_block

    #
    path_split: list[str] = path.split(".")

    #
    _current_block: lc.ModelBlock

    #
    _path_elt: str
    for _path_elt in path_split:

        #
        if crt_block_name not in model.model_blocks:

            #
            raise UserWarning(f"Error: no block named `{crt_block_name}` in model : {model}")

        #
        _current_block = model.model_blocks[crt_block_name]

        #
        if _path_elt not in _current_block.block_layers:

            #
            raise UserWarning(f"Error: no elt named `{_path_elt}` in block : {_current_block}")

        #
        _current_layer: lc.Layer = _current_block.block_layers[_path_elt]

        #
        if _current_layer.layer_type in model.model_blocks:

            #
            crt_block_name = _current_layer.layer_type

        #
        elif _current_layer.layer_type in all_layers_info:

            #
            return _current_layer

    #
    return None


#
def link_weights(pytorch_model: nn.Module, model: lc.Language_Model, current_model_block: Optional[str] = None) -> None:

    # TODO
    pass



#
if __name__ == "__main__":

    #
    all_layers_info: dict[str, ll.BaseLayerInfo] = ll.load_layers_dict(filepath="core/layers.json")

    #
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        raise UserWarning(f"Error: Usage: python {sys.argv[0]} path_to_model_script.py path_to_model_weight.ptx [--main-block <MainBlockName>]")

    #
    path_to_python_file: str = sys.argv[1]
    main_block_name: str = ""
    if len(sys.argv) == 4 and sys.argv[3].startswith("--main-block"):
        main_block_name = sys.argv[3].split("=")[1] if "=" in sys.argv[3] else ""

    #
    l1_model: lc.Language_Model = extract_from_file(filepath=path_to_python_file, main_block_name=main_block_name)

    #
    filepath_to_weights: str = sys.argv[2]

    #
    model_class: Callable[..., Any] = get_pytorch_main_model(model_arch=l1_model, filepath=path_to_python_file)

    #
    model: nn.Module = model_class()  # For the moment, we will suppose we will support only model that doesn't need parameters to initialise

    #
    if filepath_to_weights != "NO_WEIGHTS":
        #
        model.load_state_dict(torch.load(filepath_to_weights, weights_only=True))  # type: ignore

    #
    print(f"Model loaded and architecture extracted !\n\nExtracted architecture :\n\n{l1_model}\n\nArchitecture from direct pytorch module loaded:\n\n{model}\n")

    #
    name: str
    module: nn.Module
    #
    for name, module in model.named_modules():  # type: ignore
        #
        print(f"DEBUG | {name}, module = {module}, {type(module)}")  # type: ignore
