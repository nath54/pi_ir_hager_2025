#
from typing import Callable, Any, Optional, cast
#
import sys
#
import torch
from torch import nn
#
import numpy as np
from numpy.typing import NDArray
#
from lib_extract_model_architecture import extract_from_file, get_pytorch_main_model
import core.lib_impl.lib_classes as lc
import core.lib_impl.lib_layers as ll


#
def get_model_block_or_layer_from_named_pytorch_layer(path: str, l_model: lc.Language_Model, current_block: Optional[str], all_layers_info: dict[str, ll.BaseLayerInfo]) -> Optional[lc.Layer]:

    #
    crt_block_name: str = current_block if current_block is not None else l_model.main_block

    #
    path_split: list[str] = path.split(".")

    #
    _current_block: lc.ModelBlock

    #
    # print(f"\033[41m DEBUG | get_model_block_or_layer_from_named_pytorch_layer | path = `{path}` \033[m")

    #
    _path_elt: str
    #
    for _j, _path_elt in enumerate(path_split):

        #
        # print(f"\033[41m DEBUG | j = `{j}` | _path_elt = `{_path_elt}` \033[m")

        #
        if crt_block_name not in l_model.model_blocks:
            #
            raise UserWarning(f"Error: no block named `{crt_block_name}` in model : {l_model}")

        #
        _current_block = l_model.model_blocks[crt_block_name]

        #
        if _path_elt not in _current_block.block_layers:
            #
            raise UserWarning(f"Error: no elt named `{_path_elt}` in block : {_current_block}")

        #
        _current_layer: lc.Layer = _current_block.block_layers[_path_elt]

        # print(f"\033[41m DEBUG | => _current_layer = `{_current_layer}` \033[m")

        #
        if _current_layer.layer_type in l_model.model_blocks:
            #
            crt_block_name = _current_layer.layer_type
            #
            # print(f"\033[41m DEBUG | _current_layer.layer_type in l_model.model_blocks | crt_block_name = `{crt_block_name}` \033[m")

        #
        elif _current_layer.layer_type in all_layers_info:
            #
            # print(f"\033[42m DEBUG | _current_layer.layer_type in all_layers_info \033[m")
            #
            return _current_layer

    #
    return None


#
def get_class_name(o: Any) -> str:
    #
    return o.__class__.__name__


#
def link_weights(pt_model: nn.Module, l_model: lc.Language_Model, all_layers_info: dict[str, ll.BaseLayerInfo], crt_model_block_name: Optional[str] = None) -> lc.Language_Model:

    #
    module_path_name: str
    module: nn.Module
    #
    for module_path_name, module in pt_model.named_modules():  # type: ignore

        #
        if not module_path_name or get_class_name(module) not in all_layers_info:
            #
            continue

        #
        layer: Optional[lc.Layer] = get_model_block_or_layer_from_named_pytorch_layer(
            path = cast( str, module_path_name ),
            l_model = l_model,
            current_block=crt_model_block_name,
            all_layers_info=all_layers_info
        )

        #
        if layer is None:
            #
            print(f" \033[31m ERROR: layer is None for {module} at path {module_path_name} \033[m ")
            #
            continue

        #
        if layer.layer_type not in str(module):  # type: ignore
            #
            print(f" \033[31m ERROR : {layer.layer_type} is not same layer type than {module} at path {module_path_name} \033[m ")
            #
            continue

        #
        print(f" \033[34m {layer.layer_weights} | module = {module} \033[m ")

        #
        ### Get all named parameters. ###
        #
        for p_name, param in module.named_parameters():  # type: ignore

            #
            print(f"Param Name: `{p_name}`, Shape: {param.shape}")  # type: ignore
            # print(param)  # type: ignore

            #
            layer.layer_weights[p_name] = cast( NDArray[np.float32], param.data.cpu().numpy().astype(dtype=np.float32) )  # type: ignore

    #
    return l_model


#
if __name__ == "__main__":

    #
    all_layers_info: dict[str, ll.BaseLayerInfo] = ll.load_layers_dict(filepath="core/layers.json")

    #
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        #
        raise UserWarning(f"Error: Usage: python {sys.argv[0]} path_to_model_script.py path_to_model_weight.ptx [--main-block <MainBlockName>]")

    #
    path_to_python_file: str = sys.argv[1]
    main_block_name: str = ""
    #
    if len(sys.argv) == 4 and sys.argv[3].startswith("--main-block"):
        #
        main_block_name = sys.argv[3].split("=")[1] if "=" in sys.argv[3] else ""

    #
    l1_model: lc.Language_Model = extract_from_file(filepath=path_to_python_file, main_block_name=main_block_name)

    #
    filepath_to_weights: str = sys.argv[2]

    #
    model_class: Callable[..., Any] = get_pytorch_main_model(model_arch=l1_model, filepath=path_to_python_file)

    #
    pt_model: nn.Module = model_class()  # For the moment, we will suppose we will support only model that doesn't need parameters to initialise

    #
    if filepath_to_weights != "NO_WEIGHTS":
        #
        pt_model.load_state_dict(torch.load(filepath_to_weights, weights_only=True))  # type: ignore

    #
    print(f"Model loaded and architecture extracted !\n\nExtracted architecture :\n\n{l1_model}\n\n\n")
    # print(f"Model loaded and architecture extracted !\n\nExtracted architecture :\n\n{l1_model}\n\nArchitecture from direct pytorch module loaded:\n\n{pt_model}\n")

    #
    l1_model = link_weights(pt_model = pt_model, l_model = l1_model, all_layers_info=all_layers_info, crt_model_block_name = None)

