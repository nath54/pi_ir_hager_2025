#
from typing import Callable
#
import sys
#
import torch
import torch.nn as nn
#
from extract_model_architecture import extract_from_file, get_pytorch_main_model
import lib_classes as lc




#
def get_model_block_or_layer_from_named_pytorch_layer(path: str, main_block: lc.ModelBlock) -> None:

    #
    path_split: list[str] = path.split(".")

    #
    current_block: lc.ModelBlock | lc.Layer = main_block

    #
    path_elt: str
    for path_elt in path_split:

        #
        if isinstance(current_block)


#
def link_weights(pytorch_model: nn.Module, model_block: lc.ModelBlock) -> None:

    #
    pass



#
if __name__ == "__main__":
    #
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        raise UserWarning(f"Error: Usage: python {sys.argv[0]} path_to_model_script.py path_to_model_weight.ptx [--main-block <MainBlockName>]")

    #
    path_to_python_file: str = sys.argv[1]
    main_block_name: str = ""
    if len(sys.argv) == 4 and sys.argv[3].startswith("--main-block"):
        main_block_name = sys.argv[3].split("=")[1] if "=" in sys.argv[3] else ""

    #
    l1_model: lc.Language1_Model = extract_from_file(filepath=path_to_python_file, main_block_name=main_block_name)

    #
    filepath_to_weights: str = sys.argv[2]

    #
    model_class: Callable = get_pytorch_main_model(model_arch=l1_model, filepath=path_to_python_file)

    #
    model: nn.Module = model_class()  # For the moment, we will suppose we will support only model that doesn't need parameters to initialise

    #
    if filepath_to_weights != "NO_WEIGHTS":
        #
        model.load_state_dict(torch.load(filepath_to_weights, weights_only=True))

    #
    print(f"Model loaded and architecture extracted !\n\nExtracted architecture :\n\n{l1_model}\n\nArchitecture from direct pytorch module loaded:\n\n{model}\n")

    #
    for name, module in model.named_modules():
        #
        print(f"DEBUG | {name}, module = {module}, {type(module)}")



