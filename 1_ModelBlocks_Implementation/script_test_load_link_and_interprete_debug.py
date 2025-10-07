#
from typing import Callable, Any
#
import sys
#
import numpy as np
from numpy.typing import NDArray
#
import torch
from torch import nn
#z
from lib_extract_model_architecture import extract_from_file, get_pytorch_main_model
from lib_weights_link import link_weights
import core.lib_impl.lib_classes as lc
import core.lib_impl.lib_layers as ll
import core.lib_impl.lib_interpretor as li
import core.lib_impl.lib_interpretor_debug as lidbg


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

    #
    ### Create interpreter (debug). ###
    #
    interpreter: lidbg.LanguageModel_ForwardInterpreter_Debug = lidbg.LanguageModel_ForwardInterpreter_Debug(l1_model)

    #
    ### For script equivalence with non-debug version, run continuously by default. ###
    ### Users can comment the next line to enter interactive stepping by default. ###
    # interpreter.debugger_continue()

    #
    ### Validate model. ###
    #
    validation_issues: list[str] = li.ModelInterpreterUtils.validate_model_structure(l1_model)

    #
    if validation_issues:

        #
        print("Validation Issues:")
        #
        for issue in validation_issues:
            #
            print(f"  - {issue}")

        #
        exit()

    #
    ### Print model summary. ###
    #
    print(li.ModelInterpreterUtils.generate_model_summary(l1_model))

    #
    ### Prepare input. ###
    #
    batch_size: int = 3
    window_time: int = 128
    features_1: int = 16
    features_2: int = 8
    input_tensor: NDArray[np.float32] = np.random.randn(batch_size, window_time, features_1, features_2).astype(np.float32)
    #
    inputs: dict[str, NDArray[np.float32]] = {"x": input_tensor}

    #
    ### Execute forward pass with trace. ###
    #
    outputs: dict[str, NDArray[np.float32]] = li.ModelInterpreterUtils.print_execution_trace(interpreter, inputs, verbose=True)

    #
    print(f"DEBUG | outputs = {outputs}")

    #
    if "output" in outputs:
        #
        print(f"\nFinal output shape: {outputs['output'].shape}")
        print(f"Output sample: {outputs['output'][0]}")


