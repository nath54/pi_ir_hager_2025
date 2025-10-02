#
### Import Libraries. ###
#
from typing import Optional, Callable, Any
#
import os
import argparse
#
import torch
from torch import nn, Tensor
#
import numpy as np
from numpy.typing import NDArray
#
import core.lib_impl.lib_classes as lc
import core.lib_impl.lib_layers as ll
import core.lib_impl.lib_interpretor as li
#
import lib_extract_model_architecture as lema
import lib_test_link_weights_to_architecture as ltlwta
import lib_test_load_link_and_interprete as ltllai


#
### Tester class. ###
#
class Tester:

    #
    def __init__(
        self,
        code_path: str,
        input_dim: tuple[int, ...],
        main_block_name: str = "",
        weights_path: Optional[str] = None
    ) -> None:

        #
        ### Gather the base arguments here. ###
        #
        self.code_path: str = code_path
        #
        self.main_block_name: str = main_block_name
        #
        self.weights_path: Optional[str] = weights_path
        #
        self.input_dim: tuple[int, ...] = input_dim


    #
    def verify_model_extraction_and_linking(
        self,
        l_model: lc.Language_Model,
        pt_model: nn.Module,
        all_layer_infos: dict[str, ll.BaseLayerInfo]
    ) -> tuple[bool, str]:

        #
        ### Verify each pytorch layer (excepted non inferences one like dropout is there). ###
        #

        #
        ### Get all PyTorch layers that should be present in the extracted model. ###
        #
        pytorch_layers: list[tuple[str, nn.Module]] = []
        #
        name: str
        module: nn.Module
        #
        for name, module in pt_model.named_modules():
            #
            if name and module.__class__.__name__ in all_layer_infos:
                #
                pytorch_layers.append((name, module))

        #
        ### Get all extracted layers from the language model. ###
        #
        extracted_layers: list[tuple[str, lc.Layer]] = []

        #
        for block_name, block in l_model.model_blocks.items():

            #
            for layer_name, layer in block.block_layers.items():

                #
                if layer.layer_type in all_layer_infos:
                    #
                    extracted_layers.append((f"{block_name}.{layer_name}", layer))

        #
        ### 1. Verify layer count ###
        #
        if len(pytorch_layers) != len(extracted_layers):
            #
            return (False, f"Layer count mismatch: PyTorch has {len(pytorch_layers)} layers, extracted model has {len(extracted_layers)} layers")

        #
        ### 2. Verify layer types and parameters ###
        #
        pytorch_layer_dict = {name: module for name, module in pytorch_layers}
        extracted_layer_dict = {name: layer for name, layer in extracted_layers}

        #
        for pt_name, pt_module in pytorch_layers:

            #
            ### Find corresponding extracted layer ###
            #
            found_match: bool = False

            #
            for ext_name, ext_layer in extracted_layers:

                #
                if pt_module.__class__.__name__ == ext_layer.layer_type:

                    #
                    found_match = True

                    #
                    ### Verify layer type ###
                    #
                    if pt_module.__class__.__name__ != ext_layer.layer_type:
                        #
                        return (False, f"Layer type mismatch for {pt_name}: PyTorch has {pt_module.__class__.__name__}, extracted has {ext_layer.layer_type}")

                    #
                    ### Verify parameters ###
                    #
                    pt_params = dict(pt_module.named_parameters())
                    ext_params = ext_layer.layer_parameters_kwargs

                    #
                    ### Check if parameter names match (basic check) ###
                    #
                    pt_param_names = set(pt_params.keys())
                    ext_param_names = set(ext_params.keys())

                    #
                    ### For layers with weights, check if weight parameters are present and dimensions match ###
                    #
                    if hasattr(pt_module, 'weight') and 'weight' in pt_param_names:

                        #
                        if 'weight' not in ext_layer.layer_weights:

                            #
                            return (False, f"Missing weight parameter in extracted layer {ext_name}")

                        #
                        ### Verify weight dimensions match ###
                        #
                        pt_weight_shape = pt_module.weight.shape
                        #
                        ext_weight_shape = ext_layer.layer_weights['weight'].shape

                        #
                        ### For Linear layers, the weight matrix is transposed during extraction ###
                        ### PyTorch stores as (out_features, in_features) but we store as (in_features, out_features) ###
                        #
                        expected_ext_shape = pt_weight_shape
                        #
                        if ext_layer.layer_type == "Linear":
                            #
                            expected_ext_shape = (pt_weight_shape[1], pt_weight_shape[0])

                        #
                        if ext_weight_shape != expected_ext_shape:

                            #
                            return (False, f"Weight shape mismatch in layer {ext_name}: PyTorch has {pt_weight_shape}, extracted has {ext_weight_shape}, expected {expected_ext_shape}")

                    #
                    if hasattr(pt_module, 'bias') and 'bias' in pt_param_names:
                        #
                        if 'bias' not in ext_layer.layer_weights:
                            #
                            return (False, f"Missing bias parameter in extracted layer {ext_name}")

                        #
                        ### Verify bias dimensions match ###
                        #
                        pt_bias_shape = pt_module.bias.shape
                        #
                        ext_bias_shape = ext_layer.layer_weights['bias'].shape

                        #
                        if pt_bias_shape != ext_bias_shape:
                            #
                            return (False, f"Bias shape mismatch in layer {ext_name}: PyTorch has {pt_bias_shape}, extracted has {ext_bias_shape}")

                    #
                    break

            #
            if not found_match:
                #
                return (False, f"No matching extracted layer found for PyTorch layer {pt_name} of type {pt_module.__class__.__name__}")

        #
        ### 3. Verify architecture structure ###
        #

        #
        ### Check if main block exists ###
        #
        if not l_model.main_block:
            #
            return (False, "No main block specified in extracted model")

        #
        if l_model.main_block not in l_model.model_blocks:
            #
            return (False, f"Main block '{l_model.main_block}' not found in extracted model blocks")

        #
        ### Check if main block has a forward function ###
        #
        main_block = l_model.model_blocks[l_model.main_block]
        #
        if 'forward' not in main_block.block_functions:
            #
            return (False, f"Main block '{l_model.main_block}' does not have a forward function")

        #
        ### 4. Verify that all extracted layers are valid according to layer definitions ###
        #
        for ext_name, ext_layer in extracted_layers:
            #
            if ext_layer.layer_type not in all_layer_infos:
                #
                return (False, f"Extracted layer {ext_name} has unknown type {ext_layer.layer_type}")

            #
            layer_info = all_layer_infos[ext_layer.layer_type]

            #
            ### Check if required parameters are present ###
            #
            for param_name, (param_type, param_default) in layer_info.parameters.items():
                #
                if param_name not in ext_layer.layer_parameters_kwargs:
                    #
                    if isinstance(param_default, lc.ExpressionNoDefaultArguments):
                        #
                        return (False, f"Missing required parameter '{param_name}' in layer {ext_name}")

        #
        ### 5. Verify that non-inference layers (like Dropout) are excluded ###
        #
        inference_layers = [
            name for name, module in pytorch_layers
            if module.__class__.__name__ not in ['Dropout', 'Dropout2d', 'Dropout3d', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d']
        ]

        #
        if len(inference_layers) != len(extracted_layers):
            #
            return (False, f"Non-inference layer filtering issue: {len(inference_layers)} inference layers vs {len(extracted_layers)} extracted layers")

        #
        return (True, f"Model extraction verification passed: {len(extracted_layers)} layers verified")


    #
    def test_code_extractor(self) -> lc.Language_Model:

        #
        l_model: lc.Language_Model = lema.extract_from_file(filepath=self.code_path, main_block_name=self.main_block_name)

        #
        return l_model


    #
    def test_model_execution(
        self,
        l_model: lc.Language_Model,
        pt_model: nn.Module,
        interpreter: li.LanguageModel_ForwardInterpreter
    ) -> tuple[bool, str]:

        #
        ### Generate a random input. ###
        #
        inp: NDArray[np.float32] = np.random.rand(*self.input_dim)
        #
        inputs_dict: dict[str, NDArray[np.float32]] = {
            "x": inp
        }

        #
        ### Get the output from the pytorch model. ###
        #
        with torch.no_grad():

            #
            ref_output_pt: Tensor = pt_model.forward( torch.from_numpy(inp).to(dtype=torch.float32) )
            ref_output: NDArray[np.float32] = ref_output_pt.to(dtype=torch.float32, device="cpu").numpy()

        #
        ### Get the output from the extracted model. ###
        #
        extr_outputs: dict[str, NDArray[np.float32]] = interpreter.forward(inputs=inputs_dict)

        #
        extr_output: Optional[NDArray[np.float32]] = None
        #
        for var_name in ["output", "y", "x"]:

            #
            if var_name in extr_outputs:
                #
                extr_output = extr_outputs[var_name]
                #
                break

        #
        if len(extr_outputs) == 0:
            #
            return (False, "No model outputs.")

        #
        extr_output = list(extr_outputs.values())[0]

        #
        if extr_output is None:
            #
            return (False, "Model output is None.")

        #
        ### Compare Pytorch Model Output to Extracted Model Output. ###
        #

        #
        ### Compare Shape. ###
        #
        if ref_output.shape != extr_output.shape:

            #
            return (False, f"Model output don't have expected output shape: obtained = {extr_output.shape}, expected = {ref_output.shape} !")

        #
        dist: float = np.linalg.norm(ref_output - extr_output)

        #
        if dist > 1e-1:

            #
            return (False, f"Model output is too far from reference : {dist} > 1e-1 !")

        #
        return (True, f"Distance between reference and inference output is {dist}.")


    #
    def test(self) -> tuple[bool, bool, bool, str]:

        #
        ### Step 1 - Try to load language model. ###
        #
        try:

            #
            l_model: lc.Language_Model = self.test_code_extractor()

        #
        except Exception as e:

            #
            return (False, False, False, str(e))

        #
        ### Step 2 - Load the pytorch model. ###
        #

        #
        model_class: Callable[..., Any] = lema.get_pytorch_main_model(model_arch=l_model, filepath=self.code_path)

        #
        ### For the moment, we will suppose we will support only model that doesn't need parameters to initialise. ###
        #
        pt_model: nn.Module = model_class()

        #
        if self.weights_path is not None and self.weights_path not in ["", "NO_WEIGHTS"] and os.path.exists(self.weights_path):
            #
            pt_model.load_state_dict(torch.load(self.weights_path, weights_only=True))  # type: ignore

        #
        ### Need all_layers_info for weights linking. ###
        #
        all_layers_info: dict[str, ll.BaseLayerInfo] = ll.load_layers_dict(filepath="core/layers.json")

        #
        ### Step 3 - Create a model interpreter. ###
        #
        try:
            #
            interpreter: li.LanguageModel_ForwardInterpreter = li.LanguageModel_ForwardInterpreter(l_model)
        #
        except Exception as e:
            #
            return (True, True, False, str(e))


        #
        ### Step 4 - Verify if model seems to be loaded correctly. ###
        #
        #
        validation_issues: list[str] = li.ModelInterpreterUtils.validate_model_structure(l_model)

        #
        if validation_issues:

            #
            error_msg: str = "Validation Issues:\n\t" + "\n\t - ".join( validation_issues )

            #
            return (False, False, False, error_msg)

        #
        ### Step 5 - Try to link the pytorch model weights to the extracted intermediate representation of the model. ###
        #
        try:

            #
            l_model = ltlwta.link_weights(pt_model=pt_model, l_model=l_model, all_layers_info=all_layers_info)

        #
        except Exception as e:

            #
            return (True, False, False, str(e))

        #
        ### Step 6 - Verify Extracted Module Structure, Layers and Weighst. ###
        #
        res_verif: tuple[bool, str] = self.verify_model_extraction_and_linking(
            l_model=l_model,
            pt_model=pt_model,
            all_layer_infos=all_layers_info
        )

        #
        if not res_verif[0]:
            #
            return (False, False, False, res_verif[1])


        #
        ### Step 7 - Try to execute model. ###
        #
        test_exec: tuple[bool, str] = self.test_model_execution(l_model=l_model, pt_model=pt_model, interpreter=interpreter)

        #
        if not test_exec[0]:
            #
            return (True, True, False, test_exec[1])

        #
        ### Step 8 - All the tests passed correctly. ###
        #
        return (True, True, True, test_exec[1])


    #
    @staticmethod
    def test_code(
        code_path: str,
        input_dim: tuple[int, ...],
        main_block_name: str = "",
        weights_path: Optional[str] = None
    ) -> tuple[bool, bool, bool, str]:

        #
        tester: Tester = Tester(
            code_path=code_path,
            input_dim=input_dim,
            main_block_name=main_block_name,
            weights_path=weights_path
        )

        #
        return tester.test()


#
if __name__ == "__main__":

    #
    ### Argument parsing to get custom code. ###
    #
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Test parsing, weights linking, and execution of ")
    #
    parser.add_argument("--code", type=str, help="The Python code that contains the pytorch code.", required=True)
    parser.add_argument("--input_dim", type=str, help="The input dimension used to test the model inference.", required=True)
    parser.add_argument("--main_block", type=str, help="The name of the main block Module", default="")
    parser.add_argument("--weights", type=str, help="The path to weights files.")
    #
    args: argparse.Namespace = parser.parse_args()

    #
    ### Parse Input dim from text to tuple of integers. ###
    #

    #
    input_dim_str: str = args.input_dim.strip()

    #
    ### Remove parentheses if present. ###
    #
    if input_dim_str.startswith('(') and input_dim_str.endswith(')'):
        #
        input_dim_str = input_dim_str[1:-1]

    #
    ### Split by comma or whitespace and convert to integers. ###
    #
    input_dim: tuple[int, ...]
    #
    try:
        #
        if ',' in input_dim_str:
            #
            input_dim = tuple(int(x.strip()) for x in input_dim_str.split(',') if x.strip())
        #
        else:
            #
            input_dim = tuple(int(x.strip()) for x in input_dim_str.split() if x.strip())
    #
    except ValueError as e:
        #
        print(f"Error parsing input dimensions '{args.input_dim}': {e}")
        #
        exit(1)

    #
    tester: Tester = Tester(
        code_path=args.code,
        input_dim=input_dim,
        main_block_name=args.main_block,
        weights_path=args.weights
    )

    #
    print( tester.test() )


