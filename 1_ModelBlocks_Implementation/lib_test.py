#
### Import Libraries. ###
#
from typing import Optional, Callable, Any
#
import argparse
#
import torch
from torch import nn, Tensor
#
import numpy as np
from numpy.typing import NDArray
#
from .core.lib_impl import lib_classes as lc
from .core.lib_impl import lib_layers as ll
from .core.lib_impl import lib_interpretor as li
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
    def test_code_extractor(self) -> lc.Language_Model:

        #
        l_model: lc.Language_Model = lema.extract_from_file(filepath="", main_block_name="")

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
        ref_output_pt: Tensor = pt_model.forward(inp)
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
        extr_output = extr_outputs.values()[0]

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
        ### Step 2 - Create a model interpreter. ###
        #
        interpreter: li.LanguageModel_ForwardInterpreter = li.LanguageModel_ForwardInterpreter(l_model)

        #
        ### Step 3 - Verify if model seems to be loaded correctly. ###
        #
        validation_issues: list[str] = li.ModelInterpreterUtils.validate_model_structure(l_model)

        #
        if validation_issues:

            #
            error_msg: str = "Validation Issues:\n\t" + "\n\t - ".join( validation_issues )

            #
            return (False, False, False, error_msg)

        #
        ### Step 4 - Load the pytorch model. ###
        #

        #
        model_class: Callable[..., Any] = lema.get_pytorch_main_model(model_arch=l_model, filepath=self.code_path)

        #
        ### For the moment, we will suppose we will support only model that doesn't need parameters to initialise. ###
        #
        pt_model: nn.Module = model_class()

        #
        if self.weights_path not in ["", "NO_WEIGHTS"]:
            #
            pt_model.load_state_dict(torch.load(self.weights_path, weights_only=True))  # type: ignore

        #
        ### Need all_layers_info for weights linking. ###
        #
        all_layers_info: dict[str, ll.BaseLayerInfo] = ll.load_layers_dict(filepath="core/layers.json")

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
        ### Step 6 - Try to execute model. ###
        #
        test_exec: tuple[bool, str] = self.test_model_execution(l_model=l_model, pt_model=pt_model, interpreter=interpreter)

        #
        if not test_exec[0]:
            #
            return (True, True, False, test_exec[1])

        #
        ### Step 6 - All the tests passed correctly. ###
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
    parser.add_argument("--code", type=str, help="The Python code that contains the pytorch code.")
    parser.add_argument("--main_block", type=str, help="The name of the main block Module", default="")
    parser.add_argument("--weights", type=str, help="The path to weights files.")
    #
    args: argparse.Namespace = parser.parse_args()

    #
    tester: Tester = Tester(
        code_path=args.code,
        main_block_name=args.main_block,
        weights_path=args.weights
    )

    #
    print( tester.test() )


