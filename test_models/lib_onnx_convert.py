#
### Import Modules. ###
#
from typing import Any
#
import torch
from torch import nn
from torch import Tensor
#
import onnx
#
import traceback


#
class ONNX_Converter:

    #
    def __init__(self, verbose: bool = True) -> None:

        #
        self.verbose: bool = verbose

    #
    def log(self, *args: Any) -> None:

        #
        if self.verbose:
            #
            print(*args)

    #
    def convert_to_onnx(self, pt_model: nn.Module, input_shape: tuple[int, ...], onnx_filepath: str) -> bool:

        #
        try:

            #
            ### Critical: set to eval mode. ###
            #
            pt_model.eval()

            #
            ### Disable gradient computation. ###
            #
            with torch.no_grad():

                #
                ### Generate random input matrix. ###
                #
                input_tensor: Tensor = torch.randn(input_shape)

                #
                ### Use torch.jit.script instead of trace to handle the explicit slicing better. ###
                ### But first, we need to trace with specific settings. ###
                #
                self.log("\nTracing model...")

                #
                ### More permissive tracing. ###
                #
                traced_model = torch.jit.trace(  # type: ignore
                    pt_model,
                    input_tensor,
                    strict=False,  # Allow some flexibility
                    check_trace=False  # Skip trace checking for now
                )

                #
                self.log("✓ Model traced successfully")

                #
                ### Freeze the traced model. ###
                #
                traced_model = torch.jit.freeze(traced_model)  # type: ignore
                #
                self.log("✓ Model frozen")

            #
            ### Export with more conservative settings for STM32. ###
            #
            self.log("\nExporting to ONNX...")
            #
            torch.onnx.export(  # type: ignore
                traced_model,
                input_tensor,  # type: ignore
                onnx_filepath,
                export_params=True,
                # opset_version=19,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                training=torch.onnx.TrainingMode.EVAL,
                dynamic_axes=None,
                verbose=False,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX
            )

            #
            self.log(f"✅ Model exported successfully to {onnx_filepath}")

            #
            ### Verify the export. ###
            #
            self.log("\nVerifying ONNX model...")
            #
            onnx_model = onnx.load(onnx_filepath)  # type: ignore
            onnx.checker.check_model(onnx_model)  # type: ignore
            self.log("✓ ONNX model is valid")

            #
            ### log model info. ###
            #
            self.log(f"\nModel info:")
            self.log(f"  IR version: {onnx_model.ir_version}")
            self.log(f"  Opset version: {onnx_model.opset_import[0].version}")
            self.log(f"  Number of nodes: {len(onnx_model.graph.node)}")

        #
        except Exception as e:
            #
            self.log(f"❌ Export failed: {e}")
            #
            traceback.print_exception(e)  # type: ignore
            #
            return False

        #
        return True
