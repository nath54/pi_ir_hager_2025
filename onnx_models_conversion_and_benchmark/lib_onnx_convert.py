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
import os

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
        ### Create the directory if it doesn't exist. ###
        #
        os.makedirs(os.path.dirname(os.path.abspath(onnx_filepath)), exist_ok=True)

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
                ### Export directly (torch.onnx.export will handle tracing internally). ###
                #
                self.log("\nExporting to ONNX...")
                #
                torch.onnx.export(  # type: ignore
                    pt_model,
                    input_tensor,  # type: ignore
                    onnx_filepath,
                    export_params=True,
                    opset_version=15,
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
            #
            ### Force the model to be self-contained (no external data). ###
            ### This overwrites the file with a version that includes all weights. ###
            #
            onnx.save(onnx_model, onnx_filepath)
            #
            self.log("✓ ONNX model is valid and self-contained")

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
