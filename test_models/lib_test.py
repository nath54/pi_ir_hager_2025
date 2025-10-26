#
### Import Modules. ###
#
import os
import time
import json
#
import numpy as np
from numpy.typing import NDArray
#
import torch
from torch import nn
from torch import Tensor
#
import onnxruntime as ort
#
from lib_onnx_convert import ONNX_Converter


#
### Model Processing and Tester class. ###
#
class Model_Processing_and_Tester:

    #
    def __init__(self, nb_inference_test: int = 10, input_shape: tuple[int, ...] = (1, 30, 10)) -> None:

        #
        ### Parameter that indicate how many inference we run for one model to ensure statistical validity. ###
        #
        self.nb_inference_test: int = nb_inference_test

        #
        ### Model input shape. ###
        #
        self.input_shape: tuple[int, ...] = input_shape

        #
        ### ONNX converter. ###
        #
        self.onnx_converter: ONNX_Converter = ONNX_Converter()

        #
        ### Total count of parameters for each model. ###
        #
        self.model_nb_parameters: dict[str, int] = {}

        #
        ### Onnx file path for each processed model. ###
        #
        self.model_onnx_filepath: dict[str, str] = {}

        #
        ### Onnx file size for each processed model. ###
        #
        self.model_onnx_file_size: dict[str, float] = {}

        #
        ### Pytorch measured inference time for each run for each model (in ms). ###
        #
        self.model_pt_inference_times: dict[str, list[float]] = []

        #
        ### Pytorch measured peak RAM usage for each run for each model. ###
        #
        self.model_pt_ram_usage: dict[str, list[float]] = {}

        #
        ### Onnx measured inference time for each run for each model (in ms). ###
        #
        self.model_onnx_inference_times: dict[str, list[float]] = {}

        #
        ### Onnx measured peak RAM usage for each run for each model. ###
        #
        self.model_onnx_ram_usage: dict[str, list[float]] = {}

        #
        ### Distances between PyTorch and ONNX outputs for each run for each model. ###
        #
        self.model_distances: dict[str, list[float]] = {}
        #
        ### Max distances between PyTorch and ONNX outputs for each run for each model. ###
        #
        self.model_max_distances: dict[str, list[float]] = {}
        #
        ### Mean distances between PyTorch and ONNX outputs for each run for each model. ###
        #
        self.model_mean_distances: dict[str, list[float]] = {}

    #
    def measure_model(
        self,
        model_name: str,
        pt_model: nn.Module,
    ) -> None:

        #
        ### Get ONNX file path. ###
        #
        onnx_model_path: str = self.model_onnx_filepath[model_name]

        #
        ### Load ONNX model. ###
        #
        self.log(f"\n🔍 Loading ONNX model from {onnx_model_path}...")
        #
        ort_session = ort.InferenceSession(onnx_model_path)

        #
        pt_times: list[float] = []
        #
        onnx_times: list[float] = []

        #
        pt_ram: list[float] = []  # TODO: get only the ram of the pytorch inference (including model weights and structure), to simulate approximatively what would happen on the STM32, to get an idea
        #
        onnx_ram: list[float] = []  # TODO: get only the ram of the onnx inference (including model weights and structure), to simulate approximatively what would happen on the STM32, to get an idea

        #
        distances: list[float] = []
        #
        max_distances: list[float] = []
        #
        mean_distances: list[float] = []

        #
        for _ in range(self.nb_inference_test):

            #
            ### Prepare input (same as used during export) ###
            ### Assuming input_data1 is already a torch.Tensor from your previous script ###
            #
            input_tensor: Tensor = torch.random(self.input_shape)
            #
            input_numpy: NDArray[np.float32] = input_tensor.cpu().numpy()

            #
            ### Get the model input name (to avoid mismatches). ###
            #
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            #
            self.log(f"ONNX Input name: {input_name}")
            self.log(f"ONNX Output name: {output_name}")

            #
            ### Run inference. ###
            #
            self.log("\n🚀 Running ONNX inference...")
            #
            onnx_time: float = time.time()
            #
            onnx_outputs = ort_session.run([output_name], {input_name: input_numpy})
            #
            onnx_output = onnx_outputs[0]
            #
            onnx_times.append( time.time() - onnx_time )

            #
            self.log(f"ONNX output shape: {onnx_output.shape}")
            self.log(f"ONNX output sample: {onnx_output.flatten()[:10]}")  # self.log first 10 values

            #
            ### Compare with PyTorch output. ###
            #
            self.log("\n🔬 Comparing PyTorch vs ONNX outputs...")
            #
            pt_time: float = time.time()
            #
            pytorch_output = pt_model(input_tensor).detach().cpu().numpy()
            #
            pt_times.append( time.time() - pt_time )

            #
            ### Compute absolute and relative differences. ###
            #
            abs_diff = np.abs(pytorch_output - onnx_output)
            max_diff = abs_diff.max()
            mean_diff = abs_diff.mean()
            rel_diff = np.mean(abs_diff / (np.abs(pytorch_output) + 1e-8))

            #
            self.log(f"Max absolute difference: {max_diff:.6e}")
            self.log(f"Mean absolute difference: {mean_diff:.6e}")
            self.log(f"Mean relative difference: {rel_diff:.6e}")

            #
            ### Optional: sanity check. ###
            #
            if max_diff < 1e-4:
                #
                self.log("✅ ONNX model matches PyTorch model (differences within tolerance).")
            #
            else:
                #
                self.log("⚠️ Differences detected! Check custom layers or numerical precision.")

            #
            distances.append(mean_diff)
            max_distances.append(max_diff)
            mean_distances.append(mean_diff)

        #
        self.model_pt_inference_times[model_name] = pt_times
        self.model_onnx_inference_times[model_name] = onnx_times
        #
        self.model_distances[model_name] = distances
        self.model_max_distances[model_name] = max_distances
        self.model_mean_distances[model_name] = mean_distances


    #
    def test_model(self, model_name: str, pt_model: nn.Module) -> None:

        #
        ### Step 1: convert the model in onnx. ###
        #
        if not self.onnx_converter.convert_to_onnx(
            pt_model=pt_model,
            input_shape=self.input_shape,
            onnx_filepath=f"models_onnx/{model_name}.onnx"
        ):
            #
            print(f"Error with model: {model_name}, cannot convert into ONNX !")
            #
            return

        #
        ### Step 2: Measure the model inference. ###
        #
        self.measure_model(model_name = model_name, pt_model=pt_model)


    #
    def save_logs(self) -> None:

        #
        data: dict[str, dict[str, dict | str]] = {
            "model_nb_parameters": self.model_nb_parameters,
            "model_onnx_filepath": self.model_onnx_filepath,
            "model_onnx_file_size": self.model_onnx_file_size,
            "model_pt_inference_times": self.model_pt_inference_times,
            "model_pt_ram_usage": self.model_pt_ram_usage,
            "model_onnx_inference_times": self.model_onnx_inference_times,
            "model_onnx_ram_usage": self.model_onnx_ram_usage,
        }

        #
        with open("saved_models_measurements.json", "w", enoding="utf-8") as f:
            #
            json.dump(data, f, indent=4)
