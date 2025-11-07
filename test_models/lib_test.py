#
### Import Modules. ###
#
from typing import Any, Optional
#
import psutil
import os
import time
import json
import gc
import traceback
#
from copy import deepcopy
#
import numpy as np
from numpy.typing import NDArray
#
import torch
from torch import nn
from torch import Tensor
#
import onnxruntime as ort  # type: ignore
#
import onnx
#
from lib_onnx_convert import ONNX_Converter


#
def measure_onnx_ram(onnx_path: str, input_shape: tuple[int, ...]) -> dict[str, float]:
    """
    Measure ONNX model memory usage.
    Returns dict with model_size_kb, estimated_activation_kb, total_kb
    """

    #
    ### Load ONNX model and get file size. ###
    #
    model_size: int = os.path.getsize(onnx_path)

    #
    ### Parse ONNX model to get activation sizes. ###
    #
    onnx_model: onnx.ModelProto = onnx.load(f=onnx_path)  # type: ignore

    #
    ### Calculate intermediate tensor sizes from model graph. ###
    #
    activation_size: int = 0

    #
    ### Get value info for all intermediate tensors. ###
    #
    for value_info in onnx_model.graph.value_info:
        #
        tensor_type: Any = value_info.type.tensor_type
        #
        if tensor_type.HasField('shape'):
            #
            dims: list[int] = [d.dim_value for d in tensor_type.shape.dim if d.dim_value > 0]
            #
            if dims:
                #
                ### Assume float32 (4 bytes). ###
                #
                activation_size += int( np.prod(dims) * 4 )

    #
    ### Add input/output sizes. ###
    #
    for input_info in onnx_model.graph.input:
        #
        tensor_type: Any = input_info.type.tensor_type
        #
        if tensor_type.HasField('shape'):
            #
            dims: list[int] = [d.dim_value for d in tensor_type.shape.dim if d.dim_value > 0]
            #
            if dims:
                #
                activation_size += int( np.prod(dims) * 4 )

    #
    for output_info in onnx_model.graph.output:
        #
        tensor_type: Any = output_info.type.tensor_type
        #
        if tensor_type.HasField('shape'):
            #
            dims: list[int] = [d.dim_value for d in tensor_type.shape.dim if d.dim_value > 0]
            #
            if dims:
                #
                activation_size += int( np.prod(dims) * 4 )

    #
    return {
        'model_size_kb': model_size / 1024,
        'activation_size_kb': activation_size / 1024,
        'total_kb': (model_size + activation_size) / 1024
    }


#
def measure_ram_with_process_monitoring(
    model: nn.Module,
    input_tensor: Tensor,
    is_onnx: bool = False,
    onnx_session: Optional[ort.InferenceSession] = None
) -> dict[str, float]:

    """
    Measure RAM by monitoring process memory before/during/after inference.
    Most reliable for CPU inference.
    """

    #
    process: psutil.Process = psutil.Process( pid = os.getpid() )

    #
    ### Force garbage collection. ###
    #
    gc.collect()

    #
    ### Baseline memory. ###
    #
    baseline_mem: float = process.memory_info().rss / 1024

    #
    ### Deep copy model to examine it memory cost. ###
    #
    model_copy: nn.Module = deepcopy(model)

    #
    ### Load model into memory (already loaded, but measure). ###
    #
    model_loaded_mem = process.memory_info().rss / 1024
    #
    model_overhead = model_loaded_mem - baseline_mem

    #
    ### Clean model copy. ###
    #
    del(model_copy)

    #
    ### Run inference and measure peak. ###
    #
    gc.collect()
    #
    mem_before = process.memory_info().rss / 1024

    #
    if is_onnx and onnx_session is not None:
        #
        input_name = onnx_session.get_inputs()[0].name  # type: ignore
        output_name = onnx_session.get_outputs()[0].name  # type: ignore
        #
        input_numpy: NDArray[np.float32] = input_tensor.cpu().numpy()  # type: ignore
        #
        _ = onnx_session.run(output_names=[output_name], input_feed={input_name: input_numpy})  # type: ignore
    #
    else:
        #
        with torch.no_grad():
            #
            _ = model(input_tensor)

    #
    mem_after = process.memory_info().rss / 1024
    activation_mem = mem_after - mem_before

    #
    ### Clean up. ###
    #
    gc.collect()

    #
    return {
        'baseline_kb': baseline_mem,
        'model_overhead_kb': model_overhead,
        'activation_kb': activation_mem,
        'peak_kb': mem_after,
        'inference_increase_kb': activation_mem
    }


#
### Model Processing and Tester class. ###
#
class Model_Processing_and_Tester:

    #
    def __init__(self, nb_inference_test: int = 10, input_shape: tuple[int, ...] = (1, 30, 10), verbose: bool = True) -> None:

        #
        ### Verbose. ###
        #
        self.verbose: bool = verbose

        #
        ### Parameter that indicate how many inference we run for one model to ensure statistical validity. ###
        #
        self.nb_inference_test: int = nb_inference_test

        #
        ### Model input shape. ###
        #
        self.input_shape: tuple[int, ...] = input_shape

        #
        ### Onnx converter module. ###
        #
        self.onnx_converter: ONNX_Converter = ONNX_Converter(verbose=verbose)

        #
        ### Total count of parameters for each model. ###
        #
        self.model_nb_parameters: dict[str, int] = {}

        #
        ### Model settings / Kwargs / hyper parameters. ###
        #
        self.model_hyper_params: dict[str, dict[str, Any]] = {}

        #
        ### Layers counts. ###
        #
        self.model_layer_counts: dict[str, dict[str, tuple[int, int]]] = {}

        #
        ### Onnx file path for each processed model. ###
        #
        self.model_onnx_filepath: dict[str, str] = {}

        #
        ### Onnx file size for each processed model. ###
        #
        self.model_onnx_file_size: dict[str, float] = {}

        #
        ### Model family for each model name. ###
        #
        self.models_families: dict[str, int] = {}
        #
        self.models_families_script: dict[str, str] = {}

        #
        ### Pytorch measured peak RAM usage for each run for each model. ###
        #
        self.model_pt_ram_breakdown: dict[str, dict[str, float]] = {}

        #
        ### Onnx measured peak RAM usage for each run for each model. ###
        #
        self.model_onnx_ram_breakdown: dict[str, dict[str, float]] = {}

        #
        ### Pytorch measured inference time for each run for each model (in ms). ###
        #
        self.model_pt_inference_times: dict[str, list[float]] = {}

        #
        ### Onnx measured inference time for each run for each model (in ms). ###
        #
        self.model_onnx_inference_times: dict[str, list[float]] = {}

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
    def log(self, *args: Any) -> None:

        #
        if self.verbose:
            #
            print(*args)


    #
    def measure_pytorch_memory_detailed(
        self,
        model: nn.Module,
        input_tensor: Tensor
    ) -> dict[str, float]:

        """
        Comprehensive PyTorch memory measurement combining multiple approaches.
        """

        #
        gc.collect()

        #
        ### Method 1: Calculate model parameter size. ###
        #
        model_params_size: int = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )

        #
        model_buffers_size: int = sum(
            b.numel() * b.element_size()
            for b in model.buffers()
        )

        #
        model_total_size: int = model_params_size + model_buffers_size

        #
        ### Method 2: Track activations with hooks. ###
        #
        activations: list[Tensor] = []
        hooks: list[Any] = []

        #
        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            #
            if isinstance(output, Tensor):
                #
                activations.append(output)
            #
            elif isinstance(output, (tuple, list)):
                #
                activations.extend([o for o in output if isinstance(o, Tensor)])  # type: ignore

        #
        for module in model.modules():
            #
            if module != model:
                #
                hooks.append(module.register_forward_hook(hook_fn))

        #
        ### Run inference. ###
        #
        with torch.no_grad():
            #
            output: Tensor = model(input_tensor)
            #
            activations.append(output)

        #
        ### Calculate activation memory. ###
        #
        activation_size: int = sum(
            t.numel() * t.element_size()
            for t in activations
            if t is not None  # type: ignore
        )
        #
        input_size: int = input_tensor.numel() * input_tensor.element_size()

        #
        ### Remove hooks. ###
        #
        for hook in hooks:
            #
            hook.remove()

        #
        ### Method 3: Process memory delta (for validation). ###
        #
        process: psutil.Process = psutil.Process( pid = os.getpid() )
        #
        mem_before: float = process.memory_info().rss

        #
        gc.collect()

        #
        with torch.no_grad():
            #
            _ = model(input_tensor)

        #
        mem_after: float = process.memory_info().rss
        #
        process_delta: float = mem_after - mem_before

        #
        ### Clean up. ###
        #
        del(activations)
        #
        gc.collect()

        #
        return {
            'model_params_kb': model_params_size / 1024,
            'model_buffers_kb': model_buffers_size / 1024,
            'model_total_kb': model_total_size / 1024,
            'input_kb': input_size / 1024,
            'activations_kb': activation_size / 1024,
            'inference_total_kb': (model_total_size + input_size + activation_size) / 1024,
            'process_delta_kb': process_delta / 1024,  # For validation
        }

    #
    def measure_onnx_memory_detailed(
        self,
        onnx_path: str,
        onnx_session: ort.InferenceSession,
        input_numpy: NDArray[np.float32]
    ) -> dict[str, float]:

        """
        Comprehensive ONNX memory measurement.
        """

        #
        gc.collect()

        #
        ### Model file size. ###
        #
        model_file_size = os.path.getsize(onnx_path)

        #
        ### Parse model structure. ###
        #
        onnx_model: onnx.ModelProto = onnx.load(f=onnx_path)  # type: ignore

        #
        ### Calculate activation sizes from graph. ###
        #
        activation_size: int = 0

        #
        ### Intermediate tensors. ###
        #
        for value_info in onnx_model.graph.value_info:
            #
            tensor_type: Any = value_info.type.tensor_type
            #
            if tensor_type.HasField('shape'):
                #
                dims: list[int] = [d.dim_value for d in tensor_type.shape.dim if d.dim_value > 0]
                #
                if dims:
                    #
                    activation_size += int( np.prod(dims) * 4 )  # float32

        #
        ### Input tensors. ###
        #
        input_size: int = 0
        #
        for input_info in onnx_model.graph.input:
            #
            tensor_type: Any = input_info.type.tensor_type
            #
            if tensor_type.HasField('shape'):
                #
                dims: list[int] = [d.dim_value for d in tensor_type.shape.dim if d.dim_value > 0]
                #
                if dims:
                    #
                    input_size += int( np.prod(dims) * 4 )

        #
        ### Output tensors. ###
        #
        output_size: int = 0
        #
        for output_info in onnx_model.graph.output:
            #
            tensor_type: Any = output_info.type.tensor_type
            #
            if tensor_type.HasField('shape'):
                #
                dims: list[int] = [d.dim_value for d in tensor_type.shape.dim if d.dim_value > 0]
                #
                if dims:
                    #
                    output_size += int( np.prod(dims) * 4 )

        #
        ### Process memory delta. ###
        #
        process: psutil.Process = psutil.Process( pid = os.getpid() )
        mem_before: float = process.memory_info().rss

        #
        gc.collect()

        #
        input_name: str = onnx_session.get_inputs()[0].name  # type: ignore
        output_name: str = onnx_session.get_outputs()[0].name  # type: ignore

        #
        _ = onnx_session.run([output_name], {input_name: input_numpy})  # type: ignore

        #
        mem_after: float = process.memory_info().rss

        #
        process_delta: float = mem_after - mem_before

        #
        gc.collect()

        #
        return {
            'model_file_kb': model_file_size / 1024,
            'input_kb': input_size / 1024,
            'output_kb': output_size / 1024,
            'activations_kb': activation_size / 1024,
            'inference_total_kb': (model_file_size + input_size + output_size + activation_size) / 1024,
            'process_delta_kb': process_delta / 1024,  # For validation
        }

    #
    def convert_to_onnx(
        self,
        model_name: str,
        pt_model: nn.Module,
        onnx_filepath: str
    ) -> None:

        #
        self.onnx_converter.convert_to_onnx(
            pt_model=pt_model,
            input_shape=self.input_shape,
            onnx_filepath=onnx_filepath
        )

        #
        self.model_onnx_filepath[model_name] = onnx_filepath


    #
    def measure_model(
        self,
        model_name: str,
        pt_model: nn.Module,
    ) -> None:
        #
        onnx_model_path: str = self.model_onnx_filepath[model_name]

        #
        self.log(f"\n🔍 Loading ONNX model from {onnx_model_path}...")

        #
        ort_session: ort.InferenceSession = ort.InferenceSession(onnx_model_path)

        #
        ### Store time measurements for each run. ###
        #
        # pt_times: list[float] = []
        #
        onnx_times: list[float] = []

        #
        ### Store RAM measurements for each run. ###
        #
        # pt_ram_runs: list[dict[str, float]] = []
        #
        onnx_ram_runs: list[dict[str, float]] = []

        #
        ### Store Distance measurements for each run. ###
        #
        # distances: list[float] = []
        # max_distances: list[float] = []
        # mean_distances: list[float] = []

        #
        print("Compiling pytorch model for faster inference and most accurate measurements")
        #
        # pt_model_compiled = torch.compile(pt_model)  # type: ignore

        #
        ### Waking up the pytorch compiled model. ###
        #
        input_tensor: Tensor = torch.randn(self.input_shape)
        # pytorch_output = pt_model_compiled(input_tensor)

        #
        i: int
        #
        for i in range(self.nb_inference_test+1):

            #
            self.log(f"\n--- Inference run {i}/{self.nb_inference_test} ---")

            #
            ### Prepare input. ###
            #
            input_tensor = torch.randn(self.input_shape)
            #
            input_numpy: NDArray[np.float32] = input_tensor.cpu().numpy().astype(np.float32)  # type: ignore

            #
            ### Get ONNX input/output names. ###
            #
            input_name: str = ort_session.get_inputs()[0].name  # type: ignore
            output_name: str = ort_session.get_outputs()[0].name  # type: ignore

            """
            #
            ### === PyTorch Inference === ###
            #
            pt_time_start: float = time.perf_counter()
            #
            with torch.inference_mode():
                #
                pytorch_output = pt_model_compiled(input_tensor)
            #
            pt_time_end: float = time.perf_counter()
            #
            pt_times.append((pt_time_end - pt_time_start) * 1000)  # Convert to ms

            #
            ### Measure PyTorch RAM. ###
            #
            pt_ram: dict[str, Any] = self.measure_pytorch_memory_detailed(pt_model, input_tensor)
            #
            pt_ram_runs.append(pt_ram)
            """

            #
            ### === ONNX Inference === ###
            #
            onnx_time_start: float = time.perf_counter()
            #
            onnx_outputs: list[NDArray[np.float32]] = ort_session.run([output_name], {input_name: input_numpy})  # type: ignore
            #
            onnx_time_end: float = time.perf_counter()
            #
            onnx_times.append((onnx_time_end - onnx_time_start) * 1000)  # Convert to ms

            #
            # onnx_output: NDArray[np.float32] = onnx_outputs[0]

            #
            ### Measure ONNX RAM. ###
            #
            onnx_ram: dict[str, Any] = self.measure_onnx_memory_detailed(
                onnx_model_path,
                ort_session,
                input_numpy
            )
            #
            onnx_ram_runs.append(onnx_ram)

            """
            #
            ### === Compare outputs === ###
            #
            pytorch_output_np: NDArray[np.float32] = pytorch_output.detach().cpu().numpy()

            #
            abs_diff: NDArray[np.float32] = np.abs(pytorch_output_np - onnx_output)
            max_diff: float = float( abs_diff.max() )
            mean_diff: float = float( abs_diff.mean() )

            #
            distances.append(mean_diff)
            max_distances.append(max_diff)
            mean_distances.append(mean_diff)

            #
            self.log(f"  PT time: {pt_times[-1]:.3f}ms")
            self.log(f"  PT RAM: {pt_ram['inference_total_kb']:.2f}KB")
            self.log(f"  Max diff: {max_diff:.6e} | Mean diff: {mean_diff:.6e}")
            """

            self.log(f"  ONNX time: {onnx_times[-1]:.3f}ms")
            self.log(f"  ONNX RAM: {onnx_ram['inference_total_kb']:.2f}KB")

        #
        ### Store results. ###
        #
        # self.model_pt_inference_times[model_name] = pt_times[1:]
        self.model_onnx_inference_times[model_name] = onnx_times[1:]

        #
        ### Average RAM measurements across runs. ###
        #
        # self.model_pt_ram_breakdown[model_name] = {
        #     key: float( np.mean([run[key] for run in pt_ram_runs]) )
        #     for key in pt_ram_runs[0].keys()
        # }
        #
        self.model_onnx_ram_breakdown[model_name] = {
            key: float( np.mean([run[key] for run in onnx_ram_runs]) )
            for key in onnx_ram_runs[0].keys()
        }

        #
        # self.model_distances[model_name] = distances
        # self.model_max_distances[model_name] = max_distances
        # self.model_mean_distances[model_name] = mean_distances

        #
        self.model_nb_parameters[model_name] = sum(p.numel() for p in pt_model.parameters())

        #
        ### Log summary. ###
        #
        self.log(f"\n📊 Summary for {model_name}:")
        self.log(f"  PT Num Params: {self.model_nb_parameters[model_name]} parameters.")
        # self.log(f"  PT inference: {np.mean(pt_times[1:]):.3f}ms ± {np.std(pt_times[1:]):.3f}ms")
        self.log(f"  ONNX inference: {np.mean(onnx_times[1:]):.3f}ms ± {np.std(onnx_times[1:]):.3f}ms")
        # self.log(f"  PT total RAM: {self.model_pt_ram_breakdown[model_name]['inference_total_kb']:.2f}KB")
        self.log(f"  ONNX total RAM: {self.model_onnx_ram_breakdown[model_name]['inference_total_kb']:.2f}KB")


    #
    def load_logs(self, filepath: str) -> None:

        #
        if not os.path.exists(filepath):
            #
            return

        #
        try:

            #
            with open(filepath, "r", encoding="utf-8") as f:
                #
                data: dict[str, Any] = json.load(f)

            #
            self.models_families = data["model_families"]
            self.models_families_script = data["models_families_script"]
            self.model_nb_parameters = data["model_nb_parameters"]
            self.model_onnx_filepath = data["model_onnx_filepath"]
            self.model_pt_inference_times = data["model_pt_inference_times_ms"]
            self.model_onnx_inference_times = data["model_onnx_inference_times_ms"]
            self.model_pt_ram_breakdown = data["model_pt_ram_breakdown_kb"]
            self.model_onnx_ram_breakdown = data["model_onnx_ram_breakdown_kb"]
            self.model_distances = data["model_distances"]
            self.model_max_distances = data["model_max_distances"]
            self.model_mean_distances = data["model_mean_distances"]
            self.model_layer_counts = data["model_layer_counts"]

        #
        except Exception as e:

            #
            print(f"Warning: Error during loading file at path : `{filepath}`")
            print(e)
            print( traceback.extract_stack() )

            #
            ### Clearing partially loaded containers, we want to avoid corrupted data. ###
            #
            self.models_families.clear()
            self.models_families_script.clear()
            self.model_nb_parameters.clear()
            self.model_onnx_filepath.clear()
            self.model_pt_inference_times.clear()
            self.model_onnx_inference_times.clear()
            self.model_pt_ram_breakdown.clear()
            self.model_onnx_ram_breakdown.clear()
            self.model_distances.clear()
            self.model_max_distances.clear()
            self.model_mean_distances.clear()
            self.model_layer_counts.clear()


    #
    def save_logs(self, filepath: str, csv_filepath: str = "data_table.csv") -> None:

        #
        ### JSON SAVE. ###
        #

        #
        data: dict[str, dict[str, Any] | list[Any]] = {
            "model_families": self.models_families,
            "models_families_script": self.models_families_script,
            "model_nb_parameters": self.model_nb_parameters,
            "model_onnx_filepath": self.model_onnx_filepath,
            "model_pt_inference_times_ms": self.model_pt_inference_times,
            "model_onnx_inference_times_ms": self.model_onnx_inference_times,
            "model_pt_ram_breakdown_kb": self.model_pt_ram_breakdown,
            "model_onnx_ram_breakdown_kb": self.model_onnx_ram_breakdown,
            "model_distances": self.model_distances,
            "model_max_distances": self.model_max_distances,
            "model_mean_distances": self.model_mean_distances,
            "model_layer_counts": self.model_layer_counts
        }

        #
        with open(filepath, "w", encoding="utf-8") as f:
            #
            json.dump(data, f, indent=4)

        #
        self.log(f"\n✅ Measurements saved to `{filepath}`")

        #
        models_of_models_families: dict[str, list[str]] = {}

        #
        idx_model: int
        model_name: str
        model_script: str
        #
        for model_name, model_script in self.models_families_script.items():

            #
            if model_script not in models_of_models_families:
                #
                models_of_models_families[model_script] = []

            #
            models_of_models_families[model_script].append( model_name )

        #
        ### EXPORT CSV ###
        #
        # Nom parametre (str) -> idx model (int) -> valeur (str)
        #
        csv_table: dict[str, dict[int, str]] = {}

        #
        cols_base: list[str] = ["model_name", "model_script", "nb_params", "onnx_inference_time", "onnx_flash_size", "onnx_ram"]
        cols_layers: list[str] = []
        cols_params: list[str] = []

        #
        nb_models: int = len(self.models_families_script)

        #
        for idx_model, model_name in enumerate(self.models_families_script.keys()):

            #
            if model_name not in self.model_nb_parameters:
                #
                continue

            #
            ### "model_name" Attribute
            #
            if "model_name" not in csv_table:
                #
                csv_table["model_name"] = {}
            #
            csv_table["model_name"][idx_model] = model_name

            #
            ### "model_script" Attribute
            #
            if "model_script" not in csv_table:
                #
                csv_table["model_script"] = {}
            #
            csv_table["model_script"][idx_model] = self.models_families_script[model_name]

            #
            ### "nb_params" Attribute
            #
            if "nb_params" not in csv_table:
                #
                csv_table["nb_params"] = {}
            #
            csv_table["nb_params"][idx_model] = str(self.model_nb_parameters[model_name])

            #
            ### "onnx_inference_time" Attribute
            #
            if "onnx_inference_time" not in csv_table:
                #
                csv_table["onnx_inference_time"] = {}
            #
            csv_table["onnx_inference_time"][idx_model] = str(sum(self.model_onnx_inference_times[model_name])/len(self.model_onnx_inference_times[model_name]))

            #
            ### "onnx_flash_size" Attribute
            #
            if "onnx_flash_size" not in csv_table:
                #
                csv_table["onnx_flash_size"] = {}
            #
            csv_table["onnx_flash_size"][idx_model] = str(self.model_onnx_ram_breakdown[model_name]["model_file_kb"])

            #
            ### "onnx_ram" Attribute
            #
            if "onnx_ram" not in csv_table:
                #
                csv_table["onnx_ram"] = {}
            #
            csv_table["onnx_ram"][idx_model] = str(self.model_onnx_ram_breakdown[model_name]["inference_total_kb"])

            #
            ### Layers Counts. ###
            #
            layer_type: str
            #
            for layer_type in self.model_layer_counts[model_name]:

                #
                if f"nb {layer_type}" not in csv_table:
                    #
                    csv_table[f"nb {layer_type}"] = {}
                    #
                    cols_layers.append( f"nb {layer_type}" )
                #
                csv_table[f"nb {layer_type}"][idx_model] = str( self.model_layer_counts[model_name][layer_type][0] )

                #
                if self.model_layer_counts[model_name][layer_type][1] > 0:

                    #
                    if f"tot params {layer_type}" not in csv_table:
                        #
                        csv_table[f"tot params {layer_type}"] = {}
                        #
                        cols_layers.append( f"tot params {layer_type}" )
                    #
                    csv_table[f"tot params {layer_type}"][idx_model] = str( self.model_layer_counts[model_name][layer_type][1] )

            #
            ### "parameters" Attribute ###
            #
            hyper_param_key: str
            hyper_param_value: Any
            #
            for hyper_param_key, hyper_param_value in self.model_hyper_params[model_name].items():

                #
                if hyper_param_key not in csv_table:
                    #
                    csv_table[hyper_param_key] = {}
                    #
                    cols_params.append(hyper_param_key)

                #
                csv_table[hyper_param_key][idx_model] = str( hyper_param_value )

        #
        csv_text: str = ""

        #
        csv_sep: str = ";"

        #
        previous_model_script: str = ""

        #
        cols_size: dict[str, int] = {}

        #
        csv_col_names: list[str] = cols_base + cols_layers + cols_params

        #
        ### CSV HEADER. ###
        #
        for i, col_name in enumerate(csv_col_names):

            #
            cols_size[col_name] = max([len(col_name)] + [len(v) for v in csv_table[col_name].values()]) + 1

            #
            if i > 0:
                #
                csv_text += csv_sep

            #
            csv_text += col_name + " " * (cols_size[col_name] - len(col_name))

        #
        csv_text += "\n"

        #
        ### CSV ROWS - SEPARATED BY MODEL FAMILIES. ###
        #
        for idx_model in range(nb_models):

            #
            if idx_model not in csv_table["model_script"]:
                #
                continue

            #
            crt_model_script: str = csv_table["model_script"][idx_model]

            #
            ## NEED TO DISPLAY FAMILY SCRIPT ROW. ##
            #
            if crt_model_script != previous_model_script:

                #
                previous_model_script = crt_model_script

                #
                for i, col_name in enumerate(csv_col_names):
                    #
                    if i > 0:
                        #
                        csv_text += csv_sep
                    #
                    if col_name == "model_script":
                        #
                        csv_text += crt_model_script + " " * (cols_size[col_name] - len(crt_model_script))
                    #
                    else:
                        csv_text += " " * cols_size[col_name]

                #
                csv_text += "\n"

            #
            ## ADD MODEL DATA ROW ##
            #
            for i, col_name in enumerate(csv_col_names):
                #
                if i > 0:
                    #
                    csv_text += csv_sep
                #
                if idx_model in csv_table[col_name]:
                    #
                    v: str = csv_table[col_name][idx_model]
                    #
                    csv_text += v + " " * (cols_size[col_name] - len(v))
                #
                else:
                    #
                    csv_text += " " * cols_size[col_name]
            #
            csv_text += "\n"

        #
        with open(csv_filepath, "w", encoding="utf-8") as f:
            #
            f.write(csv_text)

        #
        self.log(f"\n✅ CSV Table saved at path `{csv_filepath}`")

