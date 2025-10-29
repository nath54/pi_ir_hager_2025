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
        pt_times: list[float] = []
        #
        onnx_times: list[float] = []

        #
        ### Store RAM measurements for each run. ###
        #
        pt_ram_runs: list[dict[str, float]] = []
        #
        onnx_ram_runs: list[dict[str, float]] = []

        #
        ### Store Distance measurements for each run. ###
        #
        distances: list[float] = []
        max_distances: list[float] = []
        mean_distances: list[float] = []

        #
        print("Compiling pytorch model for faster inference and most accurate measurements")
        #
        pt_model_compiled = torch.compile(pt_model)  # type: ignore

        #
        ### Waking up the pytorch compiled model. ###
        #
        input_tensor: Tensor = torch.randn(self.input_shape)
        pytorch_output = pt_model_compiled(input_tensor)

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
            onnx_output: NDArray[np.float32] = onnx_outputs[0]

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
            self.log(f"  PT time: {pt_times[-1]:.3f}ms | ONNX time: {onnx_times[-1]:.3f}ms")
            self.log(f"  PT RAM: {pt_ram['inference_total_kb']:.2f}KB | ONNX RAM: {onnx_ram['inference_total_kb']:.2f}KB")
            self.log(f"  Max diff: {max_diff:.6e} | Mean diff: {mean_diff:.6e}")

        #
        ### Store results. ###
        #
        self.model_pt_inference_times[model_name] = pt_times[1:]
        self.model_onnx_inference_times[model_name] = onnx_times[1:]

        #
        ### Average RAM measurements across runs. ###
        #
        self.model_pt_ram_breakdown[model_name] = {
            key: float( np.mean([run[key] for run in pt_ram_runs]) )
            for key in pt_ram_runs[0].keys()
        }
        #
        self.model_onnx_ram_breakdown[model_name] = {
            key: float( np.mean([run[key] for run in onnx_ram_runs]) )
            for key in onnx_ram_runs[0].keys()
        }

        #
        self.model_distances[model_name] = distances
        self.model_max_distances[model_name] = max_distances
        self.model_mean_distances[model_name] = mean_distances

        #
        self.model_nb_parameters[model_name] = sum(p.numel() for p in pt_model.parameters())

        #
        ### Log summary. ###
        #
        self.log(f"\n📊 Summary for {model_name}:")
        self.log(f"  PT Num Params: {self.model_nb_parameters[model_name]} parameters.")
        self.log(f"  PT inference: {np.mean(pt_times[1:]):.3f}ms ± {np.std(pt_times[1:]):.3f}ms")
        self.log(f"  ONNX inference: {np.mean(onnx_times[1:]):.3f}ms ± {np.std(onnx_times[1:]):.3f}ms")
        self.log(f"  PT total RAM: {self.model_pt_ram_breakdown[model_name]['inference_total_kb']:.2f}KB")
        self.log(f"  ONNX total RAM: {self.model_onnx_ram_breakdown[model_name]['inference_total_kb']:.2f}KB")


    #
    def save_logs(self) -> None:

        #
        data: dict[str, dict[str, Any] | list[Any]] = {
            "model_families": self.models_families,
            "model_nb_parameters": self.model_nb_parameters,
            "model_onnx_filepath": self.model_onnx_filepath,
            "model_pt_inference_times_ms": self.model_pt_inference_times,
            "model_onnx_inference_times_ms": self.model_onnx_inference_times,
            "model_pt_ram_breakdown_kb": self.model_pt_ram_breakdown,
            "model_onnx_ram_breakdown_kb": self.model_onnx_ram_breakdown,
            "model_distances": self.model_distances,
            "model_max_distances": self.model_max_distances,
            "model_mean_distances": self.model_mean_distances,
        }

        #
        with open("saved_models_measurements.json", "w", encoding="utf-8") as f:
            #
            json.dump(data, f, indent=4)

        #
        self.log("\n✅ Measurements saved to saved_models_measurements.json")
