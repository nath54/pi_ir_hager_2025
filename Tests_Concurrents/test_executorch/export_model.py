#
import pickle

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch import Tensor
import executorch.exir as exir
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner # Example backend
# from executorch.sdk.bundled_program import BundledProgramBuilder, FileSpec
# Assume 'MyModel' is your nn.Module class and 'model' is an instance
# Assume 'example_inputs' is a tuple of tensors representative of your model's input


#
from model_to_export import Model


#
def export_executorch_model(model: nn.Module, example_inputs: tuple) -> None:

    # --- Standard ExecuTorch Export ---
    try:
        # Capture the graph
        prog = exir.capture(model, example_inputs, exir.CaptureConfig()).to_edge()

        # Optional: Lower graph for a specific backend (e.g., XNNPACK for CPU)
        # Check ExecuTorch docs for available backends and partitioners
        # prog = prog.to_backend(XnnpackPartitioner) # Uncomment and adapt if using a specific backend

        # Convert to ExecuTorch format
        executorch_program = prog.to_executorch(
            exir.ExecutorchBackendConfig(
                #passes=[...] # Add optimization passes if needed
                #extract_constants=True, # Bundle constants/weights into the file
                #extract_delegates=True # Bundle backend delegates if used
            )
        )

        # Save the .pte file
        output_pte_path = "model_pre_compiled.pte"
        with open(output_pte_path, "wb") as f:
            f.write(executorch_program.buffer)
        print(f"Model exported successfully to {output_pte_path}")

        # --- Bundled Program (Alternative/Advanced - includes method metadata) ---
        # Bundled programs are often preferred for easier C++ loading
        # bundled_program = BundledProgramBuilder()
        # with bundled_program.add_method("forward", executorch_program.emitter_output.program) as method_builder:
        #     # Specify input/output placeholders (names must match usage in C++)
        #     method_builder.add_input_placeholder("input_0")
        #     method_builder.add_output_placeholder("output_0")
        #     # You can also add files like tokenizer configs if needed
        #     # method_builder.add_spec(FileSpec("tokenizer.json", b"{}")) # Example

        # output_bp_path = "my_model_bundled.pte"
        # bundled_program.save(output_bp_path)
        # print(f"Bundled program saved successfully to {output_bp_path}")


    except Exception as e:
        print(f"Export failed: {e}")
        # Consider logging details or using specific exception handling



#
if __name__ == "__main__":

    #
    model: nn.Module = Model()

    #
    model.load_state_dict(torch.load("model_to_export_weights.pth", weights_only=True))

    #
    with open("model_to_export_example_data.pkl", "rb") as f:
        example_data: list[NDArray[np.float32]] = pickle.load( f )

    #
    input_data1: Tensor = Tensor( example_data[0] ).unsqueeze(dim=0)
    input_data2: Tensor = Tensor( example_data[1] ).unsqueeze(dim=0)

    #
    # print( f"\ninput_data : {input_data}\n" )
    print( f"\ninput_data1 shape : {input_data1.shape}\n" )

    #
    print( f"\ninput_data2 : {input_data2}\n" )
    print( f"\ninput_data2 shape : {input_data2.shape}\n" )

    #
    model.eval()

    #
    predictions: Tensor = model( input_data1 )

    #
    print( f"\npredictions : {predictions}\n" )
    print( f"\npredictions shape : {predictions.shape}\n" )

    #
    export_executorch_model( model, (input_data1,) )
