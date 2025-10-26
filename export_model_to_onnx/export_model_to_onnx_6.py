import pickle
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch import Tensor

import onnxruntime as ort

from model_to_export_custom_layer_norm_simpl_no_float import DeepArcNet as Model


def export_executorch_model(model: nn.Module, example_inputs: tuple) -> None:
    """
    Export model to ONNX using TorchScript tracing.
    Optimized for ST Edge AI - no dynamic operations, no If nodes.
    """
    try:
        # Critical: set to eval mode
        model.eval()

        # quantize into int8
        model = model.to(dtype=torch.int8)

        # quantize input
        inp: Tensor = example_inputs[0].to(dtype=torch.int8)

        # Disable gradient computation
        with torch.no_grad():
            # Test the model first
            print("Testing model forward pass...")
            test_output = model(inp)
            print(f"✓ Forward pass successful, output shape: {test_output.shape}")

            # Use torch.jit.script instead of trace to handle the explicit slicing better
            # But first, we need to trace with specific settings
            print("\nTracing model...")

            # More permissive tracing
            traced_model = torch.jit.trace(
                model,
                example_inputs,
                strict=False,  # Allow some flexibility
                check_trace=False  # Skip trace checking for now
            )

            print("✓ Model traced successfully")

            # Freeze the traced model
            traced_model = torch.jit.freeze(traced_model)
            print("✓ Model frozen")

        #
        fp: str=  "onnx_quant.onnx"

        # Export with more conservative settings for STM32
        print("\nExporting to ONNX...")
        torch.onnx.export(
            traced_model,
            example_inputs,
            fp,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            training=torch.onnx.TrainingMode.EVAL,
            dynamic_axes=None,
            verbose=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

        print(f"✅ Model exported successfully to {fp}")

        # Verify the export
        print("\nVerifying ONNX model...")
        import onnx
        onnx_model = onnx.load("exported_model_ir9_no_if.onnx")
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")

        # Print model info
        print(f"\nModel info:")
        print(f"  IR version: {onnx_model.ir_version}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")
        print(f"  Number of nodes: {len(onnx_model.graph.node)}")

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    # Load model
    model: nn.Module = Model()

    # Load weights
    print("Loading model weights...")
    model.load_state_dict(torch.load("model_to_export_weights.pth", weights_only=True))

    # CRITICAL: Assign conv modules EXACTLY as before to maintain weight compatibility
    model.conv0 = model.ConvEncode[0]
    model.conv1 = model.ConvEncode[1]
    model.conv2 = model.ConvEncode[2]
    model.conv3 = model.ConvEncode[3]
    model.conv4 = model.ConvEncode[4]
    model.conv5 = model.ConvEncode[5]
    print("✓ Weights loaded successfully")

    # Load example data
    print("\nLoading example data...")
    with open("model_to_export_example_data.pkl", "rb") as f:
        example_data: list[NDArray[np.float32]] = pickle.load(f)

    # Prepare input
    input_data1: Tensor = Tensor(example_data[0]).unsqueeze(dim=0)
    input_data2: Tensor = Tensor(example_data[1]).unsqueeze(dim=0)

    print(f"Input shape: {input_data1.shape}")

    # Test model
    model.eval()
    with torch.no_grad():
        predictions: Tensor = model(input_data1)
        print(f"Predictions: {predictions}")
        print(f"Predictions shape: {predictions.shape}")

    # Export
    print("\n" + "="*50)
    print("Starting ONNX export...")
    print("="*50)
    export_executorch_model(model, (input_data1,))

    # Load ONNX model
    onnx_model_path = "exported_model_ir9_no_if.onnx"
    print(f"\n🔍 Loading ONNX model from {onnx_model_path}...")
    ort_session = ort.InferenceSession(onnx_model_path)

    # Prepare input (same as used during export)
    # Assuming input_data1 is already a torch.Tensor from your previous script
    input_numpy = input_data1.cpu().numpy()

    # Get the model input name (to avoid mismatches)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    print(f"ONNX Input name: {input_name}")
    print(f"ONNX Output name: {output_name}")

    # Run inference
    print("\n🚀 Running ONNX inference...")
    onnx_outputs = ort_session.run([output_name], {input_name: input_numpy})
    onnx_output = onnx_outputs[0]

    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"ONNX output sample: {onnx_output.flatten()[:10]}")  # Print first 10 values

    # Compare with PyTorch output
    print("\n🔬 Comparing PyTorch vs ONNX outputs...")
    pytorch_output = model(input_data1).detach().cpu().numpy()

    # Compute absolute and relative differences
    abs_diff = np.abs(pytorch_output - onnx_output)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    rel_diff = np.mean(abs_diff / (np.abs(pytorch_output) + 1e-8))

    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Mean relative difference: {rel_diff:.6e}")

    # Optional: sanity check
    if max_diff < 1e-4:
        print("✅ ONNX model matches PyTorch model (differences within tolerance).")
    else:
        print("⚠️ Differences detected! Check custom layers or numerical precision.")
