import pickle
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch import Tensor

from model_to_export_custom_layer_norm_simpl import DeepArcNet as Model


def export_executorch_model(model: nn.Module, example_inputs: tuple) -> None:
    """
    Export model to ONNX using TorchScript tracing.
    Optimized for ST Edge AI - no dynamic operations, no If nodes.
    """

    # Try multiple opset versions in order of preference
    opset_versions = [9, 11, 13]

    for opset in opset_versions:
        try:
            print(f"\n{'='*50}")
            print(f"Attempting export with opset version {opset}...")
            print('='*50)

            # Critical: set to eval mode
            model.eval()

            # Disable gradient computation
            with torch.no_grad():
                # Test the model first
                print("Testing model forward pass...")
                test_output = model(example_inputs[0])
                print(f"✓ Forward pass successful, output shape: {test_output.shape}")

                print("\nTracing model...")

                # More permissive tracing
                traced_model = torch.jit.trace(
                    model,
                    example_inputs,
                    strict=False,
                    check_trace=False
                )

                print("✓ Model traced successfully")

                # Freeze the traced model
                traced_model = torch.jit.freeze(traced_model)
                print("✓ Model frozen")

            # Export with specific opset
            output_filename = f"exported_model_opset{opset}.onnx"
            print(f"\nExporting to {output_filename}...")

            torch.onnx.export(
                traced_model,
                example_inputs,
                output_filename,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                training=torch.onnx.TrainingMode.EVAL,
                dynamic_axes=None,
                verbose=False,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX
            )

            print(f"✅ Model exported successfully to {output_filename}")

            # Verify the export
            print("\nVerifying ONNX model...")
            import onnx
            onnx_model = onnx.load(output_filename)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model is valid")

            # Print model info
            print(f"\nModel info:")
            print(f"  IR version: {onnx_model.ir_version}")
            print(f"  Opset version: {onnx_model.opset_import[0].version}")
            print(f"  Number of nodes: {len(onnx_model.graph.node)}")

            # Print node types
            node_types = {}
            for node in onnx_model.graph.node:
                node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
            print(f"\nNode types in model:")
            for op_type, count in sorted(node_types.items()):
                print(f"  {op_type}: {count}")

            print(f"\n✅ Successfully exported with opset {opset}")
            print(f"   Try this file with ST Edge AI: {output_filename}")

        except Exception as e:
            print(f"❌ Export with opset {opset} failed: {e}")
            if opset == opset_versions[-1]:
                import traceback
                traceback.print_exc()
            continue


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