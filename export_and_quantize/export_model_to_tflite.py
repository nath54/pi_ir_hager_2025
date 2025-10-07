import os
import pickle
import numpy as np
import torch
from torch import nn, Tensor
import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf


def export_tflite_model(model: nn.Module, example_inputs: tuple) -> None:
    """Export a PyTorch model to TFLite using ONNX → Keras → TFLite conversion."""

    model.eval()
    onnx_path = "model.onnx"

    # Step 1: Export PyTorch → ONNX
    torch.onnx.export(
        model,
        example_inputs,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"✅ Exported ONNX model to {onnx_path}")

    # Step 2: ONNX → Keras
    onnx_model = onnx.load(onnx_path)
    k_model = onnx_to_keras(onnx_model, ["input"], name_policy='short')
    print("✅ Converted ONNX → Keras model")

    # Step 3: Save Keras model and convert to TFLite
    keras_path = "keras_model"
    os.makedirs(keras_path, exist_ok=True)
    k_model.save(keras_path)
    print(f"✅ Saved Keras model to {keras_path}")

    converter = tf.lite.TFLiteConverter.from_saved_model(keras_path)
    tflite_model = converter.convert()
    with open("exported_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Exported TensorFlow Lite model: exported_model.tflite")


if __name__ == "__main__":
    # --- Load model ---
    model: nn.Module = torch.load("quantized_model.pth", map_location="cpu")
    model.eval()

    # --- Load example input data ---
    with open("model_to_export_example_data.pkl", "rb") as f:
        example_data = pickle.load(f)

    input_data1 = torch.tensor(example_data[0], dtype=torch.float32).unsqueeze(0)

    # --- Run test inference ---
    with torch.no_grad():
        predictions = model(input_data1)
        print(f"✅ Inference OK — predictions shape: {predictions.shape}")

    # --- Export ---
    export_tflite_model(model, (input_data1,))
