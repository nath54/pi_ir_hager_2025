import os
import pickle
import numpy as np
import torch
from torch import nn, Tensor
import onnx
import tensorflow as tf
import tf2onnx


def export_tflite_model(model: nn.Module, example_inputs: tuple) -> None:
    """Export a PyTorch model to TFLite using ONNX → TensorFlow → TFLite (modern path)."""
    model.eval()
    onnx_path = "model.onnx"

    # --- Step 1: PyTorch → ONNX ---
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
    print(f"✅ Exported ONNX model → {onnx_path}")

    # --- Step 2: ONNX → TensorFlow Graph ---
    tf_model_path = "saved_model"
    model_proto = onnx.load(onnx_path)
    tf_rep, _ = tf2onnx.convert.from_onnx(model_proto, output_path=None)
    tf.saved_model.save(tf_rep, tf_model_path)
    print(f"✅ Converted to TensorFlow SavedModel → {tf_model_path}")

    # --- Step 3: TensorFlow → TFLite ---
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional quantization
    tflite_model = converter.convert()
    with open("exported_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Exported TensorFlow Lite model → exported_model.tflite")


if __name__ == "__main__":
    model: nn.Module = torch.load("quantized_model.pth", map_location="cpu")
    model.eval()

    with open("model_to_export_example_data.pkl", "rb") as f:
        example_data = pickle.load(f)

    input_data1 = torch.tensor(example_data[0], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        predictions = model(input_data1)
        print(f"✅ Inference OK — predictions shape: {predictions.shape}")

    export_tflite_model(model, (input_data1,))
