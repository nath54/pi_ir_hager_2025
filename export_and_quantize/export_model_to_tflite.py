import os
import pickle
import numpy as np
import torch
from torch import nn, Tensor
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import tempfile


def export_tflite_model(model: nn.Module, example_inputs: tuple, quantize: bool = False) -> None:
    """Export a PyTorch model to TFLite using ONNX → onnx-tf → TensorFlow → TFLite."""

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

    # --- Step 2: ONNX → TensorFlow ---
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)  # onnx-tf converts to TF graph
    tf_model_path = tempfile.mkdtemp()
    tf.saved_model.save(tf_rep.tf_module, tf_model_path)
    print(f"✅ Converted to TensorFlow SavedModel → {tf_model_path}")

    # --- Step 3: TensorFlow → TFLite ---
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # post-training quantization

    tflite_model = converter.convert()
    tflite_path = "exported_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Exported TensorFlow Lite model → {tflite_path}")

    # --- Step 4: Optional numerical verification ---
    try:
        # PyTorch output
        with torch.no_grad():
            torch_out = model(*example_inputs).cpu().numpy()

        # TFLite output
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], example_inputs[0].cpu().numpy())
        interpreter.invoke()
        tflite_out = interpreter.get_tensor(output_details[0]['index'])

        # Compare outputs
        if np.allclose(torch_out, tflite_out, rtol=1e-5, atol=1e-5):
            print("✅ PyTorch and TFLite outputs match within tolerance.")
        else:
            max_diff = np.max(np.abs(torch_out - tflite_out))
            print(f"⚠️ Max difference between PyTorch and TFLite outputs = {max_diff}")

    except Exception as e:
        print(f"⚠️ Verification failed: {e}")


if __name__ == "__main__":
    # --- Load PyTorch model ---
    model: nn.Module = torch.load("quantized_model.pth", map_location="cpu")
    model.eval()

    # --- Load example input ---
    with open("model_to_export_example_data.pkl", "rb") as f:
        example_data = pickle.load(f)

    input_data1 = torch.tensor(example_data[0], dtype=torch.float32).unsqueeze(0)

    # --- Test PyTorch inference ---
    with torch.no_grad():
        predictions = model(input_data1)
        print(f"✅ PyTorch inference OK — predictions shape: {predictions.shape}")

    # --- Export TFLite ---
    export_tflite_model(model, (input_data1,), quantize=True)
