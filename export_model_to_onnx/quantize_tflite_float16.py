import tensorflow as tf
import numpy as np
import sys
import os

def quantize_model_to_float16(input_model_path, output_model_path):
    """
    Convert a TensorFlow Lite model to use 16-bit float quantization.
    
    Args:
        input_model_path: Path to the input .tflite model
        output_model_path: Path to save the quantized .tflite model
    """
    
    # Check if input file exists
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Input model not found: {input_model_path}")
    
    # Read the original TFLite model
    with open(input_model_path, 'rb') as f:
        tflite_model = f.read()
    
    print(f"Converting model from: {input_model_path}")
    print("Applying float16 quantization...")
    
    # Load interpreter to get model structure
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Import flatbuffers for model modification
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb
        import flatbuffers
    except ImportError:
        raise ImportError("Required modules not found. Please ensure TensorFlow is properly installed.")
    
    # Parse the model
    model = schema_fb.Model.GetRootAsModel(tflite_model, 0)
    
    # Create a new model with float16 quantization
    builder = flatbuffers.Builder(1024)
    
    # This is a complex approach. Let's use the simpler method:
    # Convert model buffer tensors from float32 to float16
    
    model_obj = schema_fb.ModelT.InitFromObj(model)
    
    # Iterate through all subgraphs
    for subgraph in model_obj.subgraphs:
        # Iterate through all tensors
        for tensor in subgraph.tensors:
            # Convert float32 tensors to float16
            if tensor.type == schema_fb.TensorType.FLOAT32:
                tensor.type = schema_fb.TensorType.FLOAT16
                
                # Convert buffer data if it exists
                if tensor.buffer < len(model_obj.buffers):
                    buffer = model_obj.buffers[tensor.buffer]
                    if buffer.data is not None and len(buffer.data) > 0:
                        # Convert float32 data to float16
                        float32_data = np.frombuffer(buffer.data, dtype=np.float32)
                        float16_data = float32_data.astype(np.float16)
                        buffer.data = float16_data.tobytes()
    
    # Serialize the modified model
    quantized_model = model_obj.Pack(builder)
    builder.Finish(quantized_model)
    quantized_tflite = bytes(builder.Output())
    
    # Save the quantized model
    with open(output_model_path, 'wb') as f:
        f.write(quantized_tflite)
    
    # Print size comparison
    original_size = os.path.getsize(input_model_path)
    quantized_size = os.path.getsize(output_model_path)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"\nConversion complete!")
    print(f"Original model size: {original_size / 1024:.2f} KB")
    print(f"Quantized model size: {quantized_size / 1024:.2f} KB")
    print(f"Size reduction: {reduction:.2f}%")
    print(f"Output saved to: {output_model_path}")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_model.tflite> [output_model.tflite]")
        print("\nExample:")
        print("  python script.py model.tflite model_float16.tflite")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # If output path not provided, create default name
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_float16.tflite"
    
    try:
        quantize_model_to_float16(input_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)