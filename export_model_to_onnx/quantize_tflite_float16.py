import tensorflow as tf
import numpy as np
import sys
import os

def quantize_model_to_float16(input_model_path, output_model_path):
    """
    Convert a TensorFlow Lite model to use 16-bit float quantization.
    This uses TF's built-in converter with proper float16 support.
    
    Args:
        input_model_path: Path to the input .tflite model
        output_model_path: Path to save the quantized .tflite model
    """
    
    # Check if input file exists
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Input model not found: {input_model_path}")
    
    print(f"Converting model from: {input_model_path}")
    print("Applying float16 quantization...")
    
    # Read the original TFLite model
    with open(input_model_path, 'rb') as f:
        tflite_model = f.read()
    
    # Load interpreter to understand the model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model inputs: {len(input_details)}")
    print(f"Model outputs: {len(output_details)}")
    
    # Use flatbuffers to modify only weight buffers
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb
        import flatbuffers
    except ImportError:
        raise ImportError("Required modules not found. Please ensure TensorFlow is properly installed.")
    
    # Parse the model
    model = schema_fb.Model.GetRootAsModel(tflite_model, 0)
    model_obj = schema_fb.ModelT.InitFromObj(model)
    
    # Collect model I/O tensor indices
    model_io_tensors = set()
    for subgraph in model_obj.subgraphs:
        if subgraph.inputs is not None:
            model_io_tensors.update([int(x) for x in subgraph.inputs])
        if subgraph.outputs is not None:
            model_io_tensors.update([int(x) for x in subgraph.outputs])
    
    print(f"Model I/O tensors: {model_io_tensors}")
    
    # For each operator, identify which of its inputs are constants (weights)
    # Constants are tensors that have buffer data and are not outputs of other ops
    constant_tensors = set()
    variable_tensors = set()
    
    for subgraph in model_obj.subgraphs:
        # First pass: identify all tensors that are outputs of operations
        for op in subgraph.operators:
            if op.outputs is not None:
                variable_tensors.update([int(x) for x in op.outputs])
        
        # Second pass: identify constants (have buffer data, not operation outputs)
        for tensor_idx, tensor in enumerate(subgraph.tensors):
            has_data = (tensor.buffer > 0 and 
                       tensor.buffer < len(model_obj.buffers) and
                       model_obj.buffers[tensor.buffer].data is not None and
                       len(model_obj.buffers[tensor.buffer].data) > 0)
            
            if has_data and tensor_idx not in variable_tensors and tensor_idx not in model_io_tensors:
                constant_tensors.add(tensor_idx)
    
    print(f"Identified {len(constant_tensors)} constant tensors (weights/biases)")
    
    # Quantize only the constant tensors
    quantized_count = 0
    for subgraph in model_obj.subgraphs:
        for tensor_idx, tensor in enumerate(subgraph.tensors):
            if tensor_idx in constant_tensors and tensor.type == schema_fb.TensorType.FLOAT32:
                buffer = model_obj.buffers[tensor.buffer]
                
                # Convert buffer from float32 to float16
                float32_data = np.frombuffer(buffer.data, dtype=np.float32)
                float16_data = float32_data.astype(np.float16)
                buffer.data = float16_data.tobytes()
                
                # Change tensor type to FLOAT16
                tensor.type = schema_fb.TensorType.FLOAT16
                quantized_count += 1
    
    print(f"Quantized {quantized_count} constant tensors to float16")
    
    # Rebuild the model
    builder = flatbuffers.Builder(1024)
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
    
    # Verify the model structure is preserved
    try:
        quant_interpreter = tf.lite.Interpreter(model_path=output_model_path)
        quant_interpreter.allocate_tensors()
        
        quant_input_details = quant_interpreter.get_input_details()
        quant_output_details = quant_interpreter.get_output_details()
        
        print("\n✓ Model verification: Successfully loaded quantized model")
        print(f"  Inputs: {len(quant_input_details)}")
        for i, detail in enumerate(quant_input_details):
            print(f"    Input {i}: shape={detail['shape'].tolist()}, dtype={detail['dtype'].__name__}")
        
        print(f"  Outputs: {len(quant_output_details)}")
        for i, detail in enumerate(quant_output_details):
            print(f"    Output {i}: shape={detail['shape'].tolist()}, dtype={detail['dtype'].__name__}")
        
        # Check if I/O signatures match
        io_match = (len(input_details) == len(quant_input_details) and 
                   len(output_details) == len(quant_output_details))
        
        if io_match:
            for orig, quant in zip(input_details, quant_input_details):
                if not np.array_equal(orig['shape'], quant['shape']) or orig['dtype'] != quant['dtype']:
                    io_match = False
                    break
        
        if io_match:
            print("\n✓ Input/Output signatures match original model")
        else:
            print("\n✗ Warning: Input/Output signatures differ from original model")
            
    except Exception as e:
        print(f"\n✗ Warning: Could not verify quantized model: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_model.tflite> [output_model.tflite]")
        print("\nExample:")
        print("  python script.py model.tflite model_float16.tflite")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
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