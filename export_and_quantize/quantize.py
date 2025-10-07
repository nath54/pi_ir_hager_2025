import torch
import torch.nn as nn
import os
from afdd_model_eval_sdia import model, BATCH_SIZE

def quantize_model_to_int8(model, sample_input):
    """
    Quantize a PyTorch model to INT8.
    This preserves the exact input and output shapes.
    
    Args:
        model: Your PyTorch model (nn.Module)
        sample_input: A sample input tensor with the correct shape for your model
                     Example: torch.randn(1, 100, 32, 32) for 4D input
    
    Returns:
        quantized_model: INT8 quantized version of your model
    """
    print("="*50)
    print("Quantizing Model to INT8")
    print("="*50)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Apply dynamic quantization (works on all platforms including Windows)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU, nn.Conv1d, nn.Conv2d, nn.Conv3d},
        dtype=torch.qint8
    )
    
    print(f"✓ Quantization complete!")
    return quantized_model


def verify_model(original_model, quantized_model, sample_input):
    """
    Verify that quantization preserved input/output shapes and check accuracy.
    
    Args:
        original_model: Original FP32 model
        quantized_model: Quantized INT8 model
        sample_input: Sample input tensor
    """
    print("\n" + "="*50)
    print("Verification")
    print("="*50)
    
    # Set to eval mode
    original_model.eval()
    quantized_model.eval()
    
    # Run inference
    with torch.no_grad():
        try:
            original_output = original_model(sample_input)
            quantized_output = quantized_model(sample_input)
            
            # Check shapes
            print(f"\nInput shape:            {sample_input.shape}")
            print(f"Original output shape:  {original_output.shape}")
            print(f"Quantized output shape: {quantized_output.shape}")
            
            # Verify shapes match
            if original_output.shape == quantized_output.shape:
                print("✓ Input and output shapes preserved!")
            else:
                print(f"✗ ERROR: Shape mismatch!")
                return False
            
            # Calculate accuracy difference
            diff = torch.abs(original_output - quantized_output).mean()
            max_diff = torch.abs(original_output - quantized_output).max()
            print(f"\nMean absolute difference: {diff.item():.6f}")
            print(f"Max absolute difference:  {max_diff.item():.6f}")
            
            # Show sample outputs
            print(f"\nOriginal output sample:  {original_output.flatten()[:5]}")
            print(f"Quantized output sample: {quantized_output.flatten()[:5]}")
            
        except Exception as e:
            print(f"✗ ERROR during verification: {e}")
            return False
    
    # Model size comparison
    print("\n" + "="*50)
    print("Model Size Comparison")
    print("="*50)
    
    def get_model_size(model, filename):
        torch.save(model.state_dict(), filename)
        size = os.path.getsize(filename) / 1e6  # MB
        os.remove(filename)
        return size
    
    try:
        orig_size = get_model_size(original_model, "temp_orig.p")
        quant_size = get_model_size(quantized_model, "temp_quant.p")
        
        print(f"Original model size:  {orig_size:.2f} MB")
        print(f"Quantized model size: {quant_size:.2f} MB")
        print(f"Compression ratio:    {orig_size/quant_size:.2f}x smaller")
        print(f"Size reduction:       {(1 - quant_size/orig_size)*100:.1f}%")
    except Exception as e:
        print(f"Could not calculate model size: {e}")
    
    return True


def save_quantized_model(quantized_model, filepath="quantized_model.pth"):
    """Save the quantized model to disk."""
    torch.save(quantized_model.state_dict(), filepath)
    print(f"\n✓ Quantized model saved to: {filepath}")


def quantize_model(model, sample_input):
    print("\n" + "="*50)
    print("QUANTIZING MODEL")
    print("="*50)
    
    if model is not None:
        quantized_model = quantize_model_to_int8(model, sample_input)
        
        verify_model(model, quantized_model, sample_input)
        
        # Save
        save_quantized_model(quantized_model, "quantized_model.pth")
        
        return quantized_model
    else:
        print("Empty input")
        return None


if __name__ == "__main__":
    # Run examples
    print("\n" + "="*70)
    print(" PyTorch INT8 Quantization - Preserves Input/Output Shapes")
    print("="*70)
    sample_input = torch.randn(BATCH_SIZE, 6, 38, 5)
    quantize_model(model, sample_input)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)