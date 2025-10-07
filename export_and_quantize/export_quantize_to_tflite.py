def verify_quantization(original_model, quantized_model, sample_input):
    """Verify the quantized model works correctly and maintains same I/O shapes."""
    print("\n=== Verification ===")
    
    # Set models to eval mode
    original_model.eval()
    quantized_model.eval()
    
    # Get outputs
    with torch.no_grad():
        original_output = original_model(sample_input)
        quantized_output = quantized_model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Original output shape: {original_output.shape}")
    print(f"Quantized output shape: {quantized_output.shape}")
    
    # Verify shapes match
    assert original_output.shape == quantized_output.shape, \
        f"Shape mismatch! Original: {original_output.shape}, Quantized: {quantized_output.shape}"
    print("✓ Input and output shapes preserved!")
    
    print(f"\nOriginal output sample: {original_output[0, :5]}")
    print(f"Quantized output sample: {quantized_output[0, :5]}")
    
    # Calculate difference
    diff = torch.abs(original_output - quantized_output).mean()
    print(f"Mean absolute difference: {diff.item():.6f}")