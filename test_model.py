#!/usr/bin/env python3
"""
Test script for ONNX model inference
Tests the exported RF-DETR Nano ONNX model
"""

import numpy as np
import torch
import onnx
from PIL import Image
import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import onnxruntime as ort
    from export import make_infer_image
    import rfdetr.datasets.transforms as T
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure onnxruntime is installed: pip install onnxruntime")
    sys.exit(1)


def validate_onnx_model(onnx_path):
    """Validate ONNX model structure"""
    print(f"\n{'='*60}")
    print("Validating ONNX Model Structure")
    print(f"{'='*60}")
    
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model is valid: {onnx_path}")
        
        # Print model info
        print(f"\nModel inputs:")
        for input_tensor in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else '?' 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  - {input_tensor.name}: {shape}")
        
        print(f"\nModel outputs:")
        for output_tensor in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else '?' 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"  - {output_tensor.name}: {shape}")
        
        return True
    except Exception as e:
        print(f"✗ ONNX model validation failed: {e}")
        return False


def test_onnx_inference(onnx_path, shape=(384, 384), batch_size=1, use_gpu=False, compare_pytorch=False, checkpoint_path=None):
    """Test ONNX model inference"""
    print(f"\n{'='*60}")
    print("Testing ONNX Model Inference")
    print(f"{'='*60}")
    
    # Create test input
    print(f"\nCreating test input: shape={shape}, batch_size={batch_size}")
    input_tensor = make_infer_image(None, shape, batch_size, device="cpu")
    input_numpy = input_tensor.cpu().numpy()
    
    print(f"Input shape: {input_numpy.shape}")
    print(f"Input dtype: {input_numpy.dtype}")
    print(f"Input range: [{input_numpy.min():.3f}, {input_numpy.max():.3f}]")
    
    # Create ONNX Runtime session
    print(f"\nLoading ONNX model: {onnx_path}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"✓ Model loaded successfully")
        print(f"  Providers: {session.get_providers()}")
        print(f"  Active provider: {session.get_providers()[0]}")
    except Exception as e:
        print(f"✗ Failed to load ONNX model: {e}")
        return False
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    print(f"\nInput name: {input_name}")
    print(f"Output names: {output_names}")
    
    # Run inference
    print(f"\nRunning inference...")
    try:
        outputs = session.run(output_names, {input_name: input_numpy})
        print(f"✓ Inference successful!")
        
        # Print output information
        for i, (output_name, output) in enumerate(zip(output_names, outputs)):
            print(f"\nOutput {i+1}: {output_name}")
            print(f"  Shape: {output.shape}")
            print(f"  Dtype: {output.dtype}")
            print(f"  Range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  Mean: {output.mean():.3f}, Std: {output.std():.3f}")
            
            # Show some sample values for boxes
            if 'dets' in output_name.lower() or 'boxes' in output_name.lower():
                print(f"  Sample boxes (first 3):")
                for j in range(min(3, output.shape[1])):
                    print(f"    Box {j}: {output[0, j, :]}")
            
            # Show some sample values for labels
            if 'labels' in output_name.lower() or 'logits' in output_name.lower():
                print(f"  Sample logits (first query, top 5 classes):")
                top5 = np.argsort(output[0, 0, :])[-5:][::-1]
                for class_idx in top5:
                    print(f"    Class {class_idx}: {output[0, 0, class_idx]:.3f}")
        
        # Optional: Compare with PyTorch
        if compare_pytorch and checkpoint_path:
            compare_with_pytorch(checkpoint_path, input_tensor, outputs, output_names)
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_pytorch(checkpoint_path, input_tensor, onnx_outputs, output_names):
    """Compare ONNX outputs with PyTorch model outputs"""
    print(f"\n{'='*60}")
    print("Comparing ONNX vs PyTorch Outputs")
    print(f"{'='*60}")
    
    try:
        # Import and load PyTorch model (simplified version)
        # This would require the same setup as export.py
        print("Note: PyTorch comparison requires full model setup.")
        print("Skipping for now - you can add this later if needed.")
        # TODO: Add PyTorch model loading and comparison
    except Exception as e:
        print(f"Could not compare with PyTorch: {e}")


def test_with_image(onnx_path, image_path, shape=(384, 384), use_gpu=False):
    """Test ONNX model with a real image"""
    print(f"\n{'='*60}")
    print(f"Testing ONNX Model with Image: {image_path}")
    print(f"{'='*60}")
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"✓ Image loaded: {image.size}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False
    
    # Use the same preprocessing as export
    transforms = T.Compose([
        T.SquareResize([shape[0]]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor, _ = transforms(image, None)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_numpy = input_tensor.cpu().numpy()
    
    print(f"Preprocessed input shape: {input_numpy.shape}")
    
    # Run inference
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    outputs = session.run(output_names, {input_name: input_numpy})
    
    print(f"\n✓ Inference complete!")
    for output_name, output in zip(output_names, outputs):
        print(f"  {output_name}: shape={output.shape}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test exported ONNX model')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--shape', type=int, nargs=2, default=[384, 384],
                        help='Input shape [width height] (default: 384 384)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image (optional)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with PyTorch model (requires checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to PyTorch checkpoint for comparison')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.onnx):
        print(f"Error: ONNX file not found: {args.onnx}")
        sys.exit(1)
    
    # Step 1: Validate model
    if not validate_onnx_model(args.onnx):
        sys.exit(1)
    
    # Step 2: Test with dummy input
    if not test_onnx_inference(
        args.onnx, 
        shape=tuple(args.shape), 
        batch_size=args.batch_size,
        use_gpu=args.gpu,
        compare_pytorch=args.compare,
        checkpoint_path=args.checkpoint
    ):
        sys.exit(1)
    
    # Step 3: Test with real image (if provided)
    if args.image:
        if not os.path.exists(args.image):
            print(f"Warning: Image file not found: {args.image}")
        else:
            test_with_image(args.onnx, args.image, shape=tuple(args.shape), use_gpu=args.gpu)
    
    print(f"\n{'='*60}")
    print("✓ All tests passed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()