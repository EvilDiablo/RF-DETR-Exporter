# RF-DETR TensorRT Converter

Convert RF-DETR PyTorch models (`.pth` checkpoints) to ONNX format and TensorRT engines for optimized deployment.

## Overview

This project provides tools to convert RF-DETR (Roboflow Detection Transformer) models from PyTorch checkpoints to:

1. **ONNX format** (`.onnx`) - Cross-platform model format
2. **TensorRT engines** (`.engine`) - Optimized for NVIDIA GPUs

### Conversion Workflow

```
RF-DETR Checkpoint (.pth)
    ↓
ONNX Model (.onnx)
    ↓ (optional)
Simplified ONNX (.sim.onnx)
    ↓ (optional)
TensorRT Engine (.engine)
```

## Features

- ✅ Export RF-DETR models to ONNX format
- ✅ ONNX model simplification and optimization
- ✅ TensorRT engine conversion with performance profiling
- ✅ Support for RF-DETR Nano, Small, Base, Large models
- ✅ Configurable input shapes and batch sizes
- ✅ ONNX model validation and testing

## Requirements

### System Requirements

- **OS**: Windows, Linux, or macOS
- **Python**: 3.9+
- **CUDA**: 11.8+ (for GPU operations and TensorRT)
- **GPU**: NVIDIA GPU with CUDA support (recommended)

### Software Requirements

- **PyTorch**: 2.9.0+ (will be installed with `rfdetr`)
- **ONNX**: 1.19.1 (required version for RF-DETR compatibility)
- **TensorRT**: 10.13+ (optional, only for TensorRT conversion)
- **trtexec**: Command-line tool from TensorRT (optional, only for TensorRT conversion)

## Installation

### 1. Clone or Download This Repository

```bash
cd rf-detr-tensorrt-convertor
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important Notes:**

- ONNX version is pinned to **1.19.1** for RF-DETR compatibility (see [issue #21121](https://github.com/blakeblackshear/frigate/discussions/21121))
- If you already have a different ONNX version, reinstall: `pip install onnx==1.19.1 --force-reinstall`

### 4. Install TensorRT (Optional, for TensorRT Conversion)

If you want to convert to TensorRT engines, install TensorRT:

```bash
# Using NVIDIA PyPI index
pip install nvidia-pyindex
pip install tensorrt

# Or install TensorRT manually from NVIDIA website
# Make sure trtexec is in your PATH
```

## Quick Start

### Basic Usage: Export to ONNX

```bash
python run_export.py --resume rf-detr-nano.pth --output_dir output
```

This will:

- Load the RF-DETR Nano checkpoint
- Export to ONNX format
- Save to `output/inference_model.onnx`

### Export with Simplification

```bash
python run_export.py --resume rf-detr-nano.pth --output_dir output --simplify
```

### Full Pipeline: ONNX → Simplified → TensorRT

```bash
python run_export.py --resume rf-detr-nano.pth --output_dir output --simplify --tensorrt
```

## Usage

### Command Line Arguments

#### Required Arguments

- `--resume`: Path to RF-DETR checkpoint file (e.g., `rf-detr-nano.pth`)

#### Model Configuration

- `--num_classes`: Number of object classes (default: 90 for COCO)
- `--encoder`: Encoder type (default: `dinov2_windowed_small` for RF-DETR Nano)
- `--shape`: Input image shape `[width height]` (default: `384 384` for RF-DETR Nano)
- `--batch_size`: Batch size for export (default: 1)

#### Export Options

- `--output_dir`: Output directory for exported models (default: `output`)
- `--simplify`: Enable ONNX model simplification
- `--tensorrt`: Convert ONNX to TensorRT engine
- `--backbone_only`: Export only the backbone (feature extractor)
- `--verbose`: Enable verbose output
- `--opset_version`: ONNX opset version (default: 17)
- `--force`: Force overwrite existing files

#### Device Options

- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--seed`: Random seed for reproducibility (default: 42)

#### TensorRT Options

- `--profile`: Enable profiling during TensorRT conversion (requires nsys)
- `--dry-run`: Print commands without executing

### Examples

#### Example 1: Export RF-DETR Nano to ONNX

```bash
python run_export.py \
    --resume rf-detr-nano.pth \
    --output_dir output \
    --shape 384 384
```

#### Example 2: Export with Custom Input Shape

```bash
python run_export.py \
    --resume rf-detr-nano.pth \
    --output_dir output \
    --shape 640 640 \
    --simplify
```

**Note**: Input shape must be divisible by 32 (patch_size _ num_windows = 16 _ 2 = 32)

#### Example 3: Full Pipeline with TensorRT

```bash
python run_export.py \
    --resume rf-detr-nano.pth \
    --output_dir output \
    --simplify \
    --tensorrt \
    --profile
```

#### Example 4: Export with Real Image

```bash
python run_export.py \
    --resume rf-detr-nano.pth \
    --output_dir output \
    --infer_dir path/to/test/image.jpg
```

## Testing the Exported Model

Use the provided test script to validate the exported ONNX model:

### Basic Test

```bash
python test_model.py --onnx output/inference_model.onnx
```

### Test with GPU

```bash
python test_model.py --onnx output/inference_model.onnx --gpu
```

### Test with Real Image

```bash
python test_model.py \
    --onnx output/inference_model.onnx \
    --image path/to/test/image.jpg \
    --gpu
```

### Test with Different Input Shape

```bash
python test_model.py \
    --onnx output/inference_model.onnx \
    --shape 640 640
```

## Output Files

After successful export, you'll find:

### ONNX Export Only

- `output/inference_model.onnx` - Standard ONNX model

### With Simplification

- `output/inference_model.onnx` - Original ONNX model
- `output/inference_model.sim.onnx` - Simplified/optimized ONNX model

### With TensorRT

- `output/inference_model.onnx` - ONNX model
- `output/inference_model.engine` - TensorRT engine file
- `output/*.nsys-rep` - Profiling data (if `--profile` was used)

## Model Configuration

### RF-DETR Nano (Default Configuration)

- **Encoder**: `dinov2_windowed_small`
- **Input Shape**: 384×384
- **Decoder Layers**: 2
- **Number of Queries**: 300
- **Patch Size**: 16
- **Number of Windows**: 2

### Supported Models

The script is configured for **RF-DETR Nano** by default. For other models (Small, Base, Large), you may need to adjust:

- `--encoder`: `dinov2_windowed_small` or `dinov2_windowed_base`
- `--shape`: Model-specific resolution
- `--dec_layers`: Number of decoder layers
- `--num_classes`: Number of classes

Refer to the RF-DETR documentation for model-specific configurations.

## Troubleshooting

### Error: `AttributeError: module 'onnx.helper' has no attribute 'float32_to_bfloat16'`

**Solution**: Install ONNX 1.19.1:

```bash
pip install onnx==1.19.1 --force-reinstall
```

### Error: `module 'onnxruntime' has no attribute 'InferenceSession'`

**Cause**: This usually indicates that `onnxruntime-gpu` is not properly installed or there's a conflict between `onnxruntime` and `onnxruntime-gpu`.

**Solution**:

1. **Make sure you're in the correct virtual environment**:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Clean reinstall onnxruntime-gpu**:
   ```bash
   pip uninstall onnxruntime onnxruntime-gpu -y
   pip install onnxruntime-gpu
   ```

3. **Verify installation**:
   ```bash
   python -c "import onnxruntime as ort; print('Version:', ort.__version__); print('Has InferenceSession:', hasattr(ort, 'InferenceSession'))"
   ```
   
   You should see:
   ```
   Version: 1.23.2 (or similar)
   Has InferenceSession: True
   ```

**Note**: You should only have **one** of `onnxruntime` or `onnxruntime-gpu` installed, not both. Having both can cause conflicts.

### Error: `CUDAExecutionProvider not available` or `Available providers: ['CPUExecutionProvider']`

**Cause**: You have `onnxruntime` (CPU-only) installed instead of `onnxruntime-gpu`.

**Solution**:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

**Note**: This is separate from PyTorch CUDA support. You need both:
- PyTorch with CUDA (for model training/export)
- onnxruntime-gpu (for ONNX inference on GPU)

### Error: `AssertionError` during model building

**Solution**: Make sure you're using the correct encoder name. For RF-DETR Nano, use `dinov2_windowed_small`, not `vit_tiny`.

### Error: `Weights only load failed`

**Solution**: This is already fixed in the code by using `weights_only=False` in `torch.load()`. If you still see this, update `export.py`.

### TensorRT Conversion Fails

**Possible Causes**:

1. `trtexec` command not found - Install TensorRT and add to PATH
2. ONNX model incompatible - Try using `--simplify` first
3. GPU out of memory - Reduce batch size or use a smaller model

### Warnings During Export

**TracerWarnings** are normal and expected during ONNX export with TorchScript. They indicate that some operations are being traced as constants, which is fine for fixed input shapes.

**TF32 Deprecation Warning**: This is informational and will be addressed in future PyTorch versions. The code now uses the new API to suppress this warning.

## Workflow Details

### 1. Model Loading

- Loads PyTorch checkpoint (`.pth` file)
- Builds model architecture matching the checkpoint
- Loads weights into the model

### 2. Input Preparation

- Creates test input (dummy or from image)
- Applies preprocessing:
  - Square resize to target shape
  - Normalization (ImageNet mean/std)
  - Tensor conversion

### 3. ONNX Export

- Converts PyTorch model to ONNX format
- Uses legacy TorchScript exporter for compatibility
- Saves as `.onnx` file

### 4. ONNX Simplification (Optional)

- Optimizes ONNX graph structure
- Removes redundant operations
- Validates with test runs
- Creates `.sim.onnx` file

### 5. TensorRT Conversion (Optional)

- Uses `trtexec` to build TensorRT engine
- Optimizes for target GPU
- Uses FP16 precision for speed
- Benchmarks performance

## Performance Tips

1. **Use simplification**: Simplifying the ONNX model can reduce file size and improve inference speed
2. **FP16 precision**: TensorRT engines use FP16 by default for better performance
3. **Profile your model**: Use `--profile` to identify bottlenecks
4. **Optimize batch size**: Test different batch sizes for your use case

## File Structure

```
rf-detr-tensorrt-convertor/
├── export.py              # Main export script (from RF-DETR repository)
├── run_export.py          # Convenience wrapper script
├── test_model.py          # ONNX model testing script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── rf-detr-nano.pth      # Your model checkpoint (example)
└── output/               # Output directory (created automatically)
    ├── inference_model.onnx
    ├── inference_model.sim.onnx  (if --simplify)
    └── inference_model.engine    (if --tensorrt)
```

## Advanced Usage

### Using the Export Script Directly

You can also use `export.py` directly by creating an args object:

```python
from types import SimpleNamespace
from export import main

args = SimpleNamespace(
    resume='rf-detr-nano.pth',
    output_dir='output',
    # ... other arguments
)

main(args)
```

### Custom Model Configurations

For custom configurations, modify `run_export.py` or create your own script that sets the appropriate model parameters.

## References

- [RF-DETR Repository](https://github.com/roboflow/rf-detr)
- [ONNX Documentation](https://onnx.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Frigate Discussion on ONNX Version](https://github.com/blakeblackshear/frigate/discussions/21121)

## License

This project is based on RF-DETR and LW-DETR code:

- RF-DETR: Copyright (c) 2025 Roboflow. All Rights Reserved. Licensed under Apache License 2.0
- LW-DETR: Copyright (c) 2024 Baidu. All Rights Reserved.

## Support

For issues related to:

- **RF-DETR models**: Check the [RF-DETR GitHub repository](https://github.com/roboflow/rf-detr)
- **Export process**: Check this repository's issues
- **TensorRT**: Check [NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

## Contributing

Contributions are welcome! Please ensure your changes maintain compatibility with the required ONNX version (1.19.1).
