# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
export ONNX model and TensorRT engine for deployment
"""
import os
import ast
import random
import argparse
import subprocess
import torch.nn as nn
from pathlib import Path
import time
from collections import defaultdict

import onnx
import torch
import onnxsim
import numpy as np
from PIL import Image

import rfdetr.util.misc as utils
import rfdetr.datasets.transforms as T
from rfdetr.models import build_model
from rfdetr.deploy._onnx import OnnxOptimizer
import re
import sys


def run_command_shell(command, dry_run:bool = False) -> int:
    if dry_run:
        print("")
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} {command}")
        print("")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr.decode('utf-8')}")
        raise


def make_infer_image(infer_dir, shape, batch_size, device="cuda"):
    if infer_dir is None:
        dummy = np.random.randint(0, 256, (shape[0], shape[1], 3), dtype=np.uint8)
        image = Image.fromarray(dummy, mode="RGB")
    else:
        image = Image.open(infer_dir).convert("RGB")

    transforms = T.Compose([
        T.SquareResize([shape[0]]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    inps, _ = transforms(image, None)
    inps = inps.to(device)
    # inps = utils.nested_tensor_from_tensor_list([inps for _ in range(args.batch_size)])
    inps = torch.stack([inps for _ in range(batch_size)])
    return inps

def export_onnx(output_dir, model, input_names, input_tensors, output_names, dynamic_axes, backbone_only=False, verbose=True, opset_version=17):
    export_name = "backbone_model" if backbone_only else "inference_model"
    output_file = os.path.join(output_dir, f"{export_name}.onnx")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Prepare model for export
    if hasattr(model, "export"):
        model.export()
    
    try:
        torch.onnx.export(
            model,
            input_tensors,
            output_file,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            verbose=verbose,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False)  # Use legacy TorchScript exporter for compatibility

        # Verify the exported ONNX model
        try:
            onnx_model = onnx.load(output_file)
            onnx.checker.check_model(onnx_model)
            print(f'ONNX model verification passed')
        except onnx.checker.ValidationError as e:
            print(f'Warning: ONNX model verification failed: {e}')
            # Don't raise, as the model might still be usable
        
        print(f'\nSuccessfully exported ONNX model: {output_file}')
        return output_file
    except Exception as e:
        # If export fails and we're on CUDA, try CPU as fallback
        if input_tensors.device.type == 'cuda':
            print(f'\nONNX export on CUDA failed: {e}')
            print('Falling back to CPU export...')
            model_cpu = model.cpu()
            input_tensors_cpu = input_tensors.cpu()
            try:
                torch.onnx.export(
                    model_cpu,
                    input_tensors_cpu,
                    output_file,
                    input_names=input_names,
                    output_names=output_names,
                    export_params=True,
                    keep_initializers_as_inputs=False,
                    do_constant_folding=True,
                    verbose=verbose,
                    opset_version=opset_version,
                    dynamic_axes=dynamic_axes,
                    training=torch.onnx.TrainingMode.EVAL,
                    dynamo=False)
                print(f'\nSuccessfully exported ONNX model on CPU: {output_file}')
                return output_file
            except Exception as e2:
                raise RuntimeError(f"ONNX export failed on both CUDA and CPU. CUDA error: {e}, CPU error: {e2}")
        else:
            raise


def onnx_simplify(onnx_dir:str, input_names, input_tensors, force=False):
    sim_onnx_dir = onnx_dir.replace(".onnx", ".sim.onnx")
    if os.path.isfile(sim_onnx_dir) and not force:
        return sim_onnx_dir
    
    if isinstance(input_tensors, torch.Tensor):
        input_tensors = [input_tensors]
    
    print(f'start simplify ONNX model: {onnx_dir}')
    opt = OnnxOptimizer(onnx_dir)
    opt.info('Model: original')
    opt.common_opt()
    opt.info('Model: optimized')
    opt.save_onnx(sim_onnx_dir)
    input_dict = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, input_tensors)}
    model_opt, check_ok = onnxsim.simplify(
        onnx_dir,
        check_n = 3,
        input_data=input_dict,
        dynamic_input_shape=False)
    if check_ok:
        onnx.save(model_opt, sim_onnx_dir)
    else:
        raise RuntimeError("Failed to simplify ONNX model.")
    print(f'Successfully simplified ONNX model: {sim_onnx_dir}')
    return sim_onnx_dir


def trtexec(onnx_dir:str, args) -> dict:
    engine_dir = onnx_dir.replace(".onnx", f".engine")
    
    # Base trtexec command
    trt_command = " ".join([
        "trtexec",
            f"--onnx={onnx_dir}",
            f"--saveEngine={engine_dir}",
            f"--memPoolSize=workspace:4096 --fp16",
            f"--useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10",
            f"{'--verbose' if args.verbose else ''}"])
    
    if args.profile:
        profile_dir = onnx_dir.replace(".onnx", f".nsys-rep")
        # Wrap with nsys profile command
        command = " ".join([
            "nsys profile",
                f"--output={profile_dir}",
                "--trace=cuda,nvtx",
                "--force-overwrite true",
                trt_command
        ])
        print(f'Profile data will be saved to: {profile_dir}')
    else:
        command = trt_command

    output = run_command_shell(command, args.dry_run)
    stats = parse_trtexec_output(output.stdout)
    return stats

def parse_trtexec_output(output_text):
    print(output_text)
    # Common patterns in trtexec output
    gpu_compute_pattern = r"GPU Compute Time: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms, median = (\d+\.\d+) ms"
    h2d_pattern = r"Host to Device Transfer Time: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms"
    d2h_pattern = r"Device to Host Transfer Time: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms"
    latency_pattern = r"Latency: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms"
    throughput_pattern = r"Throughput: (\d+\.\d+) qps"
    
    stats = {}
    
    # Extract compute times
    if match := re.search(gpu_compute_pattern, output_text):
        stats.update({
            'compute_min_ms': float(match.group(1)),
            'compute_max_ms': float(match.group(2)),
            'compute_mean_ms': float(match.group(3)),
            'compute_median_ms': float(match.group(4))
        })
    
    # Extract H2D times
    if match := re.search(h2d_pattern, output_text):
        stats.update({
            'h2d_min_ms': float(match.group(1)),
            'h2d_max_ms': float(match.group(2)),
            'h2d_mean_ms': float(match.group(3))
        })
    
    # Extract D2H times
    if match := re.search(d2h_pattern, output_text):
        stats.update({
            'd2h_min_ms': float(match.group(1)),
            'd2h_max_ms': float(match.group(2)),
            'd2h_mean_ms': float(match.group(3))
        })

    if match := re.search(latency_pattern, output_text):
        stats.update({
            'latency_min_ms': float(match.group(1)),
            'latency_max_ms': float(match.group(2)),
            'latency_mean_ms': float(match.group(3))
        })
    
    # Extract throughput
    if match := re.search(throughput_pattern, output_text):
        stats['throughput_qps'] = float(match.group(1))
    
    return stats

def no_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            raise ValueError("BatchNorm2d found in the model. Please remove it.")

def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    return 0

def benchmark_inference(model, input_tensors, device, num_warmup=10, num_runs=100):
    """Benchmark PyTorch model inference time"""
    model.eval()
    model.to(device)
    input_tensors = input_tensors.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensors)
    
    # Synchronize GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensors)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times)
    }

def benchmark_onnx(onnx_path, input_tensors, num_warmup=10, num_runs=100):
    """Benchmark ONNX model inference time"""
    try:
        import onnxruntime as ort
    except ImportError as e:
        print(f"Warning: onnxruntime not installed. Skipping ONNX benchmark. Error: {e}")
        return None
    
    # Verify InferenceSession exists
    if not hasattr(ort, 'InferenceSession'):
        print(f"Error: onnxruntime module does not have InferenceSession attribute.")
        print(f"  This might indicate a corrupted installation or version mismatch.")
        print(f"  Try: pip uninstall onnxruntime onnxruntime-gpu")
        print(f"  Then: pip install onnxruntime-gpu")
        return None
    
    # Convert input to numpy
    input_numpy = input_tensors.cpu().numpy()
    
    # Check available providers - handle different onnxruntime versions
    available_providers = []
    if hasattr(ort, 'get_available_providers'):
        try:
            available_providers = ort.get_available_providers()
        except (AttributeError, Exception):
            pass
    
    # Try to create session with CUDA first if available, fallback to CPU
    session = None
    providers = ['CPUExecutionProvider']
    
    if torch.cuda.is_available():
        # Try CUDA first
        try:
            session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            providers = session.get_providers()
            print(f"  Using CUDAExecutionProvider for ONNX inference")
        except Exception as e:
            # CUDA failed, try CPU
            try:
                session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                providers = session.get_providers()
                print(f"  Warning: CUDAExecutionProvider not available. Install onnxruntime-gpu for GPU support.")
                print(f"  Using CPUExecutionProvider for ONNX inference")
            except Exception as e2:
                print(f"  Error creating ONNX session with CPU: {e2}")
                print(f"  Original CUDA error: {e}")
                print(f"  Make sure onnxruntime-gpu is properly installed:")
                print(f"    pip uninstall onnxruntime onnxruntime-gpu")
                print(f"    pip install onnxruntime-gpu")
                return None
    else:
        # No CUDA available, use CPU
        try:
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            providers = session.get_providers()
            print(f"  Using CPUExecutionProvider for ONNX inference")
        except Exception as e:
            print(f"  Error creating ONNX session: {e}")
            print(f"  Make sure onnxruntime or onnxruntime-gpu is properly installed:")
            print(f"    pip install onnxruntime-gpu")
            return None
    
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Warmup
    for _ in range(num_warmup):
        _ = session.run(output_names, {input_name: input_numpy})
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(output_names, {input_name: input_numpy})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'provider': session.get_providers()[0]
    }

def print_model_statistics(model, checkpoint_path, onnx_path, tensorrt_path=None):
    """Print comprehensive model statistics before and after export"""
    print(f"\n{'='*70}")
    print("MODEL STATISTICS COMPARISON")
    print(f"{'='*70}")
    
    # PyTorch Model Statistics
    print(f"\n{'─'*70}")
    print("1. PYTORCH MODEL (.pt)")
    print(f"{'─'*70}")
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    print(f"  Total Parameters:        {n_parameters:,}")
    print(f"  Trainable Parameters:    {n_trainable:,}")
    print(f"  Model Size (in memory):  {total_size / (1024**2):.2f} MB")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_size = get_file_size_mb(checkpoint_path)
        print(f"  Checkpoint File Size:    {checkpoint_size:.2f} MB")
    
    # ONNX Model Statistics
    print(f"\n{'─'*70}")
    print("2. ONNX MODEL (.onnx)")
    print(f"{'─'*70}")
    if onnx_path and os.path.exists(onnx_path):
        onnx_size = get_file_size_mb(onnx_path)
        print(f"  ONNX File Size:          {onnx_size:.2f} MB")
        
        # Load and analyze ONNX model
        try:
            onnx_model = onnx.load(onnx_path)
            # Count parameters in ONNX
            onnx_params = 0
            from onnx.numpy_helper import to_array
            for param in onnx_model.graph.initializer:
                try:
                    # Get the actual tensor data and calculate size
                    arr = to_array(param)
                    onnx_params += arr.size
                except Exception:
                    # Fallback: try to get shape from dims
                    try:
                        if hasattr(param, 'dims') and param.dims:
                            onnx_params += np.prod(param.dims)
                    except Exception:
                        pass
            print(f"  ONNX Parameters:         {onnx_params:,}")
            
            # Calculate compression ratio
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint_size = get_file_size_mb(checkpoint_path)
                compression = (1 - onnx_size / checkpoint_size) * 100 if checkpoint_size > 0 else 0
                print(f"  Size Reduction:          {compression:.1f}%")
        except Exception as e:
            print(f"  (Could not analyze ONNX structure: {e})")
    else:
        print(f"  ONNX file not found: {onnx_path}")
    
    # TensorRT Engine Statistics
    if tensorrt_path and os.path.exists(tensorrt_path):
        print(f"\n{'─'*70}")
        print("3. TENSORRT ENGINE (.engine)")
        print(f"{'─'*70}")
        engine_size = get_file_size_mb(tensorrt_path)
        print(f"  Engine File Size:        {engine_size:.2f} MB")
        
        if onnx_path and os.path.exists(onnx_path):
            onnx_size = get_file_size_mb(onnx_path)
            compression = (1 - engine_size / onnx_size) * 100 if onnx_size > 0 else 0
            print(f"  Size Reduction vs ONNX:  {compression:.1f}%")

def print_performance_comparison(pytorch_stats, onnx_stats, tensorrt_stats=None):
    """Print performance comparison between different formats"""
    print(f"\n{'='*70}")
    print("PERFORMANCE BENCHMARK (Inference Latency)")
    print(f"{'='*70}")
    
    # PyTorch Performance
    if pytorch_stats:
        print(f"\n{'─'*70}")
        print("1. PYTORCH MODEL")
        print(f"{'─'*70}")
        print(f"  Mean Latency:    {pytorch_stats['mean_ms']:.2f} ms")
        print(f"  Median Latency:  {pytorch_stats['median_ms']:.2f} ms")
        print(f"  Min Latency:     {pytorch_stats['min_ms']:.2f} ms")
        print(f"  Max Latency:     {pytorch_stats['max_ms']:.2f} ms")
        print(f"  Std Deviation:   {pytorch_stats['std_ms']:.2f} ms")
        baseline = pytorch_stats['mean_ms']
    else:
        baseline = None
    
    # ONNX Performance
    if onnx_stats:
        print(f"\n{'─'*70}")
        print("2. ONNX MODEL")
        print(f"{'─'*70}")
        print(f"  Provider:        {onnx_stats.get('provider', 'Unknown')}")
        print(f"  Mean Latency:    {onnx_stats['mean_ms']:.2f} ms")
        print(f"  Median Latency:  {onnx_stats['median_ms']:.2f} ms")
        print(f"  Min Latency:     {onnx_stats['min_ms']:.2f} ms")
        print(f"  Max Latency:     {onnx_stats['max_ms']:.2f} ms")
        print(f"  Std Deviation:   {onnx_stats['std_ms']:.2f} ms")
        
        if baseline:
            speedup = (baseline / onnx_stats['mean_ms']) if onnx_stats['mean_ms'] > 0 else 0
            improvement = ((baseline - onnx_stats['mean_ms']) / baseline) * 100
            print(f"  vs PyTorch:      {speedup:.2f}x faster ({improvement:.1f}% improvement)")
    
    # TensorRT Performance
    if tensorrt_stats:
        print(f"\n{'─'*70}")
        print("3. TENSORRT ENGINE")
        print(f"{'─'*70}")
        trt_mean = tensorrt_stats.get('latency_mean_ms', tensorrt_stats.get('compute_mean_ms'))
        trt_median = tensorrt_stats.get('compute_median_ms', tensorrt_stats.get('latency_mean_ms'))
        trt_min = tensorrt_stats.get('latency_min_ms', tensorrt_stats.get('compute_min_ms'))
        trt_max = tensorrt_stats.get('latency_max_ms', tensorrt_stats.get('compute_max_ms'))
        
        if trt_mean:
            print(f"  Mean Latency:    {trt_mean:.2f} ms")
        if trt_median:
            print(f"  Median Latency:  {trt_median:.2f} ms")
        if trt_min:
            print(f"  Min Latency:     {trt_min:.2f} ms")
        if trt_max:
            print(f"  Max Latency:     {trt_max:.2f} ms")
        if 'throughput_qps' in tensorrt_stats:
            print(f"  Throughput:      {tensorrt_stats['throughput_qps']:.2f} qps")
        
        if baseline and trt_mean:
            speedup = baseline / trt_mean if trt_mean > 0 else 0
            improvement = ((baseline - trt_mean) / baseline) * 100
            print(f"  vs PyTorch:      {speedup:.2f}x faster ({improvement:.1f}% improvement)")
        
        if onnx_stats and trt_mean:
            speedup = onnx_stats['mean_ms'] / trt_mean if trt_mean > 0 else 0
            improvement = ((onnx_stats['mean_ms'] - trt_mean) / onnx_stats['mean_ms']) * 100
            print(f"  vs ONNX:         {speedup:.2f}x faster ({improvement:.1f}% improvement)")
    
    print(f"\n{'='*70}")

def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    
    # convert device to device_id
    if args.device == 'cuda':
        device_id = "0"
    elif args.device == 'cpu':
        device_id = ""
    else:
        device_id = str(int(args.device))
        args.device = f"cuda:{device_id}"

    # Device for export onnx - try CUDA first for faster export, fallback to CPU if it fails
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    if torch.cuda.is_available() and args.device != 'cpu' and device_id:
        export_device = torch.device("cuda")
        print("Attempting ONNX export on CUDA...")
    else:
        export_device = torch.device("cpu")
        print("Using CPU for ONNX export...")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {n_parameters}")
    n_backbone_parameters = sum(p.numel() for p in model.backbone.parameters())
    print(f"number of backbone parameters: {n_backbone_parameters}")
    n_projector_parameters = sum(p.numel() for p in model.backbone[0].projector.parameters())
    print(f"number of projector parameters: {n_projector_parameters}")
    n_backbone_encoder_parameters = sum(p.numel() for p in model.backbone[0].encoder.parameters())
    print(f"number of backbone encoder parameters: {n_backbone_encoder_parameters}")
    n_transformer_parameters = sum(p.numel() for p in model.transformer.parameters())
    print(f"number of transformer parameters: {n_transformer_parameters}")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"load checkpoints {args.resume}")

    if args.layer_norm:
        no_batch_norm(model)

    model.to(export_device)

    # Extract shape from args (default to 640x640 if not provided)
    shape = getattr(args, 'shape', (640, 640))
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        shape = tuple(shape)
    else:
        shape = (640, 640)
    
    batch_size = getattr(args, 'batch_size', 1)
    infer_dir = getattr(args, 'infer_dir', None)
    
    input_tensors = make_infer_image(infer_dir, shape, batch_size, export_device)
    input_names = ['input']
    output_names = ['features'] if args.backbone_only else ['dets', 'labels']
    dynamic_axes = None
    # Run model inference in pytorch mode for verification
    # Use CUDA for inference test if available, otherwise use export device
    inference_device = torch.device("cuda") if torch.cuda.is_available() and device_id and args.device != 'cpu' else export_device
    model.eval().to(inference_device)
    input_tensors_test = input_tensors.to(inference_device)
    with torch.no_grad():
        if args.backbone_only:
            features = model(input_tensors_test)
            print(f"PyTorch inference output shape: {features.shape}")
        elif getattr(args, 'segmentation_head', False):
            outputs = model(input_tensors_test)
            dets = outputs['pred_boxes']
            labels = outputs['pred_logits']
            masks = outputs['pred_masks']
            print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, Masks: {masks.shape}")
        else:
            outputs = model(input_tensors_test)
            dets = outputs['pred_boxes']
            labels = outputs['pred_logits']
            print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
    
    # Move model and inputs to export device for ONNX export
    # If export device is already the same as inference device, this is a no-op
    model.to(export_device)
    input_tensors = input_tensors.to(export_device)

    # Create output directory if it doesn't exist
    output_dir = getattr(args, 'output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Export to ONNX
    backbone_only = getattr(args, 'backbone_only', False)
    verbose = getattr(args, 'verbose', False)
    opset_version = getattr(args, 'opset_version', 17)
    
    output_file = export_onnx(
        output_dir,
        model,
        input_names,
        input_tensors,
        output_names,
        dynamic_axes,
        backbone_only=backbone_only,
        verbose=verbose,
        opset_version=opset_version
    )
    
    if getattr(args, 'simplify', False):
        force = getattr(args, 'force', False)
        output_file = onnx_simplify(output_file, input_names, input_tensors, force=force)

    # Print model statistics
    tensorrt_path = None
    if getattr(args, 'tensorrt', False):
        trt_stats = trtexec(output_file, args)
        tensorrt_path = output_file.replace(".onnx", ".engine")
    else:
        trt_stats = None
    
    # Print model statistics comparison
    print_model_statistics(model, args.resume, output_file, tensorrt_path)
    
    # Benchmark PyTorch inference
    print("\nBenchmarking PyTorch model...")
    pytorch_stats = None
    if (inference_device.type == 'cuda' or export_device.type == 'cuda') and torch.cuda.is_available():
        try:
            pytorch_stats = benchmark_inference(model, input_tensors, inference_device)
        except Exception as e:
            print(f"PyTorch benchmark failed: {e}")
    elif export_device.type == 'cpu':
        try:
            pytorch_stats = benchmark_inference(model, input_tensors, export_device)
        except Exception as e:
            print(f"PyTorch benchmark failed: {e}")

    # Benchmark ONNX inference
    print("\nBenchmarking ONNX model...")
    onnx_stats = None
    try:
        onnx_stats = benchmark_onnx(output_file, input_tensors)
    except Exception as e:
        print(f"ONNX benchmark failed: {e}")

    # Print performance comparison
    print_performance_comparison(pytorch_stats, onnx_stats, trt_stats)