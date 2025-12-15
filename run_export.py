#!/usr/bin/env python3
"""
Simple script to run the export.py script for converting rf-detr-nano.pth to TensorRT

Usage:
    python run_export.py --resume rf-detr-nano.pth --output_dir output --tensorrt
"""

import argparse
from types import SimpleNamespace
import sys
import os

# Add current directory to path to import export
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the export script's main function
try:
    from export import main
    import rfdetr.util.misc as utils
    from rfdetr.models import build_model
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure rfdetr is installed: pip install rfdetr")
    sys.exit(1)


def get_args_parser():
    """Create a minimal argument parser for export"""
    parser = argparse.ArgumentParser(description='Export RF-DETR model to ONNX and TensorRT')
    
    # Required arguments
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to checkpoint file (e.g., rf-detr-nano.pth)')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=90,
                        help='Number of classes (default: 90)')
    parser.add_argument('--encoder', type=str, default='dinov2_windowed_small',
                        help='Encoder type (default: dinov2_windowed_small for RF-DETR Nano)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--dec_layers', type=int, default=2,
                        help='Number of decoder layers (default: 2 for RF-DETR Nano)')
    parser.add_argument('--num_queries', type=int, default=300,
                        help='Number of queries (default: 300)')
    
    # Export arguments
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for exported models (default: output)')
    parser.add_argument('--infer_dir', type=str, default=None,
                        help='Path to inference image (optional, uses dummy if not provided)')
    parser.add_argument('--shape', type=int, nargs=2, default=[384, 384],
                        help='Input shape [width height] (default: 384 384 for RF-DETR Nano)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for export (default: 1)')
    
    # Export options
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model')
    parser.add_argument('--tensorrt', action='store_true',
                        help='Convert to TensorRT engine')
    parser.add_argument('--backbone_only', action='store_true',
                        help='Export backbone only')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--opset_version', type=int, default=17,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing files')
    parser.add_argument('--profile', action='store_true',
                        help='Run profiling during TensorRT export')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run (print commands only)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--layer_norm', action='store_true',
                        help='Use layer norm (check for BatchNorm)')
    
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("RF-DETR Model Export to ONNX and TensorRT")
    print("=" * 60)
    print(f"Checkpoint: {args.resume}")
    print(f"Output directory: {args.output_dir}")
    print(f"Input shape: {args.shape}")
    if args.simplify:
        print("✓ ONNX simplification enabled")
    if args.tensorrt:
        print("✓ TensorRT conversion enabled")
    print("=" * 60)
    print()
    
    # Convert args to a namespace object that works with the export script
    # The export script expects certain attributes, so we create a compatible object
    export_args = SimpleNamespace()
    
    # Copy all attributes from parsed args
    for key, value in vars(args).items():
        setattr(export_args, key, value)
    
    # Set any additional required attributes with defaults (RF-DETR Nano config)
    export_args.segmentation_head = False
    export_args.encoder_only = False
    export_args.position_embedding = 'sine'
    export_args.out_feature_indexes = [3, 6, 9, 12]  # RF-DETR Nano
    export_args.freeze_encoder = False
    export_args.dim_feedforward = 2048
    export_args.sa_nheads = 8
    export_args.ca_nheads = 16  # RF-DETR Nano uses 16
    export_args.group_detr = 13
    export_args.two_stage = True  # RF-DETR Nano uses two_stage
    export_args.projector_scale = ['P4']  # Must be a list for RF-DETR
    export_args.lite_refpoint_refine = True  # RF-DETR Nano uses this
    export_args.num_select = 300
    export_args.dec_n_points = 2  # RF-DETR Nano uses 2
    export_args.decoder_norm = 'LN'
    export_args.bbox_reparam = True  # RF-DETR Nano uses this
    export_args.vit_encoder_num_layers = 12
    export_args.window_block_indexes = None
    export_args.use_cls_token = False
    export_args.rms_norm = False
    export_args.backbone_lora = False
    export_args.force_no_pretrain = False
    
    # Missing attributes required by build_model (RF-DETR Nano specific)
    export_args.pretrained_encoder = None
    export_args.drop_path = 0
    export_args.dropout = 0
    export_args.gradient_checkpointing = False
    export_args.pretrain_weights = None
    export_args.patch_size = 16  # RF-DETR Nano uses 16
    export_args.num_windows = 2  # RF-DETR Nano uses 2
    export_args.positional_encoding_size = 24  # RF-DETR Nano uses 24
    export_args.aux_loss = True
    export_args.mask_downsample_ratio = 4
    export_args.layer_norm = True  # RF-DETR Nano uses layer_norm
    
    try:
        main(export_args)
        print("\n" + "=" * 60)
        print("Export completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

