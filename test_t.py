import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
import os
import sys
from pathlib import Path
import time
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd

# Add YOLOv5 to path
YOLOV5_PATH = r"E:\machine l\Falcondet_CSPDarknet_backbon_C\yolov5"
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Try importing YOLOv5 modules
try:
    from models.common import *
    from models.yolo import Model
    from utils.general import check_yaml, check_file, check_dataset, colorstr, increment_path
    from utils.torch_utils import select_device, time_sync
    from utils.dataloaders import create_dataloader  # Update this line
    from utils.metrics import ap_per_class, ConfusionMatrix
    from val import run as val_run

    print("✓ YOLOv5 modules imported successfully")
except ImportError as e:
    print(f"✗ YOLOv5 import error: {e}")
    print("Please make sure the YOLOv5 repository is cloned and the path is correct:")
    print(f"    {YOLOV5_PATH}")
    sys.exit(1)



# Your MicroFusionNet Components
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=[2, 3])
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)


class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=2):
        super(GhostConv, self).__init__()
        hidden_channels = out_channels // ratio
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        self.ghost_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels - hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels - hidden_channels),
            nn.ReLU()
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        ghost = self.ghost_conv(primary)
        return torch.cat([primary, ghost], dim=1)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6, stride=1):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv2(out))
        if self.use_res_connect:
            return x + out
        return out


# MicroFusionNet Backbone for YOLOv5
class MicroFusionBackbone(nn.Module):
    def __init__(self, ch_out=(256, 512, 1024)):
        super().__init__()
        self.ch_out = ch_out

        # Stem
        self.stem = Conv(3, 32, 6, 2, 2)  # P1/2

        # Stage 1: P1/2 -> P2/4
        self.stage1 = nn.Sequential(
            Conv(32, 64, 3, 2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # Stage 2: P2/4 -> P3/8
        self.stage2 = nn.Sequential(
            Conv(64, 128, 3, 2),
            ResidualBlock(128, 128, use_se=True),
            ResidualBlock(128, 128, use_se=True),
            ResidualBlock(128, 128, use_se=True)
        )

        # Stage 3: P3/8 -> P4/16
        self.stage3 = nn.Sequential(
            GhostConv(128, 256, stride=2),
            ResidualBlock(256, 256, use_se=True),
            ResidualBlock(256, 256, use_se=True),
            ResidualBlock(256, 256, use_se=True),
            ResidualBlock(256, 256, use_se=True),
            ResidualBlock(256, 256, use_se=True)
        )

        # Stage 4: P4/16 -> P5/32
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, use_se=True),
            ResidualBlock(512, 512, use_se=True),
            ResidualBlock(512, 512, use_se=True)
        )

        # Additional processing for P5
        self.stage5 = nn.Sequential(
            ResidualBlock(512, 1024, use_se=True),
            ResidualBlock(1024, 1024, use_se=True)
        )

    def forward(self, x):
        x = self.stem(x)  # P1/2
        x = self.stage1(x)  # P2/4, 64 channels

        x3 = self.stage2(x)  # P3/8, 128 channels
        x4 = self.stage3(x3)  # P4/16, 256 channels
        x5 = self.stage4(x4)  # P5/32, 512 channels
        x5 = self.stage5(x5)  # P5/32, 1024 channels

        # Match YOLOv5 expected output channels
        return [x3, x4, x5]  # [128, 256, 1024] channels


# Custom YOLOv5 Configuration Generator
def create_microfusion_config():
    """Create YOLOv5 config with MicroFusionNet backbone"""
    config = {
        'nc': 80,  # number of classes
        'depth_multiple': 1.0,
        'width_multiple': 1.0,
        'anchors': [
            [10, 13, 16, 30, 33, 23],  # P3/8
            [30, 61, 62, 45, 59, 119],  # P4/16
            [116, 90, 156, 198, 373, 326]  # P5/32
        ],
        'backbone': [
            # MicroFusionNet will be inserted here programmatically
        ],
        'head': [
            [[-1, 1, Conv, [512, 1, 1]],
             [-1, 1, nn.Upsample, [None, 2, 'nearest']],
             [[-1, 6], 1, Concat, [1]],  # cat backbone P4
             [-1, 3, C3, [512, False]],  # 13

             [-1, 1, Conv, [256, 1, 1]],
             [-1, 1, nn.Upsample, [None, 2, 'nearest']],
             [[-1, 4], 1, Concat, [1]],  # cat backbone P3
             [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

             [-1, 1, Conv, [256, 3, 2]],
             [[-1, 14], 1, Concat, [1]],  # cat head P4
             [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

             [-1, 1, Conv, [512, 3, 2]],
             [[-1, 10], 1, Concat, [1]],  # cat head P5
             [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

             [[17, 20, 23], 1, 'Detect', ['nc', 'anchors']],  # Detect(P3, P4, P5)
             ]
        ]
    }
    return config


class ModelComparator:
    def __init__(self, device='cuda'):
        self.device = select_device(device)
        self.results = {}

    def create_models(self, data_dict):
        """Create both MicroFusionNet and CSPDarknet YOLOv5 models"""
        models = {}

        # Original YOLOv5s model (CSPDarknet)
        print("Creating YOLOv5s (CSPDarknet) model...")
        models['yolov5s_csp'] = Model('models/yolov5s.yaml', ch=3, nc=data_dict['nc']).to(self.device)

        # MicroFusionNet YOLOv5 model
        print("Creating YOLOv5 with MicroFusionNet backbone...")
        models['yolov5s_mf'] = self.create_microfusion_model(data_dict['nc'])

        return models

    def create_microfusion_model(self, nc):
        """Create YOLOv5 model with MicroFusionNet backbone"""
        # Create a custom model by replacing the backbone
        model = Model('models/yolov5s.yaml', ch=3, nc=nc).to(self.device)

        # Replace backbone with MicroFusionNet
        backbone = MicroFusionBackbone()

        # We need to integrate this properly with YOLOv5's architecture
        # This is a simplified approach - in practice, you'd modify the yaml config
        model.model[0] = backbone  # Replace the backbone

        return model

    def count_parameters(self, model):
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def measure_inference_speed(self, model, imgsz=640, batch_size=1, runs=100):
        """Measure inference speed"""
        model.eval()
        img = torch.zeros((batch_size, 3, imgsz, imgsz), device=self.device)

        # Warmup
        for _ in range(10):
            _ = model(img)

        # Measure
        times = []
        for _ in range(runs):
            t0 = time_sync()
            _ = model(img)
            times.append(time_sync() - t0)

        return np.mean(times), np.std(times)

    def measure_memory_usage(self, model, imgsz=640):
        """Measure GPU memory usage"""
        if self.device.type != 'cuda':
            return None, None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.eval()
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)

        with torch.no_grad():
            _ = model(img)

        allocated = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB

        return allocated, peak

    def run_validation(self, model, data_dict, imgsz=640, batch_size=32):
        """Run validation on COCO dataset"""
        try:
            # Create validation dataloader
            val_loader = create_dataloader(
                data_dict['val'], imgsz, batch_size, stride=32,
                single_cls=False, pad=0.0, rect=True, workers=8,
                prefix=colorstr('val: ')
            )[0]

            # Run validation
            results = val_run(
                data=data_dict,
                weights=None,
                batch_size=batch_size,
                imgsz=imgsz,
                model=model,
                dataloader=val_loader,
                save_dir=Path('./runs/val'),
                plots=False,
                verbose=True
            )

            return results
        except Exception as e:
            print(f"Validation error: {e}")
            return None

    def run_comprehensive_comparison(self, data_path='data/coco.yaml'):
        """Run comprehensive comparison between models"""
        print("=" * 80)
        print("MicroFusionNet vs CSPDarknet YOLOv5 Comprehensive Comparison")
        print("=" * 80)

        # Load dataset configuration
        data_dict = check_dataset(data_path)
        print(f"Dataset: {data_dict['path']}")
        print(f"Classes: {data_dict['nc']}")

        # Create models
        models = self.create_models(data_dict)

        results = {}

        for name, model in models.items():
            print(f"\n{'=' * 20} {name.upper()} {'=' * 20}")
            results[name] = {}

            # 1. Parameter count
            total_params, trainable_params = self.count_parameters(model)
            results[name]['total_params'] = total_params
            results[name]['trainable_params'] = trainable_params
            print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

            # 2. Model size (MB)
            model_size = total_params * 4 / (1024 ** 2)  # Assuming float32
            results[name]['model_size_mb'] = model_size
            print(f"Model size: {model_size:.2f} MB")

            # 3. Inference speed
            mean_time, std_time = self.measure_inference_speed(model)
            fps = 1.0 / mean_time
            results[name]['inference_time'] = mean_time
            results[name]['fps'] = fps
            print(f"Inference: {mean_time * 1000:.2f}±{std_time * 1000:.2f}ms ({fps:.1f} FPS)")

            # 4. Memory usage
            if self.device.type == 'cuda':
                allocated, peak = self.measure_memory_usage(model)
                results[name]['memory_allocated'] = allocated
                results[name]['memory_peak'] = peak
                print(f"Memory: {allocated:.1f}MB allocated, {peak:.1f}MB peak")

            # 5. Validation on dataset (if available)
            print("Running validation...")
            val_results = self.run_validation(model, data_dict)
            if val_results:
                results[name]['map50'] = val_results[2]  # mAP@0.5
                results[name]['map50_95'] = val_results[3]  # mAP@0.5:0.95
                print(f"mAP@0.5: {val_results[2]:.3f}")
                print(f"mAP@0.5:0.95: {val_results[3]:.3f}")

        # Generate comparison report
        self.generate_report(results)
        return results

    def generate_report(self, results):
        """Generate detailed comparison report"""
        print("\n" + "=" * 80)
        print("DETAILED COMPARISON REPORT")
        print("=" * 80)

        # Create comparison table
        df_data = []
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Parameters (M)': f"{metrics.get('total_params', 0) / 1e6:.2f}",
                'Size (MB)': f"{metrics.get('model_size_mb', 0):.1f}",
                'FPS': f"{metrics.get('fps', 0):.1f}",
                'Memory (MB)': f"{metrics.get('memory_allocated', 0):.1f}",
                'mAP@0.5': f"{metrics.get('map50', 0):.3f}",
                'mAP@0.5:0.95': f"{metrics.get('map50_95', 0):.3f}"
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))

        # Save results
        os.makedirs('comparison_results', exist_ok=True)
        df.to_csv('comparison_results/model_comparison.csv', index=False)

        # Performance comparison
        if len(results) == 2:
            models = list(results.keys())
            mf_results = results[models[1]]  # MicroFusionNet
            csp_results = results[models[0]]  # CSPDarknet

            print(f"\n{'=' * 40}")
            print("PERFORMANCE COMPARISON SUMMARY")
            print(f"{'=' * 40}")

            # Parameter efficiency
            param_ratio = mf_results['total_params'] / csp_results['total_params']
            print(f"Parameter Ratio (MF/CSP): {param_ratio:.2f}x")

            # Speed comparison
            speed_ratio = mf_results['fps'] / csp_results['fps']
            print(f"Speed Ratio (MF/CSP): {speed_ratio:.2f}x")

            # Accuracy comparison
            if 'map50_95' in mf_results and 'map50_95' in csp_results:
                acc_diff = mf_results['map50_95'] - csp_results['map50_95']
                print(f"Accuracy Difference (mAP@0.5:0.95): {acc_diff:+.3f}")

            # Memory efficiency
            if 'memory_allocated' in mf_results and 'memory_allocated' in csp_results:
                mem_ratio = mf_results['memory_allocated'] / csp_results['memory_allocated']
                print(f"Memory Ratio (MF/CSP): {mem_ratio:.2f}x")

            print(f"\nResults saved to: comparison_results/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')

    args = parser.parse_args()

    # Initialize comparator
    comparator = ModelComparator(device=args.device)

    # Run comprehensive comparison
    results = comparator.run_comprehensive_comparison(data_path=args.data)

    print("\n✓ Comparison completed successfully!")
    print("Check 'comparison_results/' folder for detailed reports.")


if __name__ == '__main__':
    main()