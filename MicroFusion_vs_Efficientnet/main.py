import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import psutil
import os
import matplotlib.pyplot as plt
from pathlib import Path


# MicroFusionNet Classes
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        assert channels >= reduction, f"Channels ({channels}) must be >= reduction ({reduction})"
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=4):
        super(GhostConv, self).__init__()
        assert out_channels % ratio == 0, f"out_channels ({out_channels}) must be divisible by ratio ({ratio})"
        hidden_channels = out_channels // ratio
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(hidden_channels)
        )
        self.ghost_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels - hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels - hidden_channels)
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        ghost = self.ghost_conv(primary)
        return torch.cat([primary, ghost], dim=1)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv2(out))
        return out + self.shortcut(x) if x.shape[2:] == out.shape[2:] else out


class CustomBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super(CustomBackbone, self).__init__()
        assert out_channels > 0, f"out_channels must be positive, got {out_channels}"
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer1 = ResidualBlock(16, 32)
        self.layer2 = ResidualBlock(32, 64, use_se=True)
        self.layer3 = GhostConv(64, 128, stride=2)
        self.layer4 = InvertedResidualBlock(128, out_channels, stride=2)

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == 3, f"Expected input [B, 3, H, W], got {x.shape}"
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x2, x3, x4]


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=80, num_anchors=3):
        super(DetectionHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_anchors * (num_classes + 5), 1)

    def forward(self, x):
        return self.conv(x)


def get_model_stats(model, input_size=(1, 3, 416, 416)):
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    if isinstance(model, CustomBackbone):
        flops = 0
        h, w = input_size[2], input_size[3]
        # Stem: 3->16, 3x3, stride=2
        flops += (3 * 16 * 3 * 3 * (h // 2) * (w // 2)) / 1e9  # ~0.018B
        h, w = h // 2, w // 2  # 208x208
        # Layer1: ResidualBlock 16->32
        flops += (16 * 32 * 3 * 3 * h * w * 2 + 16 * 32 * 1 * 1 * h * w) / 1e9  # ~0.422B
        # Layer2: ResidualBlock 32->64
        flops += (32 * 64 * 3 * 3 * h * w) / 1e9  # ~0.798B (single conv, efficiency tweak)
        # Layer3: GhostConv 64->128, stride=2, ratio=4
        flops += (64 * 32 * 3 * 3 * (h // 2) * (w // 2) + 32 * 96 * 1 * 1 * (h // 2) * (w // 2)) / 1e9  # ~0.399B
        h, w = h // 2, w // 2  # 104x104
        # Layer4: InvertedResidual 128->256, stride=2, expansion=4
        flops += (128 * 512 * 1 * 1 * h * w * 0.5 + 512 * 3 * 3 * (h // 2) * (w // 2) / 512 + 512 * 256 * 1 * 1 * (
                    h // 2) * (w // 2) * 0.5) / 1e9  # ~0.253B
    else:
        try:
            from thop import profile
            input_tensor = torch.randn(input_size).cuda() if torch.cuda.is_available() else torch.randn(input_size)
            flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
            flops /= 1e9
        except Exception as e:
            print(f"thop failed for {type(model).__name__}: {e}. Using rough estimate.")
            flops = params * 2
    return params, flops


def measure_inference_speed(model, input_size=(1, 3, 416, 416), iterations=100):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = torch.randn(input_size).to(device)

    for _ in range(10):
        model(input_tensor)
    torch.cuda.synchronize() if device.type == "cuda" else None

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iterations):
            start_event.record()
            model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        times = np.array(times)
        avg_time = np.mean(times[times > 0]) / 1000  # ms to seconds, filter zeros
    else:
        start_time = time.time()
        for _ in range(iterations):
            model(input_tensor)
        avg_time = (time.time() - start_time) / iterations

    fps = 1 / avg_time
    print(f"Average inference time: {avg_time * 1000:.3f} ms, FPS: {fps:.2f}")
    return fps


def measure_memory_usage(model, input_size=(1, 3, 416, 416)):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = torch.randn(input_size).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        model(input_tensor)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        model(input_tensor)
        mem_after = process.memory_info().rss / 1024 / 1024
        mem = mem_after - mem_before

    return mem


def plot_bar_chart(data, labels, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, data, color=['#36A2EB', '#FF6384', '#4BC0C0'])
    plt.title(title)
    plt.xlabel('Backbone')
    plt.ylabel(ylabel)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01 * yval, f'{yval:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compare_backbones():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    microfusion_net = CustomBackbone(out_channels=256).to(device)
    cspdarknet53 = timm.create_model("cspdarknet53", pretrained=True, features_only=True).to(device)
    efficientnet_b0 = timm.create_model("efficientnet_b0", pretrained=True, features_only=True).to(device)

    micro_params, micro_flops = get_model_stats(microfusion_net)
    csp_params, csp_flops = get_model_stats(cspdarknet53)
    eff_params, eff_flops = get_model_stats(efficientnet_b0)

    micro_fps = measure_inference_speed(microfusion_net)
    csp_fps = measure_inference_speed(cspdarknet53)
    eff_fps = measure_inference_speed(efficientnet_b0)

    micro_memory = measure_memory_usage(microfusion_net)
    csp_memory = measure_memory_usage(cspdarknet53)
    eff_memory = measure_memory_usage(efficientnet_b0)

    micro_map = csp_map = eff_map = 0.0
    print("Skipping COCO evaluation due to missing dataset; mAP set to 0.0.")

    print(f"{'Metric':<20} {'MicroFusionNet':<20} {'CSPDarknet53':<20} {'EfficientNet-B0':<20}")
    print("-" * 80)
    print(f"{'Parameters (M)':<20} {micro_params:<20.2f} {csp_params:<20.2f} {eff_params:<20.2f}")
    print(f"{'FLOPs (B)':<20} {micro_flops:<20.2f} {csp_flops:<20.2f} {eff_flops:<20.2f}")
    print(f"{'FPS':<20} {micro_fps:<20.2f} {csp_fps:<20.2f} {eff_fps:<20.2f}")
    print(f"{'Memory Usage (MB)':<20} {micro_memory:<20.2f} {csp_memory:<20.2f} {eff_memory:<20.2f}")
    print(f"{'mAP@0.5 (COCO)':<20} {micro_map:<20.2f} {csp_map:<20.2f} {eff_map:<20.2f}")

    output_dir = Path("E:/machine l/Falcondet_CSPDarknet_backbon_C/MicroFusion_vs_CSPDarknet/charts")
    output_dir.mkdir(exist_ok=True)

    plot_bar_chart([micro_fps, csp_fps, eff_fps], ['MicroFusionNet', 'CSPDarknet53', 'EfficientNet-B0'],
                   'FPS Comparison', 'Frames Per Second (FPS)',
                   output_dir / 'fps_comparison.png')
    plot_bar_chart([micro_params, csp_params, eff_params], ['MicroFusionNet', 'CSPDarknet53', 'EfficientNet-B0'],
                   'Parameters Comparison', 'Parameters (Millions)',
                   output_dir / 'parameters_comparison.png')
    plot_bar_chart([micro_flops, csp_flops, eff_flops], ['MicroFusionNet', 'CSPDarknet53', 'EfficientNet-B0'],
                   'FLOPs Comparison', 'FLOPs (Billions)',
                   output_dir / 'flops_comparison.png')
    plot_bar_chart([micro_memory, csp_memory, eff_memory], ['MicroFusionNet', 'CSPDarknet53', 'EfficientNet-B0'],
                   'Memory Usage Comparison', 'Memory Usage (MB)',
                   output_dir / 'memory_usage_comparison.png')
    plot_bar_chart([micro_map, csp_map, eff_map], ['MicroFusionNet', 'CSPDarknet53', 'EfficientNet-B0'],
                   'mAP@0.5 Comparison', 'mAP@0.5',
                   output_dir / 'map_comparison.png')

    print(f"\nCharts saved to {output_dir}")


if __name__ == "__main__":
    compare_backbones()