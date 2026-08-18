#!/usr/bin/env python3
"""
Aletheia - Image Restoration Inference Script
=============================================

Loads the trained 2x super-resolution / restoration model and runs inference
on every ``.npy`` file in the input directory, writing one restored ``.npy``
file per input into the output directory.

Usage:
    python run.py <input-dir> <output-dir>

Example:
    python run.py ./Test_NoisyLR ./restored_outputs

Outputs are grayscale arrays of shape (H, W), values clipped to [0, 1],
with no NaN or Inf values.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MODEL CONFIGURATION
# (must match the values used during training)
# ============================================================

IN_CHANNELS = 1
OUT_CHANNELS = 1
SCALE = 2
WINDOW_SIZE = 8
BASE_CHANNELS = 96
NUM_BLOCKS = 10
ESTIMATOR_CHANNELS = 96
HF_BLOCKS = 12
NUM_HEADS = 2
MLP_RATIO = 2.0
CA_REDUCTION = 8
EPS = 1e-6
SPECKLE_FLOOR = 0.05



# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        return x + self.scale * self.body(x)


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg + mx)
        return x * attention


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H // window_size * W // window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        relative_size = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_size * relative_size, num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer(
            "relative_position_index", relative_position_index
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attention = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size,
               self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attention = attention.view(B_ // nW, nW, self.num_heads, N, N)
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, N, N)

        attention = F.softmax(attention, dim=-1)
        output = attention @ v
        output = output.transpose(1, 2).reshape(B_, N, C)
        return self.proj(output)


class MLP(nn.Module):

    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwinBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=8,
                 shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def calculate_mask(self, H, W, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = h_slices
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        return attn_mask.masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(H, W, x.device)
        else:
            attn_mask = None

        attn_windows = self.attn(x_windows, attn_mask)
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        x = x + shortcut.permute(0, 2, 3, 1)
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2).contiguous()


class NoiseAwareHybridBlock(nn.Module):

    def __init__(self, channels, num_heads, window_size,
                 shift_size, mlp_ratio, reduction):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.channel_attention = ChannelAttention(channels, reduction)
        self.noise_projection = nn.Sequential(
            nn.Conv2d(2, channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.swin = SwinBlock(channels, num_heads, window_size,
                              shift_size, mlp_ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, x, noise_context):
        residual = x
        local = self.local(x)
        local = self.channel_attention(local)
        gate = self.noise_projection(noise_context)
        local = local * (1.0 + gate)
        global_features = self.swin(local)
        fused = self.fusion(torch.cat([local, global_features], dim=1))
        return residual + self.res_scale * fused


class SpeckleEstimator(nn.Module):

    def __init__(self, in_channels=1, channels=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x) + EPS


class AdditiveNoiseEstimator(nn.Module):

    def __init__(self, in_channels=1, channels=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class HighFrequencyBranch(nn.Module):

    def __init__(self, in_channels, channels):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(HF_BLOCKS)]
        )
        self.highpass = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        x = self.input(x)
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        high = self.highpass(high)
        high = self.body(high)
        high = self.refine(high)
        return self.scale * high


class MultiplicativeAdditiveHybridSwinSR(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, channels=96,
                 blocks=10, scale=2):
        super().__init__()
        self.scale = scale
        self.speckle_estimator = SpeckleEstimator(in_channels, ESTIMATOR_CHANNELS)
        self.noise_estimator = AdditiveNoiseEstimator(in_channels, ESTIMATOR_CHANNELS)

        self.head = nn.Conv2d(in_channels * 3, channels, 3, 1, 1)

        self.body = nn.ModuleList()
        for i in range(blocks):
            self.body.append(
                NoiseAwareHybridBlock(
                    channels=channels,
                    num_heads=NUM_HEADS,
                    window_size=WINDOW_SIZE,
                    shift_size=(0 if i % 2 == 0 else WINDOW_SIZE // 2),
                    mlp_ratio=MLP_RATIO,
                    reduction=CA_REDUCTION,
                )
            )

        self.body_tail = nn.Conv2d(channels, channels, 3, 1, 1)

        self.hf_branch = HighFrequencyBranch(in_channels * 3, channels)
        self.hf_attention = ChannelAttention(channels, CA_REDUCTION)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

        self.upsample_conv = nn.Conv2d(
            channels, channels * scale * scale, 3, 1, 1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.upsample_activation = nn.LeakyReLU(0.1, inplace=True)

        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, out_channels, 3, 1, 1),
        )

        self.corrected_upsample_conv = nn.Conv2d(
            in_channels, out_channels * scale * scale, 3, 1, 1
        )
        self.corrected_pixel_shuffle = nn.PixelShuffle(scale)
        self.corrected_upsample_activation = nn.LeakyReLU(0.1, inplace=True)
        self.corrected_refine = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.output_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, lr):
        speckle = self.speckle_estimator(lr)
        noise = self.noise_estimator(lr)

        corrected = (lr - noise) / speckle
        residual = lr - corrected

        main_input = torch.cat([lr, corrected, residual], dim=1)
        main_features = self.head(main_input)

        noise_context = torch.cat([noise, speckle], dim=1)
        body_features = main_features
        for block in self.body:
            body_features = block(body_features, noise_context)
        main_features = main_features + self.body_tail(body_features)

        hf_input = torch.cat([lr, noise, speckle], dim=1)
        hf_features = self.hf_branch(hf_input)
        hf_features = self.hf_attention(hf_features)

        main_features = self.fusion(
            torch.cat([main_features, hf_features], dim=1)
        )

        x = self.upsample_conv(main_features)
        x = self.pixel_shuffle(x)
        x = self.upsample_activation(x)
        sr_residual = self.reconstruction(x)

        corrected_base = self.corrected_upsample_conv(corrected)
        corrected_base = self.corrected_pixel_shuffle(corrected_base)
        corrected_base = self.corrected_upsample_activation(corrected_base)
        corrected_base = self.corrected_refine(corrected_base)


        sr = corrected_base + self.output_scale * sr_residual
        return sr


# ============================================================
# INFERENCE HELPERS
# ============================================================

def load_npy(path):
    """Load a .npy file into a 1 x H x W float tensor."""
    x = np.load(path).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    if x.ndim == 2:
        x = x[None]
    elif x.ndim == 3:
        if x.shape[0] not in [1, 3] and x.shape[-1] in [1, 3]:
            x = np.transpose(x, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported shape {x.shape}: {path}")

    return torch.from_numpy(x).float()


def fix_channels(x, channels):
    """Convert an input tensor to the expected channel count."""
    if x.shape[0] == channels:
        return x
    if x.shape[0] == 3 and channels == 1:
        return x.mean(dim=0, keepdim=True)
    if x.shape[0] == 1 and channels == 3:
        return x.repeat(3, 1, 1)
    raise ValueError(
        f"Cannot convert {x.shape[0]} channels to {channels} channels"
    )


@torch.no_grad()
def restore_image(model, lr_path, device, amp):
    """Run the model on a single LR image and return a 2D (H, W) array."""
    lr = load_npy(lr_path)


    lr = fix_channels(lr, IN_CHANNELS)
    lr_input = lr.unsqueeze(0).to(device, non_blocking=True)

    with torch.amp.autocast("cuda", enabled=amp):
        sr = model(lr_input)

    sr = sr[0].cpu().numpy()
    sr = np.squeeze(sr)
    sr = np.nan_to_num(sr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(sr, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Aletheia image restoration inference."
    )
    parser.add_argument("input_dir", help="Path to directory of .npy test images")
    parser.add_argument("output_dir", help="Path to directory for restored outputs")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model weights (default: models/model.pth next to run.py)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = args.checkpoint or os.path.join(script_dir, "models", "model.pth")

    if not os.path.isdir(args.input_dir):
        sys.exit(f"ERROR: input directory not found: {args.input_dir}")
    if not os.path.isfile(checkpoint_path):
        sys.exit(f"ERROR: model weights not found: {checkpoint_path}")

    files = sorted(
        f for f in os.listdir(args.input_dir) if f.lower().endswith(".npy")
    )
    if len(files) == 0:
        sys.exit(f"ERROR: no .npy files found in: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    model = MultiplicativeAdditiveHybridSwinSR(
        IN_CHANNELS, OUT_CHANNELS, BASE_CHANNELS, NUM_BLOCKS, scale=SCALE
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    cleaned = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        sys.exit(f"ERROR: missing weights: {missing}")
    if unexpected:
        print(f"WARNING: unexpected weights ignored: {len(unexpected)} keys")

    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("=" * 60)
    print("Aletheia Image Restoration Inference")
    print("=" * 60)
    print(f"Input directory : {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Device          : {device}")
    print(f"Images found    : {len(files)}")
    print("=" * 60)

    for i, filename in enumerate(files, start=1):
        in_path = os.path.join(args.input_dir, filename)
        out_path = os.path.join(args.output_dir, filename)
        try:
            restored = restore_image(model, in_path, device, amp)
            np.save(out_path, restored)
        except Exception as exc:
            print(f"  ERROR {filename}: {exc}")
            continue
        if i % 50 == 0 or i == len(files):
            print(f"  Processed {i}/{len(files)}")

    print("=" * 60)
    print("Inference complete. Restored images written to:")
    print(f"  {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
