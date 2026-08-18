#!/usr/bin/env python3
"""
Aletheia - Training Script
==========================

Reproduces the training process of the Aletheia image-restoration model.

The script expects the dataset at:

    train/NoisyLR/   low-resolution noisy inputs (128x128 .npy)
    train/GT/        high-resolution clean targets (256x256 .npy)

Checkpoints are written to ./checkpoints/.

Dependencies (see requirements.txt):
    pip install torch torchvision numpy scikit-image lpips tqdm

Note: training uses torchvision VGG19 (IMAGENET1K_V1) and LPIPS (AlexNet)
weights, which are downloaded automatically on first use.
"""

import os
import glob
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import lpips

from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm


# =========

# ============================================================
# MODEL CONFIGURATION
# ============================================================

IN_CHANNELS = 1
OUT_CHANNELS = 1

SCALE = 2

WINDOW_SIZE = 8

TRAIN_PATCH_SIZE = 128

# TOTAL number of epochs
EPOCHS = 60

BATCH_SIZE = 1
NUM_WORKERS = 0

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

VAL_RATIO = 0.1
SEED = 42

BASE_CHANNELS = 96
NUM_BLOCKS = 10

ESTIMATOR_CHANNELS = 96
HF_BLOCKS = 12

NUM_HEADS = 2
MLP_RATIO = 2.0
CA_REDUCTION = 8

LAMBDA_SR = 1.0
LAMBDA_SPECKLE = 0.05
LAMBDA_NOISE = 0.05
LAMBDA_GRAD = 0.10
LAMBDA_LAP = 0.
LAMBDA_EDGE = 0.
LAMBDA_DEG = 0.
LAMBDA_VGG = 0.10


# ============================================================
# AUGMENTATION
# ============================================================

AUGMENT = True

AUG_SPECKLE_PROB = 0.35
AUG_SPECKLE_STD = 0.04

AUG_NOISE_PROB = 0.35
AUG_NOISE_STD = 0.015

AUG_GAIN_PROB = 0.25
AUG_GAIN_MIN = 0.97
AUG_GAIN_MAX = 1.03

CUTBLUR_PROB = 0.5
CUTBLUR_RATIO_MIN = 0.3
CUTBLUR_RATIO_MAX = 0.7

EPS = 1e-6

VALID_RANGE_MIN = 0.0
VALID_RANGE_MAX = 1.0

SPECKLE_FLOOR = 0.05

# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

AMP = DEVICE.type == "cuda"

GPU_COUNT = (
    torch.cuda.device_count()
    if torch.cuda.is_available()
    else 0
)


# ============================================================
# SEED
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = True



TRAIN_LR_DIR = "./train/NoisyLR"
TRAIN_HR_DIR = "./train/GT"


# ============================================================
# LOCAL STORAGE
# ============================================================

CHECKPOINT_DIR = "./checkpoints"
SAMPLE_DIR = "./samples"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


# ============================================================
# CHECKPOINT PATHS
# ============================================================

latest_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "latest.pth"
)

best_psnr_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "best_psnr.pth"
)

best_ssim_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "best_ssim.pth"
)

best_lpips_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "best_lpips.pth"
)

best_checkpoint = best_psnr_checkpoint


# ============================================================
# VERIFY DATASET
# ============================================================

print()
print("=" * 70)
print("VERIFYING DATASET")
print("=" * 70)

print()
print("LR directory:")
print(f"  {os.path.abspath(TRAIN_LR_DIR)}")

print()
print("HR directory:")
print(f"  {os.path.abspath(TRAIN_HR_DIR)}")


if not os.path.isdir(TRAIN_LR_DIR):
    raise FileNotFoundError(
        f"\nLR directory not found:\n"
        f"  {os.path.abspath(TRAIN_LR_DIR)}\n\n"
        f"Expected: ./train/NoisyLR"
    )


if not os.path.isdir(TRAIN_HR_DIR):
    raise FileNotFoundError(
        f"\nHR directory not found:\n"
        f"  {os.path.abspath(TRAIN_HR_DIR)}\n\n"
        f"Expected: ./train/GT"
    )


lr_check_files = glob.glob(
    os.path.join(TRAIN_LR_DIR, "*.npy")
)

hr_check_files = glob.glob(
    os.path.join(TRAIN_HR_DIR, "*.npy")
)

print()
print(f"LR .npy files found: {len(lr_check_files)}")
print(f"HR .npy files found: {len(hr_check_files)}")

print("=" * 70)


# ============================================================
# LOCAL STORAGE INFORMATION
# ============================================================

print()
print("=" * 70)
print("LOCAL STORAGE")
print("=" * 70)

print()
print("Checkpoint directory:")
print(f"  {os.path.abspath(CHECKPOINT_DIR)}")

print()
print("Sample directory:")
print(f"  {os.path.abspath(SAMPLE_DIR)}")

print()
print("Latest checkpoint:")
print(f"  {os.path.abspath(latest_checkpoint)}")

print()
print("Best PSNR checkpoint:")
print(f"  {os.path.abspath(best_psnr_checkpoint)}")

print()
print("Best SSIM checkpoint:")
print(f"  {os.path.abspath(best_ssim_checkpoint)}")

print()
print("Best LPIPS checkpoint:")
print(f"  {os.path.abspath(best_lpips_checkpoint)}")

print("=" * 70)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IN_CHANNELS = 1
OUT_CHANNELS = 1

SCALE = 2

WINDOW_SIZE = 8

TRAIN_PATCH_SIZE = 128

# TOTAL number of epochs
EPOCHS = 60

BATCH_SIZE = 1
NUM_WORKERS = 0

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

VAL_RATIO = 0.1
SEED = 42

BASE_CHANNELS = 96
NUM_BLOCKS = 10

ESTIMATOR_CHANNELS = 96
HF_BLOCKS = 12

NUM_HEADS = 2
MLP_RATIO = 2.0
CA_REDUCTION = 8

LAMBDA_SR = 1.0
LAMBDA_SPECKLE = 0.05
LAMBDA_NOISE = 0.05
LAMBDA_GRAD = 0.10
LAMBDA_LAP = 0.
LAMBDA_EDGE = 0.
LAMBDA_DEG = 0.
LAMBDA_VGG = 0.10


# ============================================================
# AUGMENTATION
# ============================================================

AUGMENT = True

AUG_SPECKLE_PROB = 0.35
AUG_SPECKLE_STD = 0.04

AUG_NOISE_PROB = 0.35
AUG_NOISE_STD = 0.015

AUG_GAIN_PROB = 0.25
AUG_GAIN_MIN = 0.97
AUG_GAIN_MAX = 1.03

CUTBLUR_PROB = 0.5
CUTBLUR_RATIO_MIN = 0.3
CUTBLUR_RATIO_MAX = 0.7

EPS = 1e-6

VALID_RANGE_MIN = 0.0
VALID_RANGE_MAX = 1.0

SPECKLE_FLOOR = 0.05


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

AMP = DEVICE.type == "cuda"

GPU_COUNT = (
    torch.cuda.device_count()
    if torch.cuda.is_available()
    else 0
)


# ============================================================
# SEED
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = True


print()
print("=" * 70)
print("DEVICE INFORMATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"GPU count: {GPU_COUNT}")

for i in range(GPU_COUNT):

    print(
        f"GPU {i}: "
        f"{torch.cuda.get_device_name(i)}"
    )

print("=" * 70)


# ============================================================
# VGG PERCEPTUAL LOSS
# ============================================================

class VGGPerceptualLoss(nn.Module):

    def __init__(
        self,
        layer_ids=(3, 8, 17, 26),
        layer_weights=None
    ):

        super().__init__()

        weights = (
            torchvision.models.VGG19_Weights.IMAGENET1K_V1
        )

        vgg = torchvision.models.vgg19(
            weights=weights
        ).features

        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

        self.layer_ids = tuple(
            sorted(layer_ids)
        )

        self.max_layer = self.layer_ids[-1]

        if layer_weights is None:
            layer_weights = [1.0] * len(self.layer_ids)

        assert len(layer_weights) == len(
            self.layer_ids
        )

        self.layer_weights = dict(
            zip(
                self.layer_ids,
                layer_weights
            )
        )

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

    def _prepare(self, x):

        x = torch.clamp(
            x,
            0.0,
            1.0
        )

        if x.shape[1] == 1:

            x = x.repeat(
                1,
                3,
                1,
                1
            )

        return (
            x - self.mean
        ) / self.std

    def forward(self, pred, target):

        pred = self._prepare(pred)
        target = self._prepare(target)

        loss = 0.0

        x = pred
        y = target

        for i, layer in enumerate(self.vgg):

            x = layer(x)

            with torch.no_grad():
                y = layer(y)

            if i in self.layer_weights:

                loss = (
                    loss
                    + self.layer_weights[i]
                    * F.l1_loss(x, y)
                )

            if i == self.max_layer:
                break

        return loss


vgg_loss_fn = VGGPerceptualLoss().to(DEVICE)
vgg_loss_fn.eval()


# ============================================================
# LPIPS
# ============================================================

LPIPS_NET = "alex"

lpips_metric = lpips.LPIPS(
    net=LPIPS_NET
).to(DEVICE)

lpips_metric.eval()

for param in lpips_metric.parameters():
    param.requires_grad = False


# ============================================================
# RAW PAIR LOADER
# ============================================================

class RawPairLoader:

    @staticmethod
    def load_npy(path):

        x = np.load(path).astype(
            np.float32
        )

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        if x.ndim == 2:

            x = x[None]

        elif x.ndim == 3:

            if (
                x.shape[0] not in [1, 3]
                and x.shape[-1] in [1, 3]
            ):

                x = np.transpose(
                    x,
                    (2, 0, 1)
                )

        else:

            raise ValueError(
                f"Unsupported shape "
                f"{x.shape}: {path}"
            )

        return torch.from_numpy(x)

    @staticmethod
    def convert_channels(
        x,
        channels
    ):

        if x.shape[0] == channels:
            return x

        if (
            x.shape[0] == 3
            and channels == 1
        ):

            return x.mean(
                0,
                keepdim=True
            )

        if (
            x.shape[0] == 1
            and channels == 3
        ):

            return x.repeat(
                3,
                1,
                1
            )

        raise ValueError(
            f"Channels {x.shape[0]} "
            f"!= {channels}"
        )

    @staticmethod
    def ensure_scale_relationship(
        lr,
        hr,
        scale
    ):

        target_h = (
            lr.shape[-2] * scale
        )

        target_w = (
            lr.shape[-1] * scale
        )

        if hr.shape[-2:] != (
            target_h,
            target_w
        ):

            hr = F.interpolate(
                hr.unsqueeze(0),
                size=(
                    target_h,
                    target_w
                ),
                mode="bicubic",
                align_corners=False
            ).squeeze(0)

        return hr

    @classmethod
    def load_pair(
        cls,
        lr_path,
        hr_path
    ):

        lr = cls.load_npy(lr_path)
        hr = cls.load_npy(hr_path)

        lr = cls.convert_channels(
            lr,
            IN_CHANNELS
        )

        hr = cls.convert_channels(
            hr,
            OUT_CHANNELS
        )

        hr = cls.ensure_scale_relationship(
            lr,
            hr,
            SCALE
        )

        lr = torch.nan_to_num(
            lr,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        hr = torch.nan_to_num(
            hr,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        return (
            lr.float(),
            hr.float()
        )


# ============================================================
# PAIR CREATION
# ============================================================

def build_pairs():

    lr_files = sorted(
        glob.glob(
            os.path.join(
                TRAIN_LR_DIR,
                "*.npy"
            )
        )
    )

    hr_files = sorted(
        glob.glob(
            os.path.join(
                TRAIN_HR_DIR,
                "*.npy"
            )
        )
    )

    if len(lr_files) == 0:

        raise FileNotFoundError(
            f"No .npy files found in "
            f"{TRAIN_LR_DIR}"
        )

    if len(hr_files) == 0:

        raise FileNotFoundError(
            f"No .npy files found in "
            f"{TRAIN_HR_DIR}"
        )

    hr_map = {
        os.path.basename(x): x
        for x in hr_files
    }

    pairs = []

    for lr in lr_files:

        name = os.path.basename(lr)

        if name in hr_map:

            pairs.append(
                (
                    lr,
                    hr_map[name]
                )
            )

    if len(pairs) == 0:

        if len(lr_files) != len(hr_files):

            raise RuntimeError(
                "LR and HR filenames do not match."
            )

        pairs = list(
            zip(
                lr_files,
                hr_files
            )
        )

    return pairs


pairs = build_pairs()

random.shuffle(pairs)

val_count = max(
    1,
    int(len(pairs) * VAL_RATIO)
)

val_pairs = pairs[:val_count]
train_pairs = pairs[val_count:]


print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    f"Total images: {len(pairs)}"
)

print(
    f"Training images: "
    f"{len(train_pairs)}"
)

print(
    f"Validation images: "
    f"{len(val_pairs)}"
)

print("=" * 70)


# ============================================================
# AUGMENTATION HELPERS
# ============================================================

def geometric_augmentation(
    lr,
    hr
):

    if random.random() < 0.5:

        lr = torch.flip(
            lr,
            dims=[2]
        )

        hr = torch.flip(
            hr,
            dims=[2]
        )

    if random.random() < 0.5:

        lr = torch.flip(
            lr,
            dims=[1]
        )

        hr = torch.flip(
            hr,
            dims=[1]
        )

    if random.random() < 0.5:

        k = random.randint(
            1,
            3
        )

        lr = torch.rot90(
            lr,
            k,
            dims=[1, 2]
        )

        hr = torch.rot90(
            hr,
            k,
            dims=[1, 2]
        )

    return (
        lr.contiguous(),
        hr.contiguous()
    )


def degradation_augmentation(lr):

    if (
        random.random()
        < AUG_SPECKLE_PROB
    ):

        speckle = (
            torch.randn_like(lr)
            * AUG_SPECKLE_STD
        )

        lr = lr * (
            1.0 + speckle
        )

    if (
        random.random()
        < AUG_NOISE_PROB
    ):

        noise = (
            torch.randn_like(lr)
            * AUG_NOISE_STD
        )

        lr = lr + noise

    if (
        random.random()
        < AUG_GAIN_PROB
    ):

        gain = random.uniform(
            AUG_GAIN_MIN,
            AUG_GAIN_MAX
        )

        lr = lr * gain

    lr = torch.clamp(
        lr,
        VALID_RANGE_MIN,
        VALID_RANGE_MAX
    )

    return lr


def cutblur_augmentation(
    lr,
    hr,
    scale,
    prob=CUTBLUR_PROB,
    ratio_range=(
        CUTBLUR_RATIO_MIN,
        CUTBLUR_RATIO_MAX
    )
):

    if random.random() >= prob:

        return (
            lr,
            hr
        )

    h, w = lr.shape[-2:]

    cut_ratio = random.uniform(
        *ratio_range
    )

    ch = max(
        1,
        int(h * cut_ratio)
    )

    cw = max(
        1,
        int(w * cut_ratio)
    )

    cy = random.randint(
        0,
        h - ch
    )

    cx = random.randint(
        0,
        w - cw
    )

    if random.random() < 0.5:

        lr_region = lr[
            :,
            cy:cy + ch,
            cx:cx + cw
        ]

        up_region = F.interpolate(
            lr_region.unsqueeze(0),
            size=(
                ch * scale,
                cw * scale
            ),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        up_region = torch.clamp(
            up_region,
            VALID_RANGE_MIN,
            VALID_RANGE_MAX
        )

        hr = hr.clone()

        hr[
            :,
            cy * scale:(cy + ch) * scale,
            cx * scale:(cx + cw) * scale
        ] = up_region

    else:

        hr_region = hr[
            :,
            cy * scale:(cy + ch) * scale,
            cx * scale:(cx + cw) * scale
        ]

        down_region = F.interpolate(
            hr_region.unsqueeze(0),
            size=(
                ch,
                cw
            ),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        down_region = torch.clamp(
            down_region,
            VALID_RANGE_MIN,
            VALID_RANGE_MAX
        )

        lr = lr.clone()

        lr[
            :,
            cy:cy + ch,
            cx:cx + cw
        ] = down_region

    return (
        lr,
        hr
    )


def pad_to_multiple(
    x,
    multiple=WINDOW_SIZE
):

    c, h, w = x.shape

    pad_h = (
        multiple
        - h % multiple
    ) % multiple

    pad_w = (
        multiple
        - w % multiple
    ) % multiple

    if pad_h or pad_w:

        x = F.pad(
            x.unsqueeze(0),
            (
                0,
                pad_w,
                0,
                pad_h
            ),
            mode="reflect"
        ).squeeze(0)

    return (
        x,
        h,
        w
    )


def random_crop_pair(
    lr,
    hr,
    patch_size,
    scale
):

    _, h, w = lr.shape

    if (
        h < patch_size
        or w < patch_size
    ):

        pad_h = max(
            0,
            patch_size - h
        )

        pad_w = max(
            0,
            patch_size - w
        )

        lr = F.pad(
            lr.unsqueeze(0),
            (
                0,
                pad_w,
                0,
                pad_h
            ),
            mode="reflect"
        ).squeeze(0)

        hr = F.pad(
            hr.unsqueeze(0),
            (
                0,
                pad_w * scale,
                0,
                pad_h * scale
            ),
            mode="reflect"
        ).squeeze(0)

        h, w = lr.shape[-2:]

    top = random.randint(
        0,
        h - patch_size
    )

    left = random.randint(
        0,
        w - patch_size
    )

    lr_patch = lr[
        :,
        top:top + patch_size,
        left:left + patch_size
    ]

    hr_patch = hr[
        :,
        top * scale:(top + patch_size) * scale,
        left * scale:(left + patch_size) * scale
    ]

    return (
        lr_patch.contiguous(),
        hr_patch.contiguous()
    )


# ============================================================
# DATASETS
# ============================================================

class TrainSRDataset(Dataset):

    def __init__(
        self,
        file_pairs,
        patch_size,
        scale,
        augment=True
    ):

        self.file_pairs = file_pairs
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.loader = RawPairLoader()

    def __len__(self):

        return len(self.file_pairs)

    def __getitem__(self, idx):

        lr_path, hr_path = (
            self.file_pairs[idx]
        )

        lr, hr = self.loader.load_pair(
            lr_path,
            hr_path
        )

        if self.augment:

            lr, hr = geometric_augmentation(
                lr,
                hr
            )

            lr = degradation_augmentation(
                lr
            )

        lr, hr = random_crop_pair(
            lr,
            hr,
            self.patch_size,
            self.scale
        )

        if self.augment:

            lr, hr = cutblur_augmentation(
                lr,
                hr,
                self.scale
            )

        lr = torch.nan_to_num(
            lr,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        hr = torch.nan_to_num(
            hr,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        return (
            lr,
            hr,
            os.path.basename(lr_path)
        )


class ValSRDataset(Dataset):

    def __init__(
        self,
        file_pairs,
        window_size=WINDOW_SIZE
    ):

        self.file_pairs = file_pairs
        self.window_size = window_size
        self.loader = RawPairLoader()

    def __len__(self):

        return len(self.file_pairs)

    def __getitem__(self, idx):

        lr_path, hr_path = (
            self.file_pairs[idx]
        )

        lr, hr = self.loader.load_pair(
            lr_path,
            hr_path
        )

        (
            lr_padded,
            orig_h,
            orig_w
        ) = pad_to_multiple(
            lr,
            self.window_size
        )

        return (
            lr_padded,
            hr,
            orig_h,
            orig_w,
            os.path.basename(lr_path)
        )


train_dataset = TrainSRDataset(
    train_pairs,
    TRAIN_PATCH_SIZE,
    SCALE,
    augment=AUGMENT
)

val_dataset = ValSRDataset(
    val_pairs,
    WINDOW_SIZE
)


# ============================================================
# DATA LOADERS
# ============================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(
        NUM_WORKERS > 0
    )
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(
        NUM_WORKERS > 0
    )
)


print()
print("=" * 70)
print("DATASET READY")
print("=" * 70)

print(
    f"Training source images: "
    f"{len(train_dataset)}"
)

print(
    f"Samples per epoch: "
    f"{len(train_dataset)}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Batches per epoch: "
    f"{len(train_loader)}"
)

print(
    f"Train patch size: "
    f"{TRAIN_PATCH_SIZE}x{TRAIN_PATCH_SIZE}"
)

print("=" * 70)


# ============================================================
# MODEL
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            )
        )

        self.scale = nn.Parameter(
            torch.tensor(0.2)
        )

    def forward(self, x):

        return (
            x
            + self.scale
            * self.body(x)
        )


class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction=8
    ):

        super().__init__()

        hidden = max(
            channels // reduction,
            4
        )

        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.max_pool = (
            nn.AdaptiveMaxPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                channels,
                1
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.mlp(
            self.avg_pool(x)
        )

        mx = self.mlp(
            self.max_pool(x)
        )

        attention = self.sigmoid(
            avg + mx
        )

        return x * attention


def window_partition(
    x,
    window_size
):

    B, H, W, C = x.shape

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )

    windows = x.permute(
        0,
        1,
        3,
        2,
        4,
        5
    ).contiguous()

    return windows.view(
        -1,
        window_size,
        window_size,
        C
    )


def window_reverse(
    windows,
    window_size,
    H,
    W
):

    B = int(
        windows.shape[0]
        / (
            H // window_size
            * W // window_size
        )
    )

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )

    x = x.permute(
        0,
        1,
        3,
        2,
        4,
        5
    ).contiguous()

    return x.view(
        B,
        H,
        W,
        -1
    )


class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        window_size,
        num_heads
    ):

        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = (
            dim // num_heads
        )

        self.scale = (
            head_dim ** -0.5
        )

        self.qkv = nn.Linear(
            dim,
            dim * 3
        )

        self.proj = nn.Linear(
            dim,
            dim
        )

        relative_size = (
            2 * window_size - 1
        )

        self.relative_position_bias_table = (
            nn.Parameter(
                torch.zeros(
                    relative_size
                    * relative_size,
                    num_heads
                )
            )
        )

        coords_h = torch.arange(
            window_size
        )

        coords_w = torch.arange(
            window_size
        )

        coords = torch.stack(
            torch.meshgrid(
                coords_h,
                coords_w,
                indexing="ij"
            )
        )

        coords_flatten = torch.flatten(
            coords,
            1
        )

        relative_coords = (
            coords_flatten[:, :, None]
            - coords_flatten[:, None, :]
        )

        relative_coords = (
            relative_coords
            .permute(
                1,
                2,
                0
            )
            .contiguous()
        )

        relative_coords[:, :, 0] += (
            window_size - 1
        )

        relative_coords[:, :, 1] += (
            window_size - 1
        )

        relative_coords[:, :, 0] *= (
            2 * window_size - 1
        )

        relative_position_index = (
            relative_coords.sum(-1)
        )

        self.register_buffer(
            "relative_position_index",
            relative_position_index
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02
        )

    def forward(
        self,
        x,
        mask=None
    ):

        B_, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B_,
            N,
            3,
            self.num_heads,
            C // self.num_heads
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2]
        )

        q = q * self.scale

        attention = (
            q
            @ k.transpose(-2, -1)
        )

        relative_position_bias = (
            self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ]
        )

        relative_position_bias = (
            relative_position_bias.view(
                self.window_size
                * self.window_size,
                self.window_size
                * self.window_size,
                -1
            )
        )

        relative_position_bias = (
            relative_position_bias
            .permute(
                2,
                0,
                1
            )
            .contiguous()
        )

        attention = (
            attention
            + relative_position_bias.unsqueeze(0)
        )

        if mask is not None:

            nW = mask.shape[0]

            attention = attention.view(
                B_ // nW,
                nW,
                self.num_heads,
                N,
                N
            )

            attention = (
                attention
                + mask.unsqueeze(1).unsqueeze(0)
            )

            attention = attention.view(
                -1,
                self.num_heads,
                N,
                N
            )

        attention = F.softmax(
            attention,
            dim=-1
        )

        output = attention @ v

        output = (
            output
            .transpose(1, 2)
            .reshape(
                B_,
                N,
                C
            )
        )

        return self.proj(output)


class MLP(nn.Module):

    def __init__(
        self,
        dim,
        mlp_ratio=2.0
    ):

        super().__init__()

        hidden_dim = int(
            dim * mlp_ratio
        )

        self.fc1 = nn.Linear(
            dim,
            hidden_dim
        )

        self.act = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_dim,
            dim
        )

    def forward(self, x):

        return self.fc2(
            self.act(
                self.fc1(x)
            )
        )


class SwinBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=2.0
    ):

        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(
            dim,
            window_size,
            num_heads
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(
            dim,
            mlp_ratio
        )

    def calculate_mask(
        self,
        H,
        W,
        device
    ):

        img_mask = torch.zeros(
            (1, H, W, 1),
            device=device
        )

        h_slices = (
            slice(0, -self.window_size),
            slice(
                -self.window_size,
                -self.shift_size
            ),
            slice(
                -self.shift_size,
                None
            )
        )

        w_slices = (
            slice(0, -self.window_size),
            slice(
                -self.window_size,
                -self.shift_size
            ),
            slice(
                -self.shift_size,
                None
            )
        )

        cnt = 0

        for h in h_slices:

            for w in w_slices:

                img_mask[
                    :,
                    h,
                    w,
                    :
                ] = cnt

                cnt += 1

        mask_windows = window_partition(
            img_mask,
            self.window_size
        )

        mask_windows = mask_windows.view(
            -1,
            self.window_size
            * self.window_size
        )

        attn_mask = (
            mask_windows.unsqueeze(1)
            - mask_windows.unsqueeze(2)
        )

        attn_mask = attn_mask.masked_fill(
            attn_mask != 0,
            -100.0
        )

        return attn_mask.masked_fill(
            attn_mask == 0,
            0.0
        )

    def forward(self, x):

        B, C, H, W = x.shape

        shortcut = x

        x = x.permute(
            0,
            2,
            3,
            1
        ).contiguous()

        x = self.norm1(x)

        if self.shift_size > 0:

            shifted_x = torch.roll(
                x,
                shifts=(
                    -self.shift_size,
                    -self.shift_size
                ),
                dims=(1, 2)
            )

        else:

            shifted_x = x

        x_windows = window_partition(
            shifted_x,
            self.window_size
        )

        x_windows = x_windows.view(
            -1,
            self.window_size
            * self.window_size,
            C
        )

        if self.shift_size > 0:

            attn_mask = self.calculate_mask(
                H,
                W,
                x.device
            )

        else:

            attn_mask = None

        attn_windows = self.attn(
            x_windows,
            attn_mask
        )

        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            C
        )

        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            H,
            W
        )

        if self.shift_size > 0:

            x = torch.roll(
                shifted_x,
                shifts=(
                    self.shift_size,
                    self.shift_size
                ),
                dims=(1, 2)
            )

        else:

            x = shifted_x

        x = (
            x
            + shortcut.permute(
                0,
                2,
                3,
                1
            )
        )

        x = (
            x
            + self.mlp(
                self.norm2(x)
            )
        )

        return x.permute(
            0,
            3,
            1,
            2
        ).contiguous()


class NoiseAwareHybridBlock(nn.Module):

    def __init__(
        self,
        channels,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio,
        reduction
    ):

        super().__init__()

        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            )
        )

        self.channel_attention = (
            ChannelAttention(
                channels,
                reduction
            )
        )

        self.noise_projection = nn.Sequential(
            nn.Conv2d(
                2,
                channels,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                1
            ),
            nn.Sigmoid()
        )

        self.swin = SwinBlock(
            channels,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            )
        )

        self.res_scale = nn.Parameter(
            torch.tensor(0.15)
        )

    def forward(
        self,
        x,
        noise_context
    ):

        residual = x

        local = self.local(x)

        local = self.channel_attention(
            local
        )

        gate = self.noise_projection(
            noise_context
        )

        local = local * (
            1.0 + gate
        )

        global_features = self.swin(
            local
        )

        fused = self.fusion(
            torch.cat(
                [
                    local,
                    global_features
                ],
                dim=1
            )
        )

        return (
            residual
            + self.res_scale * fused
        )


class SpeckleEstimator(nn.Module):

    def __init__(
        self,
        in_channels=1,
        channels=48
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                1,
                3,
                1,
                1
            ),
            nn.Softplus()
        )

    def forward(self, x):

        return self.net(x) + EPS


class AdditiveNoiseEstimator(nn.Module):

    def __init__(
        self,
        in_channels=1,
        channels=48
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                1,
                3,
                1,
                1
            )
        )

    def forward(self, x):

        return self.net(x)


class HighFrequencyBranch(nn.Module):

    def __init__(
        self,
        in_channels,
        channels
    ):

        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            )
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(HF_BLOCKS)
            ]
        )

        self.highpass = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            )
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            )
        )

        self.scale = nn.Parameter(
            torch.tensor(0.2)
        )

    def forward(self, x):

        x = self.input(x)

        low = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1
        )

        high = x - low

        high = self.highpass(high)

        high = self.body(high)

        high = self.refine(high)

        return self.scale * high


class MultiplicativeAdditiveHybridSwinSR(
    nn.Module
):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        channels=96,
        blocks=10,
        scale=2
    ):

        super().__init__()

        self.scale = scale

        self.speckle_estimator = (
            SpeckleEstimator(
                in_channels,
                ESTIMATOR_CHANNELS
            )
        )

        self.noise_estimator = (
            AdditiveNoiseEstimator(
                in_channels,
                ESTIMATOR_CHANNELS
            )
        )

        self.head = nn.Conv2d(
            in_channels * 3,
            channels,
            3,
            1,
            1
        )

        self.body = nn.ModuleList()

        for i in range(blocks):

            self.body.append(
                NoiseAwareHybridBlock(
                    channels=channels,
                    num_heads=NUM_HEADS,
                    window_size=WINDOW_SIZE,
                    shift_size=(
                        0
                        if i % 2 == 0
                        else WINDOW_SIZE // 2
                    ),
                    mlp_ratio=MLP_RATIO,
                    reduction=CA_REDUCTION
                )
            )

        self.body_tail = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            1
        )

        self.hf_branch = (
            HighFrequencyBranch(
                in_channels * 3,
                channels
            )
        )

        self.hf_attention = (
            ChannelAttention(
                channels,
                CA_REDUCTION
            )
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            )
        )

        self.upsample_conv = nn.Conv2d(
            channels,
            channels * scale * scale,
            3,
            1,
            1
        )

        self.pixel_shuffle = (
            nn.PixelShuffle(scale)
        )

        self.upsample_activation = (
            nn.LeakyReLU(
                0.1,
                inplace=True
            )
        )

        self.reconstruction = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                1,
                1
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.Conv2d(
                channels,
                out_channels,
                3,
                1,
                1
            )
        )

        self.corrected_upsample_conv = (
            nn.Conv2d(
                in_channels,
                out_channels * scale * scale,
                3,
                1,
                1
            )
        )

        self.corrected_pixel_shuffle = (
            nn.PixelShuffle(scale)
        )

        self.corrected_upsample_activation = (
            nn.LeakyReLU(
                0.1,
                inplace=True
            )
        )

        self.corrected_refine = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            1
        )

        self.output_scale = nn.Parameter(
            torch.tensor(0.2)
        )

    def forward(self, lr):

        speckle = (
            self.speckle_estimator(lr)
        )

        speckle_safe = torch.clamp(
            speckle,
            min=SPECKLE_FLOOR
        )

        noise = (
            self.noise_estimator(lr)
        )

        corrected = (
            lr - noise
        ) / speckle_safe

        corrected = torch.clamp(
            corrected,
            -1.0,
            2.0
        )

        residual = (
            lr - corrected
        )

        main_input = torch.cat(
            [
                lr,
                corrected,
                residual
            ],
            dim=1
        )

        main_features = self.head(
            main_input
        )

        noise_context = torch.cat(
            [
                noise,
                speckle
            ],
            dim=1
        )

        body_features = main_features

        for block in self.body:

            body_features = block(
                body_features,
                noise_context
            )

        main_features = (
            main_features
            + self.body_tail(
                body_features
            )
        )

        hf_input = torch.cat(
            [
                lr,
                noise,
                speckle
            ],
            dim=1
        )

        hf_features = self.hf_branch(
            hf_input
        )

        hf_features = self.hf_attention(
            hf_features
        )

        main_features = self.fusion(
            torch.cat(
                [
                    main_features,
                    hf_features
                ],
                dim=1
            )
        )

        x = self.upsample_conv(
            main_features
        )

        x = self.pixel_shuffle(x)

        x = self.upsample_activation(x)

        sr_residual = self.reconstruction(x)

        corrected_base = (
            self.corrected_upsample_conv(
                corrected
            )
        )

        corrected_base = (
            self.corrected_pixel_shuffle(
                corrected_base
            )
        )

        corrected_base = (
            self.corrected_upsample_activation(
                corrected_base
            )
        )

        corrected_base = (
            self.corrected_refine(
                corrected_base
            )
        )

        if (
            sr_residual.shape[-2:]
            != corrected_base.shape[-2:]
        ):

            sr_residual = F.interpolate(
                sr_residual,
                size=corrected_base.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        sr = (
            corrected_base
            + self.output_scale
            * sr_residual
        )

        return (
            sr,
            speckle,
            noise,
            corrected,
            residual,
            hf_features
        )


# ============================================================
# INITIALIZE MODEL
# ============================================================

def initialize_weights(model):

    for m in model.modules():

        if isinstance(m, nn.Conv2d):

            nn.init.kaiming_normal_(
                m.weight,
                a=0.1,
                mode="fan_in",
                nonlinearity="leaky_relu"
            )

            if m.bias is not None:

                nn.init.zeros_(
                    m.bias
                )

        elif isinstance(m, nn.Linear):

            nn.init.trunc_normal_(
                m.weight,
                std=0.02
            )

            if m.bias is not None:

                nn.init.zeros_(
                    m.bias
                )

        elif isinstance(m, nn.LayerNorm):

            nn.init.ones_(
                m.weight
            )

            nn.init.zeros_(
                m.bias
            )


model = MultiplicativeAdditiveHybridSwinSR(
    IN_CHANNELS,
    OUT_CHANNELS,
    BASE_CHANNELS,
    NUM_BLOCKS,
    scale=SCALE
)

initialize_weights(model)

model = model.to(DEVICE)


# ============================================================
# MULTI-GPU
# ============================================================

if GPU_COUNT > 1:

    model = nn.DataParallel(
        model,
        device_ids=list(
            range(GPU_COUNT)
        ),
        output_device=0
    )


print()
print(
    f"Using {GPU_COUNT} GPU(s)"
)

print(
    f"Generator parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# MULTI-RESOLUTION SHAPE TEST
# ============================================================

with torch.no_grad():

    for test_lr_size in (
        128,
        256
    ):

        test_input = torch.randn(
            1,
            IN_CHANNELS,
            test_lr_size,
            test_lr_size,
            device=DEVICE
        )

        test_output = model(
            test_input
        )[0]

        expected_hr_size = (
            test_lr_size * SCALE
        )

        print(
            f"Shape test: "
            f"{tuple(test_input.shape)} "
            f"-> "
            f"{tuple(test_output.shape)} "
            f"(expected HR side "
            f"{expected_hr_size})"
        )

        assert (
            test_output.shape[-2:]
            == (
                expected_hr_size,
                expected_hr_size
            )
        )


# ============================================================
# LOSSES
# ============================================================

def charbonnier(
    pred,
    target,
    eps=1e-3
):

    return torch.mean(
        torch.sqrt(
            (pred - target) ** 2
            + eps ** 2
        )
    )


def gradient_map(x):

    gx = (
        x[:, :, :, 1:]
        - x[:, :, :, :-1]
    )

    gy = (
        x[:, :, 1:, :]
        - x[:, :, :-1, :]
    )

    return gx, gy


def gradient_loss(
    pred,
    target
):

    pgx, pgy = gradient_map(pred)
    tgx, tgy = gradient_map(target)

    return (
        F.l1_loss(pgx, tgx)
        + F.l1_loss(pgy, tgy)
    )


def multiscale_gradient_loss(
    pred,
    target
):

    loss = 0.0

    for scale in [1, 2, 4]:

        if scale > 1:

            p = F.avg_pool2d(
                pred,
                scale,
                scale
            )

            t = F.avg_pool2d(
                target,
                scale,
                scale
            )

        else:

            p = pred
            t = target

        loss += gradient_loss(
            p,
            t
        )

    return loss / 3.0


def laplacian(x):

    kernel = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ],
        dtype=x.dtype,
        device=x.device
    ).view(
        1,
        1,
        3,
        3
    )

    kernel = kernel.repeat(
        x.shape[1],
        1,
        1,
        1
    )

    return F.conv2d(
        x,
        kernel,
        padding=1,
        groups=x.shape[1]
    )


def laplacian_loss(
    pred,
    target
):

    return F.l1_loss(
        laplacian(pred),
        laplacian(target)
    )


def edge_loss(
    pred,
    target
):

    return F.smooth_l1_loss(
        laplacian(pred),
        laplacian(target)
    )


def prepare_for_lpips(x):

    x = torch.clamp(
        x.float(),
        0.0,
        1.0
    )

    if x.shape[1] == 1:

        x = x.repeat(
            1,
            3,
            1,
            1
        )

    elif x.shape[1] != 3:

        raise ValueError(
            f"LPIPS expects 1 or 3 "
            f"channels, got {x.shape[1]}"
        )

    return x * 2.0 - 1.0


def lpips_metric_value(
    pred,
    target
):

    pred_lpips = (
        prepare_for_lpips(pred)
    )

    target_lpips = (
        prepare_for_lpips(target)
    )

    return lpips_metric(
        pred_lpips,
        target_lpips
    ).mean()


def vgg_perceptual_loss(
    pred,
    target
):

    return vgg_loss_fn(
        pred,
        target
    )


# ============================================================
# DEGRADATION FUNCTIONS
# ============================================================

def create_clean_lr(
    hr,
    scale=SCALE
):

    h = (
        hr.shape[-2]
        // scale
    )

    w = (
        hr.shape[-1]
        // scale
    )

    return F.interpolate(
        hr,
        size=(h, w),
        mode="bicubic",
        align_corners=False
    )


def create_speckle_target(
    lr,
    hr
):

    clean_lr = create_clean_lr(
        hr
    )

    speckle = (
        lr
        / (clean_lr + EPS)
    )

    return speckle.detach()


def degradation_loss(
    lr,
    clean_lr,
    speckle,
    noise
):

    reconstructed_lr = (
        clean_lr * speckle
        + noise
    )

    return F.l1_loss(
        reconstructed_lr,
        lr
    )


def noise_consistency_loss(
    lr,
    clean_lr,
    speckle,
    noise
):

    expected_noise = (
        lr
        - clean_lr * speckle
    )

    return charbonnier(
        noise,
        expected_noise.detach()
    )


# ============================================================
# METRICS
# ============================================================

def psnr_torch(
    pred,
    target
):

    pred = torch.clamp(
        pred,
        0,
        1
    )

    target = torch.clamp(
        target,
        0,
        1
    )

    mse = F.mse_loss(
        pred,
        target
    )

    if mse.item() <= 1e-12:

        return 100.0

    return (
        -10.0
        * torch.log10(mse)
    ).item()


def calculate_ssim(
    pred,
    target
):

    pred = np.clip(
        pred,
        0,
        1
    )

    target = np.clip(
        target,
        0,
        1
    )

    if pred.ndim == 3:

        pred = pred[0]
        target = target[0]

    return ssim(
        target,
        pred,
        data_range=1.0
    )


def calculate_mae(
    pred,
    target
):

    return float(
        np.mean(
            np.abs(
                pred - target
            )
        )
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.99)
)

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )
)

scaler = GradScaler(
    "cuda",
    enabled=AMP
)


# ============================================================
# HISTORY
# ============================================================

history = {

    "train_loss": [],
    "train_sr": [],
    "train_speckle": [],
    "train_noise": [],
    "train_grad": [],
    "train_lap": [],
    "train_edge": [],
    "train_deg": [],
    "train_vgg": [],

    "val_psnr": [],
    "val_ssim": [],
    "val_mae": [],
    "val_lpips": [],
    "val_speckle": [],
    "val_noise": [],
    "val_deg": []
}


best_psnr = -float("inf")
best_ssim = -float("inf")
best_lpips = float("inf")

start_epoch = 0


# ============================================================
# RESUME FROM LOCAL CHECKPOINT
# ============================================================

if os.path.exists(
    latest_checkpoint
):

    print()
    print("=" * 70)
    print("LOCAL CHECKPOINT FOUND")
    print("=" * 70)

    print(
        latest_checkpoint
    )

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        latest_checkpoint,
        map_location=DEVICE
    )

    model_to_load = (
        model.module
        if isinstance(
            model,
            nn.DataParallel
        )
        else model
    )

    model_to_load.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler"]
    )

    if "scaler" in checkpoint:

        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    history = checkpoint.get(
        "history",
        history
    )

    history.setdefault(
        "train_vgg",
        history.pop(
            "train_lpips",
            []
        )
    )

    history.setdefault(
        "val_lpips",
        []
    )

    best_psnr = checkpoint.get(
        "best_psnr",
        -float("inf")
    )

    best_ssim = checkpoint.get(
        "best_ssim",
        -float("inf")
    )

    best_lpips = checkpoint.get(
        "best_lpips",
        float("inf")
    )

    previous_epoch = checkpoint.get(
        "epoch",
        -1
    )

    start_epoch = (
        previous_epoch + 1
    )

    print()
    print("-" * 70)

    print(
        f"Checkpoint epoch: "
        f"{previous_epoch + 1}"
    )

    print(
        f"Resuming from epoch: "
        f"{start_epoch + 1}"
    )

    print(
        f"Best PSNR: "
        f"{best_psnr:.3f}"
    )

    print(
        f"Best SSIM: "
        f"{best_ssim:.5f}"
    )

    print(
        f"Best LPIPS: "
        f"{best_lpips:.5f}"
    )

    print(
        f"Current LR: "
        f"{optimizer.param_groups[0]['lr']:.3e}"
    )

    print("-" * 70)

else:

    print()
    print("=" * 70)
    print("NO LOCAL CHECKPOINT FOUND")
    print("=" * 70)

    print(
        "Starting training from epoch 1."
    )

    start_epoch = 0


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    epoch
):

    model.train()

    samples_this_epoch = (
        len(train_dataset)
    )

    print(
        f"\nEpoch {epoch + 1}: "
        f"training on "
        f"{samples_this_epoch} samples "
        f"(full pass)"
    )

    total_loss = 0.0
    total_sr = 0.0
    total_speckle = 0.0
    total_noise = 0.0
    total_grad = 0.0
    total_lap = 0.0
    total_edge = 0.0
    total_deg = 0.0
    total_vgg = 0.0

    progress = tqdm(
        train_loader,
        desc=(
            f"Training Epoch "
            f"{epoch + 1}"
        ),
        leave=False
    )

    for lr, hr, _ in progress:

        lr = lr.to(
            DEVICE,
            non_blocking=True
        )

        hr = hr.to(
            DEVICE,
            non_blocking=True
        )

        clean_lr = create_clean_lr(
            hr
        )

        speckle_target = (
            create_speckle_target(
                lr,
                hr
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            "cuda",
            enabled=AMP
        ):

            (
                sr,
                speckle_pred,
                noise_pred,
                corrected,
                residual,
                hf_features
            ) = model(lr)

            sr_loss = (
                0.7 * charbonnier(
                    sr,
                    hr
                )
                + 0.3 * F.l1_loss(
                    sr,
                    hr
                )
            )

            speckle_loss = charbonnier(
                speckle_pred,
                speckle_target
            )

            noise_loss = (
                noise_consistency_loss(
                    lr,
                    clean_lr,
                    speckle_pred,
                    noise_pred
                )
            )

            grad_loss = (
                multiscale_gradient_loss(
                    sr,
                    hr
                )
            )

            lap_loss = (
                laplacian_loss(
                    sr,
                    hr
                )
            )

            edge = edge_loss(
                sr,
                hr
            )

            deg_loss = (
                degradation_loss(
                    lr,
                    clean_lr,
                    speckle_pred,
                    noise_pred
                )
            )

            loss_without_vgg = (

                LAMBDA_SR * sr_loss

                + LAMBDA_SPECKLE
                * speckle_loss

                + LAMBDA_NOISE
                * noise_loss

                + LAMBDA_GRAD
                * grad_loss

                + LAMBDA_LAP
                * lap_loss

                + LAMBDA_EDGE
                * edge

                + LAMBDA_DEG
                * deg_loss
            )

        # ----------------------------------------------------
        # VGG IN FLOAT32
        # ----------------------------------------------------

        vgg_loss_value = (
            vgg_perceptual_loss(
                sr.float(),
                hr.float()
            )
        )

        loss = (
            loss_without_vgg
            + LAMBDA_VGG
            * vgg_loss_value
        )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += loss.item()

        total_sr += sr_loss.item()

        total_speckle += (
            speckle_loss.item()
        )

        total_noise += (
            noise_loss.item()
        )

        total_grad += (
            grad_loss.item()
        )

        total_lap += (
            lap_loss.item()
        )

        total_edge += (
            edge.item()
        )

        total_deg += (
            deg_loss.item()
        )

        total_vgg += (
            vgg_loss_value.item()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            sr=f"{sr_loss.item():.4f}"
        )

    n = max(
        len(train_loader),
        1
    )

    return {

        "loss":
            total_loss / n,

        "sr":
            total_sr / n,

        "speckle":
            total_speckle / n,

        "noise":
            total_noise / n,

        "grad":
            total_grad / n,

        "lap":
            total_lap / n,

        "edge":
            total_edge / n,

        "deg":
            total_deg / n,

        "vgg":
            total_vgg / n,

        "samples":
            samples_this_epoch
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    psnr_values = []
    ssim_values = []
    mae_values = []
    lpips_values = []

    speckle_values = []
    noise_values = []
    degradation_values = []

    examples = []

    progress = tqdm(
        val_loader,
        desc="Validation",
        leave=False
    )

    for i, (
        lr_padded,
        hr,
        orig_h,
        orig_w,
        names
    ) in enumerate(progress):

        lr_padded = lr_padded.to(
            DEVICE,
            non_blocking=True
        )

        hr = hr.to(
            DEVICE,
            non_blocking=True
        )

        orig_h = (
            int(orig_h[0])
            if torch.is_tensor(orig_h)
            else int(orig_h)
        )

        orig_w = (
            int(orig_w[0])
            if torch.is_tensor(orig_w)
            else int(orig_w)
        )

        clean_lr = create_clean_lr(
            hr
        )

        with autocast(
            "cuda",
            enabled=AMP
        ):

            (
                sr,
                speckle_pred,
                noise_pred,
                corrected,
                residual,
                hf_features
            ) = model(
                lr_padded
            )

        sr = sr[
            ...,
            :orig_h * SCALE,
            :orig_w * SCALE
        ]

        sr = torch.clamp(
            sr.float(),
            0,
            1
        )

        speckle_pred = (
            speckle_pred[
                ...,
                :orig_h,
                :orig_w
            ].float()
        )

        noise_pred = (
            noise_pred[
                ...,
                :orig_h,
                :orig_w
            ].float()
        )

        corrected = (
            corrected[
                ...,
                :orig_h,
                :orig_w
            ].float()
        )

        residual = (
            residual[
                ...,
                :orig_h,
                :orig_w
            ].float()
        )

        lr = lr_padded[
            ...,
            :orig_h,
            :orig_w
        ]

        speckle_target = (
            create_speckle_target(
                lr,
                hr
            )
        )

        psnr_values.append(
            psnr_torch(
                sr,
                hr
            )
        )

        lpips_values.append(
            lpips_metric_value(
                sr,
                hr
            ).item()
        )

        sr_np = (
            sr[0]
            .cpu()
            .numpy()
        )

        hr_np = (
            hr[0]
            .cpu()
            .numpy()
        )

        lr_np = (
            lr[0]
            .cpu()
            .numpy()
        )

        speckle_np = (
            speckle_pred[0]
            .cpu()
            .numpy()
        )

        noise_np = (
            noise_pred[0]
            .cpu()
            .numpy()
        )

        corrected_np = (
            corrected[0]
            .cpu()
            .numpy()
        )

        residual_np = (
            residual[0]
            .cpu()
            .numpy()
        )

        ssim_values.append(
            calculate_ssim(
                sr_np,
                hr_np
            )
        )

        mae_values.append(
            calculate_mae(
                sr_np,
                hr_np
            )
        )

        speckle_values.append(
            F.l1_loss(
                speckle_pred,
                speckle_target
            ).item()
        )

        expected_noise = (
            lr
            - clean_lr * speckle_pred
        )

        noise_values.append(
            F.l1_loss(
                noise_pred,
                expected_noise
            ).item()
        )

        reconstructed_lr = (
            clean_lr
            * speckle_pred
            + noise_pred
        )

        degradation_values.append(
            F.l1_loss(
                reconstructed_lr,
                lr
            ).item()
        )

        if i < 5:

            examples.append(
                {
                    "name":
                        names[0],

                    "lr":
                        lr_np,

                    "speckle":
                        speckle_np,

                    "noise":
                        noise_np,

                    "corrected":
                        corrected_np,

                    "residual":
                        residual_np,

                    "sr":
                        sr_np,

                    "hr":
                        hr_np,

                    "resolution":
                        (
                            f"{orig_h}x{orig_w}"
                            f" -> "
                            f"{orig_h * SCALE}x"
                            f"{orig_w * SCALE}"
                        )
                }
            )

    return {

        "psnr":
            float(
                np.mean(
                    psnr_values
                )
            ),

        "ssim":
            float(
                np.mean(
                    ssim_values
                )
            ),

        "mae":
            float(
                np.mean(
                    mae_values
                )
            ),

        "lpips":
            float(
                np.mean(
                    lpips_values
                )
            ),

        "speckle_mae":
            float(
                np.mean(
                    speckle_values
                )
            ),

        "noise_mae":
            float(
                np.mean(
                    noise_values
                )
            ),

        "degradation_mae":
            float(
                np.mean(
                    degradation_values
                )
            ),

        "examples":
            examples
    }


# ============================================================
# TRAINING LOOP
# ============================================================

if start_epoch >= EPOCHS:

    print()
    print("=" * 70)

    print(
        "TRAINING IS ALREADY COMPLETE."
    )

    print(
        f"Checkpoint contains "
        f"{start_epoch}/{EPOCHS} epochs."
    )

    print("=" * 70)

else:

    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_metrics = (
            train_one_epoch(
                epoch
            )
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        val_metrics = validate()

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history[
            "train_loss"
        ].append(
            train_metrics["loss"]
        )

        history[
            "train_sr"
        ].append(
            train_metrics["sr"]
        )

        history[
            "train_speckle"
        ].append(
            train_metrics["speckle"]
        )

        history[
            "train_noise"
        ].append(
            train_metrics["noise"]
        )

        history[
            "train_grad"
        ].append(
            train_metrics["grad"]
        )

        history[
            "train_lap"
        ].append(
            train_metrics["lap"]
        )

        history[
            "train_edge"
        ].append(
            train_metrics["edge"]
        )

        history[
            "train_deg"
        ].append(
            train_metrics["deg"]
        )

        history[
            "train_vgg"
        ].append(
            train_metrics["vgg"]
        )

        history[
            "val_psnr"
        ].append(
            val_metrics["psnr"]
        )

        history[
            "val_ssim"
        ].append(
            val_metrics["ssim"]
        )

        history[
            "val_mae"
        ].append(
            val_metrics["mae"]
        )

        history[
            "val_lpips"
        ].append(
            val_metrics["lpips"]
        )

        history[
            "val_speckle"
        ].append(
            val_metrics["speckle_mae"]
        )

        history[
            "val_noise"
        ].append(
            val_metrics["noise_mae"]
        )

        history[
            "val_deg"
        ].append(
            val_metrics["degradation_mae"]
        )

        # ----------------------------------------------------
        # CURRENT LR
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # BEST METRICS
        # ----------------------------------------------------

        is_best_psnr = (
            val_metrics["psnr"]
            > best_psnr
        )

        is_best_ssim = (
            val_metrics["ssim"]
            > best_ssim
        )

        is_best_lpips = (
            val_metrics["lpips"]
            < best_lpips
        )

        if is_best_psnr:

            best_psnr = (
                val_metrics["psnr"]
            )

        if is_best_ssim:

            best_ssim = (
                val_metrics["ssim"]
            )

        if is_best_lpips:

            best_lpips = (
                val_metrics["lpips"]
            )

        # ----------------------------------------------------
        # STATE DICT
        # ----------------------------------------------------

        state_dict = (

            model.module.state_dict()

            if isinstance(
                model,
                nn.DataParallel
            )

            else model.state_dict()
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        checkpoint = {

            "epoch":
                epoch,

            "model":
                state_dict,

            "optimizer":
                optimizer.state_dict(),

            "scheduler":
                scheduler.state_dict(),

            "scaler":
                scaler.state_dict(),

            "best_psnr":
                best_psnr,

            "best_ssim":
                best_ssim,

            "best_lpips":
                best_lpips,

            "history":
                history,

            "train_patch_size":
                TRAIN_PATCH_SIZE,

            "scale":
                SCALE
        }

        # ----------------------------------------------------
        # SAVE LATEST
        # ----------------------------------------------------

        torch.save(
            checkpoint,
            latest_checkpoint
        )

        print()
        print(
            "Latest checkpoint saved:"
        )

        print(
            latest_checkpoint
        )

        # ----------------------------------------------------
        # SAVE BEST PSNR
        # ----------------------------------------------------

        if is_best_psnr:

            torch.save(
                checkpoint,
                best_psnr_checkpoint
            )

            print(
                f"Best-PSNR model saved: "
                f"PSNR={best_psnr:.3f}"
            )

        # ----------------------------------------------------
        # SAVE BEST SSIM
        # ----------------------------------------------------

        if is_best_ssim:

            torch.save(
                checkpoint,
                best_ssim_checkpoint
            )

            print(
                f"Best-SSIM model saved: "
                f"SSIM={best_ssim:.5f}"
            )

        # ----------------------------------------------------
        # SAVE BEST LPIPS
        # ----------------------------------------------------

        if is_best_lpips:

            torch.save(
                checkpoint,
                best_lpips_checkpoint
            )

            print(
                f"Best-LPIPS model saved: "
                f"LPIPS={best_lpips:.5f}"
            )

        # ----------------------------------------------------
        # EPOCH SUMMARY
        # ----------------------------------------------------

        print()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
            f" | Samples "
            f"{train_metrics['samples']}"
            f" | Batches "
            f"{len(train_loader)}"
            f" | Loss "
            f"{train_metrics['loss']:.5f}"
            f" | SR "
            f"{train_metrics['sr']:.5f}"
            f" | Speckle "
            f"{train_metrics['speckle']:.5f}"
            f" | Noise "
            f"{train_metrics['noise']:.5f}"
            f" | Grad "
            f"{train_metrics['grad']:.5f}"
            f" | Lap "
            f"{train_metrics['lap']:.5f}"
            f" | VGG "
            f"{train_metrics['vgg']:.5f}"
            f" | Edge "
            f"{train_metrics['edge']:.5f}"
            f" | Deg "
            f"{train_metrics['deg']:.5f}"
            f" | PSNR "
            f"{val_metrics['psnr']:.3f}"
            f" | SSIM "
            f"{val_metrics['ssim']:.5f}"
            f" | MAE "
            f"{val_metrics['mae']:.6f}"
            f" | Val LPIPS "
            f"{val_metrics['lpips']:.5f}"
            f" | LR "
            f"{current_lr:.2e}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    f"Best PSNR: "
    f"{best_psnr:.3f}"
)

print(
    f"Best SSIM: "
    f"{best_ssim:.5f}"
)

print(
    f"Best LPIPS: "
    f"{best_lpips:.5f}"
)

print(
    f"Training source images: "
    f"{len(train_dataset)}"
)

print(
    f"Training samples per epoch: "
    f"{len(train_dataset)}"
)

print("=" * 70)

print(
    "LOCAL CHECKPOINTS"
)

print("=" * 70)

print(
    f"Latest:\n"
    f"  {latest_checkpoint}\n"
)

print(
    f"Best PSNR:\n"
    f"  {best_psnr_checkpoint}\n"
)

print(
    f"Best SSIM:\n"
    f"  {best_ssim_checkpoint}\n"
)

print(
    f"Best LPIPS:\n"
    f"  {best_lpips_checkpoint}"
)

print("=" * 70)