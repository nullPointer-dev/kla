# Aletheia — Image Restoration (2x Super-Resolution)

Team **Aletheia** — restoration of low-resolution, speckle- and
noise-corrupted scientific images to clean high-resolution images.

This repository contains:

| Path | Description |
|---|---|
| `run.py` | Inference script — restores every `.npy` image in a directory |
| `models/model.pth` | Trained model weights (2x SR / restoration) |
| `restored_outputs/` | Restored outputs produced by our model on the test set |
| `train.py` | Training script that reproduces our training from scratch |
| `requirements.txt` | Complete pip freeze of the training environment |

---

## 1. Setup

### Requirements

- Python 3.10+ (tested with 3.12)
- An NVIDIA GPU with CUDA (used for inference; a CPU works but is slower)
- `pip`

### Install dependencies

```bash
python -m venv aletheia_env
source aletheia_env/bin/activate        # Windows: aletheia_env\Scripts\activate
pip install -r requirements.txt
```

This installs the exact package versions used during training. Inference
itself only needs `numpy` and `torch`; the rest are required by `train.py`.

---

## 2. Running Inference

The inference script takes a directory of low-resolution `.npy` images and
writes one restored `.npy` image per input into the output directory.

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py ./Test_NoisyLR ./my_restored_outputs
```

### What `run.py` does

1. Scans `<input-dir>` for every `*.npy` file (case-insensitive).
2. Creates `<output-dir>` if it does not exist.
3. Loads the trained model from `models/model.pth` (next to `run.py`).
4. Runs inference on every input image.
5. Writes one restored `*.npy` file per input, using the **same filename**.

### Output format

- **Grayscale** 2D arrays of shape `(H, W)` (`float32`).
- Values clipped to **[0, 1]**.
- **No NaN / Inf** values.
- **Resolution**: inputs are restored at 2x scale. Test inputs are
  `128 x 128` and produce `256 x 256` outputs.
- Output filename matches the input filename (e.g. `000000.npy` in →
  `000000.npy` out).

### GPU usage

The script automatically uses the first available CUDA device. It runs fully
**offline** — no internet, no API keys, no extra model downloads, no manual
configuration. Use a custom weights path with `--checkpoint` if needed:

```bash
python run.py <input-dir> <output-dir> --checkpoint path/to/weights.pth
```

---

## 3. Restored Test Outputs

The folder `restored_outputs/` contains our model's restored output for the
entire test set (400 images, `256 x 256` each, values in `[0, 1]`). These are
the actual outputs produced by `run.py` when run on the provided test inputs.

---

## 4. Reproducing Training

### Dataset layout

`train.py` expects the training data at:

```
train/
├── NoisyLR/      # low-resolution noisy inputs   (128x128 .npy, 3200 images)
└── GT/           # high-resolution clean targets (256x256 .npy, 3200 images)
```

### Train

```bash
python train.py
```

- Trains the `MultiplicativeAdditiveHybridSwinSR` model for 60 epochs
  (batch size 1, 128x128 patches, 2x scale).
- Checkpoints are written to `./checkpoints/`:
  - `latest.pth` — latest state,
  - `best_psnr.pth` / `best_ssim.pth` / `best_lpips.pth` — best per metric.
- Training resumes automatically from `./checkpoints/latest.pth` if present.

### Notes on training dependencies

`train.py` uses torchvision VGG19 (`IMAGENET1K_V1`) and LPIPS (AlexNet)
perceptual losses. These weights are downloaded automatically the first time
training runs, so internet access is required **only for training**, not for
inference.

The final trained weights shipped in `models/model.pth` are the last saved
checkpoint from training (final epoch, validation PSNR 28.26 dB, SSIM 0.767).
No other weight files are required — `model.pth` alone contains everything
needed for inference.

---

## 5. Model

`MultiplicativeAdditiveHybridSwinSR` is a hybrid restoration network that:

- estimates multiplicative **speckle** and additive **noise** degradations,
- corrects the input using the estimated degradation,
- refines details with a Swin-Transformer body and a high-frequency branch,
- upsamples with pixel-shuffle (2x super-resolution).

~7.2M parameters. 1-channel grayscale in/out.