# Compression-Robust Trigger-Based Watermarking

PyTorch experiment comparing standard trigger-based model watermarking with a stable-aware variant under pruning and simulated weight quantization.

## Experiment

- Dataset: MNIST, 10 classes
- Model: small CNN with two convolution blocks and two linear layers
- Trigger: 4x4 white patch at the bottom-right corner
- Target label: `0`
- WSR: fraction of triggered test images classified as the target label

The stable-aware model adds a Fisher-like regularizer:

```text
clean_loss + lambda_wm * watermark_loss
    + lambda_reg * sum_i F_i * (theta_i - theta_i_clean)^2
```

where `F_i` is estimated from squared cross-entropy gradients on clean training batches.

## Setup

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python main.py --dataset mnist
```

Run CIFAR-10:

```powershell
python main.py --dataset cifar10
```

Quick CIFAR-10 smoke run:

```powershell
python main.py --dataset cifar10 --quick_test
```

Select a specific GPU:

```powershell
python main.py --dataset cifar10 --device cuda:1
```

Force CPU:

```powershell
python main.py --dataset mnist --cpu
```

Fast smoke run:

```powershell
python main.py --train-subset 5000 --clean-epochs 1 --wm-epochs 1
```

## Outputs

- `outputs/checkpoints/clean.pt`
- `outputs/checkpoints/standard_watermark.pt`
- `outputs/checkpoints/stable_aware_watermark.pt`
- `outputs/results_pruning.csv`
- `outputs/results_quantization.csv`
- `outputs/figures/wsr_vs_pruning.png`
- `outputs/figures/acc_vs_pruning.png`
- `outputs/figures/wsr_vs_quantization.png`
- `outputs/figures/acc_vs_quantization.png`
