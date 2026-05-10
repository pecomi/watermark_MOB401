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

Recommended CIFAR-10 seed repeat:

```powershell
python main.py --config configs/cifar10_seed_repeat.yaml
```

CIFAR-10 lambda regularization ablation:

```powershell
python main.py --config configs/cifar10_lambda_ablation.yaml
```

CIFAR-10 ResNet18 experiment:

```powershell
python main.py --config configs/cifar10_resnet18.yaml
```

Thesis experiments with persistent-backdoor-inspired direct mask embedding:

```powershell
python main.py --dataset cifar10 --model cifar_small --methods standard stable_mask_direct random_mask_direct --quick_test
python main.py --dataset cifar10 --model resnet18_cifar --methods standard stable_aware_reg stable_mask_direct random_mask_direct
python main.py --dataset cifar10 --model resnet18_cifar --seeds 42 43 44 --methods standard stable_mask_direct random_mask_direct
```

Direct embedding sweep for diagnosing low WSR:

```powershell
python main.py --dataset cifar10 --model resnet18_cifar --methods stable_mask_direct --direct-sweep --device cuda:0
```

ResNet18-CIFAR standard/stable-aware watermark sweep before compression:

```powershell
python main.py --dataset cifar10 --model resnet18_cifar --resnet-wm-sweep --device cuda:0
```

Run the best ResNet watermark configuration under pruning and quantization:

```powershell
python main.py --dataset cifar10 --model resnet18_cifar --seeds 42 43 44 --methods standard stable_aware_reg --target-label 1 --trigger-size 5 --poison-ratio 0.05 --wm-epochs 10 --lambda-wm 5.0 --lambda-reg 0.1 --learning-rate-watermark 0.0005 --watermark-train-mode alternating --watermark-steps-per-batch 2 --device cuda:0
```

Run a selected stronger direct-embedding configuration under pruning and quantization:

```powershell
python main.py --dataset cifar10 --model resnet18_cifar --seeds 42 43 44 --methods standard stable_mask_direct --stable-mask-percent 0.3 --lambda-wm 10.0 --poison-ratio 0.03 --watermark-steps-per-batch 2 --direct-embedding-mode wm_focused --lambda-clean 0.5 --device cuda:0
```

Compare stable and random direct masks with the same training procedure:

```powershell
python main.py --dataset cifar10 --model resnet18_cifar --seeds 42 43 44 --methods stable_mask_direct random_mask_direct --stable-mask-percent 0.3 --lambda-wm 10.0 --poison-ratio 0.03 --watermark-steps-per-batch 2 --device cuda:0
```

Mask selection options:

```powershell
python main.py --dataset cifar10 --model cifar_small --methods stable_mask_direct --mask-granularity channel --selection-mode quant_stable
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
- `outputs/cifar10_results_pruning.csv`
- `outputs/cifar10_results_quantization.csv`
- `outputs/cifar10_seed_repeat_results_pruning.csv`
- `outputs/cifar10_lambda_ablation_results_quantization.csv`
- `outputs/thesis_results/results_all.csv`
- `outputs/thesis_results/direct_embedding_sweep.csv`
- `outputs/resnet_wm_sweep.csv`
- `outputs/debug/trigger_examples.png`
- `outputs/thesis_results/checkpoints/`
- `outputs/thesis_results/figures/`
- `outputs/figures/wsr_vs_pruning.png`
- `outputs/figures/acc_vs_pruning.png`
- `outputs/figures/wsr_vs_quantization.png`
- `outputs/figures/acc_vs_quantization.png`
