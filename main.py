import argparse
import copy
import csv
from pathlib import Path

import torch
import yaml

from compression import apply_fake_quantization, apply_pruning
from data import make_loaders, set_seed
from evaluate import evaluate_acc, evaluate_wsr
from importance import compute_importance
from models import SmallCNN
from plot_results import plot_all
from watermark import train_clean, train_watermark


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(cfg):
    set_seed(cfg["seed"])
    device_name = cfg.get("device")
    if device_name is None:
        use_cuda = torch.cuda.is_available() and not cfg.get("cpu", False)
        device_name = "cuda" if use_cuda else "cpu"
    device = torch.device(device_name)
    print(f"device={device}")

    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"

    train_loader, test_loader = make_loaders(
        cfg["data_dir"],
        cfg["batch_size"],
        cfg["num_workers"],
        cfg["seed"],
        cfg.get("train_subset"),
    )

    clean_model = SmallCNN().to(device)
    train_clean(clean_model, train_loader, device, cfg["clean_epochs"], cfg["lr"])
    clean_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in clean_model.state_dict().items()
    }
    save_checkpoint(clean_model, checkpoint_dir / "clean.pt")

    importance = compute_importance(
        clean_model,
        train_loader,
        device,
        cfg["importance_batches"],
    )
    importance = {name: score.detach() for name, score in importance.items()}

    standard_model = SmallCNN().to(device)
    standard_model.load_state_dict(clean_state)
    train_watermark(
        standard_model,
        clean_state,
        train_loader,
        device,
        cfg["wm_epochs"],
        cfg["lr"],
        cfg["lambda_wm"],
        target_label=cfg["target_label"],
    )
    save_checkpoint(standard_model, checkpoint_dir / "standard_watermark.pt")

    stable_model = SmallCNN().to(device)
    stable_model.load_state_dict(clean_state)
    train_watermark(
        stable_model,
        clean_state,
        train_loader,
        device,
        cfg["wm_epochs"],
        cfg["lr"],
        cfg["lambda_wm"],
        lambda_reg=cfg["lambda_reg"],
        importance=importance,
        target_label=cfg["target_label"],
    )
    save_checkpoint(stable_model, checkpoint_dir / "stable_aware_watermark.pt")

    models = {
        "standard": standard_model,
        "stable_aware": stable_model,
    }

    print("pre-compression metrics")
    for name, model in models.items():
        acc = evaluate_acc(model, test_loader, device)
        wsr = evaluate_wsr(model, test_loader, device, cfg["target_label"])
        print(f"{name}: acc={acc:.4f} wsr={wsr:.4f}")

    pruning_rows = []
    for ratio in cfg["pruning_ratios"]:
        for name, model in models.items():
            compressed = copy.deepcopy(model).to(device)
            apply_pruning(compressed, ratio)
            acc = evaluate_acc(compressed, test_loader, device)
            wsr = evaluate_wsr(compressed, test_loader, device, cfg["target_label"])
            pruning_rows.append(
                {
                    "model": name,
                    "pruning_ratio": ratio,
                    "acc": acc,
                    "wsr": wsr,
                }
            )
            save_checkpoint(compressed, checkpoint_dir / f"{name}_pruned_{ratio:.1f}.pt")
            print(f"prune {ratio:.1f} {name}: acc={acc:.4f} wsr={wsr:.4f}")

    quantization_rows = []
    for bits in cfg["quantization_bits"]:
        for name, model in models.items():
            compressed = apply_fake_quantization(model, bits).to(device)
            acc = evaluate_acc(compressed, test_loader, device)
            wsr = evaluate_wsr(compressed, test_loader, device, cfg["target_label"])
            quantization_rows.append(
                {
                    "model": name,
                    "bits": bits,
                    "acc": acc,
                    "wsr": wsr,
                }
            )
            save_checkpoint(compressed, checkpoint_dir / f"{name}_fake_quant_{bits}bit.pt")
            print(f"quant {bits}bit {name}: acc={acc:.4f} wsr={wsr:.4f}")

    write_csv(
        output_dir / "results_pruning.csv",
        pruning_rows,
        ["model", "pruning_ratio", "acc", "wsr"],
    )
    write_csv(
        output_dir / "results_quantization.csv",
        quantization_rows,
        ["model", "bits", "acc", "wsr"],
    )
    plot_all(pruning_rows, quantization_rows, figure_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compression-robust trigger-based model watermarking on MNIST."
    )
    parser.add_argument("--config", default="configs/mnist.yaml")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", default=None, help="Device string, e.g. cpu, cuda, cuda:0, cuda:1")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--clean-epochs", type=int, default=None)
    parser.add_argument("--wm-epochs", type=int, default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    if args.cpu:
        cfg["cpu"] = True
        cfg["device"] = "cpu"
    if args.device is not None:
        cfg["device"] = args.device
    for key in ["train_subset", "clean_epochs", "wm_epochs"]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    run(config)
