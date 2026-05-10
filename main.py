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
from models import build_model
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


def resolve_epochs(cfg):
    clean_epochs = cfg.get("clean_epochs", cfg.get("epochs_clean"))
    wm_epochs = cfg.get("wm_epochs", cfg.get("epochs_watermark"))
    return clean_epochs, wm_epochs


def get_quantization_bits(cfg):
    return cfg.get("quantization_bits", cfg.get("quant_bits"))


def run(cfg):
    set_seed(cfg["seed"])
    dataset = cfg.get("dataset", "mnist")
    model_name = cfg.get("model_name", "small_cnn" if dataset == "mnist" else "cifar_small_cnn")
    trigger_size = cfg.get("trigger_size")
    poison_ratio = cfg.get("poison_ratio", 1.0)
    clean_epochs, wm_epochs = resolve_epochs(cfg)
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
        dataset,
    )

    clean_model = build_model(dataset).to(device)
    train_clean(clean_model, train_loader, device, clean_epochs, cfg["lr"])
    clean_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in clean_model.state_dict().items()
    }
    clean_checkpoint = "clean.pt" if dataset == "mnist" else f"{dataset}_clean.pt"
    standard_checkpoint = (
        "standard_watermark.pt" if dataset == "mnist" else f"{dataset}_standard_wm.pt"
    )
    stable_checkpoint = (
        "stable_aware_watermark.pt" if dataset == "mnist" else f"{dataset}_stable_aware_wm.pt"
    )
    save_checkpoint(clean_model, checkpoint_dir / clean_checkpoint)

    importance = compute_importance(
        clean_model,
        train_loader,
        device,
        cfg["importance_batches"],
    )
    importance = {name: score.detach() for name, score in importance.items()}

    standard_model = build_model(dataset).to(device)
    standard_model.load_state_dict(clean_state)
    train_watermark(
        standard_model,
        clean_state,
        train_loader,
        device,
        wm_epochs,
        cfg["lr"],
        cfg["lambda_wm"],
        target_label=cfg["target_label"],
        dataset=dataset,
        trigger_size=trigger_size,
        poison_ratio=poison_ratio,
    )
    save_checkpoint(standard_model, checkpoint_dir / standard_checkpoint)

    stable_model = build_model(dataset).to(device)
    stable_model.load_state_dict(clean_state)
    train_watermark(
        stable_model,
        clean_state,
        train_loader,
        device,
        wm_epochs,
        cfg["lr"],
        cfg["lambda_wm"],
        lambda_reg=cfg["lambda_reg"],
        importance=importance,
        target_label=cfg["target_label"],
        dataset=dataset,
        trigger_size=trigger_size,
        poison_ratio=poison_ratio,
    )
    save_checkpoint(stable_model, checkpoint_dir / stable_checkpoint)

    models = {
        "standard": standard_model,
        "stable_aware": stable_model,
    }

    print("pre-compression metrics")
    for name, model in models.items():
        acc = evaluate_acc(model, test_loader, device)
        wsr = evaluate_wsr(
            model,
            test_loader,
            device,
            cfg["target_label"],
            dataset=dataset,
            trigger_size=trigger_size,
        )
        print(f"{name}: acc={acc:.4f} wsr={wsr:.4f}")

    pruning_rows = []
    for ratio in cfg["pruning_ratios"]:
        for name, model in models.items():
            compressed = copy.deepcopy(model).to(device)
            apply_pruning(compressed, ratio)
            acc = evaluate_acc(compressed, test_loader, device)
            wsr = evaluate_wsr(
                compressed,
                test_loader,
                device,
                cfg["target_label"],
                dataset=dataset,
                trigger_size=trigger_size,
            )
            if dataset == "mnist":
                row = {"model": name, "pruning_ratio": ratio, "acc": acc, "wsr": wsr}
            else:
                row = {
                    "dataset": dataset,
                    "model": model_name,
                    "method": name,
                    "pruning_ratio": ratio,
                    "acc": acc,
                    "wsr": wsr,
                }
            pruning_rows.append(row)
            save_checkpoint(compressed, checkpoint_dir / f"{name}_pruned_{ratio:.1f}.pt")
            print(f"prune {ratio:.1f} {name}: acc={acc:.4f} wsr={wsr:.4f}")

    quantization_rows = []
    for bits in get_quantization_bits(cfg):
        for name, model in models.items():
            compressed = apply_fake_quantization(model, bits).to(device)
            acc = evaluate_acc(compressed, test_loader, device)
            wsr = evaluate_wsr(
                compressed,
                test_loader,
                device,
                cfg["target_label"],
                dataset=dataset,
                trigger_size=trigger_size,
            )
            if dataset == "mnist":
                row = {"model": name, "bits": bits, "acc": acc, "wsr": wsr}
            else:
                row = {
                    "dataset": dataset,
                    "model": model_name,
                    "method": name,
                    "bits": bits,
                    "acc": acc,
                    "wsr": wsr,
                }
            quantization_rows.append(row)
            save_checkpoint(compressed, checkpoint_dir / f"{name}_fake_quant_{bits}bit.pt")
            print(f"quant {bits}bit {name}: acc={acc:.4f} wsr={wsr:.4f}")

    if dataset == "mnist":
        pruning_path = output_dir / "results_pruning.csv"
        quantization_path = output_dir / "results_quantization.csv"
        pruning_fields = ["model", "pruning_ratio", "acc", "wsr"]
        quantization_fields = ["model", "bits", "acc", "wsr"]
        plot_all(pruning_rows, quantization_rows, figure_dir)
    else:
        pruning_path = output_dir / f"{dataset}_results_pruning.csv"
        quantization_path = output_dir / f"{dataset}_results_quantization.csv"
        pruning_fields = ["dataset", "model", "method", "pruning_ratio", "acc", "wsr"]
        quantization_fields = ["dataset", "model", "method", "bits", "acc", "wsr"]
        plot_all(pruning_rows, quantization_rows, figure_dir, prefix=dataset, group_key="method")

    write_csv(pruning_path, pruning_rows, pruning_fields)
    write_csv(quantization_path, quantization_rows, quantization_fields)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compression-robust trigger-based model watermarking."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", default=None, help="Device string, e.g. cpu, cuda, cuda:0, cuda:1")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--clean-epochs", type=int, default=None)
    parser.add_argument("--wm-epochs", type=int, default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.cpu:
        cfg["cpu"] = True
        cfg["device"] = "cpu"
    if args.device is not None:
        cfg["device"] = args.device
    for key in ["train_subset", "clean_epochs", "wm_epochs"]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.quick_test:
        cfg["clean_epochs"] = 3
        cfg["wm_epochs"] = 1
        cfg["pruning_ratios"] = [0.0, 0.5, 0.9]
        cfg["quantization_bits"] = [8, 4]
        cfg["quant_bits"] = [8, 4]
    return cfg


if __name__ == "__main__":
    args = parse_args()
    dataset = "mnist" if args.dataset is None else args.dataset
    config_path = args.config if args.config is not None else f"configs/{dataset}.yaml"
    config = apply_cli_overrides(load_config(config_path), args)
    run(config)
