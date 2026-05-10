import argparse
import copy
import csv
import itertools
from pathlib import Path

import torch
import yaml

from compression import apply_fake_quantization, apply_pruning
from data import make_loaders, set_seed
from evaluate import evaluate_acc, evaluate_wsr
from importance import compute_importance
from models import build_model
from plot_results import plot_all
from thesis import (
    default_thesis_config,
    run_direct_embedding_sweep,
    run_direct_embedding_diagnostic,
    run_resnet_precompression_diagnostic,
    run_resnet_watermark_sweep,
    run_thesis,
)
from watermark import train_clean, train_watermark


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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


def format_float(value):
    return str(value).replace(".", "p")


def experiment_tag(cfg, multi_run):
    if not multi_run:
        return ""
    parts = [f"seed{cfg['seed']}"]
    if cfg.get("method") == "lambda_reg_ablation":
        parts.append(f"reg{format_float(cfg['lambda_reg'])}")
    return "_" + "_".join(parts)


def checkpoint_name(dataset, model_name, role, tag, multi_run):
    if dataset == "mnist":
        names = {
            "clean": "clean.pt",
            "standard": "standard_watermark.pt",
            "stable_aware": "stable_aware_watermark.pt",
        }
        return names[role]
    if dataset == "cifar10" and model_name == "cifar_small_cnn" and not multi_run:
        names = {
            "clean": "cifar10_clean.pt",
            "standard": "cifar10_standard_wm.pt",
            "stable_aware": "cifar10_stable_aware_wm.pt",
        }
        return names[role]
    role_name = {
        "clean": "clean",
        "standard": "standard_wm",
        "stable_aware": "stable_aware_wm",
    }[role]
    return f"{dataset}_{model_name}_{role_name}{tag}.pt"


def plot_series(cfg, method):
    if cfg.get("method") == "lambda_reg_ablation" and method == "stable_aware":
        return f"{method}_reg_{cfg['lambda_reg']}"
    return method


def output_prefix(dataset, cfg):
    prefix = cfg.get("output_prefix")
    if prefix is not None:
        return prefix
    return "" if dataset == "mnist" else dataset


def run_single(cfg, device, multi_run=False):
    set_seed(cfg["seed"])
    dataset = cfg.get("dataset", "mnist")
    model_name = cfg.get("model_name", "small_cnn" if dataset == "mnist" else "cifar_small_cnn")
    trigger_size = cfg.get("trigger_size")
    poison_ratio = cfg.get("poison_ratio", 1.0)
    clean_epochs, wm_epochs = resolve_epochs(cfg)

    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"

    train_loader, test_loader = make_loaders(
        cfg["data_dir"],
        cfg["batch_size"],
        cfg["num_workers"],
        cfg["seed"],
        cfg.get("train_subset"),
        dataset,
    )

    clean_model = build_model(dataset, model_name).to(device)
    train_clean(clean_model, train_loader, device, clean_epochs, cfg["lr"])
    clean_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in clean_model.state_dict().items()
    }
    tag = experiment_tag(cfg, multi_run)
    clean_checkpoint = checkpoint_name(dataset, model_name, "clean", tag, multi_run)
    standard_checkpoint = checkpoint_name(dataset, model_name, "standard", tag, multi_run)
    stable_checkpoint = checkpoint_name(dataset, model_name, "stable_aware", tag, multi_run)
    save_checkpoint(clean_model, checkpoint_dir / clean_checkpoint)

    importance = compute_importance(
        clean_model,
        train_loader,
        device,
        cfg["importance_batches"],
    )
    importance = {name: score.detach() for name, score in importance.items()}

    standard_model = build_model(dataset, model_name).to(device)
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

    stable_model = build_model(dataset, model_name).to(device)
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
                    "series": plot_series(cfg, name),
                    "seed": cfg["seed"],
                    "lambda_reg": cfg["lambda_reg"],
                    "pruning_ratio": ratio,
                    "acc": acc,
                    "wsr": wsr,
                }
            pruning_rows.append(row)
            compressed_name = f"{dataset}_{model_name}_{name}_pruned_{ratio:.1f}{tag}.pt"
            save_checkpoint(compressed, checkpoint_dir / compressed_name)
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
                    "series": plot_series(cfg, name),
                    "seed": cfg["seed"],
                    "lambda_reg": cfg["lambda_reg"],
                    "bits": bits,
                    "acc": acc,
                    "wsr": wsr,
                }
            quantization_rows.append(row)
            compressed_name = f"{dataset}_{model_name}_{name}_fake_quant_{bits}bit{tag}.pt"
            save_checkpoint(compressed, checkpoint_dir / compressed_name)
            print(f"quant {bits}bit {name}: acc={acc:.4f} wsr={wsr:.4f}")

    return pruning_rows, quantization_rows


def expand_experiments(cfg):
    seeds = cfg.get("seeds", [cfg["seed"]])
    lambda_regs = cfg.get("lambda_reg_values", [cfg["lambda_reg"]])
    experiments = []
    for seed, lambda_reg in itertools.product(seeds, lambda_regs):
        run_cfg = copy.deepcopy(cfg)
        run_cfg["seed"] = seed
        run_cfg["lambda_reg"] = lambda_reg
        experiments.append(run_cfg)
    return experiments


def run(cfg):
    dataset = cfg.get("dataset", "mnist")
    device_name = cfg.get("device")
    if device_name is None:
        use_cuda = torch.cuda.is_available() and not cfg.get("cpu", False)
        device_name = "cuda" if use_cuda else "cpu"
    device = torch.device(device_name)
    print(f"device={device}")

    output_dir = Path(cfg["output_dir"])
    figure_dir = output_dir / "figures"
    experiments = expand_experiments(cfg)
    multi_run = len(experiments) > 1

    pruning_rows = []
    quantization_rows = []
    for index, run_cfg in enumerate(experiments, start=1):
        print(
            f"run {index}/{len(experiments)} "
            f"dataset={run_cfg.get('dataset', 'mnist')} "
            f"model={run_cfg.get('model_name', 'small_cnn' if dataset == 'mnist' else 'cifar_small_cnn')} "
            f"seed={run_cfg['seed']} lambda_reg={run_cfg['lambda_reg']}"
        )
        run_pruning_rows, run_quantization_rows = run_single(run_cfg, device, multi_run)
        pruning_rows.extend(run_pruning_rows)
        quantization_rows.extend(run_quantization_rows)

    if dataset == "mnist":
        pruning_path = output_dir / "results_pruning.csv"
        quantization_path = output_dir / "results_quantization.csv"
        pruning_fields = ["model", "pruning_ratio", "acc", "wsr"]
        quantization_fields = ["model", "bits", "acc", "wsr"]
        plot_all(pruning_rows, quantization_rows, figure_dir)
    else:
        prefix = output_prefix(dataset, cfg)
        pruning_path = output_dir / f"{prefix}_results_pruning.csv"
        quantization_path = output_dir / f"{prefix}_results_quantization.csv"
        pruning_fields = ["dataset", "model", "method", "pruning_ratio", "acc", "wsr"]
        quantization_fields = ["dataset", "model", "method", "bits", "acc", "wsr"]
        if multi_run:
            pruning_fields = [
                "dataset",
                "model",
                "method",
                "seed",
                "lambda_reg",
                "pruning_ratio",
                "acc",
                "wsr",
            ]
            quantization_fields = [
                "dataset",
                "model",
                "method",
                "seed",
                "lambda_reg",
                "bits",
                "acc",
                "wsr",
            ]
        group_key = "series" if cfg.get("method") == "lambda_reg_ablation" else "method"
        plot_all(pruning_rows, quantization_rows, figure_dir, prefix=prefix, group_key=group_key)

    write_csv(pruning_path, pruning_rows, pruning_fields)
    write_csv(quantization_path, quantization_rows, quantization_fields)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compression-robust trigger-based model watermarking."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--model", choices=["cifar_small", "cifar_small_cnn", "resnet18_cifar"], default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["standard", "stable_aware_reg", "stable_mask_direct", "random_mask_direct"],
        default=None,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--stable-mask-percent", type=float, default=None)
    parser.add_argument("--mask-granularity", choices=["parameter", "channel"], default=None)
    parser.add_argument("--selection-mode", choices=["fisher_top", "quant_stable", "random"], default=None)
    parser.add_argument("--quant-stable-bits", type=int, default=None)
    parser.add_argument("--quant-error-alpha", type=float, default=None)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--direct-sweep", action="store_true")
    parser.add_argument("--resnet-wm-sweep", action="store_true")
    parser.add_argument("--poster-resnet-diagnostic", action="store_true")
    parser.add_argument("--poster-direct-diagnostic", action="store_true")
    parser.add_argument("--lambda-wm", type=float, default=None)
    parser.add_argument("--lambda-reg", type=float, default=None)
    parser.add_argument("--poison-ratio", type=float, default=None)
    parser.add_argument("--trigger-size", type=int, default=None)
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--learning-rate-watermark", type=float, default=None)
    parser.add_argument("--watermark-steps-per-batch", type=int, default=None)
    parser.add_argument("--watermark-train-mode", choices=["joint", "alternating"], default=None)
    parser.add_argument("--direct-embedding-mode", choices=["joint", "wm_focused"], default=None)
    parser.add_argument("--lambda-clean", type=float, default=None)
    parser.add_argument("--use-activation-guidance", action="store_true")
    parser.add_argument("--activation-layer", default=None)
    parser.add_argument("--lambda-act", type=float, default=None)
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
    if (
        args.methods is not None
        or args.model is not None
        or args.seeds is not None
        or args.direct_sweep
        or args.resnet_wm_sweep
        or args.poster_resnet_diagnostic
        or args.poster_direct_diagnostic
    ):
        thesis_config = default_thesis_config(args)
        device_name = args.device
        if device_name is None:
            use_cuda = torch.cuda.is_available() and not args.cpu
            device_name = "cuda" if use_cuda else "cpu"
        if args.direct_sweep:
            run_direct_embedding_sweep(thesis_config, torch.device(device_name))
        elif args.resnet_wm_sweep:
            run_resnet_watermark_sweep(thesis_config, torch.device(device_name))
        elif args.poster_resnet_diagnostic:
            run_resnet_precompression_diagnostic(args, torch.device(device_name))
        elif args.poster_direct_diagnostic:
            run_direct_embedding_diagnostic(args, torch.device(device_name))
        else:
            run_thesis(thesis_config, torch.device(device_name))
        raise SystemExit(0)

    dataset = "mnist" if args.dataset is None else args.dataset
    config_path = args.config if args.config is not None else f"configs/{dataset}.yaml"
    config = apply_cli_overrides(load_config(config_path), args)
    run(config)
