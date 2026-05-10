import copy
import csv
from pathlib import Path

import torch

from compression import apply_fake_quantization, apply_pruning
from data import make_loaders, set_seed
from evaluate import evaluate_thesis_metrics
from importance import compute_importance
from masks import create_direct_masks, selected_quant_error, selected_survival_rate
from models import build_model
from watermark import train_clean, train_mask_direct_watermark, train_watermark


THESIS_FIELDS = [
    "dataset",
    "model",
    "seed",
    "method",
    "compression_type",
    "compression_level",
    "acc",
    "wsr",
    "wsr_non_target",
    "clean_target_rate",
    "pred_label_distribution",
    "selected_survival_rate",
    "selected_quant_error",
]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=THESIS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _method_label(method):
    return "stable_aware" if method == "stable_aware_reg" else method


def _save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _train_method(method, cfg, clean_state, importance, train_loader, device, seed):
    model = build_model(cfg["dataset"], cfg["model_name"]).to(device)
    model.load_state_dict(clean_state)

    if method == "standard":
        train_watermark(
            model,
            clean_state,
            train_loader,
            device,
            cfg["epochs_watermark"],
            cfg["lr"],
            cfg["lambda_wm"],
            target_label=cfg["target_label"],
            dataset=cfg["dataset"],
            trigger_size=cfg["trigger_size"],
            poison_ratio=cfg["poison_ratio"],
        )
        return model, {}

    if method == "stable_aware_reg":
        train_watermark(
            model,
            clean_state,
            train_loader,
            device,
            cfg["epochs_watermark"],
            cfg["lr"],
            cfg["lambda_wm"],
            lambda_reg=cfg["lambda_reg"],
            importance=importance,
            target_label=cfg["target_label"],
            dataset=cfg["dataset"],
            trigger_size=cfg["trigger_size"],
            poison_ratio=cfg["poison_ratio"],
        )
        return model, {}

    if method in ["stable_mask_direct", "random_mask_direct"]:
        selection_mode = "random" if method == "random_mask_direct" else cfg["selection_mode"]
        masks = create_direct_masks(
            model,
            importance,
            cfg["stable_mask_percent"],
            cfg["mask_granularity"],
            selection_mode,
            cfg["quant_stable_bits"],
            cfg["quant_error_alpha"],
            random_mask=(method == "random_mask_direct"),
            seed=seed,
        )
        train_mask_direct_watermark(
            model,
            train_loader,
            masks,
            device,
            cfg["epochs_watermark"],
            cfg["lr"],
            cfg["lambda_wm"],
            target_label=cfg["target_label"],
            dataset=cfg["dataset"],
            trigger_size=cfg["trigger_size"],
            poison_ratio=cfg["poison_ratio"],
        )
        return model, masks

    raise ValueError(f"Unsupported thesis method: {method}")


def _result_row(cfg, seed, method, compression_type, compression_level, metrics, survival, quant_error):
    return {
        "dataset": cfg["dataset"],
        "model": cfg["model_name"],
        "seed": seed,
        "method": method,
        "compression_type": compression_type,
        "compression_level": compression_level,
        "acc": metrics["acc"],
        "wsr": metrics["wsr"],
        "wsr_non_target": metrics["wsr_non_target"],
        "clean_target_rate": metrics["clean_target_rate"],
        "pred_label_distribution": metrics["pred_label_distribution"],
        "selected_survival_rate": survival,
        "selected_quant_error": quant_error,
    }


def _plot_metric(rows, compression_type, metric, output_path, ylabel):
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["compression_type"] == compression_type]
    if not selected:
        return
    x_key = "compression_level"
    plt.figure(figsize=(6, 4))
    for method in sorted({row["method"] for row in selected}):
        method_rows = [row for row in selected if row["method"] == method]
        levels = sorted({float(row[x_key]) for row in method_rows})
        values = []
        for level in levels:
            level_values = [float(row[metric]) for row in method_rows if float(row[x_key]) == level]
            values.append(sum(level_values) / len(level_values))
        plt.plot(levels, values, marker="o", label=method)
    plt.xlabel("Pruning ratio" if compression_type == "pruning" else "Quantization bits")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {compression_type}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_all(rows, figure_dir):
    figure_dir = Path(figure_dir)
    _plot_metric(rows, "pruning", "wsr", figure_dir / "wsr_vs_pruning.png", "WSR")
    _plot_metric(rows, "pruning", "acc", figure_dir / "acc_vs_pruning.png", "Accuracy")
    _plot_metric(rows, "quantization", "wsr", figure_dir / "wsr_vs_quantization.png", "WSR")
    _plot_metric(rows, "quantization", "acc", figure_dir / "acc_vs_quantization.png", "Accuracy")
    _plot_metric(
        rows,
        "quantization",
        "clean_target_rate",
        figure_dir / "clean_target_rate_vs_quantization.png",
        "Clean target rate",
    )
    _plot_metric(
        rows,
        "pruning",
        "selected_survival_rate",
        figure_dir / "selected_survival_rate_vs_pruning.png",
        "Selected survival rate",
    )


def run_thesis(cfg, device):
    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"
    rows = []

    for seed in cfg["seeds"]:
        print(f"thesis seed={seed}")
        set_seed(seed)
        train_loader, test_loader = make_loaders(
            cfg["data_dir"],
            cfg["batch_size"],
            cfg["num_workers"],
            seed,
            cfg.get("train_subset"),
            cfg["dataset"],
        )

        clean_model = build_model(cfg["dataset"], cfg["model_name"]).to(device)
        train_clean(clean_model, train_loader, device, cfg["epochs_clean"], cfg["lr"])
        clean_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in clean_model.state_dict().items()
        }
        _save_checkpoint(
            clean_model,
            checkpoint_dir / f"{cfg['dataset']}_{cfg['model_name']}_seed{seed}_clean.pt",
        )

        importance = compute_importance(clean_model, train_loader, device, cfg["importance_batches"])
        importance = {name: score.detach() for name, score in importance.items()}

        trained = {}
        masks_by_method = {}
        for method in cfg["methods"]:
            print(f"training method={method}")
            model, masks = _train_method(method, cfg, clean_state, importance, train_loader, device, seed)
            trained[method] = model
            masks_by_method[method] = masks
            _save_checkpoint(
                model,
                checkpoint_dir / f"{cfg['dataset']}_{cfg['model_name']}_seed{seed}_{method}.pt",
            )

        for method, model in trained.items():
            masks = masks_by_method[method]
            for ratio in cfg["pruning_ratios"]:
                compressed = copy.deepcopy(model).to(device)
                apply_pruning(compressed, ratio)
                metrics = evaluate_thesis_metrics(
                    compressed,
                    test_loader,
                    device,
                    target_label=cfg["target_label"],
                    dataset=cfg["dataset"],
                    trigger_size=cfg["trigger_size"],
                )
                survival = selected_survival_rate(compressed, masks) if masks else float("nan")
                rows.append(
                    _result_row(
                        cfg,
                        seed,
                        method,
                        "pruning",
                        ratio,
                        metrics,
                        survival,
                        float("nan"),
                    )
                )

            for bits in cfg["quant_bits"]:
                compressed = apply_fake_quantization(model, bits).to(device)
                metrics = evaluate_thesis_metrics(
                    compressed,
                    test_loader,
                    device,
                    target_label=cfg["target_label"],
                    dataset=cfg["dataset"],
                    trigger_size=cfg["trigger_size"],
                )
                quant_error = selected_quant_error(model, compressed, masks) if masks else float("nan")
                rows.append(
                    _result_row(
                        cfg,
                        seed,
                        method,
                        "quantization",
                        bits,
                        metrics,
                        float("nan"),
                        quant_error,
                    )
                )

    _write_csv(output_dir / "results_all.csv", rows)
    _plot_all(rows, figure_dir)
    return rows


def default_thesis_config(args):
    epochs_clean = 3 if args.quick_test else 20
    epochs_watermark = 1 if args.quick_test else 5
    pruning_ratios = [0.0, 0.5, 0.9] if args.quick_test else [0.0, 0.5, 0.7, 0.9]
    quant_bits = [8, 4] if args.quick_test else [8, 4, 3]
    model_name = args.model or "cifar_small"
    methods = args.methods or ["standard", "stable_aware_reg", "stable_mask_direct", "random_mask_direct"]
    seeds = args.seeds or [42, 43, 44]
    stable_mask_percent = 0.1 if args.stable_mask_percent is None else args.stable_mask_percent
    mask_granularity = args.mask_granularity or "channel"
    selection_mode = args.selection_mode or "fisher_top"
    quant_stable_bits = 4 if args.quant_stable_bits is None else args.quant_stable_bits
    quant_error_alpha = 0.5 if args.quant_error_alpha is None else args.quant_error_alpha
    return {
        "dataset": args.dataset or "cifar10",
        "model_name": model_name,
        "methods": methods,
        "seeds": seeds,
        "data_dir": "data",
        "output_dir": "outputs/thesis_results",
        "batch_size": 128,
        "num_workers": 2,
        "train_subset": args.train_subset,
        "epochs_clean": epochs_clean,
        "epochs_watermark": epochs_watermark,
        "lr": 0.001,
        "poison_ratio": 0.01,
        "trigger_size": 3,
        "target_label": 0,
        "lambda_wm": 1.0,
        "lambda_reg": 0.1,
        "importance_batches": 100,
        "stable_mask_percent": stable_mask_percent,
        "mask_granularity": mask_granularity,
        "selection_mode": selection_mode,
        "quant_stable_bits": quant_stable_bits,
        "quant_error_alpha": quant_error_alpha,
        "pruning_ratios": pruning_ratios,
        "quant_bits": quant_bits,
    }
