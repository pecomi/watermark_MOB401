import copy
import csv
import itertools
import math
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
    "stable_mask_percent",
    "lambda_wm",
    "poison_ratio",
    "watermark_steps_per_batch",
    "direct_embedding_mode",
    "use_activation_guidance",
    "lambda_act",
    "diagnosis",
]

RESNET_SWEEP_FIELDS = [
    "dataset",
    "model",
    "seed",
    "method",
    "target_label",
    "poison_ratio",
    "trigger_size",
    "epochs_watermark",
    "lambda_wm",
    "lambda_reg",
    "learning_rate_watermark",
    "watermark_steps_per_batch",
    "watermark_train_mode",
    "acc",
    "wsr",
    "wsr_non_target",
    "clean_target_rate",
    "pred_label_distribution",
    "diagnosis",
]

POSTER_DIAGNOSTIC_FIELDS = [
    "dataset",
    "model",
    "seed",
    "method",
    "stable_mask_percent",
    "acc",
    "wsr",
    "wsr_non_target",
    "clean_target_rate",
    "pred_label_distribution",
    "diagnosis",
]

POSTER_SUMMARY_METRICS = ["acc", "wsr", "wsr_non_target", "clean_target_rate"]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=THESIS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _std(values):
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _majority(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else ""


def _summarize_rows(rows, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group_rows in groups.items():
        out = {group_key: value for group_key, value in zip(group_keys, key)}
        for metric in POSTER_SUMMARY_METRICS:
            values = [float(row[metric]) for row in group_rows]
            out[f"{metric}_mean"] = _mean(values)
            out[f"{metric}_std"] = _std(values)
        out["diagnosis_majority"] = _majority([row["diagnosis"] for row in group_rows])
        summary.append(out)
    return summary


def _poster_table(summary_rows, fields):
    return [{field: row.get(field, "") for field in fields} for row in summary_rows]


def _method_label(method):
    return "stable_aware" if method == "stable_aware_reg" else method


def _save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _passes_precompression_gate(metrics):
    return (
        metrics["acc"] >= 0.75
        and metrics["wsr_non_target"] >= 0.60
        and metrics["clean_target_rate"] <= 0.20
    )


def _gate_diagnosis(metrics):
    return metrics["diagnosis"] if _passes_precompression_gate(metrics) else "weak_precompression_watermark"


def _save_trigger_debug_grid(data_loader, dataset, trigger_size, output_path):
    from torchvision.utils import make_grid, save_image
    from watermark import add_trigger

    x, _ = next(iter(data_loader))
    clean = x[:8]
    triggered = add_trigger(clean, dataset=dataset, trigger_size=trigger_size)
    grid = make_grid(torch.cat([clean, triggered], dim=0), nrow=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)


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
            cfg.get("learning_rate_watermark", cfg["lr"]),
            cfg["lambda_wm"],
            target_label=cfg["target_label"],
            dataset=cfg["dataset"],
            trigger_size=cfg["trigger_size"],
            poison_ratio=cfg["poison_ratio"],
            watermark_train_mode=cfg.get("watermark_train_mode", "joint"),
            watermark_steps_per_batch=cfg.get("watermark_steps_per_batch", 1),
        )
        return model, {}

    if method == "stable_aware_reg":
        train_watermark(
            model,
            clean_state,
            train_loader,
            device,
            cfg["epochs_watermark"],
            cfg.get("learning_rate_watermark", cfg["lr"]),
            cfg["lambda_wm"],
            lambda_reg=cfg["lambda_reg"],
            importance=importance,
            target_label=cfg["target_label"],
            dataset=cfg["dataset"],
            trigger_size=cfg["trigger_size"],
            poison_ratio=cfg["poison_ratio"],
            watermark_train_mode=cfg.get("watermark_train_mode", "joint"),
            watermark_steps_per_batch=cfg.get("watermark_steps_per_batch", 1),
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
            cfg.get("learning_rate_watermark", cfg["lr"]),
            cfg["lambda_wm"],
            target_label=cfg["target_label"],
            dataset=cfg["dataset"],
            trigger_size=cfg["trigger_size"],
            poison_ratio=cfg["poison_ratio"],
            watermark_steps_per_batch=cfg["watermark_steps_per_batch"],
            direct_embedding_mode=cfg["direct_embedding_mode"],
            lambda_clean=cfg["lambda_clean"],
            use_activation_guidance=cfg["use_activation_guidance"],
            activation_layer=cfg["activation_layer"],
            lambda_act=cfg["lambda_act"],
        )
        return model, masks

    raise ValueError(f"Unsupported thesis method: {method}")


def _diagnostic_row(cfg, seed, method, metrics, stable_mask_percent=""):
    return {
        "dataset": cfg["dataset"],
        "model": cfg["model_name"],
        "seed": seed,
        "method": method,
        "stable_mask_percent": stable_mask_percent,
        "acc": metrics["acc"],
        "wsr": metrics["wsr"],
        "wsr_non_target": metrics["wsr_non_target"],
        "clean_target_rate": metrics["clean_target_rate"],
        "pred_label_distribution": metrics["pred_label_distribution"],
        "diagnosis": metrics["diagnosis"],
    }


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
        "stable_mask_percent": cfg["stable_mask_percent"],
        "lambda_wm": cfg["lambda_wm"],
        "poison_ratio": cfg["poison_ratio"],
        "watermark_steps_per_batch": cfg["watermark_steps_per_batch"],
        "direct_embedding_mode": cfg["direct_embedding_mode"],
        "use_activation_guidance": cfg["use_activation_guidance"],
        "lambda_act": cfg["lambda_act"],
        "diagnosis": metrics["diagnosis"],
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
            pre_metrics = evaluate_thesis_metrics(
                model,
                test_loader,
                device,
                target_label=cfg["target_label"],
                dataset=cfg["dataset"],
                trigger_size=cfg["trigger_size"],
            )
            if not _passes_precompression_gate(pre_metrics):
                pre_metrics = dict(pre_metrics)
                pre_metrics["diagnosis"] = "weak_precompression_watermark"
                rows.append(
                    _result_row(
                        cfg,
                        seed,
                        method,
                        "none",
                        0.0,
                        pre_metrics,
                        float("nan"),
                        float("nan"),
                    )
                )
                print(f"skipping compression for {method}: weak pre-compression watermark")
                continue
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


def _sweep_configs(base_cfg):
    for percent, lambda_wm, steps, poison_ratio, use_act, lambda_act in itertools.product(
        base_cfg["sweep_stable_mask_percents"],
        base_cfg["sweep_lambda_wm_values"],
        base_cfg["sweep_watermark_steps"],
        base_cfg["sweep_poison_ratios"],
        base_cfg["sweep_activation_guidance"],
        base_cfg["sweep_lambda_act_values"],
    ):
        if not use_act and lambda_act != base_cfg["sweep_lambda_act_values"][0]:
            continue
        cfg = copy.deepcopy(base_cfg)
        cfg["stable_mask_percent"] = percent
        cfg["lambda_wm"] = lambda_wm
        cfg["watermark_steps_per_batch"] = steps
        cfg["poison_ratio"] = poison_ratio
        cfg["use_activation_guidance"] = use_act
        cfg["lambda_act"] = lambda_act if use_act else 0.0
        yield cfg


def run_direct_embedding_sweep(cfg, device):
    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    rows = []
    seed = cfg["seeds"][0]
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
    _save_checkpoint(clean_model, checkpoint_dir / f"{cfg['dataset']}_{cfg['model_name']}_sweep_clean.pt")
    importance = compute_importance(clean_model, train_loader, device, cfg["importance_batches"])
    importance = {name: score.detach() for name, score in importance.items()}

    for index, sweep_cfg in enumerate(_sweep_configs(cfg), start=1):
        print(
            "sweep "
            f"{index}: mask={sweep_cfg['stable_mask_percent']} "
            f"lambda_wm={sweep_cfg['lambda_wm']} "
            f"steps={sweep_cfg['watermark_steps_per_batch']} "
            f"poison={sweep_cfg['poison_ratio']} "
            f"act={sweep_cfg['use_activation_guidance']} "
            f"lambda_act={sweep_cfg['lambda_act']}"
        )
        model, masks = _train_method(
            "stable_mask_direct",
            sweep_cfg,
            clean_state,
            importance,
            train_loader,
            device,
            seed,
        )
        metrics = evaluate_thesis_metrics(
            model,
            test_loader,
            device,
            target_label=sweep_cfg["target_label"],
            dataset=sweep_cfg["dataset"],
            trigger_size=sweep_cfg["trigger_size"],
        )
        row = _result_row(
            sweep_cfg,
            seed,
            "stable_mask_direct",
            "none",
            0.0,
            metrics,
            selected_survival_rate(model, masks),
            float("nan"),
        )
        rows.append(row)
        if metrics["acc"] >= 0.75 and metrics["wsr_non_target"] >= 0.6 and metrics["clean_target_rate"] <= 0.2:
            print("candidate found: acc>=0.75, wsr_non_target>=0.6, clean_target_rate<=0.2")

    _write_csv(output_dir / "direct_embedding_sweep.csv", rows)
    return rows


def _resnet_sweep_configs(base_cfg):
    for poison_ratio, trigger_size, epochs, lambda_wm, lr_wm, steps, target_label in itertools.product(
        [0.01, 0.03, 0.05, 0.10],
        [3, 4, 5],
        [5, 10, 20],
        [1.0, 2.0, 5.0, 10.0],
        [0.001, 0.0005, 0.0001],
        [1, 2, 3],
        [0, 1, 2, 3, 5, 7],
    ):
        for method in ["standard", "stable_aware_reg"]:
            lambda_regs = [0.0] if method == "standard" else [0.01, 0.1, 1.0]
            for lambda_reg in lambda_regs:
                cfg = copy.deepcopy(base_cfg)
                cfg.update(
                    {
                        "method": method,
                        "poison_ratio": poison_ratio,
                        "trigger_size": trigger_size,
                        "epochs_watermark": epochs,
                        "lambda_wm": lambda_wm,
                        "lambda_reg": lambda_reg,
                        "learning_rate_watermark": lr_wm,
                        "watermark_steps_per_batch": steps,
                        "target_label": target_label,
                    }
                )
                yield cfg


def run_resnet_watermark_sweep(cfg, device):
    output_dir = Path("outputs")
    debug_dir = output_dir / "debug"
    rows = []
    seed = cfg["seeds"][0]
    set_seed(seed)
    train_loader, test_loader = make_loaders(
        cfg["data_dir"],
        cfg["batch_size"],
        cfg["num_workers"],
        seed,
        cfg.get("train_subset"),
        cfg["dataset"],
    )
    _save_trigger_debug_grid(train_loader, cfg["dataset"], 3, debug_dir / "trigger_examples.png")
    print("ResNet18-CIFAR check: conv1=3x3 stride1 padding1, maxpool=Identity, fc=10")
    print("CIFAR transform: ToTensor only; trigger_pixel_value=1.0")

    clean_model = build_model(cfg["dataset"], cfg["model_name"]).to(device)
    train_clean(clean_model, train_loader, device, cfg["epochs_clean"], cfg["lr"])
    clean_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in clean_model.state_dict().items()
    }
    importance = compute_importance(clean_model, train_loader, device, cfg["importance_batches"])
    importance = {name: score.detach() for name, score in importance.items()}

    for index, sweep_cfg in enumerate(_resnet_sweep_configs(cfg), start=1):
        print(
            f"resnet sweep {index}: method={sweep_cfg['method']} "
            f"target={sweep_cfg['target_label']} poison={sweep_cfg['poison_ratio']} "
            f"trigger={sweep_cfg['trigger_size']} epochs={sweep_cfg['epochs_watermark']} "
            f"lambda_wm={sweep_cfg['lambda_wm']} lr_wm={sweep_cfg['learning_rate_watermark']} "
            f"steps={sweep_cfg['watermark_steps_per_batch']} triggered_samples_per_batch="
            f"{max(1, math.floor(sweep_cfg['poison_ratio'] * cfg['batch_size']))}"
        )
        model = build_model(sweep_cfg["dataset"], sweep_cfg["model_name"]).to(device)
        model.load_state_dict(clean_state)
        train_watermark(
            model,
            clean_state,
            train_loader,
            device,
            sweep_cfg["epochs_watermark"],
            sweep_cfg["learning_rate_watermark"],
            sweep_cfg["lambda_wm"],
            lambda_reg=sweep_cfg["lambda_reg"],
            importance=importance,
            target_label=sweep_cfg["target_label"],
            dataset=sweep_cfg["dataset"],
            trigger_size=sweep_cfg["trigger_size"],
            poison_ratio=sweep_cfg["poison_ratio"],
            watermark_train_mode=sweep_cfg["watermark_train_mode"],
            watermark_steps_per_batch=sweep_cfg["watermark_steps_per_batch"],
        )
        metrics = evaluate_thesis_metrics(
            model,
            test_loader,
            device,
            target_label=sweep_cfg["target_label"],
            dataset=sweep_cfg["dataset"],
            trigger_size=sweep_cfg["trigger_size"],
        )
        row = {
            "dataset": sweep_cfg["dataset"],
            "model": sweep_cfg["model_name"],
            "seed": seed,
            "method": sweep_cfg["method"],
            "target_label": sweep_cfg["target_label"],
            "poison_ratio": sweep_cfg["poison_ratio"],
            "trigger_size": sweep_cfg["trigger_size"],
            "epochs_watermark": sweep_cfg["epochs_watermark"],
            "lambda_wm": sweep_cfg["lambda_wm"],
            "lambda_reg": sweep_cfg["lambda_reg"],
            "learning_rate_watermark": sweep_cfg["learning_rate_watermark"],
            "watermark_steps_per_batch": sweep_cfg["watermark_steps_per_batch"],
            "watermark_train_mode": sweep_cfg["watermark_train_mode"],
            "acc": metrics["acc"],
            "wsr": metrics["wsr"],
            "wsr_non_target": metrics["wsr_non_target"],
            "clean_target_rate": metrics["clean_target_rate"],
            "pred_label_distribution": metrics["pred_label_distribution"],
            "diagnosis": _gate_diagnosis(metrics),
        }
        rows.append(row)
        if _passes_precompression_gate(metrics):
            print("candidate found for ResNet watermark gate")

    _write_rows(output_dir / "resnet_wm_sweep.csv", rows, RESNET_SWEEP_FIELDS)
    return rows


def _poster_base_cfg(args, methods):
    cfg = default_thesis_config(args)
    cfg.update(
        {
            "dataset": "cifar10",
            "model_name": "resnet18_cifar",
            "methods": methods,
            "seeds": args.seeds or [42, 43, 44],
            "output_dir": "outputs/poster_results",
            "mask_granularity": "channel",
        }
    )
    return cfg


def run_resnet_precompression_diagnostic(args, device):
    cfg = _poster_base_cfg(args, ["standard", "stable_aware_reg"])
    output_dir = Path(cfg["output_dir"])
    rows = []
    for seed in cfg["seeds"]:
        print(f"poster resnet diagnostic seed={seed}")
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
        importance = compute_importance(clean_model, train_loader, device, cfg["importance_batches"])
        importance = {name: score.detach() for name, score in importance.items()}

        for method in cfg["methods"]:
            model, _ = _train_method(method, cfg, clean_state, importance, train_loader, device, seed)
            metrics = evaluate_thesis_metrics(
                model,
                test_loader,
                device,
                target_label=cfg["target_label"],
                dataset=cfg["dataset"],
                trigger_size=cfg["trigger_size"],
            )
            rows.append(_diagnostic_row(cfg, seed, method, metrics))

    summary = _summarize_rows(rows, ["model", "method"])
    table_fields = ["model", "method", "acc_mean", "wsr_non_target_mean", "clean_target_rate_mean", "diagnosis_majority"]
    _write_rows(output_dir / "resnet_precompression_diagnostic.csv", rows, POSTER_DIAGNOSTIC_FIELDS)
    _write_rows(
        output_dir / "resnet_precompression_summary.csv",
        summary,
        ["model", "method"]
        + [f"{metric}_{suffix}" for metric in POSTER_SUMMARY_METRICS for suffix in ["mean", "std"]]
        + ["diagnosis_majority"],
    )
    _write_rows(output_dir / "table_resnet_diagnostic.csv", _poster_table(summary, table_fields), table_fields)
    return rows


def run_direct_embedding_diagnostic(args, device):
    cfg = _poster_base_cfg(args, ["stable_mask_direct", "random_mask_direct"])
    output_dir = Path(cfg["output_dir"])
    rows = []
    for seed in cfg["seeds"]:
        print(f"poster direct diagnostic seed={seed}")
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
        importance = compute_importance(clean_model, train_loader, device, cfg["importance_batches"])
        importance = {name: score.detach() for name, score in importance.items()}

        for stable_mask_percent in [0.1, 0.3]:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["stable_mask_percent"] = stable_mask_percent
            for method in cfg["methods"]:
                model, _ = _train_method(method, run_cfg, clean_state, importance, train_loader, device, seed)
                metrics = evaluate_thesis_metrics(
                    model,
                    test_loader,
                    device,
                    target_label=run_cfg["target_label"],
                    dataset=run_cfg["dataset"],
                    trigger_size=run_cfg["trigger_size"],
                )
                rows.append(_diagnostic_row(run_cfg, seed, method, metrics, stable_mask_percent))

    summary = _summarize_rows(rows, ["model", "method", "stable_mask_percent"])
    table_fields = [
        "model",
        "method",
        "stable_mask_percent",
        "acc_mean",
        "wsr_non_target_mean",
        "clean_target_rate_mean",
        "diagnosis_majority",
    ]
    _write_rows(output_dir / "direct_embedding_diagnostic.csv", rows, POSTER_DIAGNOSTIC_FIELDS)
    _write_rows(
        output_dir / "direct_embedding_diagnostic_summary.csv",
        summary,
        ["model", "method", "stable_mask_percent"]
        + [f"{metric}_{suffix}" for metric in POSTER_SUMMARY_METRICS for suffix in ["mean", "std"]]
        + ["diagnosis_majority"],
    )
    _write_rows(
        output_dir / "table_direct_embedding_diagnostic.csv",
        _poster_table(summary, table_fields),
        table_fields,
    )
    return rows


def default_thesis_config(args):
    epochs_clean = 3 if args.quick_test else 20
    epochs_watermark = 1 if args.quick_test else 5
    if args.clean_epochs is not None:
        epochs_clean = args.clean_epochs
    if args.wm_epochs is not None:
        epochs_watermark = args.wm_epochs
    pruning_ratios = [0.0, 0.5, 0.9] if args.quick_test else [0.0, 0.5, 0.7, 0.9]
    quant_bits = [8, 4] if args.quick_test else [8, 4, 3]
    model_name = args.model or "cifar_small"
    methods = args.methods or ["standard", "stable_aware_reg", "stable_mask_direct", "random_mask_direct"]
    seeds = args.seeds or ([42] if args.direct_sweep else [42, 43, 44])
    stable_mask_percent = 0.1 if args.stable_mask_percent is None else args.stable_mask_percent
    mask_granularity = args.mask_granularity or "channel"
    selection_mode = args.selection_mode or "fisher_top"
    quant_stable_bits = 4 if args.quant_stable_bits is None else args.quant_stable_bits
    quant_error_alpha = 0.5 if args.quant_error_alpha is None else args.quant_error_alpha
    lambda_wm = 5.0 if args.model == "resnet18_cifar" and args.lambda_wm is None else 1.0
    lambda_wm = lambda_wm if args.lambda_wm is None else args.lambda_wm
    poison_ratio = 0.01 if args.poison_ratio is None else args.poison_ratio
    trigger_size = 3 if args.trigger_size is None else args.trigger_size
    target_label = 0 if args.target_label is None else args.target_label
    lambda_reg = 0.1 if args.lambda_reg is None else args.lambda_reg
    watermark_steps = 2 if args.model == "resnet18_cifar" and args.watermark_steps_per_batch is None else 1
    watermark_steps = watermark_steps if args.watermark_steps_per_batch is None else args.watermark_steps_per_batch
    watermark_train_mode = (
        "alternating"
        if args.model == "resnet18_cifar" and args.watermark_train_mode is None
        else "joint"
    )
    watermark_train_mode = watermark_train_mode if args.watermark_train_mode is None else args.watermark_train_mode
    direct_embedding_mode = args.direct_embedding_mode or "joint"
    lambda_clean = 0.5 if args.lambda_clean is None else args.lambda_clean
    use_activation_guidance = args.use_activation_guidance
    activation_layer = args.activation_layer or "layer4"
    lambda_act = 0.1 if args.lambda_act is None else args.lambda_act
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
        "learning_rate_watermark": 0.001
        if args.learning_rate_watermark is None
        else args.learning_rate_watermark,
        "poison_ratio": poison_ratio,
        "trigger_size": trigger_size,
        "target_label": target_label,
        "lambda_wm": lambda_wm,
        "lambda_reg": lambda_reg,
        "importance_batches": 100,
        "stable_mask_percent": stable_mask_percent,
        "mask_granularity": mask_granularity,
        "selection_mode": selection_mode,
        "quant_stable_bits": quant_stable_bits,
        "quant_error_alpha": quant_error_alpha,
        "watermark_steps_per_batch": watermark_steps,
        "watermark_train_mode": watermark_train_mode,
        "direct_embedding_mode": direct_embedding_mode,
        "lambda_clean": lambda_clean,
        "use_activation_guidance": use_activation_guidance,
        "activation_layer": activation_layer,
        "lambda_act": lambda_act,
        "pruning_ratios": pruning_ratios,
        "quant_bits": quant_bits,
        "sweep_stable_mask_percents": [0.1, 0.2, 0.3, 0.5],
        "sweep_lambda_wm_values": [1.0, 5.0, 10.0],
        "sweep_watermark_steps": [1, 2],
        "sweep_poison_ratios": [0.01, 0.03],
        "sweep_activation_guidance": [False, True],
        "sweep_lambda_act_values": [0.01, 0.1, 1.0],
    }
