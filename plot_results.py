from pathlib import Path


def plot_metric(rows, x_key, y_key, title, xlabel, ylabel, output_path, group_key="model"):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for group_name in sorted({row[group_key] for row in rows}):
        model_rows = [row for row in rows if row[group_key] == group_name]
        model_rows = sorted(model_rows, key=lambda row: row[x_key])
        plt.plot(
            [row[x_key] for row in model_rows],
            [row[y_key] for row in model_rows],
            marker="o",
            label=group_name,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_all(pruning_rows, quantization_rows, figure_dir, prefix="", group_key="model"):
    figure_dir = Path(figure_dir)
    prefix = f"{prefix}_" if prefix else ""
    plot_metric(
        pruning_rows,
        "pruning_ratio",
        "wsr",
        "WSR vs Pruning Ratio",
        "Pruning ratio",
        "WSR",
        figure_dir / f"{prefix}wsr_vs_pruning.png",
        group_key,
    )
    plot_metric(
        pruning_rows,
        "pruning_ratio",
        "acc",
        "ACC vs Pruning Ratio",
        "Pruning ratio",
        "Accuracy",
        figure_dir / f"{prefix}acc_vs_pruning.png",
        group_key,
    )
    plot_metric(
        quantization_rows,
        "bits",
        "wsr",
        "WSR vs Quantization Bits",
        "Bits",
        "WSR",
        figure_dir / f"{prefix}wsr_vs_quantization.png",
        group_key,
    )
    plot_metric(
        quantization_rows,
        "bits",
        "acc",
        "ACC vs Quantization Bits",
        "Bits",
        "Accuracy",
        figure_dir / f"{prefix}acc_vs_quantization.png",
        group_key,
    )
