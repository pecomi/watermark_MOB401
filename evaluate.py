import torch
import json

from watermark import add_trigger


@torch.no_grad()
def evaluate_acc(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def evaluate_wsr(model, data_loader, device, target_label=0, dataset="mnist", trigger_size=None):
    model.eval()
    success = 0
    total = 0
    for x, _ in data_loader:
        x = add_trigger(x.to(device), dataset=dataset, trigger_size=trigger_size)
        pred = model(x).argmax(dim=1)
        success += (pred == target_label).sum().item()
        total += pred.numel()
    return success / total


@torch.no_grad()
def evaluate_thesis_metrics(model, data_loader, device, target_label=0, dataset="cifar10", trigger_size=3):
    model.eval()
    correct = 0
    total = 0
    clean_target = 0
    trigger_success = 0
    non_target_success = 0
    non_target_total = 0
    label_counts = torch.zeros(10, dtype=torch.long)

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        clean_pred = model(x).argmax(dim=1)
        correct += (clean_pred == y).sum().item()
        clean_target += (clean_pred == target_label).sum().item()
        total += y.numel()
        label_counts += torch.bincount(clean_pred.cpu(), minlength=10)

        triggered = add_trigger(x, dataset=dataset, trigger_size=trigger_size)
        triggered_pred = model(triggered).argmax(dim=1)
        trigger_success += (triggered_pred == target_label).sum().item()
        non_target_mask = y != target_label
        non_target_success += (triggered_pred[non_target_mask] == target_label).sum().item()
        non_target_total += non_target_mask.sum().item()

    distribution = {str(i): int(label_counts[i].item()) for i in range(label_counts.numel())}
    return {
        "acc": correct / total,
        "wsr": trigger_success / total,
        "wsr_non_target": float("nan") if non_target_total == 0 else non_target_success / non_target_total,
        "clean_target_rate": clean_target / total,
        "pred_label_distribution": json.dumps(distribution, sort_keys=True),
    }
