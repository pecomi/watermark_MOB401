import torch

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
