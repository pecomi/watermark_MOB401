import torch
import torch.nn.functional as F

from watermark import add_trigger


def compute_importance(model, train_loader, device, num_batches):
    model.eval()
    importance = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    seen_batches = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] += param.grad.detach().pow(2)
        seen_batches += 1
        if seen_batches >= num_batches:
            break

    scale = max(seen_batches, 1)
    return {name: score / scale for name, score in importance.items()}


def compute_watermark_importance(
    model,
    train_loader,
    device,
    num_batches,
    target_label=0,
    dataset="cifar10",
    trigger_size=None,
    poison_ratio=0.1,
):
    model.eval()
    importance = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    seen_batches = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        poison_count = x.size(0) if poison_ratio >= 1.0 else max(1, int(x.size(0) * poison_ratio))
        wm_x = add_trigger(x[:poison_count], dataset=dataset, trigger_size=trigger_size)
        wm_y = torch.full((wm_x.size(0),), target_label, dtype=y.dtype, device=device)

        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(wm_x), wm_y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] += param.grad.detach().pow(2)
        seen_batches += 1
        if seen_batches >= num_batches:
            break

    scale = max(seen_batches, 1)
    return {name: score / scale for name, score in importance.items()}
