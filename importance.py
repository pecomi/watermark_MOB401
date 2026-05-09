import torch
import torch.nn.functional as F


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
