import torch
import torch.nn.functional as F

from data import MNIST_MEAN, MNIST_STD


def add_trigger(x, dataset="mnist", trigger_size=None):
    triggered = x.clone()
    if dataset == "mnist":
        size = 4 if trigger_size is None else trigger_size
        white_value = (1.0 - MNIST_MEAN) / MNIST_STD
    elif dataset == "cifar10":
        size = 3 if trigger_size is None else trigger_size
        white_value = 1.0
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    triggered[:, :, -size:, -size:] = white_value
    return triggered


def train_clean(model, train_loader, device, epochs, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
        print(f"clean epoch {epoch + 1}/{epochs} loss={total_loss / total:.4f}")
    return model


def stable_regularizer(model, clean_state, importance):
    reg = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in importance:
            clean_param = clean_state[name].to(param.device)
            reg = reg + (importance[name] * (param - clean_param).pow(2)).sum()
    return reg


def train_watermark(
    model,
    clean_state,
    train_loader,
    device,
    epochs,
    lr,
    lambda_wm,
    lambda_reg=0.0,
    importance=None,
    target_label=0,
    dataset="mnist",
    trigger_size=None,
    poison_ratio=1.0,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if poison_ratio >= 1.0:
                wm_source = x
            else:
                poison_count = max(1, int(x.size(0) * poison_ratio))
                wm_source = x[:poison_count]
            wm_x = add_trigger(wm_source, dataset=dataset, trigger_size=trigger_size)
            wm_y = torch.full((wm_x.size(0),), target_label, dtype=y.dtype, device=device)

            optimizer.zero_grad(set_to_none=True)
            clean_loss = F.cross_entropy(model(x), y)
            wm_loss = F.cross_entropy(model(wm_x), wm_y)
            loss = clean_loss + lambda_wm * wm_loss
            if lambda_reg > 0.0:
                loss = loss + lambda_reg * stable_regularizer(model, clean_state, importance)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
        print(f"watermark epoch {epoch + 1}/{epochs} loss={total_loss / total:.4f}")
    return model


def train_mask_direct_watermark(
    model,
    train_loader,
    masks,
    device,
    epochs,
    lr,
    lambda_wm,
    target_label=0,
    dataset="cifar10",
    trigger_size=3,
    poison_ratio=0.01,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            poison_count = x.size(0) if poison_ratio >= 1.0 else max(1, int(x.size(0) * poison_ratio))
            wm_x = add_trigger(x[:poison_count], dataset=dataset, trigger_size=trigger_size)
            wm_y = torch.full((wm_x.size(0),), target_label, dtype=y.dtype, device=device)

            optimizer.zero_grad(set_to_none=True)
            clean_loss = F.cross_entropy(model(x), y)
            clean_loss.backward()
            clean_grads = {
                name: None if param.grad is None else param.grad.detach().clone()
                for name, param in model.named_parameters()
            }

            optimizer.zero_grad(set_to_none=True)
            wm_loss = lambda_wm * F.cross_entropy(model(wm_x), wm_y)
            wm_loss.backward()

            for name, param in model.named_parameters():
                clean_grad = clean_grads[name]
                wm_grad = param.grad
                if wm_grad is not None:
                    mask = masks.get(name)
                    if mask is None:
                        wm_grad.zero_()
                    else:
                        wm_grad.mul_(mask.to(device=wm_grad.device, dtype=wm_grad.dtype))
                    if clean_grad is not None:
                        wm_grad.add_(clean_grad)
                elif clean_grad is not None:
                    param.grad = clean_grad

            optimizer.step()
            total_loss += (clean_loss.item() + wm_loss.item()) * x.size(0)
            total += x.size(0)
        print(f"mask-direct epoch {epoch + 1}/{epochs} loss={total_loss / total:.4f}")
    return model
