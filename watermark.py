import torch
import torch.nn.functional as F

from data import MNIST_MEAN, MNIST_STD


def add_trigger(x):
    triggered = x.clone()
    white_value = (1.0 - MNIST_MEAN) / MNIST_STD
    triggered[:, :, -4:, -4:] = white_value
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
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            wm_x = add_trigger(x)
            wm_y = torch.full_like(y, target_label)

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
