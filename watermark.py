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
    watermark_train_mode="joint",
    watermark_steps_per_batch=1,
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

            if watermark_train_mode == "alternating":
                optimizer.zero_grad(set_to_none=True)
                clean_loss = F.cross_entropy(model(x), y)
                clean_step_loss = clean_loss
                if lambda_reg > 0.0:
                    clean_step_loss = clean_step_loss + lambda_reg * stable_regularizer(
                        model, clean_state, importance
                    )
                clean_step_loss.backward()
                optimizer.step()

                loss = clean_step_loss.detach()
                for _ in range(watermark_steps_per_batch):
                    wm_x = add_trigger(wm_source, dataset=dataset, trigger_size=trigger_size)
                    optimizer.zero_grad(set_to_none=True)
                    wm_step_loss = lambda_wm * F.cross_entropy(model(wm_x), wm_y)
                    if lambda_reg > 0.0:
                        wm_step_loss = wm_step_loss + lambda_reg * stable_regularizer(
                            model, clean_state, importance
                        )
                    wm_step_loss.backward()
                    optimizer.step()
                    loss = loss + wm_step_loss.detach()
            else:
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


def _get_module(model, layer_name):
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Activation layer not found: {layer_name}")
    return modules[layer_name]


def _selected_activation_channels(masks, activation_layer):
    selected = None
    for name, mask in masks.items():
        if not name.startswith(f"{activation_layer}.") or mask.ndim != 4:
            continue
        channel_mask = mask.detach().flatten(1).any(dim=1)
        selected = channel_mask if selected is None else (selected | channel_mask)
    if selected is None or not selected.any():
        return None
    return selected.nonzero(as_tuple=False).flatten()


def _masked_grad_step(model, optimizer, masks):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        mask = masks.get(name)
        if mask is None:
            param.grad.zero_()
        else:
            param.grad.mul_(mask.to(device=param.grad.device, dtype=param.grad.dtype))
    optimizer.step()


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
    watermark_steps_per_batch=1,
    direct_embedding_mode="joint",
    lambda_clean=0.5,
    use_activation_guidance=False,
    activation_layer="layer4",
    lambda_act=0.0,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    selected_channels = _selected_activation_channels(masks, activation_layer)
    activation_cache = {}
    hook_handle = None
    if use_activation_guidance and selected_channels is not None:
        def _capture_activation(module, inputs, output):
            activation_cache["value"] = output

        try:
            hook_handle = _get_module(model, activation_layer).register_forward_hook(_capture_activation)
        except ValueError:
            print(f"warning: activation layer {activation_layer} not found; disabling activation guidance.")
            use_activation_guidance = False

    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            poison_count = x.size(0) if poison_ratio >= 1.0 else max(1, int(x.size(0) * poison_ratio))
            wm_y = torch.full((poison_count,), target_label, dtype=y.dtype, device=device)

            optimizer.zero_grad(set_to_none=True)
            clean_loss = F.cross_entropy(model(x), y)
            clean_loss.backward()
            optimizer.step()

            batch_wm_loss = 0.0
            for _ in range(watermark_steps_per_batch):
                wm_x = add_trigger(x[:poison_count], dataset=dataset, trigger_size=trigger_size)
                activation_cache.clear()
                optimizer.zero_grad(set_to_none=True)
                logits = model(wm_x)
                wm_loss = F.cross_entropy(logits, wm_y)
                if use_activation_guidance and selected_channels is not None and "value" in activation_cache:
                    activation = activation_cache["value"]
                    channels = selected_channels.to(activation.device)
                    channels = channels[channels < activation.size(1)]
                    if channels.numel() > 0:
                        act_loss = -activation[:, channels].mean()
                        wm_loss = wm_loss + lambda_act * act_loss
                if direct_embedding_mode == "wm_focused":
                    clean_reg = F.cross_entropy(model(x), y)
                    wm_loss = lambda_wm * wm_loss + lambda_clean * clean_reg
                else:
                    wm_loss = lambda_wm * wm_loss
                wm_loss.backward()
                _masked_grad_step(model, optimizer, masks)
                batch_wm_loss += wm_loss.item()

            total_loss += (clean_loss.item() + batch_wm_loss) * x.size(0)
            total += x.size(0)
        print(f"mask-direct epoch {epoch + 1}/{epochs} loss={total_loss / total:.4f}")
    if hook_handle is not None:
        hook_handle.remove()
    return model
