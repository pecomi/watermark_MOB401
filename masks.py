import math

import torch
import torch.nn as nn


def _minmax_normalize(x):
    x_min = x.min()
    x_max = x.max()
    if torch.isclose(x_min, x_max):
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def _fake_quant_error_tensor(weight, bits):
    levels = (2**bits) - 1
    w_min = weight.min()
    w_max = weight.max()
    if torch.isclose(w_min, w_max):
        return torch.zeros_like(weight)
    scale = (w_max - w_min) / levels
    quantized = torch.round((weight - w_min) / scale).clamp(0, levels) * scale + w_min
    return (quantized - weight).abs()


def _topk_binary(scores, percent):
    flat = scores.flatten()
    k = max(1, math.ceil(flat.numel() * percent))
    k = min(k, flat.numel())
    selected = torch.zeros_like(flat)
    indices = torch.topk(flat, k).indices
    selected[indices] = 1.0
    return selected.view_as(scores)


def _random_binary_like(scores, selected_count, seed):
    generator = torch.Generator().manual_seed(seed)
    flat = torch.zeros(scores.numel(), device=scores.device, dtype=scores.dtype)
    selected_count = min(max(1, selected_count), flat.numel())
    indices = torch.randperm(flat.numel(), generator=generator)[:selected_count].to(scores.device)
    flat[indices] = 1.0
    return flat.view_as(scores)


def _module_lookup(model):
    return dict(model.named_modules())


def _quant_channel_error(param, bits):
    error = _fake_quant_error_tensor(param.detach(), bits)
    if param.ndim == 4:
        return error.flatten(1).mean(dim=1)
    if param.ndim == 2:
        return error.mean(dim=1)
    return error


def _combine_scores(fisher_score, watermark_score=None, watermark_alpha=0.0):
    score = fisher_score.detach()
    if watermark_score is not None and watermark_alpha > 0.0:
        score = _minmax_normalize(score) + watermark_alpha * _minmax_normalize(
            watermark_score.detach()
        )
    return score


def _channel_scores(
    model,
    importance,
    selection_mode,
    quant_bits,
    alpha,
    watermark_importance=None,
    watermark_alpha=0.0,
):
    modules = _module_lookup(model)
    scores = {}
    for module_name, module in modules.items():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        weight_name = f"{module_name}.weight" if module_name else "weight"
        if weight_name not in importance:
            continue
        fisher = importance[weight_name].detach()
        if fisher.ndim == 4:
            score = fisher.flatten(1).sum(dim=1)
        elif fisher.ndim == 2:
            score = fisher.sum(dim=1)
        else:
            continue
        wm_score = None
        if watermark_importance is not None and weight_name in watermark_importance:
            wm = watermark_importance[weight_name].detach()
            if wm.ndim == 4:
                wm_score = wm.flatten(1).sum(dim=1)
            elif wm.ndim == 2:
                wm_score = wm.sum(dim=1)
        score = _combine_scores(score, wm_score, watermark_alpha)
        if selection_mode == "quant_stable":
            quant_error = _quant_channel_error(module.weight, quant_bits)
            score = _minmax_normalize(score) - alpha * _minmax_normalize(quant_error)
        scores[weight_name] = score
    return scores


def make_parameter_masks(
    model,
    importance,
    percent,
    selection_mode,
    quant_bits,
    alpha,
    random_mask,
    seed,
    watermark_importance=None,
    watermark_alpha=0.0,
):
    scores = {}
    for name, param in model.named_parameters():
        if name not in importance:
            continue
        wm_score = watermark_importance.get(name) if watermark_importance is not None else None
        score = _combine_scores(importance[name], wm_score, watermark_alpha)
        if selection_mode == "quant_stable":
            quant_error = _fake_quant_error_tensor(param.detach(), quant_bits)
            score = _minmax_normalize(score) - alpha * _minmax_normalize(quant_error)
        scores[name] = score

    if not scores:
        return {}

    flat_scores = torch.cat([score.flatten() for score in scores.values()])
    selected_count = max(1, math.ceil(flat_scores.numel() * percent))
    if random_mask:
        flat_mask = _random_binary_like(flat_scores, selected_count, seed)
    else:
        flat_mask = _topk_binary(flat_scores, percent).flatten()

    masks = {}
    offset = 0
    for name, score in scores.items():
        count = score.numel()
        masks[name] = flat_mask[offset : offset + count].view_as(score).detach()
        offset += count
    return masks


def make_channel_masks(
    model,
    importance,
    percent,
    selection_mode,
    quant_bits,
    alpha,
    random_mask,
    seed,
    watermark_importance=None,
    watermark_alpha=0.0,
):
    channel_scores = _channel_scores(
        model,
        importance,
        selection_mode,
        quant_bits,
        alpha,
        watermark_importance=watermark_importance,
        watermark_alpha=watermark_alpha,
    )
    masks = {}
    modules = _module_lookup(model)

    for module_name, module in modules.items():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        weight_name = f"{module_name}.weight" if module_name else "weight"
        bias_name = f"{module_name}.bias" if module_name else "bias"
        if weight_name not in channel_scores:
            continue
        scores = channel_scores[weight_name]
        selected_count = max(1, math.ceil(scores.numel() * percent))
        channel_mask = (
            _random_binary_like(scores, selected_count, seed + len(masks))
            if random_mask
            else _topk_binary(scores, percent)
        )
        if module.weight.ndim == 4:
            masks[weight_name] = channel_mask[:, None, None, None].expand_as(module.weight).detach()
        else:
            masks[weight_name] = channel_mask[:, None].expand_as(module.weight).detach()
        if module.bias is not None:
            masks[bias_name] = channel_mask.detach()
    return masks


def create_direct_masks(
    model,
    importance,
    percent,
    granularity,
    selection_mode,
    quant_bits,
    alpha,
    random_mask=False,
    seed=0,
    watermark_importance=None,
    watermark_alpha=0.0,
):
    if selection_mode == "random":
        random_mask = True
    if granularity == "parameter":
        return make_parameter_masks(
            model,
            importance,
            percent,
            selection_mode,
            quant_bits,
            alpha,
            random_mask,
            seed,
            watermark_importance=watermark_importance,
            watermark_alpha=watermark_alpha,
        )
    if granularity == "channel":
        return make_channel_masks(
            model,
            importance,
            percent,
            selection_mode,
            quant_bits,
            alpha,
            random_mask,
            seed,
            watermark_importance=watermark_importance,
            watermark_alpha=watermark_alpha,
        )
    raise ValueError(f"Unsupported mask granularity: {granularity}")


def selected_survival_rate(model, masks):
    if not masks:
        return float("nan")
    state = dict(model.named_parameters())
    survived = 0.0
    selected = 0.0
    for name, mask in masks.items():
        if name not in state:
            continue
        mask = mask.to(device=state[name].device, dtype=torch.bool)
        selected_values = state[name].detach()[mask]
        survived += (selected_values != 0).float().sum().item()
        selected += selected_values.numel()
    return float("nan") if selected == 0 else survived / selected


def selected_quant_error(original_model, quantized_model, masks):
    if not masks:
        return float("nan")
    original = dict(original_model.named_parameters())
    quantized = dict(quantized_model.named_parameters())
    total_error = 0.0
    selected = 0
    for name, mask in masks.items():
        if name not in original or name not in quantized:
            continue
        mask = mask.to(device=original[name].device, dtype=torch.bool)
        before = original[name].detach()[mask]
        after = quantized[name].detach()[mask]
        error = (after - before).abs() / before.abs().clamp_min(1e-8)
        total_error += error.sum().item()
        selected += error.numel()
    return float("nan") if selected == 0 else total_error / selected
