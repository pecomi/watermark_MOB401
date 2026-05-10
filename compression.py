import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def prunable_modules(model):
    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def apply_pruning(model, ratio):
    if ratio <= 0.0:
        return model
    parameters = prunable_modules(model)
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=ratio)
    for module, name in parameters:
        prune.remove(module, name)
    return model


@torch.no_grad()
def apply_fake_quantization(model, bits):
    # Simulated per-tensor weight quantize-dequantize for sensitivity testing.
    # This is not full INT8 deployment quantization with quantized kernels.
    quantized = copy.deepcopy(model)
    levels = (2**bits) - 1
    for module in quantized.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            w_min = weight.min()
            w_max = weight.max()
            if torch.isclose(w_min, w_max):
                continue
            scale = (w_max - w_min) / levels
            q = torch.round((weight - w_min) / scale).clamp(0, levels)
            module.weight.data.copy_(q * scale + w_min)
    return quantized
