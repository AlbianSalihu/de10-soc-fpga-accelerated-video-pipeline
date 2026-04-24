# Shared utilities used across all ML scripts.

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Integer quantization constants
# ---------------------------------------------------------------------------
INT8_MAX  = 127     # max positive value for symmetric int8
UINT8_MAX = 255     # max value for uint8 (post-ReLU activations)
EPS       = 1e-12   # minimum clamp to avoid zero-division in scale computation

# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def next_run_id(base: Path) -> int:
    """Return the next available runN ID by scanning base for existing folders."""
    i = 0
    while (base / f"run{i}").exists():
        i += 1
    return i


def latest_run_id(base: Path) -> int:
    """Return the ID of the most recent existing runN folder in base.

    Raises:
        RuntimeError: if no runN folders exist in base.
    """
    i = 0
    while (base / f"run{i}").exists():
        i += 1
    if i == 0:
        raise RuntimeError(f"No runs found in {base}. Run train.py first.")
    return i - 1


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a torch.device.

    Args:
        device_str: 'cpu', 'cuda', or '' for auto-detect.

    Returns:
        torch.device
    """
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model inspection helpers (Sequential-based models)
# ---------------------------------------------------------------------------

def ordered_conv_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Collect Conv2d and Linear modules in named_modules() order.

    Note:
        For branching/skip-connection models, named_modules() order is not
        guaranteed to match true execution order.

    Returns:
        List of (layer_name, layer_module) for every Conv2d and Linear in model.
    """
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear))
    ]


def find_next_relu_name(
    model: nn.Module,
    layer_name: str,
    max_lookahead: int = 6,
) -> Optional[str]:
    """Find the first nn.ReLU after layer_name inside the same nn.Sequential.

    Args:
        model:          PyTorch model.
        layer_name:     Dot-separated module name (e.g. "features.0").
        max_lookahead:  How many siblings to scan forward before giving up.

    Returns:
        Module name of the next nn.ReLU, or None if not found.
    """
    if "." not in layer_name:
        return None

    parent_name, child_key = layer_name.rsplit(".", 1)

    try:
        parent = model.get_submodule(parent_name)
    except AttributeError:
        return None

    if not isinstance(parent, nn.Sequential):
        return None

    keys = list(parent._modules.keys())
    if child_key not in keys:
        return None

    i0 = keys.index(child_key)
    for j in range(i0 + 1, min(i0 + 1 + max_lookahead, len(keys))):
        k   = keys[j]
        mod = parent._modules[k]
        if isinstance(mod, nn.ReLU):
            return f"{parent_name}.{k}"
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            break   # another compute layer before relu — stop

    return None
