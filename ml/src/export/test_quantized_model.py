# Quantized inference verification script for AlexNet64Gray
# This compares the results between quantized and unquantized 

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ml.src.data.mnist64 import MNIST64Config, get_dataloaders
from ml.src.models.alexnet64gray import AlexNet64Gray
from ml.src.utils import (
    INT8_MAX, UINT8_MAX,
    find_next_relu_name, latest_run_id, ordered_conv_linear_modules, resolve_device,
)

def clamp_int(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def rounding_right_shift(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Arithmetic right shift with sign-aware rounding. 

    We want (x/2^r) rounded to nearest, but done in integer math.
    For positive values:
       (x + 2^(r-1)) >> r
    For negative values:
       -(((-x) + 2^(r-1))>>r)

    Args:
        x (torch.Tensor): int64 tensor
        r (torch.Tensor): int tensor broadcastable to x 

    Returns:
        torch.Tensor: int64 tensor after rounding and shifting
    """
    r = r.to(dtype=torch.int64)

    add = (1 << (r - 1))
    pos = (x >= 0)
    x_pos = (x + add) >> r
    x_neg = -(((-x) + add) >> r)
    return torch.where(pos, x_pos, x_neg)


def requant_u8_from_acc(
    acc: torch.Tensor,
    m: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Requantize int accumulator into uint8 activation (post-ReLU)

    Given per channel parameters (m[c], r[c]) approximating:
       M_c = (s_x * s_w[c]) / s_y = m[c] / 2^r[c]

    Compute:
       q_y = round(acc * m / 2^r)
    Then Clamp into [0,255] (uint8), consistent with post-ReLU activations and z=0

    Args:
        acc (torch.Tensor): int64 tensor [N, C, H, W] (conv) or [N, C] (linear)
        m (torch.Tensor): int32 tensor [C]
        r (torch.Tensor): int32 tensor [C]

    Returns:
        torch.Tensor: uint8 tensor same shape as acc, values in [0,255]
    """
    m64 = m.to(dtype=torch.int64)
    r64 = r.to(dtype=torch.int64)

    if acc.dim() == 4:
        m64 = m64.view(1, -1, 1, 1)
        r64 = r64.view(1, -1, 1, 1)
    elif acc.dim() == 2:
        m64 = m64.view(1, -1)
        r64 = r64.view(1, -1)
    else:
        raise ValueError(f"Unsupported acc dim {acc.dim()}")

    prod = acc * m64  # int64
    # right shift with rounding
    out = rounding_right_shift(prod, r64)
    out = clamp_int(out, 0, UINT8_MAX).to(torch.uint8)
    return out


def conv2d_int_acc(
    x_u8_or_i8: torch.Tensor,
    w_i8: torch.Tensor,
    b_i32: torch.Tensor | None,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> torch.Tensor:
    """Compute conv2d accumulator exactly (integer domain)

     acc = conv2d(x,w) + b

    Args:
        x_u8_or_i8 (torch.Tensor): [N, C, H, W] uint8 or int8 activations
        w_i8 (torch.Tensor): [O, C, KH, KW] int8 weights
        b_i32 (torch.Tensor | None): int32 bias accumulator units, or None
        stride (Tuple[int, int]): conv stride
        padding (Tuple[int, int]): conv padding

    Returns:
        torch.Tensor: int64 accumulator tensor [N, O, H', W']
    """
    # Use float64 conv for exact integer sums (avoid float32 overflow/rounding).
    x64 = x_u8_or_i8.to(torch.int32).to(torch.float64)
    w64 = w_i8.to(torch.int32).to(torch.float64)

    y64 = F.conv2d(x64, w64, bias=None, stride=stride, padding=padding)
    # y64 is exact integer in float64; convert to int64
    acc = torch.round(y64).to(torch.int64)

    if b_i32 is not None:
        b64 = b_i32.to(torch.int64).view(1, -1, 1, 1)
        acc = acc + b64
    return acc


def linear_int_acc(
    x_u8_or_i8: torch.Tensor,
    w_i8: torch.Tensor,
    b_i32: torch.Tensor | None,
) -> torch.Tensor:
    """Compute linear accumulator exactly (integer domain).

    acc = x @ w^T + b

    Args:
        x_u8_or_i8 (torch.Tensor): [N, I] uint8 or int8
        w_i8 (torch.Tensor): [O, I] int8
        b_i32 (torch.Tensor | None): [O] int32 or None

    Returns:
        torch.Tensor: int64 tensor [N, O]
    """
    x64 = x_u8_or_i8.to(torch.int32).to(torch.float64)
    w64 = w_i8.to(torch.int32).to(torch.float64)

    y64 = x64 @ w64.t()
    acc = torch.round(y64).to(torch.int64)
    if b_i32 is not None:
        acc = acc + b_i32.to(torch.int64).view(1, -1)
    return acc

class QParams:
    """Loads exported quantization parameters and exposes per-layer tensors.

    The exporter produces:
        - .npz: arrays (int8 weights, int32 bias, int32 m/r, and optional debug scales)
        - .json: manifest describing layers and which arrays correspond to which layer

    The class provides:
        - name -> exported array keys
        - tensors moved to correct device
        - scale metadata s_x and s_y from manifest
    """
    def __init__(self, npz_path: Path, meta_path: Path, device: torch.device):
        self.device = device
        self.arr = np.load(npz_path)
        self.meta = json.loads(meta_path.read_text())

        # Map module name -> keys
        self.layer_meta: Dict[str, Dict[str, Any]] = {}
        for entry in self.meta["layers"]:
            self.layer_meta[entry["name"]] = entry

    def get_layer_tensors(self, layer_name: str):
        """
        Returns:
            _type_: (W, B, m, r, s_w_debug, entry) for a layer
        """
        entry = self.layer_meta[layer_name]
        keys = entry["export_keys"]

        W = torch.from_numpy(self.arr[keys["W_q"]]).to(self.device)  # int8
        B = None
        if keys.get("B_q"):
            B = torch.from_numpy(self.arr[keys["B_q"]]).to(self.device)  # int32

        m = r = None
        if keys.get("m") and keys.get("r"):
            m = torch.from_numpy(self.arr[keys["m"]]).to(self.device)  # int32
            r = torch.from_numpy(self.arr[keys["r"]]).to(self.device)  # int32

        # For last layer logits argmax, need s_w (debug array)
        s_w = None
        if keys.get("s_w_debug"):
            s_w = torch.from_numpy(self.arr[keys["s_w_debug"]]).to(self.device)  # float32

        return W, B, m, r, s_w, entry

    def sx_for_layer(self, layer_name: str) -> float:
        return float(self.layer_meta[layer_name]["s_x"])

    def sy_for_layer(self, layer_name: str) -> float | None:
        v = self.layer_meta[layer_name]["s_y"]
        return None if v is None else float(v)


@torch.no_grad()
def forward_quantized(
    model: AlexNet64Gray,
    qp: QParams,
    x0_q: torch.Tensor,
    collect: dict | None = None,
) -> torch.Tensor:
    """Run quantized inference following the PQT design.

    Args:
        model (AlexNet64Gray): Architecture instance
        qp (QParams): Quantization params loaded from export
        x0_q (torch.Tensor): Quantized input tensor int8 [N, 1, 64, 64]
        collect: Optional dict populated with dequantized per-layer outputs keyed by
                 conv/linear name; used by the per-layer error report.

    Returns:
        torch.Tensor: Float logits for argmax comparison
    """
    x = x0_q  # int8 [N,1,64,64] initially

    assert isinstance(model.features, nn.Sequential)
    for i, mod in enumerate(model.features):
        name = f"features.{i}"
        if isinstance(mod, nn.Conv2d):
            W, B, m, r, _s_w, entry = qp.get_layer_tensors(name)
            acc = conv2d_int_acc(
                x_u8_or_i8=x,
                w_i8=W,
                b_i32=B,
                stride=mod.stride,
                padding=mod.padding,
            )
            # ReLU on accumulator
            acc = torch.clamp(acc, min=0)

            if m is None or r is None:
                raise RuntimeError(f"Missing m/r for {name} (expected post-ReLU quantized layer).")
            x = requant_u8_from_acc(acc, m=m, r=r)  # uint8
            if collect is not None:
                s_y = qp.sy_for_layer(name)
                if s_y is not None:
                    collect[name] = x.float() * float(s_y)

        elif isinstance(mod, nn.ReLU):
            # Already applied ReLU in acc domain; skip
            continue

        elif isinstance(mod, nn.MaxPool2d):
            # Pool operates on uint8 activations (comparison). Use int32 for F.max_pool2d.
            x_pool = x.to(torch.int32)
            x_pool = F.max_pool2d(
                x_pool,
                kernel_size=mod.kernel_size,
                stride=mod.stride,
                padding=mod.padding,
                dilation=mod.dilation,
                ceil_mode=mod.ceil_mode,
            )
            x = x_pool.to(torch.uint8)

        else:
            # If your AlexNet has other ops (e.g., AdaptiveAvgPool), add here.
            raise NotImplementedError(f"Unsupported module in features: {name} -> {mod.__class__.__name__}")

    # Flatten
    x = torch.flatten(x, 1)  # [N, I] uint8

    assert isinstance(model.classifier, nn.Sequential)
    last_linear_name = None
    for i, mod in enumerate(model.classifier):
        if isinstance(mod, nn.Linear):
            last_linear_name = f"classifier.{i}"

    logits_float = None

    for i, mod in enumerate(model.classifier):
        name = f"classifier.{i}"

        if isinstance(mod, nn.Flatten):
            # Some models include Flatten as the first classifier op
            x = torch.flatten(x, 1)
            continue

        if isinstance(mod, nn.Linear):
            W, B, m, r, s_w, entry = qp.get_layer_tensors(name)
            acc = linear_int_acc(x_u8_or_i8=x, w_i8=W, b_i32=B)

            if name == last_linear_name:
                # Last layer: no ReLU, no requant in our export.
                s_x = qp.sx_for_layer(name)
                if s_w is None:
                    raise RuntimeError(f"Missing s_w_debug for last layer {name}; needed for logits scaling.")
                scale = (s_w.to(torch.float64) * float(s_x)).view(1, -1)
                logits_float = acc.to(torch.float64) * scale
                break

            # Hidden FC: apply ReLU then requant to uint8
            acc = torch.clamp(acc, min=0)
            if m is None or r is None:
                raise RuntimeError(f"Missing m/r for {name} (expected post-ReLU quantized layer).")
            x = requant_u8_from_acc(acc, m=m, r=r)  # uint8
            if collect is not None:
                s_y = qp.sy_for_layer(name)
                if s_y is not None:
                    collect[name] = x.float() * float(s_y)
            continue

        if isinstance(mod, nn.ReLU):
            continue

        if isinstance(mod, nn.Dropout):
            continue

        raise NotImplementedError(f"Unsupported module in classifier: {name} -> {mod.__class__.__name__}")

    if logits_float is None:
        raise RuntimeError("Failed to produce logits from quantized forward (did not hit last linear).")
    return logits_float

def quantize_input_from_normalized(x_norm: torch.Tensor, s0: float) -> torch.Tensor:
    """Quantize already normalized float input into int8 for the first layer

    q_0 = round(x_norm / s0), clamped to [-128, 127] as int8

    Args:
        x_norm (torch.Tensor): float input [N, 1, 64, 64]
        s0 (float): first layer activation scale

    Returns:
        torch.Tensor: int8 tensor
    """
    q = torch.round(x_norm / float(s0))
    q = torch.clamp(q, -(INT8_MAX + 1), INT8_MAX).to(torch.int8)
    return q

@torch.no_grad()
def evaluate_float(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="  float eval  ", unit="batch", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def evaluate_quantized(model: AlexNet64Gray, qp: QParams, loader, device: torch.device, s0: float) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="  quant eval  ", unit="batch", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x0_q = quantize_input_from_normalized(x, s0=s0)  # int8
        logits = forward_quantized(model, qp, x0_q=x0_q)  # float64 logits
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for quantized verification."""
    p = argparse.ArgumentParser(description="Test quantized AlexNet64Gray using exported qparams.")
    p.add_argument("--data-dir", type=str, default="ml/data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-normalize", action="store_true", help="Disable dataset normalization (normally keep OFF for this test).")
    p.add_argument("--augment", action="store_true", help="Enable augmentation (keep OFF for eval).")

    p.add_argument("--checkpoints-dir", type=str, default="ml/checkpoints", help="Base dir for model weights")
    p.add_argument("--outputs-dir",     type=str, default="ml/outputs",     help="Base dir for export outputs")
    p.add_argument("--run-id",          type=int, default=-1, help="Run ID to use (default: latest)")
    p.add_argument("--ckpt", type=str,  default="", help="Override checkpoint path (overrides --run-id)")
    p.add_argument("--npz",  type=str,  default="", help="Override fpgaqparms.npz path (overrides --run-id)")
    p.add_argument("--meta", type=str,  default="", help="Override fpgaqparms.json path (overrides --run-id)")
    p.add_argument("--s0", type=float, default=None,
                   help="First-layer input scale. If omitted, read from fpgaqparms.json.")

    p.add_argument("--device", type=str, default="", help="cpu or cuda; empty=auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _per_layer_error_report(
    model: AlexNet64Gray,
    qp: QParams,
    x: torch.Tensor,
    device: torch.device,
    s0: float,
) -> None:
    """Print a per-layer L1 error table comparing float vs dequantized quantized activations.

    For each conv/linear layer that has a post-ReLU quantization point, runs both the float
    model (hooks on nn.ReLU) and the quantized model (dequantized uint8 output) on a single
    batch and reports mean absolute error.

    Args:
        model:  float AlexNet64Gray
        qp:     loaded QParams
        x:      one float input batch already on device (normalized)
        device: torch.device
        s0:     first-layer input scale
    """
    # Float: hook every ReLU, collect post-relu activations
    float_post_relu: dict[str, torch.Tensor] = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ReLU):
            def _make(n: str):
                def _h(_m, _i, out): float_post_relu[n] = out.detach().float()
                return _h
            handles.append(mod.register_forward_hook(_make(name)))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()

    # Quantized: collect dequantized per-layer outputs
    quant_collect: dict[str, torch.Tensor] = {}
    x0_q = quantize_input_from_normalized(x, s0)
    forward_quantized(model, qp, x0_q, collect=quant_collect)

    # Build conv/linear → relu name map for display
    layers = ordered_conv_linear_modules(model)

    print(f"\n[per-layer error]  float vs dequantized quantized (one batch, mean absolute error)")
    print(f"  {'Layer':<20} {'Shape':<24} {'Float μ':>10} {'Quant μ':>10} {'L1 err':>10}")
    print(f"  {'-'*76}")

    for lname, _mod in layers:
        relu_name = find_next_relu_name(model, lname)
        if relu_name is None:
            continue  # logits layer — no requant
        f_act = float_post_relu.get(relu_name)
        q_dq  = quant_collect.get(lname)
        if f_act is None or q_dq is None:
            continue
        l1 = (f_act - q_dq).abs().mean().item()
        print(
            f"  {lname:<20} {str(tuple(f_act.shape)):<24}"
            f" {f_act.mean().item():>10.4f} {q_dq.mean().item():>10.4f} {l1:>10.4f}"
        )
    print()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    # -- Resolve run-id based paths -------------------------------------------
    checkpoints_base = Path(args.checkpoints_dir).expanduser().resolve()
    outputs_base     = Path(args.outputs_dir).expanduser().resolve()
    run_id           = args.run_id if args.run_id >= 0 else latest_run_id(checkpoints_base)

    ckpt_path  = Path(args.ckpt).expanduser().resolve()  if args.ckpt \
                 else checkpoints_base / f"run{run_id}" / "best.pth"
    npz_path   = Path(args.npz).expanduser().resolve()   if args.npz  \
                 else outputs_base / f"run{run_id}" / "fpgaqparms.npz"
    meta_path  = Path(args.meta).expanduser().resolve()  if args.meta \
                 else outputs_base / f"run{run_id}" / "fpgaqparms.json"

    device = resolve_device(args.device)

    cfg = MNIST64Config(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=not args.no_normalize,
        augment=args.augment,
    )

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # Load float model
    model = AlexNet64Gray(num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load quant params
    qp = QParams(npz_path=npz_path, meta_path=meta_path, device=device)

    # Resolve s0: CLI arg takes priority, else read from the exported JSON
    if args.s0 is not None:
        s0 = float(args.s0)
    else:
        s0 = float(qp.meta["inputs"]["s0"])

    if s0 <= 0.0:
        raise SystemExit(f"s0 must be > 0, got {s0}")

    n_layers = len(qp.meta["layers"])
    n_test   = len(test_loader.dataset)

    # -- Startup banner -------------------------------------------------------
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  test_quantized_model  run{run_id}")
    print(f"  device      : {device}")
    print(f"  checkpoint  : {ckpt_path}")
    print(f"  qparams npz : {npz_path}")
    print(f"  qparams json: {meta_path}")
    print(f"  s0          : {s0}  (first-layer input scale)")
    print(f"  layers      : {n_layers}  test samples : {n_test:,}")
    print(f"{sep}\n")

    # -- Float accuracy (baseline) --------------------------------------------
    print("  [1/3] Float model evaluation...")
    float_acc = evaluate_float(model, test_loader, device=device)
    print(f"        → acc {float_acc * 100:.2f}%\n")

    # -- Quantized accuracy ---------------------------------------------------
    print("  [2/3] Quantized PTQ evaluation...")
    q_acc = evaluate_quantized(model, qp, test_loader, device=device, s0=s0)
    print(f"        → acc {q_acc * 100:.2f}%\n")

    # -- Summary --------------------------------------------------------------
    print(f"{sep}")
    print(f"  Float accuracy    : {float_acc * 100:.2f}%")
    print(f"  Quantized accuracy: {q_acc * 100:.2f}%")
    print(f"  Accuracy drop     : {(float_acc - q_acc) * 100:+.2f}%")
    print(f"{sep}\n")

    # -- Per-layer error breakdown --------------------------------------------
    print("  [3/3] Per-layer quantization error (one batch)...")
    x_sample, _ = next(iter(test_loader))
    x_sample = x_sample.to(device)
    _per_layer_error_report(model, qp, x_sample, device, s0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())