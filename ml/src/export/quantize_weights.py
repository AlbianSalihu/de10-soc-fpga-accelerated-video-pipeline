# This file is an export script that takes a trained PyTorch model and produces the quantized parameters for fpga inference.
# It produces : 
#       - int8 weights (symmetric, per output channel scales)
#       - int32 biases (in accumulator units)
#       - per output channel requantization parameters (m, r) such that:
#               M = (s_x * s_w) / s_y = m / 2^r
#             enabling integer only requantization:
#               q_y = (acc * m + rounding) >> r (if r>0) 
#               q_y = (acc * m) << (-r) (if r<0, saturate)
#       - a JSON manifest describing the exported arrays and assumptions 
#
# Run:  
#   python -m ml.src.export.quantize_weights \
#       --ckpt ml/checkpoints/best.pth \
#       --sy ml/checkpoints/act_scales_sy.json \
#       --out ml/checkpoints/fpgaqparms \
#       --s0 0.02221642937

from __future__ import annotations

import argparse
import json 
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from ml.src.models.alexnet64gray import AlexNet64Gray
from ml.src.utils import (
    EPS, INT8_MAX,
    find_next_relu_name, latest_run_id, ordered_conv_linear_modules, resolve_device,
)


def quantize_M_to_mr(M: float)->Tuple[int, int]:
    """Convert positive float M into integer pair (m,r) such that:
                        M ~= m/2^r

    For the inference to be able to do: 
    if r>0: q_y = (acc*m+rounding)>>r
    r>0: q_y = (acc*m)<<(-r) with saturation

    Args:
        M (float): s_x*s_w/s_y

    Returns:
        Tuple[int, int]: 
            - m (int32-like, 0 <= m <= 2^31-1)
            - r (int): shift ammonut (can be negative if M is huge)
    """
    # M should be positive (it's a ratio of positive scales)
    if M <= 0.0 or not math.isfinite(M):
        return 0,0
    
    # Decompose M = q * 2^e with q in [0.5, 1]
    q,e = math.frexp(M) # M = q * 2**e
    
    # Convert q to q31
    m = int(round(q*(1<<31)))

    # Fix edge case where rounding pushes to 1.0
    # If q is close to 1 then m = 2^31 except that it is meant to be : 0 <= m <=2^31-1 
    if m == (1<<31):
        m//=2
        e += 1

    # M ~= m / 2^(31-e)
    r = 31-e

    # Clamp m to signed int32 positive range to avoid float behaviours that could cause issues
    m = min(max(m,0),(1<<31)-1) 
    return m,r

def per_out_channel_weight_scale(w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Compute per-output-channel symmetric scale for int8 weights:
        s_w[c] = max(abs(w[c]))/127

    So that q_w[c] = round(w[c]/s_w[c]) in [-127,127]
    This implies that zero-point = 0 -> easier

    Works for conv2d weights [O,I,KH,KW] and linear [O,I]
    O : Number of output channel
    I : Input channel
    KH, KW : Kernel size
        
    Args:
        w (torch.Tensor): Floating point weight tensor for Conv2d or Linear
        eps (float, optional): Minimum clamp value for per-channel max to avoid zero scales. Defaults to 1e-12.

    Returns:
        torch.Tensor: 1D tensor of shape [O] containing the per output channel scales s_w.
    """

    # Take the absolute max of the module:
    if w.dim()==4:
        # [O, I, KH, KW] conv 
        a = w.abs().amax(dim=(1,2,3))
    elif w.dim()==2:
        # [O,I] linear
        a = w.abs().amax(dim=1)
    else:
        raise ValueError(f"Unsupported weight dimension {w.dim()} for per-channel scales")
    
    a = torch.clamp(a, min=eps)
    return a / float(INT8_MAX)

def quantize_weights_int8(w: torch.Tensor, s_w: torch.Tensor)->torch.Tensor:
    """Quantize float weights to int8 per output channel: 
               q_w[c] = round(w[c]/s_w[c]) clipped to [-128, 127]

    Args:
        w (torch.Tensor): Floating point weigth tensor for Conv2d or Linear 
        s_w (torch.Tensor): Per-Output channel weight scales

    Returns:
        torch.Tensor: Quantized weight tensor q_w 
    """
    # Reshape s_w for it to match the weights shape
    if w.dim() == 4:
        s = s_w.view(-1,1,1,1)
    elif w.dim()==2:
        s = s_w.view(-1,1)
    else:
        raise ValueError(f"Unsupported weight dim {w.dim()}")
    
    q = torch.round(w/s) # quantization 
    q = torch.clamp(q,-128,127).to(torch.int8)
    return q

def quantize_bias_int32(b: torch.Tensor, s_x: float, s_w: torch.Tensor)->torch.Tensor:
    """Quantize Bias to int32:
            q_b[c] = round(b[c]/(s_x*s_w[c]))
        returned as int32 tensor length 0

    Args:
        b (torch.Tensor): Floating point bias tensor 
        s_x (float): Input activation scale for the layer 
        s_w (torch.Tensor): Per output channel weight scales

    Returns:
        torch.Tensor: Quantized biase tensor q_b    
    """
    denom = (s_w*float(s_x)).to(b.device)
    q_b =torch.round(b/denom).to(torch.int32)
    return q_b

def parse_args()->argparse.Namespace:
    """Parse command-line arguments for the quantization export script.

    Returns:
        argparse.Namespace with fields:
          - ckpt: path to model checkpoint (.pth)
          - sy: path to activation scale JSON produced by calibration (find_scales.py)
          - out: output file prefix (writes <out>.npz and <out>.json)
          - device: "cpu", "cuda", or empty for auto
          - s0: first-layer input activation scale (must match FPGA input encoding)
          - zx: input zero-point (default 0; reserved for asymmetric quantization)
          - zy: output zero-point (default 0; reserved for asymmetric quantization)
          - export_last_layer_mr: if set, attempt to export (m,r) for final logits layer
    """
    p = argparse.ArgumentParser(description="Quantize weights + compute (m,r) using activation scales from calibration.")
    p.add_argument("--checkpoints-dir", type=str, default="ml/checkpoints", help="Base dir for model weights")
    p.add_argument("--outputs-dir",     type=str, default="ml/outputs",     help="Base dir for export outputs")
    p.add_argument("--run-id",          type=int, default=-1, help="Run ID to use (default: latest)")
    p.add_argument("--ckpt", type=str,  default="", help="Override checkpoint path (overrides --run-id)")
    p.add_argument("--sy",   type=str,  default="", help="Override activation scales path (overrides --run-id)")
    p.add_argument("--out",  type=str,  default="", help="Override output prefix (overrides --run-id)")
    p.add_argument("--device", type=str, default="", help="cpu or cuda, empty = auto")

    # First layer input scale
    p.add_argument("--s0", type=float, required=True,
                   help="First-layer input activation scale s0 (must match FPGA input encoding).")
    p.add_argument("--zx", type=int, default=0, help="Input zero-point (default 0).")
    p.add_argument("--zy", type=int, default=0, help="Output zero-point (default 0).")

    # Option: handle final logits layer
    p.add_argument("--export_last_layer_mr", action="store_true",
                   help="Try to compute m/r for last Linear too (requires a corresponding s_y entry; usually not present).")
    return p.parse_args()

@torch.no_grad()
def main() -> int:
    """Entry point: export integer quantization parameters for FPGA inference.

    Workflow:
      1) Load activation output scales (s_y) from calibration JSON (find_scales.py).
      2) Load a trained PyTorch checkpoint into the model architecture.
      3) Iterate Conv2d/Linear layers in forward order:
           - compute per-output-channel weight scales s_w
           - quantize weights to int8 (q_w)
           - quantize bias to int32 accumulator units (q_b) using current input scale s_x
           - locate post-ReLU output scale s_y (if present)
           - compute per-channel requant multiplier M = (s_x*s_w)/s_y and approximate via (m, r)
           - chain s_x := s_y for the next layer when a post-ReLU quantized activation exists
      4) Write:
           - <out>.npz containing arrays (q_w, q_b, m, r, optional debug scales)
           - <out>.json containing metadata and per-layer export keys

    Returns:
        0 on success. Raises exceptions on unrecoverable configuration errors.
    """
    args = parse_args()

    if args.s0 <= 0.0:
        raise SystemExit(f"--s0 must be > 0, got {args.s0}")

    # -- Resolve run-id based paths -------------------------------------------
    checkpoints_base = Path(args.checkpoints_dir).expanduser().resolve()
    outputs_base     = Path(args.outputs_dir).expanduser().resolve()
    run_id           = args.run_id if args.run_id >= 0 else latest_run_id(checkpoints_base)

    ckpt_path  = Path(args.ckpt).expanduser().resolve() if args.ckpt \
                 else checkpoints_base / f"run{run_id}" / "best.pth"
    sy_path    = Path(args.sy).expanduser().resolve()   if args.sy   \
                 else outputs_base / f"run{run_id}" / "act_scales_sy.json"
    out_prefix = Path(args.out).expanduser().resolve()  if args.out  \
                 else outputs_base / f"run{run_id}" / "fpgaqparms"

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"[quantize_weights] run{run_id}  ckpt   → {ckpt_path}")
    print(f"[quantize_weights] run{run_id}  scales → {sy_path}")
    print(f"[quantize_weights] run{run_id}  output → {out_prefix}.(npz|json)")

    device = resolve_device(args.device)

    # Load calibration JSON produced by find_scales.py
    sy_payload = json.loads(sy_path.read_text())
    sy_dict: Dict[str, float] = sy_payload["sy"]

    # Build model + load weights
    model = AlexNet64Gray(num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # get list of conv/linear layers to quantize in order
    layers = ordered_conv_linear_modules(model)

    export: Dict[str, Any] = {
        "layers": [],
        "assumptions": {
            "weights": "int8 symmetric per-output-channel",
            "activations": "uint8 post-ReLU with z=0 (as calibrated)",
            "bias": "int32 in accumulator units",
            "requant": "qy ~= (acc*m + rounding) >> r",
        },
        "inputs": {
            "s0": float(args.s0),
            "zx": int(args.zx),
            "zy": int(args.zy),
        },
        "sources": {
            "checkpoint": str(ckpt_path),
            "sy_json": str(sy_path),
        },
        "mnist64_cfg": sy_payload.get("mnist64_cfg", None),
    }

    arrays: Dict[str, np.ndarray] = {}

    # Scale chaining:
    # s_x for first layer = s0, then s_x(next) = s_y(current)
    s_x = float(args.s0)

    # Helper to fetch s_y via "next ReLU" name
    def get_sy_for_layer_output(layer_name: str) -> Optional[float]:
        relu_name = find_next_relu_name(model, layer_name)
        if relu_name is None:
            return None
        return sy_dict.get(relu_name, None)

    for li, (lname, mod) in enumerate(layers):
        # extract weight and biases in float32
        w = mod.weight.detach().to(torch.float32)
        b = mod.bias.detach().to(torch.float32) if mod.bias is not None else None

        # Weight scale per output channel + quantize weights
        s_w = per_out_channel_weight_scale(w)
        q_w = quantize_weights_int8(w, s_w)

        # Bias quantization requires s_x
        if b is not None:
            q_b = quantize_bias_int32(b, s_x=s_x, s_w=s_w)
        else:
            q_b = None

        # Find output activation scale for this layer: s_y comes from the ReLU after it (post-ReLU)
        s_y = get_sy_for_layer_output(lname)

        # Decide whether this is a "requantized" layer output or last/logits
        has_relu_after = s_y is not None

        # Compute per-channel M and then (m,r) only if we have s_y (i.e., quantize point exists)
        m_list: Optional[torch.Tensor] = None
        r_list: Optional[torch.Tensor] = None

        if has_relu_after:
            # Per-output-channel M
            # M_c = (s_x * s_w[c]) / s_y
            M = (s_w * float(s_x)) / float(s_y)
            # Convert each M to (m,r)
            m_vals = []
            r_vals = []
            for Mc in M.cpu().tolist():
                m_c, r_c = quantize_M_to_mr(float(Mc))
                m_vals.append(m_c)
                r_vals.append(r_c)
            m_list = torch.tensor(m_vals, dtype=torch.int32)
            r_list = torch.tensor(r_vals, dtype=torch.int32)  # store signed shifts safely
        else:
            # No ReLU scale entry: likely last logits layer.
            # By default we DO NOT requantize logits; keep acc/int32 and do argmax.
            # If you really want m/r here, you must provide a scale for logits and enable export_last_layer_mr.
            if args.export_last_layer_mr:
                raise RuntimeError(
                    f"No post-ReLU s_y found for layer '{lname}'. "
                    f"To export last layer m/r, add a logits scale entry (enable --include-logits in find_scales.py) "
                    f"or provide a dedicated logits s_y."
                )

        # Save arrays with stable keys
        layer_key = lname.replace(".", "_")
        arrays[f"{layer_key}__W_q_int8"] = q_w.cpu().numpy().astype(np.int8)

        arrays[f"{layer_key}__s_w_fp32"] = s_w.cpu().numpy().astype(np.float32)  # optional debug; not needed on FPGA

        if q_b is not None:
            arrays[f"{layer_key}__B_q_int32"] = q_b.cpu().numpy().astype(np.int32)

        if m_list is not None and r_list is not None:
            arrays[f"{layer_key}__m_int32"] = m_list.cpu().numpy().astype(np.int32)
            arrays[f"{layer_key}__r_int32"] = r_list.cpu().numpy().astype(np.int32)

        # Metadata for this layer
        entry: Dict[str, Any] = {
            "name": lname,
            "type": mod.__class__.__name__,
            "weight_shape": list(w.shape),
            "bias_shape": (list(b.shape) if b is not None else None),
            "s_x": float(s_x),
            "s_y": (float(s_y) if s_y is not None else None),
            "weight_scales_per_out_channel": True,
            "post_relu_quantized": bool(has_relu_after),
            "export_keys": {
                "W_q": f"{layer_key}__W_q_int8",
                "B_q": (f"{layer_key}__B_q_int32" if q_b is not None else None),
                "m": (f"{layer_key}__m_int32" if m_list is not None else None),
                "r": (f"{layer_key}__r_int32" if r_list is not None else None),
                "s_w_debug": f"{layer_key}__s_w_fp32",
            },
        }
        export["layers"].append(entry)

        # Chain s_x forward if we have s_y
        if has_relu_after:
            s_x = float(s_y)

    npz_path = out_prefix.with_suffix(".npz")
    json_path = out_prefix.with_suffix(".json")

    np.savez(npz_path, **arrays)
    json_path.write_text(json.dumps(export, indent=2))

    print(f"[quantize_weights.py] Wrote arrays: {npz_path}")
    print(f"[quantize_weights.py] Wrote metadata: {json_path}")
    print(f"[quantize_weights.py] Layers exported: {len(export['layers'])}")
    return 0    

if __name__ == "__main__":
    raise SystemExit(main())