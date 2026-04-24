# find_scales.py
# This file finds the right s_y scales for each layer.
# It does not actually quantize anything. Computes per-layer activation output scales s_y at chosen 
# quantization points (default post-ReLU) using P99.9 and saves to JSON.
# - Does not quantize; only produces calibration constants used by later quantization.

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ml.src.data.mnist64 import MNIST64Config, get_dataloaders
from ml.src.models.alexnet64gray import AlexNet64Gray
from ml.src.utils import EPS, INT8_MAX, UINT8_MAX, latest_run_id, resolve_device

class RunningTensorStats:
    """
    Statistics collector for layer output tensors.
    
    This helper is used during calibration to estimate a robust activation range
    for quantization. For each tensor it sees, it tracks:
      - exact maximum value observed
      - approximate high-percentile via random subsampling 
    """
    def __init__(self, percentile: float = 0.999, sample_per_batch: int = 20000):
        """Initialize the running statistics container.

        Args:
            percentile (float, optional): Quantile used for the max activation value (less sensitive to outliers). Defaults to 0.999.
            sample_per_batch (int, optional): Number of activation values randomly keept from each tensor batch. Defaults to 20000.
        """
        self.percentile = percentile 
        self.sample_per_batch = sample_per_batch

        self.max_val: float = 0.0
        self._samples: List[torch.Tensor] = [] 

    @torch.no_grad()
    def update(self, t: torch.Tensor, take_abs: bool = False) -> None:
        """Update running statistics using a new layer-output tensor.

        This method is intended to be called from a forward hook. It:
            - optionally converts the tensor to magnitudes (abs) for signed tensors,
            - updates the exact maximum observed so far, and
            - appends a random subsample of tensor elements (moved to CPU) for later
              percentile estimation.

        Args:
            t (torch.Tensor): Layer output
            take_abs (bool, optional): Wether we take the raw values or magnitudes. This is useful for
            calibrating signed tensors for symmetric int8 quantization. Defaults to False.
        """
        # Normally t is a tensor but if not we "should" recieve either: (outputs, ..) or [outputs, ...]
        if isinstance(t, (tuple, list)):
            t = t[0]
        t = t.detach() # prevents unnecessary memory usage
        
        # Only do absolute if needed (not needed after ReLU for instance)
        if take_abs:
            t_for_stats = t.abs()
        else:
            t_for_stats = t

        # stores the current exact max
        cur_max = t_for_stats.amax().item() # amax for GPU optimization and item to get a scalar
        if cur_max > self.max_val:
            self.max_val = cur_max

        # subsample max for percentile
        if self.sample_per_batch and self.sample_per_batch > 0:
            flat = t_for_stats.flatten()
            n = flat.numel()
            k = min(self.sample_per_batch, n) # don't sample more values than exist
            if k > 0:
                idx = torch.randint(0, n, (k,), device=flat.device)
                samp = flat[idx].to("cpu", non_blocking=True) # free memory from GPU because samp will accumulate
                self._samples.append(samp)

    def percentile_value(self) -> float:
        """Compute the configured percentile value from accumulated samples.

        Returns:
            float: The estimated percentile of the observed tensor values. If no samples were
                   collected, returns the exact maximum observed. The returned value is clamped to 
                   be non-negative.
        """
        # if the samples is empty returns maximum value 
        if not self._samples:
            return self.max_val
        
        all_s = torch.cat(self._samples, dim=0) # makes a 1D tensor

        # torch.quantile expects float; ensure float32
        if not all_s.is_floating_point():
            all_s = all_s.float()
        q = torch.quantile(all_s, self.percentile).item() # returns the value such that percentile% of samples < q
        return max(float(q), 0.0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments controlling calibration.
    
    The arguments configure:
      - dataset loading and preprocessing (normalization/augmentation),
      - which model checkpoint to load,
      - which split and how many batches to use for calibration,
      - how to estimate the activation range (percentile and sampling),
      - which module outputs to hook (ReLU/Conv/Linear),
      - where to save the resulting scales JSON.
    
    Returns:
        argparse.Namespace: contains all parsed arguments.
    """
    p = argparse.ArgumentParser(description="Calibrate per-layer activation scales s_y (post-ReLU) for AlexNet64Gray.")
    p.add_argument("--data-dir", type=str, default="ml/data", help="Indicate where the data is stored for the forward pass")
    p.add_argument("--batch-size", type=int, default=128, help="Control how many images go through the model at once")
    p.add_argument("--num-workers", type=int, default=2, help="How many CPU workers load data in parallel")
    p.add_argument("--val-ratio", type=float, default=0.1, help="defines how dataset is split into train/val")
    p.add_argument("--seed", type=int, default=1234, help="Seed to make calibration reproductible")
    p.add_argument("--no-normalize", action="store_true", help="Disable MNIST mean/std normalization")
    p.add_argument("--augment", action="store_true", help="Enable light RandomAffine augmentation")

    p.add_argument("--checkpoints-dir", type=str, default="ml/checkpoints", help="Base dir for model weights")
    p.add_argument("--outputs-dir",     type=str, default="ml/outputs",     help="Base dir for export outputs")
    p.add_argument("--run-id",          type=int, default=-1, help="Run ID to use (default: latest)")
    p.add_argument("--ckpt", type=str,  default="", help="Override checkpoint path (overrides --run-id)")
    p.add_argument("--out",  type=str,  default="", help="Override output JSON path (overrides --run-id)")
    p.add_argument("--device", type=str, default="", help="Force device: 'cpu' or 'cuda'. Empty = auto")

    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                   help="Which split to use for calibration (default: train)")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Limit number of batches (0 = no limit / full split)")
    p.add_argument("--percentile", type=float, default=0.999, help="Percentile for range (e.g. 0.999 for 99.9%)")
    p.add_argument("--sample-per-batch", type=int, default=20000,
                   help="How many activation values to subsample per layer per batch")

    p.add_argument("--hook", type=str, default="relu", choices=["relu", "conv", "linear"],
                   help="Which module outputs to treat as quantization points. For your design use 'relu'.")
    p.add_argument("--include-logits", action="store_true",
                   help="Also compute a scale for final logits (symmetric int8 style, abs-percentile/127).")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Seed PyTorch RNGs for reproducible calibration.

    Args:
        seed (int): Random seed used for torch CPU RNG and (if available) all CUDA RNGs.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_loader(split: str, train_loader, val_loader, test_loader):
    """Select the dataloader corresponding to the requested dataset split.

    Args:
        split (str): Which split to use ("train", "val", or "test").
        train_loader (_type_): Dataloader for the training split.
        val_loader (_type_): Dataloader for the validation split.
        test_loader (_type_): Dataloader for the test split.

    Returns:
        _type_: The dataloader matching the requested split.
    """
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    return test_loader


def register_hooks(
    model: nn.Module,
    hook_what: str,
    stats: Dict[str, RunningTensorStats],
    percentile: float,
    sample_per_batch: int,
    include_logits: bool,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks to collect activation statistics at chosen quantization points.

    For each module in `model` whose type matches `hook_what`:
      - create (if needed) a RunningTensorStats entry in `stats` keyed by the module name
      - attach a forward hook that records the module's output tensor into that stats entry

    Hook modes:
      - "relu": hooks nn.ReLU outputs (post-ReLU activations; non-negative)
      - "conv": hooks nn.Conv2d outputs (typically pre-ReLU; signed)
      - "linear": hooks nn.Linear outputs (typically signed)
      - If `include_logits` is True, an additional hook is attached to the final nn.Linear
        module to collect a separate stats entry labeled with "__logits" for symmetric int8
        logits scaling.

    Args:
        model (nn.Module): Pytorch model
        hook_what (str): Which module type to hook ("relu", "conv", or "linear").
        stats (Dict[str, RunningTensorStats]): Dictionary that will be populated/updated with RunningTensorStats objects.
        percentile (float): Percentile used inside each RunningTensorStats
        sample_per_batch (int): Number of values sampled per batch per hooked module.
        include_logits (bool): If True, also collect stats for final logits output of the last Linear.

    Returns:
        List[torch.utils.hooks.RemovableHandle]: A list of RemovableHandle objects. Call handle.remove() on each to unregister hooks.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = [] # keep the handles if we want to unhook functions

    def make_hook(name: str, take_abs: bool):
        def _hook(_mod, _inp, out): # ignore module object and inputs, we only need outputs 
            stats[name].update(out, take_abs=take_abs)
        return _hook

    # Which modules to hook
    if hook_what == "relu":
        capture_types = (nn.ReLU,)
        take_abs = False  # ReLU outputs are >= 0
    elif hook_what == "conv":
        capture_types = (nn.Conv2d,)
        take_abs = True   # conv outputs can be signed if pre-ReLU
    else:
        capture_types = (nn.Linear,)
        take_abs = True

    # Create stats entries on demand. Ensures there is a runningtensorstats object for the given layer name
    def get_stats(name: str) -> RunningTensorStats:
        if name not in stats:
            stats[name] = RunningTensorStats(percentile=percentile, sample_per_batch=sample_per_batch)
        return stats[name]

    #loop over model modules and attach hooks
    for name, mod in model.named_modules():
        if isinstance(mod, capture_types):
            # hook output of this module
            _ = get_stats(name)
            handles.append(mod.register_forward_hook(make_hook(name, take_abs=take_abs)))

    # Optionally hook final logits (output of last Linear)
    if include_logits:
        last_linear_name: Optional[str] = None
        last_linear_mod: Optional[nn.Linear] = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                last_linear_name, last_linear_mod = name, mod

        if last_linear_name is not None and last_linear_mod is not None:
            logits_name = f"{last_linear_name}__logits"
            stats[logits_name] = RunningTensorStats(percentile=percentile, sample_per_batch=sample_per_batch)
            handles.append(last_linear_mod.register_forward_hook(make_hook(logits_name, take_abs=True)))

    return handles


@torch.no_grad()
def main() -> int:
    """Run activation-scale calibration and write per-layer scales to JSON.

    This function:
      1) parses CLI args and seeds RNG,
      2) builds MNIST64 dataloaders for the requested split,
      3) loads AlexNet64Gray and its checkpoint weights in eval mode,
      4) registers forward hooks to collect activation statistics,
      5) runs forward passes over some or all batches to accumulate stats,
      6) converts percentile values to quantization scales (s_y) using:
           - post-ReLU uint8: s_y = P(y) / 255
           - signed symmetric int8: s = P(|y|) / 127
         and optionally logits: s_logits = P(|z|) / 127,
      7) saves scales and metadata to the output JSON path.

    Returns:
        int: 0 on success
    """
    args = parse_args()

    if not (0.0 < args.percentile < 1.0):
        raise SystemExit(f"--percentile must be in (0, 1), got {args.percentile}")

    set_seed(args.seed)

    # -- Resolve run-id based paths -------------------------------------------
    checkpoints_base = Path(args.checkpoints_dir).expanduser().resolve()
    outputs_base     = Path(args.outputs_dir).expanduser().resolve()
    run_id           = args.run_id if args.run_id >= 0 else latest_run_id(checkpoints_base)

    ckpt_path = Path(args.ckpt).expanduser().resolve() if args.ckpt \
                else checkpoints_base / f"run{run_id}" / "best.pth"
    out_path  = Path(args.out).expanduser().resolve()  if args.out  \
                else outputs_base / f"run{run_id}" / "act_scales_sy.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[find_scales] run{run_id}  ckpt    → {ckpt_path}")
    print(f"[find_scales] run{run_id}  output  → {out_path}")

    # -- Config ---------------------------------------------------------------
    cfg = MNIST64Config(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=not args.no_normalize,
        augment=args.augment,
    )

    device = resolve_device(args.device)

    # -- Data -----------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    loader = pick_loader(args.split, train_loader, val_loader, test_loader)

    # -- Model + checkpoint ---------------------------------------------------
    model = AlexNet64Gray(num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Create stats dict and attach hooks
    stats: Dict[str, RunningTensorStats] = {}
    handles = register_hooks(
        model=model,
        hook_what=args.hook,
        stats=stats,
        percentile=args.percentile,
        sample_per_batch=args.sample_per_batch,
        include_logits=args.include_logits,
    )

    max_batches = None if args.max_batches == 0 else args.max_batches
    total = max_batches if max_batches is not None else len(loader)

    try:
        for bi, batch in enumerate(tqdm(loader, total=total, desc="[find_scales] calibrating", unit="batch")):
            if max_batches is not None and bi >= max_batches:
                break
            x, _y = batch
            x = x.to(device, non_blocking=True)
            _ = model(x)
    finally:
        for h in handles:
            h.remove()

    # Convert stats -> s_y:
    #   - If hook == relu: post-ReLU => uint8 => s_y = P99.9(y) / 255
    #   - If include_logits: logits scale is symmetric int8 => s_logits = P99.9(|z|) / 127
    sy: Dict[str, float] = {}
    meta: Dict[str, Dict] = {}

    for name, st in stats.items():
        p = st.percentile_value()
        p = max(p, EPS)

        if name.endswith("__logits"):
            s = p / float(INT8_MAX)
            kind = "logits_symmetric_int8"
            denom = INT8_MAX
        else:
            if args.hook == "relu":
                s = p / float(UINT8_MAX)
                kind = "post_relu_uint8"
                denom = UINT8_MAX
            else:
                s = p / float(INT8_MAX)
                kind = "signed_symmetric_int8"
                denom = INT8_MAX

        sy[name] = float(s)
        meta[name] = {
            "percentile_value": float(p),
            "max_value": float(st.max_val),
            "denom": denom,
            "kind": kind,
        }

    payload = {
        "calibration": {
            "split": args.split,
            "max_batches": args.max_batches,
            "percentile": args.percentile,
            "sample_per_batch": args.sample_per_batch,
            "hook": args.hook,
            "include_logits": bool(args.include_logits),
            "device": str(device),
        },
        "mnist64_cfg": asdict(cfg),
        "checkpoint": str(ckpt_path),
        "sy": sy,
        "details": meta,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[find_scales.py] Wrote activation scales to: {out_path}")
    print(f"[find_scales.py] Collected {len(sy)} scale entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())