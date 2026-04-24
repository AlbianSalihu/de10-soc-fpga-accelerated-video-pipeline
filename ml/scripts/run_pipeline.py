# run_pipeline.py
# Runs the full ML pipeline end-to-end from a single config file.
#
# Steps (in order):
#   1. train              → ml/checkpoints/runN/{best,last}.pth
#                         → ml/runs/runN/{run_meta,final_report}.json
#   2. find_scales        → ml/outputs/runN/act_scales_sy.json
#   3. quantize_weights   → ml/outputs/runN/fpgaqparms.{npz,json}
#   4. test_quantized     → accuracy comparison + per-layer error table (stdout)
#   5. export_weights     → ml/outputs/runN/fpgaqparms.bin
#
# Usage:
#   python -m ml.scripts.run_pipeline
#   python -m ml.scripts.run_pipeline --config ml/config/default.yaml --device cuda
#   python -m ml.scripts.run_pipeline --skip-train --run-id 0
#
# Each step's main() is called in-process (no subprocess overhead).
# sys.argv is patched per step so each script's parse_args() sees the right flags.

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml

from ml.src.export.export_weights import main as export_main
from ml.src.export.find_scales import main as calibrate_main
from ml.src.export.quantize_weights import main as quantize_main
from ml.src.export.test_quantized_model import main as test_main
from ml.src.train.train import main as train_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _argv(args: List[str]):
    """Temporarily replace sys.argv so parse_args()-based mains see the right flags."""
    old = sys.argv
    sys.argv = [old[0]] + args
    try:
        yield
    finally:
        sys.argv = old


def _run_step(name: str, fn: Callable[[], int | None], args: List[str]) -> None:
    """Run one pipeline step, printing a clear header and timing it."""
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  STEP: {name}")
    print(f"  args: {' '.join(args) if args else '(none)'}")
    print(bar)
    t0 = time.time()
    with _argv(args):
        rc = fn()
    elapsed = time.time() - t0
    if rc not in (None, 0):
        raise RuntimeError(f"Step '{name}' returned code {rc}. Pipeline aborted.")
    print(f"\n  [{name}] Finished in {elapsed:.1f}s")


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse pipeline-level arguments."""
    p = argparse.ArgumentParser(
        description="Run the full ML pipeline (train → calibrate → quantize → test → export)."
    )
    p.add_argument(
        "--config", type=str, default="ml/config/default.yaml",
        help="Pipeline config YAML (default: ml/config/default.yaml)",
    )
    p.add_argument(
        "--run-id", type=int, default=-1,
        help="Pin all steps to a specific run ID. "
             "Default (-1): train auto-increments, export steps pick latest.",
    )
    p.add_argument(
        "--device", type=str, default="",
        help="Device override for all steps: 'cpu', 'cuda', or empty=auto.",
    )
    p.add_argument("--skip-train",     action="store_true", help="Skip training step.")
    p.add_argument("--skip-calibrate", action="store_true", help="Skip find_scales step.")
    p.add_argument("--skip-quantize",  action="store_true", help="Skip quantize_weights step.")
    p.add_argument("--skip-test",      action="store_true", help="Skip test_quantized_model step.")
    p.add_argument("--skip-export",    action="store_true", help="Skip export_weights step.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point: load config, run each pipeline step in order."""
    args = parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    cfg  = _load_config(cfg_path)
    tr   = cfg.get("train",        {})
    cal  = cfg.get("calibration",  {})
    qnt  = cfg.get("quantization", {})

    # Load FPGA config referenced inside pipeline config (or fall back to default path)
    fpga_config_path = cfg.get("fpga_config", "ml/config/fpga.yaml")

    # Common flags passed to every relevant step
    run_id_args = ["--run-id", str(args.run_id)] if args.run_id >= 0 else []
    device_args = ["--device", args.device]       if args.device       else []

    t_pipeline = time.time()

    # -- 1. Train -------------------------------------------------------------
    if not args.skip_train:
        step_args = [
            "--epochs",       str(tr.get("epochs",       10)),
            "--batch-size",   str(tr.get("batch_size",   128)),
            "--lr",           str(tr.get("lr",           1e-3)),
            "--weight-decay", str(tr.get("weight_decay", 0.0)),
            "--seed",         str(tr.get("seed",         1234)),
            "--val-ratio",    str(tr.get("val_ratio",    0.1)),
        ]
        if tr.get("augment", False):
            step_args.append("--augment")
        step_args += run_id_args + device_args
        _run_step("train", train_main, step_args)

    # -- 2. Calibrate activation scales ---------------------------------------
    if not args.skip_calibrate:
        step_args = [
            "--split",            cal.get("split",            "train"),
            "--percentile",       str(cal.get("percentile",   0.999)),
            "--hook",             cal.get("hook",             "relu"),
            "--sample-per-batch", str(cal.get("sample_per_batch", 20000)),
            "--max-batches",      str(cal.get("max_batches",  0)),
        ]
        step_args += run_id_args + device_args
        _run_step("find_scales", calibrate_main, step_args)

    # -- 3. Quantize weights --------------------------------------------------
    if not args.skip_quantize:
        s0 = qnt.get("s0")
        if s0 is None:
            raise SystemExit("quantization.s0 is required in the pipeline config.")
        step_args = ["--s0", str(s0)] + run_id_args + device_args
        _run_step("quantize_weights", quantize_main, step_args)

    # -- 4. Test quantized model ----------------------------------------------
    # s0 is read automatically from fpgaqparms.json — no --s0 needed here.
    if not args.skip_test:
        step_args = run_id_args + device_args
        _run_step("test_quantized_model", test_main, step_args)

    # -- 5. Export FPGA binary ------------------------------------------------
    if not args.skip_export:
        step_args = ["--fpga-config", fpga_config_path] + run_id_args
        _run_step("export_weights", export_main, step_args)

    total = time.time() - t_pipeline
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  PIPELINE COMPLETE  ({total:.1f}s total)")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
