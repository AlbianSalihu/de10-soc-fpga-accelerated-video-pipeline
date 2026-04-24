# This script trains a PyTorch model on MNIST digits resized to 64x64 grayscale. 
# 
# It produces:
#       - last.pth: checkpoint saved after every epoch (latest training state)
#       - best.pth: checkpoint saved whenever validation accuracy improves
#       - run_meta.json: the run configuration (CLI args + dataset config + model name)
#       - final_report.json: final test metrics (evaluated using best.pth)
# Run: 
#   python -m ml.src.train.train \
#           --data-dir ml/data \
#           --epochs 10 \
#           --batch-size 128 \
#           --lr 1e-3 \
#           --weight-decay 0.0 \
#           --val-ratio 0.1 \
#           --seed 1234 \
#           --augment \
#           --checkpoints-dir ml/checkpoints \
#           --runs-dir ml/runs \
#           --run-id 0 \
#           --device cuda
#
# Notes:
#       - Classification uses CrossEntropyLoss
#       - DataLoaders are provided by ml.src.data.mnist64.get_dataloaders
#       - Validation split is done inside get_dataloaders based on val_ratio and seed

from __future__ import annotations

import argparse
import json 
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch 
import torch.nn as nn
import torch.optim as optim

from ml.src.data.mnist64 import MNIST64Config, get_dataloaders
from ml.src.models.alexnet64gray import AlexNet64Gray
from ml.src.utils import next_run_id, resolve_device

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top 1 accuracy for a batch from raw logits

    Args:
        logits (torch.Tensor): Model outputs before softmax, shape[batch size, number of class]
        targets (torch.Tensor): Ground truth class indices, shape[batch size]

    Returns:
        float: top 1 accuracy for the batch in [0.0, 1.0].
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds==targets).sum().item()
    return correct/targets.size(0)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
)->Tuple[float, float]:
    """Evaluate a model on a dataloader. 

    This runs the model in eval mode and computes the average loss and classification accuracy over 
    the whole loader.

    Args:
        model (nn.Module): The model to evaluate
        loader (torch.utils.data.DataLoader): DataLoader for validation or test data
        loss_fn (nn.Module): Loss function (mainly CrossEntropy)
        device (torch.device): Device to run on (CPU or cuda)

    Returns:
        Tuple[float, float]: (avg_loss, avg_accuracy) over loader.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Forward pass
        logits=model(x)
        loss=loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1)==y).sum().item()
        total_n += bs

    return total_loss/total_n, total_correct/total_n

def train_one_epoch(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch over the training DataLoader.

    Args:
        model (nn.Module): The model to train
        loader (torch.utils.data.DataLoader): DataLoader for the training data
        loss_fn (nn.Module): Loss function (CrossEntropyLoss)
        optimizer (optim.Optimizer): Optimizer (Adam)
        device (torch.device): device to run on (CPU or cuda)

    Returns:
        Tuple[float, float]: (avg_loss, avg_accuracy) over the epoch
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x,y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward -> loss -> backward -> update
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        # Accumulate epoch stats
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1)==y).sum().item()
        total_n += bs

    return total_loss / total_n, total_correct / total_n

def save_checkpoint(
        out_dir: Path,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        best_val_acc: float,
        cfg: MNIST64Config, 
        extra: Dict,
        filename: str,
)->None:
    """Save a training checkpoint to disk

    The checkpoints includes:
    - current epoch
    - model parameters (state_dict)
    - optimizer state
    - best validation accuracy so far
    - dataset config used for training
    - extra metrics (loss/acc)

    Args:
        out_dir (Path): Directory where checkpoints file is saved
        model (nn.Module): Model whose weights to save
        optimizer (optim.Optimizer): Optimizer state to save
        epoch (int): Current epoch number (1-based)
        best_val_acc (float): Best validation accuracy observed so far
        cfg (MNIST64Config): Dataset/Training config snapshot
        extra (Dict): Additional metadata/metrics to store
        filename (str): Checkpoint filename (last.pth or best.pth)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "mnist64_cfg": asdict(cfg),
            "extra": extra, 
        },
        path,
    )

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training

    Returns:
        argparse.Namespace: contains all parsed arguments.
    """
    p = argparse.ArgumentParser(description="Train AlexNet64Gray on MNIST resized to 64x64.")
    p.add_argument("--data-dir", type=str, default="ml/data", help="Dataset cache dir (default: ml/data)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-normalize", action="store_true", help="Disable MNIST mean/std normalization")
    p.add_argument("--augment", action="store_true", help="Enable light RandomAffine augmentation")
    p.add_argument("--checkpoints-dir", type=str, default="ml/checkpoints", help="Base dir for model weights")
    p.add_argument("--runs-dir", type=str, default="ml/runs", help="Base dir for run metadata")
    p.add_argument("--run-id", type=int, default=-1, help="Run ID (default: auto-increment from existing runs)")
    p.add_argument("--device", type=str, default="", help="Force device: 'cpu' or 'cuda'. Empty = auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproductibilty.

    Args:
        seed (int): number used to generate the seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main()->int:
    """Main training entrypoint

    - Builds MNIST64 dataloaders (train/val/test)
    - Trains AlexNet64Gray for N epoch
    - Saves "last.pth" every epoch and "best.pth" when validation improves
    - Evaluates best checkpoint on the test set and writes a final report

    Returns:
        int: Failed or not. 0 if success
    """
    args = parse_args()

    if args.epochs < 1:
        raise SystemExit(f"--epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        raise SystemExit(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.lr <= 0.0:
        raise SystemExit(f"--lr must be > 0, got {args.lr}")

    set_seed(args.seed)

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

    checkpoints_base = Path(args.checkpoints_dir).expanduser().resolve()
    runs_base        = Path(args.runs_dir).expanduser().resolve()

    run_id  = args.run_id if args.run_id >= 0 else next_run_id(checkpoints_base)
    ckpt_dir = checkpoints_base / f"run{run_id}"
    runs_dir = runs_base        / f"run{run_id}"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] run{run_id}  checkpoints → {ckpt_dir}")
    print(f"[train] run{run_id}  metadata    → {runs_dir}")

    # Build dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    model = AlexNet64Gray(num_classes=10).to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_meta = {
        "run_id": run_id,
        "args": vars(args),
        "dataset_cfg": asdict(cfg),
        "model": "AlexNet64Gray",
    }
    (runs_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    best_val_acc = -1.0
    best_epoch   = -1

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} |"
            f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% |"
            f"val loss {val_loss:.4f} acc {val_acc*100:.2f}% |"
        )

        save_checkpoint(
            out_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            cfg=cfg,
            extra={"train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc},
            filename="last.pth",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            save_checkpoint(
                out_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                cfg=cfg,
                extra={"train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc},
                filename="best.pth",
            )

    best_path = ckpt_dir / "best.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"Test loss {te_loss:.4f} acc {te_acc*100:.2f}% | best val acc {best_val_acc*100:.2f}% (epoch {best_epoch})")

    report = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": te_loss,
        "test_acc": te_acc,
    }
    (runs_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())