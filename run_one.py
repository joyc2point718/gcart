"""
Run one training configuration (model + seed). Designed for SLURM job arrays.

Examples:
    # Single config:
    python run_one.py --model gcart --seed 42 --epochs 100

    # Classical baselines (no learnable enhancer; just preprocessing + ResNet):
    python run_one.py --model classical_he --seed 42 --epochs 100
    python run_one.py --model classical_clahe --seed 42 --epochs 100
    python run_one.py --model classical_gamma_2.2 --seed 42 --epochs 100

    # GC-ART with monotonicity penalty disabled (no_mono ablation):
    python run_one.py --model gcart_no_mono --seed 42 --epochs 100 --mono-weight 0
"""

from __future__ import annotations
import argparse
import os

import torch

from models import get_model, count_parameters, MODEL_REGISTRY
from classical import ClassicalSystem
from training import (
    TrainConfig, train_one, make_loaders, reset_seed, detect_amp_dtype,
)


def build_model(name: str) -> torch.nn.Module:
    """Construct a model from a name. Supports classical aliases like
    'classical_he', 'classical_clahe', 'classical_gamma_2.2'."""
    if name in MODEL_REGISTRY:
        return get_model(name)
    if name.startswith("classical_"):
        rest = name[len("classical_"):]
        if rest == "he":
            return ClassicalSystem(method="he")
        if rest == "clahe":
            return ClassicalSystem(method="clahe")
        if rest.startswith("gamma_"):
            try:
                gamma = float(rest[len("gamma_"):])
            except ValueError:
                raise ValueError(f"Bad gamma in '{name}'")
            return ClassicalSystem(method="gamma", gamma=gamma)
        if rest == "identity":
            return ClassicalSystem(method="identity")
    raise KeyError(f"Unknown model name: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--mono-weight", type=float, default=10.0,
                   help="GC-ART monotonicity-loss weight (set 0 for no_mono)")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile (useful for older GPUs)")
    args = p.parse_args()

    # ---- TF32 / cuDNN tuning ----
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"(compute capability {torch.cuda.get_device_capability(0)})")

    amp_dtype, use_scaler = detect_amp_dtype()
    print(f"AMP dtype: {amp_dtype} | grad scaler: {use_scaler}")

    reset_seed(args.seed)

    # If running gcart_no_mono ablation, force the weight to zero regardless
    # of CLI value, to keep the bookkeeping clear.
    if args.model == "gcart_no_mono" and args.mono_weight != 0.0:
        print("[run_one] forcing mono_weight=0 because model is gcart_no_mono")
        args.mono_weight = 0.0

    # Build loaders
    trainloader, cleanloader, corr_loaders = make_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Build model
    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model} | params: {n_params:,}")

    # torch.compile for backbone-bearing systems; classical preprocessors
    # have python control flow that can confuse Inductor.
    if not args.no_compile:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    cfg = TrainConfig(
        name=args.model,
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        mono_weight=args.mono_weight,
        out_dir=args.out_dir,
    )

    train_one(model, cfg, trainloader, cleanloader, corr_loaders,
              device, amp_dtype, use_scaler)


if __name__ == "__main__":
    main()
