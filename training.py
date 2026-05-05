"""
Training + evaluation utilities.

Adds, on top of the original script:
    - precision auto-detection (bf16 on Ampere+/Hopper, fp16 + GradScaler else)
    - multi-seed support
    - configurable monotonicity-loss weight (for the no-mono ablation)
    - structured JSON result dumps
    - CIFAR-10-C evaluation across {brightness, contrast, darken} x severities 1..5
"""

from __future__ import annotations
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data import HFCifar10, CorruptedCifar10, CORRUPTIONS


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def reset_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------------
# Precision auto-detection
# ------------------------------------------------------------------
def detect_amp_dtype():
    """Returns (dtype, use_grad_scaler).

    bf16 needs Ampere+ (compute capability >= 8.0). On Turing/Volta we fall
    back to fp16 + GradScaler.
    """
    if not torch.cuda.is_available():
        return torch.float32, False
    cc_major, _ = torch.cuda.get_device_capability(0)
    if cc_major >= 8:
        return torch.bfloat16, False
    return torch.float16, True


# ------------------------------------------------------------------
# Standard transforms
# ------------------------------------------------------------------
def standard_train_transform(brightness_jitter=(0.5, 1.0)):
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=brightness_jitter),
        T.ToTensor(),
    ])


def standard_test_transform():
    return T.Compose([T.ToTensor()])


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
def make_loaders(
    batch_size: int = 1024,
    num_workers: int = 8,
    seed: int = 42,
    brightness_jitter=(0.5, 1.0),
    pin_memory: bool = True,
):
    train_t = standard_train_transform(brightness_jitter)
    test_t = standard_test_transform()

    trainset = HFCifar10("train", train_t)
    testset_clean = HFCifar10("test", test_t)

    g = torch.Generator()
    g.manual_seed(seed)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )

    trainloader = DataLoader(trainset, shuffle=True, **common)
    cleanloader = DataLoader(testset_clean, shuffle=False, **common)

    # Build corruption loaders: 3 corruptions x 5 severities.
    corruption_loaders = {}
    for cname in CORRUPTIONS:
        for sev in range(1, 6):
            corrupted = CorruptedCifar10(testset_clean, cname, sev)
            corruption_loaders[(cname, sev)] = DataLoader(
                corrupted, shuffle=False, **common
            )

    return trainloader, cleanloader, corruption_loaders


# ------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> float:
    model.eval()
    correct, total = 0, 0
    autocast = torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda" and amp_dtype != torch.float32),
    )
    with autocast:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / max(total, 1)


# ------------------------------------------------------------------
# Train one (model, seed) -> dict of results
# ------------------------------------------------------------------
@dataclass
class TrainConfig:
    name: str                           # human-readable name
    model_name: str                     # for the factory
    seed: int = 42
    epochs: int = 100
    batch_size: int = 1024
    lr: float = 1e-3
    num_workers: int = 8
    mono_weight: float = 10.0           # only used for GC-ART variants
    brightness_jitter: tuple = (0.5, 1.0)
    eval_every: int = 0                 # 0 = only at the end
    out_dir: str = "results"
    extra: dict = field(default_factory=dict)


def _has_aux_loss(name: str) -> bool:
    return name.startswith("gcart")


def train_one(
    model: nn.Module,
    cfg: TrainConfig,
    trainloader,
    cleanloader,
    corruption_loaders: dict,
    device,
    amp_dtype,
    use_scaler: bool,
):
    """Train a single (model, seed) configuration and return a results dict."""

    has_aux = _has_aux_loss(cfg.model_name)
    mono_w = cfg.mono_weight if has_aux else 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device=device.type) if use_scaler else None

    history = {"epoch": [], "train_loss": [], "clean_acc": []}
    log_every = max(1, len(trainloader) // 4)

    print(f"[{cfg.name} | seed={cfg.seed}] start training "
          f"(mono_w={mono_w}, amp_dtype={amp_dtype})")

    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, running_n = 0.0, 0
        for i, (x, y) in enumerate(trainloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=(device.type == "cuda" and amp_dtype != torch.float32),
            ):
                out = model(x)
                if isinstance(out, tuple):
                    logits, aux = out
                else:
                    logits, aux = out, torch.zeros((), device=device)
                loss = criterion(logits, y) + mono_w * aux

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_n += y.size(0)

            if i % log_every == 0:
                print(f"  [{cfg.name}|s{cfg.seed}] epoch {epoch+1}/{cfg.epochs} "
                      f"batch {i}/{len(trainloader)} loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / max(running_n, 1)
        clean_acc = evaluate(model, cleanloader, device, amp_dtype)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["clean_acc"].append(clean_acc)
        print(f"[{cfg.name}|s{cfg.seed}] epoch {epoch+1} "
              f"train_loss={avg_loss:.4f}  clean_acc={clean_acc:.2f}%")

    # Final corruption sweep
    final_corruption = {}
    for (cname, sev), loader in corruption_loaders.items():
        acc = evaluate(model, loader, device, amp_dtype)
        final_corruption[f"{cname}_s{sev}"] = acc
        print(f"  [{cfg.name}|s{cfg.seed}] {cname}@{sev}: {acc:.2f}%")

    elapsed = time.time() - t0

    result = {
        "config": asdict(cfg),
        "amp_dtype": str(amp_dtype),
        "elapsed_sec": elapsed,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "final_clean_acc": history["clean_acc"][-1],
        "final_corruption_acc": final_corruption,
        "history": history,
    }

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(
        cfg.out_dir, f"{cfg.model_name}_s{cfg.seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{cfg.name}|s{cfg.seed}] saved -> {out_path}")
    return result
