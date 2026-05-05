"""
Data loaders.

We avoid downloading the 2.5GB CIFAR-10-C archive by computing the
brightness/contrast corruptions on the fly using the official Hendrycks &
Dietterich (ICLR 2019) formulas, plus a "darken" corruption that better
matches the low-light setting GC-ART targets.

    HFCifar10:        wraps the HuggingFace CIFAR-10 split + a transform.
    CorruptedCifar10: applies one of {brightness, contrast, darken} at a
                      given severity to a base CIFAR-10 test set.

Note on the "brightness" corruption:
    The standard CIFAR-10-C `brightness` corruption ADDS to the HSV V
    channel (i.e., it brightens images). It is included for protocol
    compatibility with the published benchmark even though our headline
    setting is low-light.
"""

from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset


# ------------------------------------------------------------------
# Wrapper around HF cifar10
# ------------------------------------------------------------------
class HFCifar10(Dataset):
    """HuggingFace uoft-cs/cifar10 with a torchvision transform."""

    def __init__(self, split: str, transform):
        self.ds = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["img"]                # PIL.Image, RGB, 32x32
        label = int(item["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ------------------------------------------------------------------
# RGB <-> HSV in pytorch (no scikit-image dep)
# ------------------------------------------------------------------
def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (..., 3) in [0, 1] -> hsv (..., 3) in [0, 1]."""
    r, g, b = rgb.unbind(-1)
    maxc, _ = rgb.max(-1)
    minc, _ = rgb.min(-1)
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc > 0, delta / maxc.clamp(min=1e-8), torch.zeros_like(maxc))
    rc = (maxc - r) / delta.clamp(min=1e-8)
    gc = (maxc - g) / delta.clamp(min=1e-8)
    bc = (maxc - b) / delta.clamp(min=1e-8)
    h = torch.where(r == maxc, bc - gc,
        torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = torch.where(delta == 0, torch.zeros_like(h), h)
    return torch.stack([h, s, v], dim=-1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """hsv: (..., 3) in [0, 1] -> rgb (..., 3) in [0, 1]."""
    h, s, v = hsv.unbind(-1)
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_int = i.long() % 6
    out = torch.zeros_like(hsv)

    def _set(mask, r, g, b):
        out[..., 0] = torch.where(mask, r, out[..., 0])
        out[..., 1] = torch.where(mask, g, out[..., 1])
        out[..., 2] = torch.where(mask, b, out[..., 2])

    _set(i_int == 0, v, t, p)
    _set(i_int == 1, q, v, p)
    _set(i_int == 2, p, v, t)
    _set(i_int == 3, p, q, v)
    _set(i_int == 4, t, p, v)
    _set(i_int == 5, v, p, q)
    return out


# ------------------------------------------------------------------
# Hendrycks & Dietterich corruption functions (CIFAR-10-C variant)
# ------------------------------------------------------------------
_BRIGHTNESS_C = [0.1, 0.2, 0.3, 0.4, 0.5]
_CONTRAST_C   = [0.4, 0.3, 0.2, 0.1, 0.05]
# darkening: multiplicative attenuation. Severity 5 ~ 0.1x.
_DARKEN_C     = [0.8, 0.6, 0.4, 0.25, 0.1]


def corrupt_brightness(x: torch.Tensor, severity: int) -> torch.Tensor:
    """x: (C, H, W) in [0, 1]. Returns same shape."""
    c = _BRIGHTNESS_C[severity - 1]
    img = x.permute(1, 2, 0).clamp(0, 1)         # (H, W, 3)
    hsv = rgb_to_hsv(img)
    hsv[..., 2] = (hsv[..., 2] + c).clamp(0, 1)
    rgb = hsv_to_rgb(hsv)
    return rgb.permute(2, 0, 1).clamp(0, 1)


def corrupt_contrast(x: torch.Tensor, severity: int) -> torch.Tensor:
    """x: (C, H, W) in [0, 1]. Returns same shape."""
    c = _CONTRAST_C[severity - 1]
    means = x.mean(dim=(1, 2), keepdim=True)
    return ((x - means) * c + means).clamp(0, 1)


def corrupt_darken(x: torch.Tensor, severity: int) -> torch.Tensor:
    """x: (C, H, W) in [0, 1]. Multiplicative attenuation."""
    c = _DARKEN_C[severity - 1]
    return (x * c).clamp(0, 1)


CORRUPTIONS: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "brightness": corrupt_brightness,
    "contrast":   corrupt_contrast,
    "darken":     corrupt_darken,
}


# ------------------------------------------------------------------
# Corrupted dataset wrapper
# ------------------------------------------------------------------
class CorruptedCifar10(Dataset):
    """Applies a CIFAR-10-C-style corruption to each item of a base dataset.

    The base dataset must yield (tensor in [0,1], label).
    """

    def __init__(self, base: Dataset, corruption: str, severity: int):
        if corruption not in CORRUPTIONS:
            raise KeyError(
                f"Unknown corruption {corruption}. Options: {list(CORRUPTIONS)}"
            )
        if severity not in (1, 2, 3, 4, 5):
            raise ValueError(f"Severity must be in 1..5, got {severity}")
        self.base = base
        self.fn = CORRUPTIONS[corruption]
        self.severity = severity
        self.corruption = corruption

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = self.fn(x, self.severity)
        return x, y
