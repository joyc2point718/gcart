"""
Classical preprocessing baselines for the GC-ART paper.

All three are parameter-free (or have a single fixed gamma) and run on GPU.
They are wrapped as nn.Modules so they plug into the same train/eval pipeline
as the learned methods.

    - HEModule:    per-image, per-channel histogram equalization
    - CLAHEModule: applies cv2.createCLAHE on CPU (slower; CIFAR-10 sizes only)
    - GammaModule: applies x ** (1/gamma) with a fixed gamma

Usage:
    from classical import ClassicalSystem
    sys = ClassicalSystem(method="he")
    sys = ClassicalSystem(method="gamma", gamma=2.2)
"""

from __future__ import annotations
import torch
import torch.nn as nn

from models import make_resnet18_cifar


# ------------------------------------------------------------------
# Histogram equalization on GPU
# ------------------------------------------------------------------
class HEModule(nn.Module):
    """Per-image, per-channel histogram equalization on the GPU.

    For each (batch, channel) we quantize to `num_bins` levels, compute the
    CDF, and remap pixel values via gather. No learnable parameters; not
    differentiable but we never train through it.
    """

    def __init__(self, num_bins: int = 256):
        super().__init__()
        self.num_bins = num_bins

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_clip = x.clamp(0.0, 1.0)
        x_flat = x_clip.view(B, C, -1)
        x_q = (x_flat * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)

        hist = torch.zeros(B, C, self.num_bins, device=x.device, dtype=torch.float32)
        ones = torch.ones_like(x_q, dtype=torch.float32)
        hist.scatter_add_(2, x_q, ones)
        cdf = hist.cumsum(dim=2)
        cdf = cdf / cdf[:, :, -1:].clamp(min=1.0)

        y_flat = torch.gather(cdf, 2, x_q)
        return y_flat.view(B, C, H, W).to(x.dtype)


# ------------------------------------------------------------------
# CLAHE on CPU via cv2
# ------------------------------------------------------------------
class CLAHEModule(nn.Module):
    """Contrast Limited Adaptive Histogram Equalization via cv2.

    Runs on the CPU; for CIFAR-10 sizes this is fine (~few ms per batch).
    Falls back to plain HE if cv2 is unavailable.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile = tile_grid_size
        try:
            import cv2  # noqa: F401
            self._cv2_available = True
        except ImportError:
            self._cv2_available = False
            print("[CLAHEModule] cv2 not available; falling back to plain HE")

        self._he_fallback = HEModule()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._cv2_available:
            return self._he_fallback(x)

        import cv2  # local import to keep top-level lighter
        device = x.device
        x_cpu = (x.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        # x_cpu shape: (B, C, H, W); cv2 wants (H, W) per channel
        B, C, H, W = x_cpu.shape
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=(self.tile, self.tile))
        out = x_cpu.copy()
        for b in range(B):
            for c in range(C):
                out[b, c] = clahe.apply(x_cpu[b, c])
        out_t = torch.from_numpy(out).to(device, dtype=x.dtype) / 255.0
        return out_t


# ------------------------------------------------------------------
# Gamma correction (fixed gamma)
# ------------------------------------------------------------------
class GammaModule(nn.Module):
    """Applies x ** (1/gamma). Fixed gamma; train multiple to grid-search."""

    def __init__(self, gamma: float = 2.2):
        super().__init__()
        self.gamma = float(gamma)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=1e-6).pow(1.0 / self.gamma)


# ------------------------------------------------------------------
# System wrapper for any classical preprocessing method
# ------------------------------------------------------------------
class ClassicalSystem(nn.Module):
    """Classical preprocessing + ResNet-18 backbone."""

    def __init__(self, method: str, num_classes: int = 10, **kwargs):
        super().__init__()
        if method == "he":
            self.enhancer = HEModule(**kwargs)
        elif method == "clahe":
            self.enhancer = CLAHEModule(**kwargs)
        elif method == "gamma":
            self.enhancer = GammaModule(**kwargs)
        elif method == "identity":
            self.enhancer = nn.Identity()
        else:
            raise ValueError(f"Unknown classical method: {method}")
        self.backbone = make_resnet18_cifar(num_classes)

    def forward(self, x):
        x_e = self.enhancer(x)
        return self.backbone(x_e), torch.zeros((), device=x.device)
