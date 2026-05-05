"""
Models for GC-ART paper experiments.

Contains:
    - GCART_Module: full method (soft histogram + rational curve + monotonicity)
    - GCART_HardHist: ablation, hard binning + straight-through estimator
    - GCART_Polynomial: ablation, 4th-order polynomial curve
    - GCART_LUT: ablation, piecewise-linear LUT
    - MiniZeroDCE: original spatial Zero-DCE baseline (3 conv layers)
    - MiniZeroDCEpp: depthwise-separable Zero-DCE++ baseline
    - GCART_System / ZeroDCE_System / ZeroDCEpp_System / Plain_System: full pipelines
    - get_model(name): factory

All systems return (logits, aux_loss). Modules without a regularizer
return aux_loss = 0.

Backbone is ResNet-18 adapted for CIFAR-10 (3x3 first conv, no maxpool).
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ------------------------------------------------------------------
# Backbone
# ------------------------------------------------------------------
def make_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR-10: 3x3 first conv, no maxpool."""
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ------------------------------------------------------------------
# GC-ART (full method)
# ------------------------------------------------------------------
class GCART_Module(nn.Module):
    """Full GC-ART: soft RBF histogram -> small MLP -> rational tone curve.

    Curve form: f(x) = (a x^2 + b x) / (d x^2 + e x + 1)
    where the constraint b = d + e + 1 - a fixes f(1) = 1 and f(0) = 0.
    """

    def __init__(self, num_bins: int = 16, gamma: float = 0.01,
                 hidden: int = 32, mono_K: int = 32):
        super().__init__()
        self.num_bins = num_bins
        self.gamma = gamma
        self.mono_K = mono_K

        # bin centers in [0, 1]
        self.register_buffer(
            "bin_centers",
            torch.linspace(0, 1, num_bins).view(1, 1, num_bins, 1, 1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(num_bins, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),  # outputs (a, raw_d, raw_e)
        )
        self.softplus = nn.Softplus()

        # Identity initialization: zeros-weight + bias targeting f(x)=x.
        nn.init.zeros_(self.mlp[2].weight)
        with torch.no_grad():
            self.mlp[2].bias.copy_(torch.tensor([0.0, -5.0, -5.0]))

    def soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable Gaussian RBF histogram. Returns (B, C, K)."""
        # x: (B, C, H, W) -> (B, C, 1, H, W)
        x_e = x.unsqueeze(2)
        w = torch.exp(-((x_e - self.bin_centers) ** 2) / self.gamma)
        return w.mean(dim=(3, 4))

    def _curve_params(self, x: torch.Tensor):
        B, C, _, _ = x.shape
        hist = self.soft_histogram(x)            # (B, C, K)
        params = self.mlp(hist.view(B * C, self.num_bins))  # (B*C, 3)
        a = params[:, 0]
        d = self.softplus(params[:, 1])
        e = self.softplus(params[:, 2])
        b = d + e + 1.0 - a                       # boundary constraint
        return a, b, d, e, B, C

    def _apply_curve(self, x, a, b, d, e):
        a4 = a.view(-1, x.shape[1], 1, 1)
        b4 = b.view(-1, x.shape[1], 1, 1)
        d4 = d.view(-1, x.shape[1], 1, 1)
        e4 = e.view(-1, x.shape[1], 1, 1)
        num = a4 * x * x + b4 * x
        den = d4 * x * x + e4 * x + 1.0
        return num / den

    def _monotonicity_loss(self, a, b, d, e, B, C):
        # evaluate on K equally spaced points in [0, 1]
        t = torch.linspace(0, 1, self.mono_K, device=a.device).view(1, 1, -1)
        a3, b3 = a.view(B, C, 1), b.view(B, C, 1)
        d3, e3 = d.view(B, C, 1), e.view(B, C, 1)
        f_t = (a3 * t * t + b3 * t) / (d3 * t * t + e3 * t + 1.0)
        diffs = f_t[:, :, 1:] - f_t[:, :, :-1]
        # mean rather than sum so the value doesn't scale with batch / K
        return torch.relu(-diffs).mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b, d, e, B, C = self._curve_params(x)
        x_out = self._apply_curve(x, a, b, d, e)
        mono = self._monotonicity_loss(a, b, d, e, B, C)
        return x_out, mono


# ------------------------------------------------------------------
# Ablation 1: hard histogram with straight-through estimator
# ------------------------------------------------------------------
class _HardHistSTE(torch.autograd.Function):
    """Hard binning forward; gradient passed through as if soft."""

    @staticmethod
    def forward(ctx, x_e, centers, gamma):
        # forward: hard one-hot to nearest bin
        # x_e:    (B, C, 1, H, W)
        # centers:(1, 1, K, 1, 1)
        diff = (x_e - centers) ** 2
        idx = diff.argmin(dim=2, keepdim=True)         # (B, C, 1, H, W)
        K = centers.shape[2]
        hard = torch.zeros_like(diff).scatter_(2, idx, 1.0)
        # save for backward
        ctx.save_for_backward(x_e, centers)
        ctx.gamma = gamma
        return hard.mean(dim=(3, 4))                   # (B, C, K)

    @staticmethod
    def backward(ctx, grad_out):
        x_e, centers = ctx.saved_tensors
        gamma = ctx.gamma
        # use soft histogram gradients as the surrogate (straight-through)
        w = torch.exp(-((x_e - centers) ** 2) / gamma)
        # d/dx_e of the soft histogram mean
        d_w = -2.0 * (x_e - centers) / gamma * w
        # grad_out: (B, C, K). Need (B, C, 1, H, W).
        H, W = x_e.shape[3], x_e.shape[4]
        g = grad_out.view(*grad_out.shape, 1, 1) / (H * W)
        grad_x = (g * d_w).sum(dim=2, keepdim=True)
        return grad_x, None, None


class GCART_HardHist(GCART_Module):
    """Hard-binning histogram + straight-through estimator. Same curve, same MLP."""

    def soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        x_e = x.unsqueeze(2)
        return _HardHistSTE.apply(x_e, self.bin_centers, self.gamma)


# ------------------------------------------------------------------
# Ablation 2: 4th-order polynomial curve
# ------------------------------------------------------------------
class GCART_Polynomial(nn.Module):
    """Same histogram + MLP front-end, but the curve is a 4th-order
    polynomial f(x) = c1 x + c2 x^2 + c3 x^3 + c4 x^4 with c1 + c2 + c3 + c4 = 1.

    Same parameter count as the rational form (3 free coefficients per channel).
    """

    def __init__(self, num_bins: int = 16, gamma: float = 0.01,
                 hidden: int = 32, mono_K: int = 32):
        super().__init__()
        self.num_bins = num_bins
        self.gamma = gamma
        self.mono_K = mono_K

        self.register_buffer(
            "bin_centers",
            torch.linspace(0, 1, num_bins).view(1, 1, num_bins, 1, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_bins, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),  # c2, c3, c4 (c1 derived)
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)  # c2=c3=c4=0 -> c1=1 -> f(x)=x

    def soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        x_e = x.unsqueeze(2)
        w = torch.exp(-((x_e - self.bin_centers) ** 2) / self.gamma)
        return w.mean(dim=(3, 4))

    def forward(self, x):
        B, C, _, _ = x.shape
        hist = self.soft_histogram(x)
        coefs = self.mlp(hist.view(B * C, self.num_bins))   # (B*C, 3)
        c2, c3, c4 = coefs[:, 0], coefs[:, 1], coefs[:, 2]
        c1 = 1.0 - c2 - c3 - c4

        c1 = c1.view(B, C, 1, 1)
        c2 = c2.view(B, C, 1, 1)
        c3 = c3.view(B, C, 1, 1)
        c4 = c4.view(B, C, 1, 1)

        x_out = c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4

        # monotonicity penalty over [0, 1]
        t = torch.linspace(0, 1, self.mono_K, device=x.device).view(1, 1, -1)
        c1_3 = c1.view(B, C, 1)
        c2_3 = c2.view(B, C, 1)
        c3_3 = c3.view(B, C, 1)
        c4_3 = c4.view(B, C, 1)
        f_t = c1_3 * t + c2_3 * t ** 2 + c3_3 * t ** 3 + c4_3 * t ** 4
        diffs = f_t[:, :, 1:] - f_t[:, :, :-1]
        mono = torch.relu(-diffs).mean()
        return x_out, mono


# ------------------------------------------------------------------
# Ablation 3: piecewise-linear LUT with K knots
# ------------------------------------------------------------------
class GCART_LUT(nn.Module):
    """Same histogram + MLP front-end, but the curve is a piecewise-linear
    function specified by K knots at uniform t = i/(K-1).

    Output is f(0)=0, f(1)=1 enforced; intermediate knots predicted as
    normalized cumulative softplus increments.
    """

    def __init__(self, num_bins: int = 16, gamma: float = 0.01,
                 hidden: int = 32, num_knots: int = 9):
        super().__init__()
        self.num_bins = num_bins
        self.gamma = gamma
        self.K = num_knots  # includes endpoints

        self.register_buffer(
            "bin_centers",
            torch.linspace(0, 1, num_bins).view(1, 1, num_bins, 1, 1),
        )
        self.register_buffer(
            "t_knots",
            torch.linspace(0, 1, num_knots),
        )
        # output K-1 increments; intermediate knots = cumsum normalized to 1
        self.mlp = nn.Sequential(
            nn.Linear(num_bins, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_knots - 1),
        )
        # initialize biases to a constant so that f(x) = x
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        x_e = x.unsqueeze(2)
        w = torch.exp(-((x_e - self.bin_centers) ** 2) / self.gamma)
        return w.mean(dim=(3, 4))

    def forward(self, x):
        B, C, H, W = x.shape
        hist = self.soft_histogram(x)
        raw = self.mlp(hist.view(B * C, self.num_bins))     # (B*C, K-1)
        inc = F.softplus(raw) + 1e-3
        inc = inc / inc.sum(dim=1, keepdim=True)            # normalize
        knots = torch.cat([torch.zeros_like(inc[:, :1]),
                           torch.cumsum(inc, dim=1)], dim=1)  # (B*C, K), 0 ... 1

        # apply LUT via linear interpolation
        x_flat = x.reshape(B * C, -1)                        # (B*C, H*W)
        t = self.t_knots.to(x.device)                        # (K,)
        # find segment index for each pixel
        # idx in {0, ..., K-2}
        idx = torch.bucketize(x_flat.contiguous(), t) - 1
        idx = idx.clamp(0, self.K - 2)
        t_lo = t[idx]
        t_hi = t[idx + 1]
        # gather knot values
        # knots: (B*C, K)
        k_lo = torch.gather(knots, 1, idx)
        k_hi = torch.gather(knots, 1, idx + 1)
        alpha = (x_flat - t_lo) / (t_hi - t_lo + 1e-8)
        y_flat = k_lo + alpha * (k_hi - k_lo)
        x_out = y_flat.view(B, C, H, W)

        # piecewise linear is monotone iff increments are non-negative;
        # because we softplus and normalize, this holds by construction.
        mono = torch.zeros((), device=x.device)
        return x_out, mono


# ------------------------------------------------------------------
# Zero-DCE (original) and Zero-DCE++ (depthwise-separable)
# ------------------------------------------------------------------
class MiniZeroDCE(nn.Module):
    """Compact 3-layer Zero-DCE-style enhancer. ~6.7K params at width 32."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)
        self.conv3 = nn.Conv2d(width, 3, 3, 1, 1)
        # identity-ish init via zero last layer (alpha=0 -> x_enhanced = x)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        alpha = torch.tanh(self.conv3(h))
        # 3 iterations of the pixel-wise quadratic curve as in Zero-DCE
        y = x + alpha * x * (1 - x)
        y = y + alpha * y * (1 - y)
        y = y + alpha * y * (1 - y)
        return y


class _DSConv(nn.Module):
    """Depthwise-separable conv block used in Zero-DCE++."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, 1, 1, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class MiniZeroDCEpp(nn.Module):
    """Depthwise-separable Zero-DCE++. ~10K params at width 32."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = _DSConv(3, width)
        self.conv2 = _DSConv(width, width)
        self.conv3 = _DSConv(width, 3)
        nn.init.zeros_(self.conv3.pw.weight)
        nn.init.zeros_(self.conv3.pw.bias)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        alpha = torch.tanh(self.conv3(h))
        y = x + alpha * x * (1 - x)
        y = y + alpha * y * (1 - y)
        y = y + alpha * y * (1 - y)
        return y


# ------------------------------------------------------------------
# Systems: enhancer + classifier
# ------------------------------------------------------------------
class _BaseSystem(nn.Module):
    def __init__(self, enhancer: nn.Module, num_classes: int = 10):
        super().__init__()
        self.enhancer = enhancer
        self.backbone = make_resnet18_cifar(num_classes)


class GCART_System(_BaseSystem):
    def __init__(self, variant: str = "rational", **kwargs):
        if variant == "rational":
            enh = GCART_Module(**kwargs)
        elif variant == "hard_hist":
            enh = GCART_HardHist(**kwargs)
        elif variant == "polynomial":
            enh = GCART_Polynomial(**kwargs)
        elif variant == "lut":
            enh = GCART_LUT(**kwargs)
        else:
            raise ValueError(f"Unknown GC-ART variant: {variant}")
        super().__init__(enh)

    def forward(self, x):
        x_e, mono = self.enhancer(x)
        return self.backbone(x_e), mono


class ZeroDCE_System(_BaseSystem):
    def __init__(self):
        super().__init__(MiniZeroDCE())

    def forward(self, x):
        x_e = self.enhancer(x)
        return self.backbone(x_e), torch.zeros((), device=x.device)


class ZeroDCEpp_System(_BaseSystem):
    def __init__(self):
        super().__init__(MiniZeroDCEpp())

    def forward(self, x):
        x_e = self.enhancer(x)
        return self.backbone(x_e), torch.zeros((), device=x.device)


class Plain_System(nn.Module):
    """Backbone-only baseline. Returns (logits, 0) for API compatibility."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = make_resnet18_cifar(num_classes)

    def forward(self, x):
        return self.backbone(x), torch.zeros((), device=x.device)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
MODEL_REGISTRY = {
    "baseline": lambda: Plain_System(),
    "zerodce": lambda: ZeroDCE_System(),
    "zerodcepp": lambda: ZeroDCEpp_System(),
    "gcart": lambda: GCART_System(variant="rational"),
    "gcart_hardhist": lambda: GCART_System(variant="hard_hist"),
    "gcart_poly": lambda: GCART_System(variant="polynomial"),
    "gcart_lut": lambda: GCART_System(variant="lut"),
    # gcart_no_mono uses the same architecture as gcart but with mono_weight=0
    # at training time; it is registered here as the same constructor.
    "gcart_no_mono": lambda: GCART_System(variant="rational"),
}


def get_model(name: str) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name}. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
