"""
Smoke test: instantiate every model, run a forward pass, and verify
shapes / aux-loss types. Run this once before launching real experiments.

    python smoke_test.py
"""

from __future__ import annotations
import torch

from models import (
    GCART_Module, GCART_HardHist, GCART_Polynomial, GCART_LUT,
    MiniZeroDCE, MiniZeroDCEpp, get_model, MODEL_REGISTRY, count_parameters,
)
from classical import ClassicalSystem


def _check_system(name, m, x):
    out = m(x)
    assert isinstance(out, tuple) and len(out) == 2, \
        f"{name} did not return (logits, aux_loss)"
    logits, aux = out
    assert logits.shape == (x.shape[0], 10), \
        f"{name} bad logits shape: {logits.shape}"
    assert aux.dim() == 0, f"{name} aux_loss must be scalar, got {aux.shape}"
    print(f"  OK {name:20s}  params={count_parameters(m):>10,}  "
          f"aux={aux.item():.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    x = torch.rand(4, 3, 32, 32, device=device)

    print("\n[1] Bare enhancer modules")
    for name, M in [
        ("GCART_Module",     GCART_Module),
        ("GCART_HardHist",   GCART_HardHist),
        ("GCART_Polynomial", GCART_Polynomial),
        ("GCART_LUT",        GCART_LUT),
        ("MiniZeroDCE",      MiniZeroDCE),
        ("MiniZeroDCEpp",    MiniZeroDCEpp),
    ]:
        m = M().to(device)
        out = m(x)
        if isinstance(out, tuple):
            y, mono = out
        else:
            y, mono = out, None
        assert y.shape == x.shape, f"{name} shape mismatch: {y.shape}"
        info = f"params={count_parameters(m):>8,}"
        if mono is not None:
            info += f"  mono={mono.item():.6f}"
        print(f"  OK {name:20s} {info}")

    print("\n[2] Full Systems via factory")
    for name in MODEL_REGISTRY:
        m = get_model(name).to(device)
        _check_system(name, m, x)

    print("\n[3] Classical systems")
    for method, kwargs in [("identity", {}), ("he", {}), ("clahe", {}),
                           ("gamma", {"gamma": 2.2})]:
        m = ClassicalSystem(method=method, **kwargs).to(device)
        _check_system(f"classical_{method}", m, x)

    print("\n[4] Backprop sanity for GC-ART")
    m = get_model("gcart").to(device).train()
    out, mono = m(x)
    loss = out.sum() + 10.0 * mono
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.parameters())
    assert has_grad, "no gradients flowed in GC-ART"
    print("  OK gradients flow through GC-ART")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
