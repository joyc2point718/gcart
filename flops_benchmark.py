"""
FLOPs and wall-clock latency benchmark for the enhancement modules across
resolutions. Measures ENHANCERS ONLY (excluding the ResNet-18 backbone, which
is identical across configurations).

For each module x resolution, reports:
    - parameter count
    - "Param-est MACs": MACs in Conv/Linear layers only. This isolates the
      cost of LEARNED parameter prediction. For GC-ART this is independent
      of resolution (just the MLP); for spatial CNNs it scales as O(HW).
    - "Pixel-op FLOPs": analytical estimate of the per-pixel arithmetic
      (histogram + curve application or curve iterations). This component
      scales as O(HW) for every method.
    - "Total FLOPs": sum of the two, comparable across methods.
    - GPU wall-clock latency (mean +/- std over N iters after M warmup).

Saves results/flops_benchmark.json and prints a markdown-friendly table.
"""

from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn

from models import (
    GCART_Module, GCART_HardHist, GCART_Polynomial, GCART_LUT,
    MiniZeroDCE, MiniZeroDCEpp,
)


# ------------------------------------------------------------------
# Hook-based MAC counter
# ------------------------------------------------------------------
class _MacCounter:
    def __init__(self):
        self.total = 0
        self.handles = []

    def attach(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._conv_hook))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._linear_hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _conv_hook(self, mod, inp, out):
        # MACs = output_elements * (kernel_elements * in_channels / groups)
        out_t = out
        if isinstance(out_t, tuple):
            out_t = out_t[0]
        k = mod.kernel_size[0] * mod.kernel_size[1]
        per_out = k * (mod.in_channels // mod.groups)
        n_out = out_t.numel()
        self.total += n_out * per_out

    def _linear_hook(self, mod, inp, out):
        # MACs = batch_dim * in * out
        out_t = out
        if isinstance(out_t, tuple):
            out_t = out_t[0]
        n = out_t.numel() // mod.out_features
        self.total += n * mod.in_features * mod.out_features


def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    counter = _MacCounter()
    counter.attach(model)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    counter.detach()
    return counter.total


# ------------------------------------------------------------------
# Analytical pixel-op FLOPs (operations not in conv/linear layers)
# ------------------------------------------------------------------
def pixel_op_flops(name: str, H: int, W: int, C: int = 3,
                   K: int = 16) -> int:
    """Approximate FLOPs for the per-pixel arithmetic that hooks miss.

    GC-ART rational/hardhist:
        soft histogram:    ~5 * C * H * W * K   (sub, square, exp, accumulate)
        rational curve:    ~7 * C * H * W       (a*x*x + b*x in num, d*x*x + e*x + 1 in den, divide)

    GC-ART polynomial:
        soft histogram:    ~5 * C * H * W * K
        polynomial curve:  ~10 * C * H * W      (4 powers, 4 mults, 3 adds)

    GC-ART LUT:
        soft histogram:    ~5 * C * H * W * K
        LUT lookup:        ~6 * C * H * W       (bucketize, gather, lerp)

    Zero-DCE / Zero-DCE++:
        3 iterations of pixel-wise curve, each ~5 * C * H * W FLOPs.
    """
    HW = H * W
    if name in ("gcart_rational", "gcart_hardhist"):
        return 5 * C * HW * K + 7 * C * HW
    if name == "gcart_polynomial":
        return 5 * C * HW * K + 10 * C * HW
    if name == "gcart_lut":
        return 5 * C * HW * K + 6 * C * HW
    if name in ("zerodce", "zerodcepp"):
        return 3 * 5 * C * HW
    if name == "he":
        # quantize, histogram, cumsum, gather: ~6 ops/pixel + scan
        return 6 * C * HW + C * 256
    if name.startswith("gamma"):
        return 2 * C * HW   # clamp + pow
    return 0


# ------------------------------------------------------------------
# Latency
# ------------------------------------------------------------------
def measure_latency(model, x, warmup=20, iters=100):
    """Returns mean ms, std ms over `iters` runs after `warmup`."""
    model.eval()
    if x.is_cuda:
        torch.cuda.synchronize()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()

        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            ts.append((t1 - t0) * 1000.0)

    import statistics
    return statistics.fmean(ts), statistics.pstdev(ts)


# ------------------------------------------------------------------
# Benchmark harness
# ------------------------------------------------------------------
def make_modules() -> Dict[str, nn.Module]:
    from classical import HEModule, GammaModule
    return {
        "gcart_rational":     GCART_Module(),
        "gcart_hardhist":     GCART_HardHist(),
        "gcart_polynomial":   GCART_Polynomial(),
        "gcart_lut":          GCART_LUT(),
        "zerodce":            MiniZeroDCE(),
        "zerodcepp":          MiniZeroDCEpp(),
        "he":                 HEModule(),
        "gamma_2.2":          GammaModule(gamma=2.2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resolutions", type=int, nargs="+",
                   default=[32, 128, 512, 1024, 2048, 4096])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", type=str, default="results/flops_benchmark.json")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    modules = make_modules()
    for m in modules.values():
        m.to(device)

    # Param counts
    params = {n: sum(p.numel() for p in m.parameters()) for n, m in modules.items()}

    results: List[dict] = []
    print("\n#  module           | resolution | params | param-est MACs | pixel-op FLOPs | total FLOPs   | latency (ms)")
    print("-- --------------    | ---------- | ------ | -------------- | -------------- | ------------- | ----------------")
    for res in args.resolutions:
        # Skip resolutions that would OOM on common GPUs
        try:
            x = torch.rand(1, 3, res, res, device=device)
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM at {res}, skipping]")
            continue

        for name, mod in modules.items():
            try:
                # Param-estimation MACs (hooks: conv + linear only)
                pe_macs = count_macs(mod, x)
                # Pixel-op FLOPs (analytical)
                px_flops = pixel_op_flops(name, res, res)
                total_flops = pe_macs + px_flops
                # Latency
                mean_ms, std_ms = measure_latency(mod, x, args.warmup, args.iters)
                row = {
                    "module": name,
                    "resolution": res,
                    "n_params": params[name],
                    "param_est_macs": pe_macs,
                    "pixel_op_flops": px_flops,
                    "total_flops": total_flops,
                    "latency_mean_ms": mean_ms,
                    "latency_std_ms": std_ms,
                }
                results.append(row)
                print(f"  {name:18s} | {res:>10d} | {params[name]:>6d} | "
                      f"{pe_macs:>14,} | {px_flops:>14,} | {total_flops:>13,} | "
                      f"{mean_ms:>6.3f} +/- {std_ms:.3f}")
            except torch.cuda.OutOfMemoryError:
                print(f"  {name:18s} | {res:>10d} | OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {name:18s} | {res:>10d} | ERROR: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
