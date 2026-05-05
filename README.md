# GC-ART — Tier-1 experiment package

This directory contains the minimum-viable experimental package for the
GC-ART paper: standardized CIFAR-10-C evaluation, classical and learned
baselines, three architectural ablations, multi-seed support, and a
FLOPs/latency benchmark.

## Files

```
models.py          # GC-ART (rational/hardhist/poly/lut) + Zero-DCE/++ + ResNet-18
classical.py       # HE, CLAHE, gamma preprocessing baselines
data.py            # HF CIFAR-10 loader + on-the-fly CIFAR-10-C corruptions
training.py        # train/eval loop with seed support and JSON logging
run_one.py         # CLI entry: one (model, seed). For SLURM array jobs.
run_all.py         # Sequential dispatcher (alternative to SLURM)
flops_benchmark.py # FLOPs + wall-clock latency across resolutions
aggregate.py       # Multi-seed aggregation -> markdown tables + plots
smoke_test.py      # 30-second sanity check; run this first
submit_slurm.sh    # SLURM array template (39 jobs = 13 models * 3 seeds)
```

## Quickstart

```bash
# 1. Sanity check (CPU is fine here)
python smoke_test.py

# 2. Single config, mostly to verify training works
python run_one.py --model gcart --seed 42 --epochs 2

# 3. FLOPs/latency benchmark (works without datasets)
python flops_benchmark.py --resolutions 32 256 1024 2048 4096

# 4. Full sweep, sequentially:
python run_all.py --seeds 42 43 44 --epochs 100

# 5. After all results files are present:
python aggregate.py --results-dir results
```

## SLURM dispatch

```bash
mkdir -p logs results
sbatch submit_slurm.sh                # runs all 13 models x 3 seeds
sbatch --array=0-2 submit_slurm.sh    # just GC-ART (3 seeds) for quick test
```

The SLURM script's `--array=0-38` maps task IDs to (model, seed) pairs.
After all jobs finish, run `python aggregate.py` to produce the paper
tables and plots from the per-job JSON files.

## Outputs

Each `run_one.py` invocation produces `results/<model>_s<seed>.json`
containing:
- the full training history (loss, clean accuracy per epoch)
- the final clean accuracy
- the final accuracy on every (corruption, severity) pair
- parameter count and elapsed time

After running `aggregate.py`:

```
results/table_main.md          # one-row-per-model summary
results/table_severities.md    # detailed per-severity numbers
results/learning_curves.png    # mean +/- std training curves
results/corruption_curves.png  # accuracy vs severity, three subplots
```

## Hardware notes

The script auto-detects bf16 vs fp16:
- Ampere/Hopper (A100, H100, RTX 30xx/40xx) -> bfloat16, no GradScaler
- Turing/Volta (TITAN RTX, V100, RTX 20xx) -> float16 + GradScaler

If `torch.compile` causes issues (older PyTorch, exotic backbones), pass
`--no-compile` to `run_one.py`.

## Expected runtime per config

ResNet-18 / CIFAR-10 / 100 epochs / batch 1024:
- A100:        ~12-18 minutes
- TITAN RTX:   ~25-35 minutes
- 3090 / 4090: ~10-15 minutes

Tier 1 is 13 models * 3 seeds = 39 runs. On an A100 with 4 parallel jobs
that's roughly 2-3 hours of wall-clock time end to end.

## What this package covers (Tier 1)

- [x] CIFAR-10-C-style brightness, contrast, darken corruptions
      (computed on-the-fly; no 2.5GB download required)
- [x] Multiple seeds with mean/std reporting
- [x] HE / CLAHE / gamma classical baselines
- [x] Zero-DCE and Zero-DCE++ learned baselines
- [x] Soft-vs-hard histogram ablation
- [x] Rational vs polynomial vs LUT curve-family ablation
- [x] Monotonicity-penalty on/off ablation
- [x] FLOPs and latency benchmark across 32 -> 4096 px

## Honest caveats to put in the paper

1. The "param-estimation MACs" stat is ~10^3, but the *total* per-image
   compute is dominated by the pixel-wise histogram (O(HWK)) and curve
   application (O(HW)). At 1024^2 GC-ART totals ~274M FLOPs vs Zero-DCE++'s
   ~1.95G — a meaningful ~7x advantage, but not the 10^4 reduction the
   v1 manuscript claimed.

2. Brightness corruption in CIFAR-10-C *increases* brightness; only
   contrast and the custom darken corruption test the low-light setting.
   Report all three.

3. CIFAR-10 is 32x32. The O(K)-vs-O(HWK) story is most visible at higher
   resolution; the FLOPs benchmark exists specifically to make that
   visible without needing a higher-res classification dataset.
