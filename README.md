# GC-ART

GC-ART is a lightweight, global-curve image enhancement module evaluated on CIFAR-10-C-style corruptions. This repository contains the code for training GC-ART, learned enhancement baselines, fixed-preprocessing baselines, ablations, aggregation scripts, and FLOPs/latency benchmarking.

## Files

```text
models.py          # GC-ART variants, Zero-DCE/Zero-DCE++, and ResNet-18 wrappers
classical.py       # Fixed HE/CLAHE/gamma preprocessing + trained ResNet-18 baselines
data.py            # CIFAR-10 loader and on-the-fly corruptions
training.py        # Train/eval loop with seed support and JSON logging
run_one.py         # CLI entry point for one model/seed run
run_all.py         # Sequential dispatcher for multi-model/multi-seed sweeps
flops_benchmark.py # FLOPs and latency benchmark for enhancement modules
aggregate.py       # Aggregates JSON results into tables and plots
smoke_test.py      # Quick sanity check
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Then run the smoke test:

```bash
python smoke_test.py
```

## Quickstart

Run one short training job:

```bash
python run_one.py --model gcart --seed 42 --epochs 2 --batch-size 128 --num-workers 0 --no-compile
```

Run a small end-to-end reproducibility check:

```bash
mkdir -p results
python smoke_test.py
python run_one.py --model baseline --seed 0 --epochs 1 --batch-size 128 --num-workers 0 --no-compile
python run_one.py --model gcart --seed 0 --epochs 1 --batch-size 128 --num-workers 0 --no-compile
python aggregate.py --results-dir results
python flops_benchmark.py --resolutions 32 --iters 2 --warmup 1 --device cpu
```

Run the full sequential sweep:

```bash
python run_all.py --seeds 42 43 44 --epochs 100
python aggregate.py --results-dir results
```

## Outputs

Each `run_one.py` invocation writes a JSON file under `results/`, containing the training history, final clean accuracy, corruption accuracies, parameter count, and elapsed time.

After running `aggregate.py`, the `results/` directory should contain summary tables and plots, including:

```text
results/table_main.md
results/table_severities.md
results/learning_curves.png
results/corruption_curves.png
```

## Baselines

This package includes:

- GC-ART and architectural ablations
- Zero-DCE and Zero-DCE++ learned enhancement baselines
- Fixed preprocessing baselines: histogram equalization, CLAHE, and gamma correction

For the fixed preprocessing baselines, the preprocessing operation is fixed/parameter-free, but the downstream ResNet-18 classifier is still trained. They should therefore be described as “fixed preprocessing + trained ResNet-18” baselines rather than fully parameter-free systems.

## FLOPs and latency benchmark

Run:

```bash
python flops_benchmark.py --resolutions 32 256 1024 2048 --device cpu
```

or, on a CUDA machine:

```bash
python flops_benchmark.py --resolutions 32 256 1024 2048 --device cuda
```

The benchmark reports enhancer-only costs, excluding the ResNet-18 classifier that is shared across configurations.

## Hardware notes

If `torch.compile` causes issues on your PyTorch version or GPU, pass:

```bash
--no-compile
```

to `run_one.py`.

## Caveats

- CIFAR-10 is only 32x32. The resolution-scaling argument is most visible in the FLOPs/latency benchmark.
- CIFAR-10-C brightness corruption increases brightness; contrast and the custom darken corruption better test low-light behavior.
- FLOPs in `flops_benchmark.py` are approximate analytical/hook-based estimates intended for comparison between enhancement modules, not a replacement for profiler-level hardware analysis.

## License

This code is released under the MIT License. See `LICENSE`.
