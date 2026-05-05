"""
Run all configurations sequentially. Useful for local testing or single-GPU
nodes where SLURM array dispatch isn't available.

Examples:
    # Full Tier-1 sweep with 3 seeds:
    python run_all.py --seeds 42 43 44 --epochs 100

    # Quick smoke test:
    python run_all.py --seeds 42 --epochs 2 --models baseline gcart
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import os


# Default Tier-1 model set
DEFAULT_MODELS = [
    # Baselines
    "baseline",
    "classical_he",
    "classical_clahe",
    "classical_gamma_1.5",
    "classical_gamma_2.2",
    "classical_gamma_3.0",
    # Learned competitors
    "zerodce",
    "zerodcepp",
    # Our method + ablations
    "gcart",
    "gcart_no_mono",
    "gcart_hardhist",
    "gcart_poly",
    "gcart_lut",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    cmds = []
    for m in args.models:
        for s in args.seeds:
            cmd = [
                sys.executable, "run_one.py",
                "--model", m,
                "--seed", str(s),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--out-dir", args.out_dir,
            ]
            if args.no_compile:
                cmd.append("--no-compile")
            cmds.append(cmd)

    print(f"Running {len(cmds)} configurations...")
    fails = 0
    for i, cmd in enumerate(cmds, 1):
        # Skip configs whose result file already exists, so we can resume.
        m = cmd[cmd.index("--model") + 1]
        s = cmd[cmd.index("--seed") + 1]
        result_path = os.path.join(args.out_dir, f"{m}_s{s}.json")
        if os.path.exists(result_path):
            print(f"\n[{i}/{len(cmds)}] SKIP (exists): {result_path}")
            continue

        print(f"\n[{i}/{len(cmds)}] {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            fails += 1
            print(f"  -> FAILED (exit {rc})")
            if not args.continue_on_error:
                sys.exit(rc)

    if fails:
        print(f"\n{fails}/{len(cmds)} runs failed.")
    else:
        print(f"\nAll {len(cmds)} runs completed.")


if __name__ == "__main__":
    main()
