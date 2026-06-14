#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SGPE simulations in parallel.")
    parser.add_argument("--mode", choices=["full", "final"], default="final")
    parser.add_argument("--program", default="./target/release/aquid")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--noise", "--noise-realisations", type=int, dest="noise_realisations", default=1)
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--threads-per-job", type=int, default=1)
    parser.add_argument("--probed-file", default="")
    return parser.parse_args()


def generate_samples(path: Path, count: int, seed: int) -> Iterable[Tuple[float, float]]:
    rng = random.Random(seed)
    samples = [(round(rng.uniform(20.0, 80.0), 2), round(rng.uniform(1.0, 200.0), 2)) for _ in range(count)]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mu_nK", "temp_nK"])
        for mu, temp in samples:
            writer.writerow([f"{mu:.2f}", f"{temp:.2f}"])
    return samples


def format_label(mu: float, temp: float) -> str:
    return f"{mu:.2f}_{temp:.2f}"


def launch_simulation(
    idx: int,
    mu: float,
    temp: float,
    save_full: bool,
    args: argparse.Namespace,
    log_dir: Path,
) -> Tuple[int, int]:
    label = format_label(mu, temp)
    log_path = log_dir / f"sim_{idx:04d}_{label}.log"

    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(args.threads_per_job)

    cmd = [
        args.program,
        f"{mu:.2f}",
        f"{temp:.2f}",
        "true" if save_full else "false",
        str(args.noise_realisations),
    ]

    with log_path.open("w") as log_file:
        process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    return idx, process.returncode


def main() -> int:
    args = parse_args()

    if not Path(args.program).is_file():
        print(f"Simulation binary not found: {args.program}", file=sys.stderr)
        return 1
    if not os.access(args.program, os.X_OK):
        print(f"Simulation binary is not executable: {args.program}", file=sys.stderr)
        return 1
    if args.count < 1:
        print("--count must be >= 1", file=sys.stderr)
        return 1
    if args.threads_per_job < 1:
        print("--threads-per-job must be >= 1", file=sys.stderr)
        return 1
    if args.noise_realisations < 1:
        print("--noise must be >= 1", file=sys.stderr)
        return 1

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if args.probed_file:
        probed_path = Path(args.probed_file)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        probed_path = data_dir / f"probed_mu_T_{timestamp}_seed{args.seed}.csv"

    samples = list(generate_samples(probed_path, args.count, args.seed))
    total = len(samples)
    if total == 0:
        print("No samples generated", file=sys.stderr)
        return 1

    cpu_cores = os.cpu_count() or 1
    max_parallel = max(1, cpu_cores // args.threads_per_job)

    if args.mode == "final":
        full_indices = {1, total}
        if total > 2:
            full_indices.add((total + 1) // 2)
    else:
        full_indices = set(range(1, total + 1))

    print("Configuration:")
    print(f"  - Mode: {args.mode}")
    print(f"  - Sample count: {total} (seed: {args.seed})")
    print(f"  - Log directory: {log_dir}")
    print(f"  - Probed values file: {probed_path}")
    print(f"  - Noise realisations: {args.noise_realisations}")
    print(f"  - CPU cores: {cpu_cores}")
    print(f"  - Threads per job: {args.threads_per_job}")
    print(f"  - Parallel jobs: {max_parallel}")
    if args.mode == "final":
        print(f"  - Full trajectories retained for entries: {sorted(full_indices)}")

    completed = 0
    failed = []
    lock = threading.Lock()
    start_time = time.monotonic()
    frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    frame_index = 0

    def make_progress_bar(fraction: float, width: int = 24) -> str:
        filled = min(width, int(round(fraction * width)))
        bar = "=" * filled
        bar += ">" if filled < width else ""
        bar += " " * max(0, width - len(bar))
        return f"|{bar}|"

    def update_progress():
        nonlocal frame_index
        fraction = completed / total if total else 0.0
        percent = math.floor(fraction * 100)
        elapsed = time.monotonic() - start_time
        bar = make_progress_bar(fraction)
        spinner = frames[frame_index % len(frames)]
        frame_index += 1
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(
            f"\r{spinner} {bar} {percent:3d}% ({completed}/{total}) elapsed {elapsed_str}",
            end="",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_idx = {}
        for idx, (mu, temp) in enumerate(samples, start=1):
            save_full = idx in full_indices
            future = executor.submit(launch_simulation, idx, mu, temp, save_full, args, log_dir)
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                _, returncode = future.result()
            except Exception as exc:  # pragma: no cover
                with lock:
                    failed.append(idx)
                    completed += 1
                    print(f"\n[run.zsh] simulation idx={idx} raised exception: {exc}", file=sys.stderr)
                    update_progress()
                continue

            with lock:
                if returncode != 0:
                    failed.append(idx)
                    print(
                        f"\n[run.zsh] simulation idx={idx} exited with status {returncode}",
                        file=sys.stderr,
                    )
                completed += 1
                update_progress()

    print()
    if not failed:
        print("All simulations completed successfully.")
    else:
        failed_str = ", ".join(str(i) for i in sorted(failed))
        print(f"Simulations completed with failures for indices: {failed_str}", file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
