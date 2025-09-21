# SGPE-rs

`sgpe-rs` implements the finite-temperature stochastic Gross–Pitaevskii equation (SGPE) in Rust. The project provides:

- A stochastic RK4 solver with harmonic/toroidal traps.
- A 2‑D Fourier-space kinetic operator using `rustfft` with proper normalisation.
- A parallel sweep launcher (`run.zsh`) written in Python
- A marimo notebook (`data/visualise.py`) to inspect atom numbers and density snapshots.

---
## Getting started

### Requirements
- Rust (stable toolchain)

### Optional
Python 3.10+ (only for visualising data, otherwise optional)

### Build
```bash
cargo build --release
```

---
## Running simulations

### Single run
```bash
./target/release/sgpe MU TEMP SAVE_TRAJECTORY NOISE_REALISATIONS
```
- `MU`, `TEMP` in nK (floats)
- `SAVE_TRAJECTORY`: `true`/`false`
- `NOISE_REALISATIONS`: integer ≥ 1

Results land in `data/<MU>_<TEMP>/` (grid, params, per-run folders, trajectory if saved).

### Batch sweep (`run.zsh`)
```bash
./run.zsh [options]
```
Key options:
- `--mode full|final`
- `--count N` (`16` default)
- `--seed S` (default `42069`)
- `--noise N` (per run)
- `--threads-per-job N`
- `--program PATH` (default `./target/release/sgpe`)
- `--log-dir PATH` (`./logs`)
- `--probed-file PATH` to reuse sample CSV

> [!NOTE]
> Simulating 3,000 distinct realisations of the SGPE on an Apple M3 Ultra takes around 4.5 minutes of wall-clock time.

---
## Visualisation
`data/visualise.py` (marimo) loads datasets, plots atom number trajectories, and renders $|\phi(x,y)|^2$.
```bash
marimo run data/visualise.py
```

---
## Numerical details
- System: $i\hbar \partial\Phi/\partial t = (1 - i\gamma)(H - μ)\Phi + \eta$
- Integration: explicit RK4 with stochastic noise
- Kinetic term: 2‑D FFT (forward on both axes, multiply by k^2/2, inverse) normalised by `nx * ny`
- Convergence: moving window (50 steps) with $\leq 10^{-3}$ relative band; early failure if ⟨N⟩ after 4 steps < 100

---
## Maintenance scripts
- `clean.zsh`: removes generated datasets (`data/<mu>_<temp>/`), sampled CSVs, and logs—keeps source files/notebook

---
## Troubleshooting
- **Only one dataset appears:** use the Python-based `run.zsh`; watch logs for failures

---
## Contributing & License
PRs welcome (new trap types, analysis tools, optimisations). Licensed under MIT.
