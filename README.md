# aquid

Simulation of an atomtronic quantum interference device in a 2D trapped toroidal
geometry with two weak-link lasers, demonstrating Josephson effects and quantum
interference of critical currents.

## Installation

Requires a [Rust](https://www.rust-lang.org/) stable toolchain.

```bash
git clone https://github.com/QuantumSensing/aquid
cd aquid
cargo build --release
```

## Development

```bash
cargo test              # run tests
cargo fmt --check       # format check
cargo clippy -- -D warnings  # lint
```

## Structure

| Directory | Contents |
|---|---|
| `src/` | Core simulation code |
| `tests/` | Integration tests |
| `data/` | Data files (tracked via Git LFS) |
| `plots/` | Generated figures (gitignored) |
| `notes/` | Private notes (gitignored) |
| `configs/` | Configuration files |

## Numerical details

- System: $i\hbar \partial\Phi/\partial t = (1 - i\gamma)(H - \mu)\Phi + \eta$
- Integration: explicit RK4 with stochastic noise
- Kinetic term: 2-D FFT (forward on both axes, multiply by $k^2/2$, inverse) normalised by $n_x \cdot n_y$
- Geometry: toroidal trap with two weak-link laser barriers

## Maintenance scripts

- `run.zsh` — batch sweep launcher
- `clean.zsh` — removes generated datasets, sampled CSVs, and logs

## License

MIT
