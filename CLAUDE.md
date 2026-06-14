# CLAUDE.md — aquid

> Simulation of an atomtronic quantum interference device in a 2D trapped toroidal geometry with two weak-link lasers, demonstrating Josephson effects and quantum interference of critical currents.

This file is loaded by Claude Code at the start of every session. It encodes all
project conventions, context, and instructions. Do not override these without
explicit instruction from the human.

---

## Project type

Physics simulation

---

## Physics context

The system is governed by the finite-temperature stochastic Gross–Pitaevskii
equation (SGPE) in 2D (x–y plane, z integrated out via tight harmonic
confinement):

\[
i\hbar \frac{\partial\psi}{\partial t} = (1-i\gamma)\left[-\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r}) + g|\psi|^2 - \mu\right]\psi + \eta(\mathbf{r},t)
\]

where \(\psi\) is the condensate wavefunction, \(\gamma\) the dimensionless
damping, \(V(\mathbf{r})\) the trapping potential, \(g = \sqrt{8\pi}\,\hbar^2 a_s / (m l_z)\)
the 2D interaction strength, \(\mu\) the chemical potential, and
\(\eta(\mathbf{r},t)\) a complex Gaussian stochastic noise term with magnitude
\(\sqrt{2\gamma k_B T\,dt/(dx\,dy)}\).

**Trap geometry:** toroidal with two laser weak links forming an atomtronic
SQUID analogue. Trap types supported: `Harmonic`, `Toroidal`.

**Unit system:** SI physical constants → dimensionless simulation units via
harmonic oscillator scalings (length \(l_x = \sqrt{\hbar/m\omega_x}\), time
\(1/\omega_x\), energy \(\hbar\omega_x\)).

**Numerical methods:**
- Explicit RK4 time integration with stochastic noise
- Kinetic term via 2D FFT (forward on both axes, multiply by \(k^2/2\), inverse)
  with normalisation factor \(1/(n_x n_y)\)
- Complex noise: Wiener (standard normal) × exp(i·2π·uniform[0,1])
- Convergence: moving window of 5 consecutive steps with ≤10⁻⁴ relative change
  in ⟨|ψ|²⟩ triggers early termination
- CFL condition: \(dt < 0.5 \cdot \min(dx, dy)\)

**Invariants to check on every numerical implementation:**
- Wavefunction normalisation: atom number \(N = \int |\psi|^2 \,dx\,dy\) must
  remain physically reasonable (no blow-up or collapse to zero)
- k = 0 Fourier mode must be exactly zero — FFT normalisation must produce this
- Potential must be real, non-negative everywhere on the grid
- Noise magnitude must scale correctly with timestep and grid spacing; verify
  against the fluctuation–dissipation relation
- RK4 coefficients must sum correctly (1/6, 2/6, 2/6, 1/6)
- Raise an explicit comment if an implementation may violate these.

---

## Language

Rust

---

## Repository structure

| Path | Contents |
|---|---|
| `src/` | Core simulation library and binary — `lib.rs` + `main.rs` |
| `data/` | Data files tracked via Git LFS; never gitignored |
| `plots/` | Generated figures; gitignored; always PDF at 300 DPI |
| `notes/` | Private working notes; gitignored |
| `tests/` | Rust integration tests (`tests/*.rs`) |
| `configs/` | YAML/TOML configuration files |
| `.claude/` | Claude Code configuration: agents, hooks, settings |

Unit tests are inline (`#[cfg(test)] mod tests`). Integration tests go in
`tests/*.rs`.

---

## Toolchain

| Tool | Role | Command |
|---|---|---|
| `cargo` | Build, test, dependency management | `cargo build --release`, `cargo test` |
| `rustfmt` | Formatting | `cargo fmt` |
| `clippy` | Linting | `cargo clippy -- -D warnings` |

All checks must pass before any commit. The pre-commit gate enforces this as a
hard block.

---

## Code conventions

**Rust**
- All public items must have `///` doc comments; use LaTeX notation for
  mathematics.
- Prefer explicit error types; use `anyhow` for binaries, `thiserror` for
  libraries.
- Named `const` for all magic numbers; include units in the name.
- No `unwrap()` in production code; propagate errors explicitly.
- Simulation/analysis code never produces figures directly. Save results to
  `data/<name>.csv`.
- Notes and working documents go in `notes/`; never commit them.
- No ASCII approximations of mathematical notation in comments or docstrings.
  Use LaTeX throughout: \(\alpha\), \(x^2\) not `x**2`,
  \(\partial x / \partial t\), \(x \in \mathbb{R}\).

---

## Writing conventions

- British English throughout (colour, behaviour, optimise, etc.).
- Terse declarative prose; semicolons over conjunctions where appropriate.
- No filler phrases ("it is important to note", "in order to", "various").
- No ASCII approximations of mathematical notation in code comments or
  docstrings. Use LaTeX throughout: \(\alpha\), \(x^2\) not `x**2`,
  \(\partial x / \partial t\), \(x \in \mathbb{R}\).

---

## Active skills

The following skills govern this project. Claude Code loads them automatically.

| Skill | Trigger |
|---|---|
| `research-project-init` | New repo or fork setup |
| `github-projects` | Session start/end, task planning, board updates |
| `code-review` | PR creation, post-merge cleanup |
| `research-figures` | Any matplotlib or plotting code |
| `experiment-log` | W&B run summary |
| `agent-team` | Any non-trivial implementation task |
| `hooks` | Pre-commit gate configuration |

---

## Agent team

This project uses a two-agent team: `implementer` and `physics-reviewer`.
The `test-writer` agent is always active regardless of project type.
See `.claude/agents/` for full agent definitions.

Active agents:
- `implementer` — writes and edits code, runs bash
- `physics-reviewer` — read-only numerical/physics review, web search enabled
- `test-writer` — writes and runs cargo tests

---

## Session protocol

### Session start
1. Run `github-projects` Workflow A: fetch board, propose tasks, wait for
   approval.
2. Confirm all checks pass:
   `cargo fmt --check && cargo clippy -- -D warnings && cargo test`
3. Move the first issue to `In Progress` on the board.

### During session
- Commit regularly; each commit should be small and purposeful.
- Branch per issue: `git checkout -b issue-<number>-<short-description>`
- Never commit directly to `main`.
- Invoke `agent-team` for any implementation task involving more than one file
  or any physics/numerical logic.

### Session end
1. Open PR via `code-review` skill; AI review posted to GitHub.
2. Move issue to `In Review`; block until human confirms merge.
3. On merge: move issue to `Done`; run `github-projects` Workflow C.

---

## GitHub

- Organisation: `QuantumSensing`
- Repo: `QuantumSensing/aquid`
- Project board: `https://github.com/orgs/QuantumSensing/projects/2`
- PR target: `main` on this repo (not upstream, if forked)
- Max PR size: 200 lines changed

---

## Environment

Secrets live in `.env` (gitignored). Required keys:

```
GITHUB_TOKEN=        # repo + project scopes
```
