# Project History and Roadmap

This document summarizes the major development milestones of the Gene Regulatory Networks (GRN) shape homeostasis simulation project and links to the active GitHub Issues derived from the original 2017 thesis notes.

---

## Release & Milestone History

### `v0.3.0` — Architecture Refactoring, `uv`, and CI/Test Suite (2026)
- **Modern Python Packaging**: Migrated to PEP 621 [`pyproject.toml`](../pyproject.toml) and [`uv`](https://docs.astral.sh/uv/) for dependency management (`Python 3.11+`).
- **Repository Restructuring**: Organized modules into PEP 420 namespace packages (`core/`, `evolution/`, `visualization/`, `analysis/`, `config/`, `examples/`, `scripts/`, `tests/`).
- **API & Engine Modernization**: Updated NetworkX 3.x (`from_numpy_array`, `to_numpy_array`), Numba (`@njit`), and Matplotlib (`draw_idle`) calls; wired `visualization/web_app.py` and `examples/run_evolution.py` to the live cellular automata and reaction-diffusion engine.
- **Continuous Integration & Unit Testing**: Added GitHub Actions workflow (`.github/workflows/ci.yml`) and a `pytest` unit test suite (`tests/`).
- **Scientific Documentation & Topologies**: Consolidated mathematical and architectural documentation in [`docs/architecture.md`](./architecture.md) and added multi-topology graph generation (`erdos-renyi`, `watts-strogatz`, `barabasi-albert`) in `analysis/graph_generator.py`.

### `v0.2.0` — Containerization and Interactive Web Interface (2025)
- **Docker Support**: Added [`Dockerfile`](../Dockerfile) (`python:3.12-slim` + `uv`), [`docker-compose.yml`](../docker-compose.yml), and launcher scripts for reproducible execution.
- **Streamlit & Plotly Web Interface**: Created [`visualization/web_app.py`](../visualization/web_app.py) for interactive parameter exploration and real-time visualization of cell states, SGF, and LGF concentration fields.
- **Modular Codebase**: Separated the flat root scripts into `core/`, `evolution/`, `visualization/`, `analysis/`, and `examples/`.

### `v0.1.0` — Original Master's Thesis Research (2017)
- **Baseline Commit**: [`j0sees/Simulation-of-Gene-Regulatory-Networks-Driving-Shape-Homeostasis@6c3fae7`](https://github.com/j0sees/Simulation-of-Gene-Regulatory-Networks-Driving-Shape-Homeostasis/tree/6c3fae7834a2ca502b1a439d7535c4adab1673f3) (*Sept–Dec 2017*, M.Sc. J. Esteban Pérez-Hidalgo, supervised by Ph.D. Philip Gerlee).
- **Core Model Implementation**:
  - 2D discrete cellular automaton with four cellular actions (`Quiet`, `Move`, `Split`, `Die`) and polarization compass.
  - Non-toroidal boundary enforcement via `flatList` and Crank-Nicolson reaction-diffusion solver (`LGFDiffEq`) with zero-flux Neumann boundary conditions (`GenerateTMatrix`).
  - Generalization of Gerlee et al.'s (2015) two-layer feedforward network into an intracellular Recurrent Neural Network (`RecurrentNeuralNetwork`).
  - Parallel Genetic Algorithm (`evolution/main_ga.py`), `neat-python` neuroevolution integration (`evolution/neat_cell_agent.py`, `evolution/neat_parallel.py`), and `(SGF, LGF)` phase-space mapping (`MapPlot`).
  - Presented at the *9th Swedish Meeting on Mathematical Biology* (Västerås, Sweden, Dec 7–8, 2017).

---

## Active Roadmap (Extracted from Original Thesis Notes)

All remaining open tasks and future improvements from the 2017 research logs are tracked as GitHub Issues in [`userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis/issues`](https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis/issues):

- **[Issue #8](https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis/issues/8)** — `docs(architecture): re-infuse v0.1.0 (6c3fae7) thesis context and multi-topology CLI support`
- **[Issue #9](https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis/issues/9)** — `refactor(core): decouple simulation engine from Matplotlib plotting and consolidate duplicate sim() loops`
- **[Issue #10](https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis/issues/10)** — `perf(core): optimize LGFDiffEq Crank-Nicolson solver, standardize coordinates, and add flow diagram`
- **[Issue #11](https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis/issues/11)** — `feat(visualization): adopt colorblind-safe palette for cell states and add timelapse snapshot export`
- **[Issue #12](https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis/issues/12)** — `feat(evolution): improve GA offspring diversity and align NEAT config-ca with [0, 1] fitness scale`
- **[Issue #13](https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis/issues/13)** — `docs(history): convert raw 2017 scratch notes in docs/history-log.md into structured milestone history`
