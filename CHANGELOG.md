# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-11-04

### Added
- **Release Baseline & Community Standards**: Added `VERSION` (`0.3.0`), `CHANGELOG.md`, `CONTRIBUTING.md` (Trunk-Based Development, Conventional Commits, issue guidelines), and `CODE_OF_CONDUCT.md` (`2026-11-04`).
- **Podman & Docker Container Launcher**: Replaced the Docker-only launcher with `scripts/run_container.sh` supporting both Podman (Podman Desktop / rootless Podman) and Docker on macOS and Linux (`adc3acd`, `2026-11-04T08:27:38-06:00`).
- **Multi-Topology CLI & Thesis Context**: Re-infused `v0.1.0` mathematical and biological context (`flatList` closed boundaries, Neumann zero-flux Laplacian, 2-layer feedforward to RNN generalization, and `MapPlot` phase-space maps) into `docs/architecture.md`, and added CLI/programmatic support for `erdos-renyi`, `watts-strogatz`, and `barabasi-albert` topologies in `analysis/graph_generator.py` (`17ae423`, `2026-11-04T07:48:28-06:00`).
- **Citation & Author Metadata**: Added `CITATION.cff` and standardized author/maintainer attribution across `README.md`, `pyproject.toml`, `LICENSE`, `docs/architecture.md`, and `visualization/web_app.py` (`39bf506`, `2026-11-04T07:24:37-06:00`).
- **GitHub Actions CI & Unit Test Suite**: Added `.github/workflows/ci.yml` (`ruff`, `compileall`, `pytest`, container smoke build) and `pytest` unit tests under `tests/` (`5a59441`, `2026-11-03T23:12:03-06:00`).

### Changed
- **Roadmap Tracking**: Converted raw 2017 scratch notes in `docs/history-log.md` into a milestone history and issue tracker (`8d68da2`, `2026-11-04T08:03:03-06:00`).
- **Root Directory Organization & Namespace Packages**: Moved `neat_cell_agent.py` and `neat_parallel.py` to `evolution/`, `web_app.py` and `plots.gp` to `visualization/`, and shell launchers to `scripts/`; removed empty `__init__.py` files in favor of PEP 420 namespace packages and bumped version to `0.3.0` (`dd99a8d`, `2026-11-04T07:11:15-06:00`).
- **PEP 8 Cleanup & Ignore Rules**: Removed commented-out legacy code across all Python modules, enforced PEP 8 formatting (`<= 100` chars/line), and aligned `.gitignore` and `.dockerignore` (`a65cc53`, `2026-11-03T23:02:24-06:00`).
- **Dependency Modernization & `uv` Migration**: Migrated from `pip`/`requirements.txt` to `pyproject.toml` and `uv`, upgraded `Dockerfile` to `python:3.12-slim` + `uv`, updated NetworkX 3.x (`from_numpy_array`, `to_numpy_array`), Numba (`@njit`), and Matplotlib (`draw_idle`) calls, and connected `visualization/web_app.py` and `examples/run_evolution.py` to the live simulation engine (`ed8e9cb`, `2026-11-03T22:49:40-06:00`).
- **Documentation Polish**: Removed decorative emojis across Markdown files, shell scripts, and Streamlit UI, and improved documentation clarity (`62df17d`, `2026-11-03T22:39:14-06:00`).

---

## [0.2.0] - 2025-09-03

### Added
- **MIT License**: Added repository `LICENSE` (`0f78f98`, `2025-09-03T11:24:01-06:00`).
- **Interactive Web Interface & Containerization**: Added Streamlit and Plotly web interface (`web_app.py`), `Dockerfile`, `docker-compose.yml`, launcher scripts, and documentation figures (`5fde409`, `2025-09-03T11:07:55-06:00`; `2e6a67f`, `2025-08-31T21:18:02-06:00`; `a7591ad`, `2025-08-31T22:00:49-06:00`).

### Changed
- **Modular Directory Structure**: Reorganized flat root scripts into `core/`, `evolution/`, `visualization/`, `analysis/`, and `examples/` and fixed `tools_ga` imports (`dd0a2b0`, `2025-08-31T20:40:08-06:00`; `2084c7b`, `2025-08-31T21:35:17-06:00`).

---

## [0.1.0] - 2017-12-15 (Thesis Baseline Updated 2025-08-06)

### Added
- **Thesis Overview & Documentation Baseline**: Documented master's thesis objectives and model summary in `README.md` and `log` (`6c3fae7`, `2025-08-06T11:54:17-06:00`; `615f383`, `2017-12-15T17:31:18+01:00`).
- **Graph Topology Generation & Evaluation**: Added `graph_generator.py` and `graph_evaluation.py` for GRN topology analysis (`5aee700`, `2017-12-12T16:50:58+01:00`).
- **NEAT Neuroevolution**: Added `neat-python` cell agent (`neat_cell_agent.py`), parallel runner (`neat_parallel.py`), and `config-ca` (`77e4595`, `2017-12-12T16:50:09+01:00`).
- **Closed Lattice Boundaries (`flatList`) & Compass Fix**: Separated `cellgrid` and `chemgrid` using `flatList` to prevent negative-index toroidal wrapping and corrected compass neighbor selection (`4073a36`, `2017-11-17T16:54:08+01:00`; `d42c147`, `2017-11-14T18:14:26+01:00`).
- **Parallel Genetic Algorithm & Numba JIT**: Implemented `multiprocessing.Pool` evolution (`main_GA.py`, `tools_GA.py`), shape-homeostasis fitness function, and Numba `@jit` acceleration (`3c879b4`, `2017-11-01T16:29:56+01:00`; `1422e07`, `2017-10-08T21:05:25+02:00`; `697d2bc`, `2017-10-06T02:59:41+02:00`; `dc206d5`, `2017-10-04T17:53:05+02:00`).
- **Recurrent Neural Network & SGF/LGF Reaction-Diffusion**: Implemented intracellular `RecurrentNeuralNetwork`, explicit Euler `SGFDiffEq`, and Crank-Nicolson `LGFDiffEq` with Neumann boundary matrix `GenerateTMatrix` (`466a79a`, `2017-10-02T01:29:59+02:00`; `d349fce`, `2017-09-25T15:41:21+02:00`).
- **2D Cellular Automaton Core**: Implemented initial cell agent states (`Quiet`, `Move`, `Split`, `Die`), polarization compass, and Matplotlib environment plotting (`0657f64`, `2017-09-14T18:00:45+02:00`; `c3ec607`, `2017-09-04T17:26:01+02:00`; `1f1ebe2`, `2017-08-31T00:37:55+02:00`).
