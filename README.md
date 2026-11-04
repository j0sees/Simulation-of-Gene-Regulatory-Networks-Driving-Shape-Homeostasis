# Simulation of Gene Regulatory Networks Driving Shape Homeostasis

Computational platform for simulating and classifying **Gene Regulatory Networks (GRNs)** that drive **shape homeostasis** in multicellular systems.

- **Original Research & Mathematical Model (`v0.1`)**: **M.Sc. J. Esteban Pérez-Hidalgo** ([@j0sees](https://github.com/j0sees), School of Physics, Costa Rica Institute of Technology), supervised by **Ph.D. Philip Gerlee** (Department of Mathematical Sciences, University of Gothenburg & Chalmers University of Technology, Sweden) — [Original Baseline (`6c3fae7`)](https://github.com/j0sees/Simulation-of-Gene-Regulatory-Networks-Driving-Shape-Homeostasis/tree/6c3fae7834a2ca502b1a439d7535c4adab1673f3)
- **Software Architecture & Lead Maintainer (`v0.2.0`–`v0.3.0`)**: **Reymer Vargas** ([@userravs](https://github.com/userravs), Independent Research Software Engineer)

---

## Overview

This software simulates multicellular collectives where shape homeostasis emerges from local cellular interactions. It integrates a 2D discrete **cellular automaton**, reaction-diffusion solvers for **Short-range (SGF)** and **Long-range (LGF)** growth factors, intracellular **recurrent neural networks**, and **evolutionary optimization** (Genetic Algorithms and NEAT), based on the framework of *Gerlee et al. (2015)*.

![Cell System Example](./docs/images/cell_system_example.png)

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis.git
cd simulation-of-gene-regulatory-networks-driving-shape-homeostasis

# Build and start the Streamlit web application at http://localhost:8501
./scripts/run_docker.sh
```

### Option 2: Local Setup with `uv`

Requires **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)**:

```bash
# Install dependencies (including dev tools)
uv sync --dev

# Launch the interactive web interface at http://localhost:8501
./scripts/run_web_app.sh
```

### Command-Line Execution & Testing

```bash
# Run a standalone simulation or genetic algorithm example
uv run python examples/run_simulation.py
uv run python examples/run_evolution.py

# Regenerate documentation figures
uv run python examples/generate_documentation_images.py

# Run linter and unit test suite
uv run ruff check .
uv run pytest -v
```

---

## Repository Structure

```text
├── analysis/             # Network topology generation and evaluation (Erdős-Rényi, etc.)
├── config/               # NEAT configuration parameters (config-ca)
├── core/                 # Core cellular automata engine, cell agent, and PDE solvers
├── docs/                 # Project documentation and example figures
├── evolution/            # Genetic Algorithm (main_ga.py) and NEAT neuroevolution
├── examples/             # Runnable simulation, evolution, and figure generation scripts
├── scripts/              # Helper scripts (run_docker.sh, run_web_app.sh)
├── tests/                # Pytest unit test suite
├── visualization/        # Matplotlib plotting utilities, Gnuplot script, and Streamlit app
├── CITATION.cff          # Citation metadata for GitHub and academic references
├── Dockerfile            # Container image definition (Python 3.12 + uv)
├── docker-compose.yml    # Docker Compose service configuration
└── pyproject.toml        # Project metadata, dependencies, and tool configuration
```

---

## Documentation

Detailed guides are located in the [`docs/`](./docs/README.md) directory:

- **[Documentation Index](./docs/README.md)** — Overview of all guides and testing instructions
- **[Model & Architecture](./docs/architecture.md)** — Mathematical model, cellular automata rules, reaction-diffusion PDEs, and code architecture
- **[Web Interface Guide](./docs/web-interface.md)** — Interactive Streamlit & Plotly simulation controls
- **[Docker Guide](./docs/docker.md)** — Container setup, volume mounts, and troubleshooting
- **[Development History](./docs/history-log.md)** — Chronological research and modernization log

---

## Authors & License

- **J. Esteban Pérez-Hidalgo** ([@j0sees](https://github.com/j0sees), School of Physics, Costa Rica Institute of Technology) — Original research, mathematical model, and initial simulation engine (`v0.1`, master's thesis *"Classification of Gene Regulatory Networks Driving Shape Homeostasis"*, `jose.perez at tec.ac.cr`)
- **Reymer Vargas** ([@userravs](https://github.com/userravs), Independent Research Software Engineer) — Software architecture, Python 3 refactoring, interactive web platform, containerization, CI/test suite, and lead maintainer (`v0.2.0`–`v0.3.0`)
- **License**: Released under the [MIT License](LICENSE).
