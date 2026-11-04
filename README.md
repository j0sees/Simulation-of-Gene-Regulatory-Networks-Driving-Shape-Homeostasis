# Simulation of Gene Regulatory Networks Driving Shape Homeostasis

Computational platform for simulating and classifying **Gene Regulatory Networks (GRNs)** that drive **shape homeostasis** in multicellular systems.

- **Original Research & Mathematical Model (`v0.1.0`)**: **M.Sc. J. Esteban Pérez-Hidalgo** ([@j0sees](https://github.com/j0sees), School of Physics, Costa Rica Institute of Technology), supervised by **Ph.D. Philip Gerlee** (Department of Mathematical Sciences, University of Gothenburg & Chalmers University of Technology, Sweden) — [Original Baseline (`6c3fae7`)](https://github.com/j0sees/Simulation-of-Gene-Regulatory-Networks-Driving-Shape-Homeostasis/tree/6c3fae7834a2ca502b1a439d7535c4adab1673f3)
- **Software Architecture & Lead Maintainer (`v0.2.0`–`v0.3.0`)**: **Reymer Vargas** ([@userravs](https://github.com/userravs), Independent Research Software Engineer)

---

## Overview

This software simulates multicellular collectives where shape homeostasis emerges from local cellular interactions. It integrates a 2D discrete **cellular automaton**, reaction-diffusion solvers for **Short-range (SGF)** and **Long-range (LGF)** growth factors, intracellular **recurrent neural networks**, and **evolutionary optimization** (Genetic Algorithms and NEAT), based on the framework of *Gerlee et al. (2015)*.

Key scientific objectives include:

- Simulating multicellular collectives and observing adaptive morphogenesis under dynamic chemical environments.
- Generalizing the intracellular signaling network into a fully connected **Recurrent Neural Network (RNN)** to expand the evolutionary search space.
- Classifying evolved GRNs into **three major behavioral classes** that achieve shape homeostasis through a dynamic equilibrium of proliferation (`Split`), migration (`Move`), quiescence (`Quiet`), and apoptosis (`Die`).
- Exploring network topology effects across **Erdős-Rényi**, **Watts-Strogatz**, and **Barabási-Albert** graph architectures.

![Cell System Example](./docs/images/cell_system_example.png)

---

## Quick Start

### Option 1: Container — Podman or Docker (Recommended on macOS & Linux)

```bash
git clone https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis.git
cd simulation-of-gene-regulatory-networks-driving-shape-homeostasis

# Auto-detects Podman or Docker, builds the image, and starts the web app at http://localhost:8501
./scripts/run_container.sh
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
├── scripts/              # Helper scripts (run_container.sh, run_web_app.sh)
├── tests/                # Pytest unit test suite
├── visualization/        # Matplotlib plotting utilities, Gnuplot script, and Streamlit app
├── CHANGELOG.md          # Release changelog with commit references and timestamps
├── CITATION.cff          # Citation metadata for GitHub and academic references
├── CODE_OF_CONDUCT.md    # Contributor Covenant Code of Conduct
├── CONTRIBUTING.md       # Contribution workflow, coding standards, and pre-PR checks
├── Dockerfile            # Container image definition (Python 3.12 + uv)
├── docker-compose.yml    # Docker Compose service configuration
├── pyproject.toml        # Project metadata, dependencies, and tool configuration
└── VERSION               # Canonical semantic version file
```

---

## Documentation

Detailed guides are located in the [`docs/`](./docs/README.md) directory:

- **[Documentation Index](./docs/README.md)** — Overview of all guides and testing instructions
- **[Model & Architecture](./docs/architecture.md)** — Mathematical model, cellular automata rules, reaction-diffusion PDEs, and code architecture
- **[Web Interface Guide](./docs/web-interface.md)** — Interactive Streamlit & Plotly simulation controls
- **[Docker Guide](./docs/docker.md)** — Container setup, volume mounts, and troubleshooting
- **[Development History](./docs/history-log.md)** — Chronological research and modernization log
- **[Changelog](./CHANGELOG.md)** — Versioned release notes with commit references and timestamps
- **[Contributing Guide](./CONTRIBUTING.md)** — Trunk-Based Development workflow, `uv` setup, and pre-PR verification
- **[Code of Conduct](./CODE_OF_CONDUCT.md)** — Community standards for scientific and open-source collaboration

---

## Authors & License

- **J. Esteban Pérez-Hidalgo** ([@j0sees](https://github.com/j0sees), School of Physics, Costa Rica Institute of Technology) — Original research, mathematical model, and initial simulation engine (`v0.1.0`, master's thesis *"Classification of Gene Regulatory Networks Driving Shape Homeostasis"*, `jose.perez at tec.ac.cr`)
- **Reymer Vargas** ([@userravs](https://github.com/userravs), Independent Research Software Engineer) — Software architecture, Python 3 refactoring, interactive web platform, containerization, CI/test suite, and lead maintainer (`v0.2.0`–`v0.3.0`)
- **License**: Released under the [MIT License](LICENSE).
