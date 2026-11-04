# Simulation of Gene Regulatory Networks Driving Shape Homeostasis

This repository contains the software developed as part of the master's thesis:

**"Classification of Gene Regulatory Networks Driving Shape Homeostasis"**

by **M.Sc. J. Esteban Pérez-Hidalgo** (School of Physics, Costa Rica Institute of Technology), under the supervision of **Ph.D. Philip Gerlee** (Department of Mathematical Sciences, University of Gothenburg & Chalmers University of Technology, Sweden).

---

## Overview

This software enables the **simulation and classification of Gene Regulatory Networks (GRNs)** that drive **shape homeostasis** in multicellular systems—a key emergent phenomenon in developmental biology and artificial life.

The simulation is written in **Python** and builds upon the computational framework proposed by **Gerlee et al.** in *"The influence of cellular characteristics on the evolution of shape homeostasis"*, alongside foundational cellular automata and neural network models in artificial life.

---

## Original Repository

This project is a **fork** of the original research repository by **M.Sc. J. Esteban Pérez-Hidalgo**:

- **Original Repository**: [https://github.com/j0sees/Simulation-of-Gene-Regulatory-Networks-Driving-Shape-Homeostasis](https://github.com/j0sees/Simulation-of-Gene-Regulatory-Networks-Driving-Shape-Homeostasis)

This fork adds modernization improvements—including Docker containerization, an interactive Streamlit web interface, modular package structure, and expanded documentation—while preserving the original scientific implementation.

---

## Purpose

- Simulate multicellular collectives and observe adaptive behavior under dynamic environments.
- Classify GRNs capable of maintaining **shape homeostasis** across varying environmental conditions.
- Explore the role of **diffusible growth factors**, **cell-cell interactions**, and **internal cellular actions** in emergent morphogenesis.
- Demonstrate how distinct behavioral classes lead to stable multicellular structures.
- Evaluate cellular responses as a function of environmental and network parameters.

---

## Methods and Technologies

The simulation integrates three core computational approaches:

- **Cellular Automata** — Models discrete 2D spatial environments and local cell state transitions.
- **Artificial Neural Networks** — Implements recurrent gene regulatory networks within each individual cell.
- **Evolutionary Optimization Algorithms** — Evolves network weights and topologies toward homeostatic target dynamics.

Collective multicellular behavior emerges from:

1. A dynamic equilibrium of cellular actions (quiet, move, divide, die),
2. Diffusible chemical signaling via Short-range Growth Factor (SGF) and Long-range Growth Factor (LGF), and
3. Intracellular recurrent signaling networks acting as **gene regulatory networks**.

---

## Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended; no local Python setup required)
- **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)** (for local execution and development)

### Installation

#### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis.git
cd simulation-of-gene-regulatory-networks-driving-shape-homeostasis

# Build and start the containerized web application
./run_docker.sh
```

#### Option 2: Local Installation with `uv`

```bash
# Clone the repository
git clone https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis.git
cd simulation-of-gene-regulatory-networks-driving-shape-homeostasis

# Create virtual environment and sync dependencies using uv
uv sync
```

### Usage

#### Web Interface (Recommended)

```bash
# Using Docker
./run_docker.sh

# Or locally using uv via the launcher script
./run_web_app.sh
```

Open your browser at `http://localhost:8501`. The web interface provides:

- **Interactive simulation controls** (Start, Pause, Reset) with real-time parameter adjustment
- **Live visualization** of cell states, SGF concentrations, and LGF concentrations
- **Fitness analysis** views for inspecting evolutionary progress
- **Topology selection** across random, Erdős-Rényi, Watts-Strogatz, and Barabási-Albert networks

#### Command Line

```bash
# Run a basic cellular automata simulation example
uv run python examples/run_simulation.py

# Run a genetic algorithm evolution example
uv run python examples/run_evolution.py

# Regenerate documentation example plots
uv run python examples/generate_documentation_images.py

# Run core simulation or genetic algorithm modules directly
uv run python core/main.py [filename] [nNodes] [individual]
uv run python evolution/main_ga.py [filename] [parameters]
```

#### Testing and Linting

```bash
# Sync development dependencies (pytest, ruff)
uv sync --dev

# Run the unit test suite
uv run pytest -v

# Run Ruff linter across the codebase
uv run ruff check .
```

---

## Project Structure

```text
├── .github/workflows/                       # GitHub Actions CI workflows
│   └── ci.yml                               # Automated linting, unit tests, and Docker build check
├── core/                                    # Core simulation engine
│   ├── cell_agent.py                        # Cell agent class and intracellular neural network
│   ├── main.py                              # Cellular automata simulation loop
│   └── tools.py                             # Differential equations and utility functions
├── evolution/                               # Evolutionary optimization
│   ├── main_ga.py                           # Parallel genetic algorithm runner
│   └── tools_ga.py                          # Selection, crossover, mutation, and fitness tools
├── visualization/                           # Matplotlib plotting routines
│   └── plot.py                              # Grid and fitness visualization utilities
├── analysis/                                # Network topology generation and evaluation
│   ├── graph_generator.py                   # Random, small-world, and scale-free graph generators
│   └── graph_evaluation.py                  # Network evaluation scripts
├── config/                                  # Configuration files
│   └── config-ca                            # NEAT configuration parameters
├── examples/                                # Runnable example scripts
│   ├── run_simulation.py                    # Standalone simulation example
│   ├── run_evolution.py                     # Standalone genetic algorithm example
│   └── generate_documentation_images.py     # Script to generate documentation figures
├── tests/                                   # Unit test suite (pytest)
│   ├── test_tools.py                        # PDE solver, RNN, and matrix utility tests
│   ├── test_cell_agent.py                   # Cell agent sensing and action transition tests
│   ├── test_evolution.py                    # Genetic operators and headless simulation tests
│   └── test_web_and_topologies.py           # Network topology and Plotly figure tests
├── docs/                                    # Project documentation
│   ├── README.md                            # Documentation index
│   ├── web-interface.md                     # Web application guide
│   ├── docker.md                            # Docker deployment and troubleshooting guide
│   ├── thesis-summary.md                    # Scientific background and model summary
│   ├── history-log.md                       # Research and modernization changelog
│   └── images/                              # Generated example figures
├── web_app.py                               # Streamlit interactive web application
├── neat_cell_agent.py                       # NEAT-based cell agent implementation
├── neat_parallel.py                         # Parallel NEAT evaluation runner
├── plots_gp                                 # Gnuplot script for fitness curves
├── run_docker.sh                            # Docker build and launch script
├── run_web_app.sh                           # Local Streamlit launch script
├── Dockerfile                               # Container image definition (Python 3.12 + uv)
├── docker-compose.yml                       # Docker Compose service configuration
├── pyproject.toml                           # Project metadata and uv dependencies
├── requirements.txt                         # Python package dependencies
├── LICENSE                                  # MIT License
└── README.md                                # Main project documentation
```

### Architecture Summary

- **`core/`**: Implements the 2D cellular automaton grid, SGF/LGF reaction-diffusion PDEs, and recurrent neural network evaluation per cell.
- **`evolution/`**: Implements genetic algorithms to evolve weight matrices that maximize shape homeostasis fitness.
- **`visualization/`**: Provides Matplotlib visualization classes for rendering grid states and chemical concentration fields.
- **`analysis/`**: Generates and evaluates graph topologies (Erdős-Rényi, Watts-Strogatz, Barabási-Albert) mapped onto GRN weight matrices.
- **`examples/`**: Self-contained scripts demonstrating simulation execution, evolutionary runs, and figure generation.
- **`tests/`**: Automated unit tests covering mathematical solvers, cellular automata rules, evolutionary operators, and graph topologies.
- **`web_app.py`**: Browser-based interface built with Streamlit and Plotly.

---

## Scientific Contribution

This project combines methods from three disciplines:

- **Computational Physics** — Lattice cellular automata and reaction-diffusion dynamics.
- **Applied Mathematics** — Finite-difference numerical integration and evolutionary optimization.
- **Computational Systems Biology** — Gene regulatory network modeling and multicellular morphogenesis.

### Example Visualizations

#### Cell System Simulation
![Cell System Example](./docs/images/cell_system_example.png)

*Simulation snapshot showing discrete cell states (empty, quiet, moving, dividing) alongside Short-range Growth Factor (SGF) and Long-range Growth Factor (LGF) concentration fields.*

#### Gene Regulatory Network
![Network Example](./docs/images/network_example.png)

*Directed graph representation of a gene regulatory network with feedback loops and regulatory interactions.*

#### Evolutionary Progress
![Evolution Example](./docs/images/evolution_example.png)

*Best and average population fitness across generations during evolutionary optimization.*

### Research Applications

- **Developmental Biology**: Investigating tissue shape formation and homeostatic maintenance.
- **Artificial Life**: Studying self-organization and emergent spatial behavior in agent collectives.
- **Systems Biology**: Analyzing regulatory motifs and feedback dynamics in GRNs.
- **Evolutionary Computation**: Benchmarking neuroevolution and genetic algorithms on spatial morphogenesis tasks.

---

## Author

- **M.Sc. J. Esteban Pérez-Hidalgo** (School of Physics, Costa Rica Institute of Technology)
  - Master's Thesis — University of Gothenburg & Chalmers University of Technology
  - Contact: `jose.perez at tec.ac.cr`

---

## Development History

### Original Research
- Master's thesis research on classifying gene regulatory networks driving shape homeostasis.
- Implementation of the 2D cellular automata engine, recurrent neural network controller, SGF/LGF diffusion solver, and parallel genetic algorithm.

### Repository Modernization
- **Modular Structure**: Reorganized codebase into `core/`, `evolution/`, `visualization/`, `analysis/`, `config/`, and `examples/`.
- **Containerization**: Added `Dockerfile`, `docker-compose.yml`, and launcher scripts for reproducible execution.
- **Web Interface**: Built interactive Streamlit and Plotly application (`web_app.py`).
- **Documentation**: Added guides under `docs/` covering architecture, Docker workflows, web interface usage, and thesis context.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
