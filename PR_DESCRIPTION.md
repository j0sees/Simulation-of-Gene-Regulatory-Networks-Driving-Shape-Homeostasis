# Project Modernization: Containerization, Web Interface, and Documentation Enhancement

## Overview

This Pull Request modernizes the Gene Regulatory Networks simulation project, organizing the research codebase into a modular Python package with Docker containerization, an interactive Streamlit web interface, and comprehensive documentation.

## Objectives

- **Containerization**: Make the simulation reproducible and deployable across environments without manual dependency setup.
- **Web Interface**: Provide an interactive browser application for configuring parameters and visualizing simulations in real time.
- **Documentation**: Supply clear guides, architecture summaries, and visual examples.
- **Code Quality**: Improve module organization, PEP 8 compliance, and maintainability.

## Major Changes

### 1. Project Restructuring
Reorganized the codebase into dedicated packages:
- `core/` — Core simulation components (`cell_agent.py`, `main.py`, `tools.py`)
- `evolution/` — Genetic algorithm implementation (`main_ga.py`, `tools_ga.py`)
- `visualization/` — Plotting and visualization utilities (`plot.py`)
- `analysis/` — Network topology generation and evaluation (`graph_generator.py`, `graph_evaluation.py`)
- `examples/` — Standalone example scripts and figure generators
- `config/` — Configuration files (`config-ca`)
- `docs/` — Project documentation and reference figures

### 2. Containerization
- **`Dockerfile`**: Builds a lightweight image based on `python:3.9-slim` running as a non-root user (`appuser`).
- **`docker-compose.yml`**: Configures service ports (`8501:8501`) and volume mounts for local development and data persistence.
- **`run_docker.sh`**: Helper script to validate Docker status, build the image, and launch the application.
- **`.dockerignore`**: Excludes build artifacts, caches, and version control metadata from the image context.

### 3. Interactive Web Interface
- **Streamlit Application (`web_app.py`)**: Interactive controls for running simulations in the browser.
- **Plotly Visualizations**: Side-by-side heatmaps for cell states, SGF concentrations, and LGF concentrations.
- **Three Main Views**:
  - **Simulation**: Live playback controls (Start, Pause, Reset) and parameter sliders.
  - **Analysis**: Fitness progression plots across evolutionary generations.
  - **About**: Scientific background, model summary, and authorship details.

### 4. Enhanced Documentation
- **Updated `README.md`**: Quick start instructions (Docker and local), project tree, scientific overview, and usage examples.
- **Dedicated `docs/` Guides**:
  - `docs/README.md` — Documentation index
  - `docs/web-interface.md` — Web application usage guide
  - `docs/docker.md` — Container setup and troubleshooting reference
  - `docs/thesis-summary.md` — Mathematical model and thesis overview
  - `docs/history-log.md` — Development timeline

### 5. Dependency Management and Code Quality
- **`requirements.txt`**: Pinned minimum versions for `numpy`, `scipy`, `matplotlib`, `networkx`, `numba`, `neat-python`, `streamlit`, and `plotly`.
- **Module Imports**: Updated relative and package-level imports across restructured directories.
- **Docstrings and Formatting**: Improved docstrings and comments across Python modules.

## Backward Compatibility

- **Core Algorithms Preserved**: Original cellular automata rules, reaction-diffusion equations, and genetic algorithm operators remain intact.
- **CLI Support Maintained**: Command-line execution of simulation and evolutionary scripts continues to work alongside the web interface.

## Quick Start

```bash
# Clone the repository and start with Docker
git clone https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis.git
cd simulation-of-gene-regulatory-networks-driving-shape-homeostasis
./run_docker.sh

# Open the web interface at: http://localhost:8501
```

## Contributors

- **M.Sc. J. Esteban Pérez-Hidalgo** (School of Physics, Costa Rica Institute of Technology)
  - Master's Thesis Author — University of Gothenburg & Chalmers University of Technology
  - Original research and simulation implementation
- **Reymer Vargas** — Platform Engineer
  - Project restructuring, containerization, web interface development, and documentation
