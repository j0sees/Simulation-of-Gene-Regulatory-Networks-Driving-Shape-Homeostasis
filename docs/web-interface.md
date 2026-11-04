# Web Interface Documentation

## Overview

The Gene Regulatory Network simulation includes an interactive web application (`visualization/web_app.py`) built with **Streamlit** and **Plotly**. It provides browser-based parameter configuration, simulation playback controls, and real-time heatmaps of cell states and chemical signaling fields.

```mermaid
flowchart LR
    subgraph Browser["Browser UI (http://localhost:8501)"]
        Sidebar["Sidebar Controls\nNodes (8..50), Lattice (10..100),\nSteps (10..500), Topology"]
        Tabs["Interactive Tabs\n1. Simulation | 2. Analysis | 3. About"]
    end

    subgraph Backend["Streamlit Session & Engine (visualization/web_app.py)"]
        Topo["Topology Generator\nRandom | Erdos-Renyi |\nWatts-Strogatz | Barabasi-Albert"]
        Sim["Live CA & PDE Stepper\ncell + SGFDiffEq + LGFDiffEq"]
        Evo["Multi-Generation Fitness Runner"]
    end

    subgraph Charts["Plotly Visualizations"]
        Heatmaps["3-Panel Heatmaps\nCell Grid, SGF Field, LGF Field"]
        Curves["Best vs. Mean Fitness Curves"]
    end

    Sidebar --> Topo --> Sim --> Heatmaps --> Tabs
    Sidebar --> Evo --> Curves --> Tabs
```

## Quick Start

### Option 1: Container — Podman or Docker (Recommended)

```bash
# Auto-detects Podman or Docker and starts the containerized web application
./scripts/run_container.sh

# Open in your browser: http://localhost:8501
```

### Option 2: Local Execution with `uv`

```bash
# Use the local launcher script (syncs dependencies via uv automatically)
./scripts/run_web_app.sh

# Or run Streamlit directly with uv
uv sync
uv run streamlit run visualization/web_app.py --server.port 8501 --server.address 0.0.0.0
```

## Interface Features

### 1. Simulation Tab
- **Start / Pause / Reset Controls**: Manage simulation execution interactively.
- **Real-Time Parameter Adjustment**: Configure network size (`8`–`50` nodes), lattice dimensions (`10`–`100`), and time steps (`10`–`500`) from the sidebar.
- **Network Topologies**: Choose between **Random**, **Erdős-Rényi**, **Watts-Strogatz**, and **Barabási-Albert** graph architectures.
- **Live Visualization**: Inspect side-by-side Plotly heatmaps of cell states (`Empty`, `Quiet`, `Moving`, `Divided`), SGF concentrations, and LGF concentrations.

![Simulation Visualization](./images/cell_system_example.png)

### 2. Analysis Tab
- **Fitness Analysis**: Run a multi-generation evolutionary pass and inspect best vs. average fitness progression curves.
- **Interactive Charts**: Hover, zoom, pan, and export plots as PNG images via the Plotly toolbar.

### 3. About Tab
- **Model Summary**: Overview of the cellular automata, chemical signaling, and recurrent neural network architecture.

## Troubleshooting

- **Port `8501` already in use**: Check active processes with `lsof -i :8501` or launch on an alternate port (`uv run streamlit run visualization/web_app.py --server.port 8502`).
- **Container import or build issues**: Rebuild cleanly via `docker-compose down && docker-compose up --build`.
- **Faster interactive rendering**: Reduce **Lattice Size** (e.g., `25`–`40`) when running long simulations in the browser.
