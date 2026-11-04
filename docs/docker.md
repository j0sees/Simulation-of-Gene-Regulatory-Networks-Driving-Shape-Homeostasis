# Docker Documentation

## Overview

This repository includes Docker and Docker Compose configurations to run the simulation and web interface in a reproducible container environment without requiring a local Python setup.

## Quick Start

### Prerequisites
- **Docker** (Docker Engine or Docker Desktop)
- **Docker Compose** (`docker-compose` or `docker compose`)

### Running the Web Interface

```bash
# Build and start using the helper script
./scripts/run_docker.sh

# Or build and start directly with Docker Compose
docker-compose up --build
```

Open `http://localhost:8501` in your browser.

## Container Components

### `Dockerfile`
- Uses `python:3.12-slim` as the base image and copies the `uv` binary from `ghcr.io/astral-sh/uv:latest`.
- Installs `gcc` and `g++` for compiled scientific extensions.
- Installs project dependencies directly from `pyproject.toml` via `uv pip install --system --no-cache -r pyproject.toml`.
- Runs as a non-root user (`appuser`, UID `1000`).
- Exposes port `8501` and launches `streamlit run visualization/web_app.py`.

### `docker-compose.yml`
- Builds the image from the local `Dockerfile`.
- Maps host port `8501` to container port `8501`.
- Mounts the workspace directory (`.:/app`) and persistent data directory (`./data:/app/data`).

## Common Commands

```bash
# Run in detached (background) mode
docker-compose up -d --build

# Follow container logs
docker-compose logs -f

# Stop and remove containers
docker-compose down

# Run CLI scripts interactively inside the container
docker-compose run --rm grn-simulation bash
python examples/run_simulation.py
python examples/run_evolution.py
```
