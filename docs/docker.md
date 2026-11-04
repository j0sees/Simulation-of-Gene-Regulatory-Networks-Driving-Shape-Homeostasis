# Docker Documentation

## Overview

This project includes Docker and Docker Compose configurations to provide a consistent, reproducible environment across platforms without requiring a local Python setup.

## Quick Start

### Prerequisites
- **Docker** (Docker Engine or Docker Desktop) installed and running
- **Docker Compose** (`docker-compose` or `docker compose`)

### Running with Docker

```bash
# Build and start the web interface using the helper script
./run_docker.sh

# Or build and start directly with Docker Compose
docker-compose up --build
```

Once started, open `http://localhost:8501` in your browser.

## Docker Components

### `Dockerfile`
Defines the container image:
- Uses `python:3.9-slim` for a compact base image.
- Installs system build dependencies (`gcc`, `g++`) required by scientific Python packages.
- Installs Python dependencies from `requirements.txt`.
- Creates and switches to a non-root user (`appuser`, UID `1000`).
- Exposes port `8501` and launches the Streamlit web interface by default.

### `docker-compose.yml`
Configures the `grn-simulation` service:
- Builds the image from the local `Dockerfile`.
- Maps host port `8501` to container port `8501`.
- Mounts the workspace directory (`.:/app`) and data directory (`./data:/app/data`).
- Sets `PYTHONPATH=/app` and `PYTHONUNBUFFERED=1`.

### `.dockerignore`
Excludes non-essential files from the Docker build context:
- Version control metadata (`.git`, `.gitignore`)
- Python bytecode and caches (`__pycache__`, `*.pyc`)
- Markdown documentation (`*.md`)
- Editor configurations (`.vscode`, `.idea`)
- Generated media and data artifacts (`*.png`, `*.avi`, `*.csv`)

### `run_docker.sh`
Convenience launcher script that:
1. Verifies that the Docker daemon is running.
2. Ensures the local `./data` directory exists.
3. Builds the container image via `docker-compose build`.
4. Starts the service via `docker-compose up`.

## Container Architecture

### Application Layout Inside `/app`
```text
/app/
├── core/               # Core simulation modules
├── evolution/          # Genetic algorithm implementation
├── visualization/      # Matplotlib plotting utilities
├── analysis/           # Network topology and evaluation tools
├── config/             # Configuration files
├── examples/           # Runnable example scripts
├── data/               # Mounted directory for persistent outputs
├── web_app.py          # Streamlit web application
├── requirements.txt    # Python package dependencies
└── Dockerfile          # Container build definition
```

### Port Mapping and Volumes
- **Port**: Host `8501` -> Container `8501` (HTTP)
- **Workspace Mount**: `.:/app` (enables live code updates during development)
- **Data Mount**: `./data:/app/data` (persists simulation outputs on the host)

## Common Workflows

### Standard Web Interface Startup
```bash
./run_docker.sh
```

### Running in Detached (Background) Mode
```bash
# Start container in the background
docker-compose up -d --build

# Follow container logs
docker-compose logs -f

# Stop and remove containers
docker-compose down
```

### Running on a Custom Host Port
```bash
# Map host port 8080 to container port 8501
docker-compose run --rm -p 8080:8501 grn-simulation
```

### Running CLI Scripts Inside the Container
```bash
# Open an interactive shell inside the container
docker-compose run --rm grn-simulation bash

# Inside the container shell:
python examples/run_simulation.py
python examples/run_evolution.py
```

## Troubleshooting

### Container Fails to Start
```bash
# Verify Docker daemon status
docker info

# Check if port 8501 is already in use on the host
lsof -i :8501

# Inspect service logs
docker-compose logs
```

### Module Import or Dependency Errors
```bash
# Stop containers and rebuild the image from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Permission Issues on `run_docker.sh` or `data/`
```bash
# Ensure the launcher script is executable
chmod +x run_docker.sh

# Verify container user identity
docker-compose run --rm grn-simulation whoami
```

## References

- **Docker Documentation**: https://docs.docker.com/
- **Docker Compose Documentation**: https://docs.docker.com/compose/
- **Deploy Streamlit Using Docker**: https://docs.streamlit.io/deploy/tutorials/docker
