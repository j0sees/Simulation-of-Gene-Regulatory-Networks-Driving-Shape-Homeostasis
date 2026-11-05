#!/usr/bin/env bash

# Script to run the Gene Regulatory Network simulation in Podman or Docker (macOS & Linux)

set -euo pipefail

cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Verify supported operating system (macOS or Linux)
OS_NAME="$(uname -s)"
case "${OS_NAME}" in
    Darwin|Linux) ;;
    *)
        echo -e "${RED}Error: Unsupported operating system '${OS_NAME}'. Only macOS (Darwin) and Linux are supported.${NC}" >&2
        exit 1
        ;;
esac

echo -e "${GREEN}Gene Regulatory Network Simulation (${OS_NAME})${NC}"
echo "=========================================="

# 2. Detect active container engine (allow override via CONTAINER_ENGINE=podman|docker)
ENGINE="${CONTAINER_ENGINE:-}"

if [[ -z "${ENGINE}" ]]; then
    if command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
        ENGINE="podman"
    elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        ENGINE="docker"
    else
        echo -e "${RED}Error: Neither a running Podman nor Docker engine was detected.${NC}" >&2
        if [[ "${OS_NAME}" == "Darwin" ]]; then
            echo "  - For Podman Desktop: start Podman Desktop or run 'podman machine start'" >&2
            echo "  - For Docker Desktop: start Docker Desktop" >&2
        else
            echo "  - Start the Podman socket/service or Docker daemon and try again." >&2
        fi
        exit 1
    fi
else
    if ! command -v "${ENGINE}" >/dev/null 2>&1 || ! "${ENGINE}" info >/dev/null 2>&1; then
        echo -e "${RED}Error: Requested container engine '${ENGINE}' is not available or not running.${NC}" >&2
        exit 1
    fi
fi

echo -e "${GREEN}Using container engine:${NC} ${ENGINE}"

# Preserve host user ID mapping for rootless Podman bind mounts (./data:/app/data)
if [[ "${ENGINE}" == "podman" ]]; then
    export PODMAN_USERNS="${PODMAN_USERNS:-keep-id}"
fi

# 3. Detect Compose command (if available) for the chosen engine
COMPOSE_CMD=()
if [[ "${ENGINE}" == "podman" ]]; then
    if command -v podman-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(podman-compose)
    elif podman compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(podman compose)
    fi
elif [[ "${ENGINE}" == "docker" ]]; then
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(docker compose)
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(docker-compose)
    fi
fi

# Create data directory if it does not exist
mkdir -p data

echo ""
echo "Web Interface: http://localhost:8501"
echo ""
echo "The web application provides:"
echo "  - Interactive simulation controls"
echo "  - Real-time visualization"
echo "  - Parameter adjustment"
echo "  - Analysis tools"
echo ""
echo "To stop the container, press Ctrl+C"
echo ""

# 4. Build and run via Compose (if available) or direct engine CLI fallback
if [[ ${#COMPOSE_CMD[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Building container image with ${COMPOSE_CMD[*]}...${NC}"
    "${COMPOSE_CMD[@]}" build
    echo -e "${YELLOW}Starting container...${NC}"
    "${COMPOSE_CMD[@]}" up
else
    IMAGE_NAME="grn-simulation:latest"
    echo -e "${YELLOW}Building container image directly with ${ENGINE}...${NC}"
    "${ENGINE}" build -t "${IMAGE_NAME}" .
    echo -e "${YELLOW}Starting container with ${ENGINE}...${NC}"
    "${ENGINE}" run --rm -it \
        --name grn-simulation \
        -p 8501:8501 \
        -v "$(pwd):/app" \
        -v "$(pwd)/data:/app/data" \
        -e PYTHONPATH=/app \
        -e PYTHONUNBUFFERED=1 \
        "${IMAGE_NAME}"
fi

