#!/usr/bin/env bash

# Script to stop and tear down Gene Regulatory Network simulation containers and images
# Supports Podman and Docker on macOS and Linux with safety confirmation.

set -euo pipefail

cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

echo -e "${RED}Gene Regulatory Network Simulation — Teardown (${OS_NAME})${NC}"
echo "=========================================================="

# 2. Parse arguments (-y / --yes / --force for non-interactive execution)
FORCE=false
CLEAN_DATA=false

for arg in "$@"; do
    case "${arg}" in
        -y|--yes|--force)
            FORCE=true
            ;;
        --clean-data)
            CLEAN_DATA=true
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Stops and removes simulation containers, images, and cache directories."
            echo ""
            echo "Options:"
            echo "  -y, --yes, --force   Skip confirmation prompt"
            echo "  --clean-data         Also remove generated simulation output in data/"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: ${arg}${NC}" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

# 3. Detect active container engine (allow override via CONTAINER_ENGINE=podman|docker)
ENGINE="${CONTAINER_ENGINE:-}"

if [[ -z "${ENGINE}" ]]; then
    if command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
        ENGINE="podman"
    elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        ENGINE="docker"
    else
        echo -e "${YELLOW}Warning: Neither running Podman nor Docker was detected.${NC}"
        echo "Will only clean up local workspace caches."
        ENGINE="none"
    fi
else
    if ! command -v "${ENGINE}" >/dev/null 2>&1 || ! "${ENGINE}" info >/dev/null 2>&1; then
        echo -e "${RED}Error: Requested container engine '${ENGINE}' is not available or not running.${NC}" >&2
        exit 1
    fi
fi

if [[ "${ENGINE}" != "none" ]]; then
    echo -e "${BLUE}Target container engine:${NC} ${ENGINE}"
fi

# 4. Confirmation prompt
echo ""
echo -e "${YELLOW}WARNING: This action is destructive and will perform the following:${NC}"
if [[ "${ENGINE}" != "none" ]]; then
    echo "  - Stop and remove running 'grn-simulation' container(s)"
    echo "  - Remove local 'grn-simulation:latest' and 'grn-simulation:ci' images"
fi
echo "  - Remove Python bytecode and test cache directories (__pycache__, .pytest_cache, .ruff_cache)"
if [[ "${CLEAN_DATA}" == "true" ]]; then
    echo "  - Remove generated simulation output files in 'data/'"
fi
echo ""

if [[ "${FORCE}" != "true" ]]; then
    read -r -p "Are you sure you want to proceed with teardown? [y/N]: " CONFIRMATION
    case "${CONFIRMATION}" in
        y|Y|yes|Yes|YES) ;;
        *)
            echo -e "${YELLOW}Teardown cancelled.${NC}"
            exit 0
            ;;
    esac
fi

echo ""
echo -e "${YELLOW}Starting teardown...${NC}"

# 5. Stop and remove containers via Compose or direct CLI
if [[ "${ENGINE}" != "none" ]]; then
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

    # Attempt compose teardown first if available
    if [[ ${#COMPOSE_CMD[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Stopping and removing containers via ${COMPOSE_CMD[*]}...${NC}"
        "${COMPOSE_CMD[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
    fi

    # Ensure standalone container is stopped and removed
    if "${ENGINE}" ps -a -q -f name=^grn-simulation$ | grep -q .; then
        echo -e "${YELLOW}Removing container 'grn-simulation'...${NC}"
        "${ENGINE}" rm -f grn-simulation >/dev/null 2>&1 || true
    fi

    # Remove built images
    for img in "grn-simulation:latest" "grn-simulation:ci"; do
        if "${ENGINE}" images -q "${img}" | grep -q .; then
            echo -e "${YELLOW}Removing image '${img}'...${NC}"
            "${ENGINE}" rmi -f "${img}" >/dev/null 2>&1 || true
        fi
    done
fi

# 6. Clean local caches
echo -e "${YELLOW}Cleaning workspace cache directories...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf .pytest_cache .ruff_cache

# 7. Optionally clean data directory
if [[ "${CLEAN_DATA}" == "true" ]]; then
    echo -e "${YELLOW}Cleaning data directory...${NC}"
    rm -rf data/*
fi

echo -e "${GREEN}Teardown complete successfully.${NC}"

