#!/bin/bash

# Script to run the Gene Regulatory Network Simulation Web App using uv

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Gene Regulatory Network Simulation Web App${NC}"
echo "=================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed. Install it from https://docs.astral.sh/uv/ first.${NC}"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

echo -e "${YELLOW}Syncing dependencies with uv...${NC}"
uv sync

echo -e "${GREEN}Starting web application...${NC}"
echo ""
echo "The web app will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

# Run Streamlit via uv
uv run streamlit run web_app.py --server.port 8501 --server.address 0.0.0.0
