#!/bin/bash

# Script to run the Gene Regulatory Network simulation in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Gene Regulatory Network Simulation${NC}"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Build the image if it doesn't exist or if Dockerfile changed
echo -e "${YELLOW}Building Docker image...${NC}"
docker-compose build

# Run the container
echo -e "${YELLOW}Starting container...${NC}"
echo -e "${GREEN}Web interface is starting.${NC}"
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

# Start the container with web interface
docker-compose up
