#!/bin/bash

# Script to run the Gene Regulatory Network Simulation Web App

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Gene Regulatory Network Simulation Web App${NC}"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi

# Check if requirements are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python3 -c "import streamlit, plotly" &> /dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip3 install -r requirements.txt
fi

# Create data directory if it doesn't exist
mkdir -p data

echo -e "${GREEN}Starting web application...${NC}"
echo ""
echo "The web app will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

# Run Streamlit
streamlit run web_app.py --server.port 8501 --server.address 0.0.0.0
