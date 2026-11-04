#!/usr/bin/env python3
"""
Basic simulation example.

This script demonstrates how to run a simple cellular automata simulation
with a gene regulatory network.
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import main


def run_basic_simulation() -> None:
    """Run a basic simulation with default parameters."""
    n_nodes = 25
    time_steps = 50
    n_lattice = 20
    mode = False  # Visualization mode

    w_matrix = np.random.rand(n_nodes, n_nodes) * 2.0 - 1.0

    print("Running simulation with:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Time steps: {time_steps}")
    print(f"  Lattice size: {n_lattice}x{n_lattice}")
    print(f"  Mode: {'Fitness' if mode else 'Visualization'}")

    main.sim(w_matrix, time_steps, n_nodes, n_lattice, mode)
    print("Simulation completed!")


if __name__ == "__main__":
    run_basic_simulation()
