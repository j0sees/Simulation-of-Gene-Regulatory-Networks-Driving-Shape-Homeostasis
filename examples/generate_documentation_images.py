#!/usr/bin/env python3
"""
Generate example images for documentation.

This script creates sample visualizations from the simulation
to include in the project documentation.
"""

from __future__ import annotations

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_cell_system_example() -> None:
    """Generate an example cell system visualization."""
    print("Generating cell system example...")

    field_size = 50
    cells_figure, (cells_subplot, sgf_subplot, lgf_subplot) = plt.subplots(
        1, 3, figsize=(15, 5)
    )

    cell_grid = np.zeros((field_size, field_size))
    sgf_grid = np.zeros((field_size, field_size))
    lgf_grid = np.zeros((field_size, field_size))

    cell_grid[20:25, 20:25] = 1  # Quiet cells
    cell_grid[30:35, 30:35] = 2  # Moving cells
    cell_grid[40:45, 40:45] = 3  # Dividing cells

    sgf_grid[15:35, 15:35] = np.random.uniform(0.0, 0.05, (20, 20))
    lgf_grid[10:40, 10:40] = np.random.uniform(0.0, 3.0, (30, 30))

    c_map = ListedColormap(["w", "g", "b", "r"])

    cells_subplot.set_aspect("equal")
    sgf_subplot.set_aspect("equal")
    lgf_subplot.set_aspect("equal")

    cells_figure.suptitle("Cell System Example", fontsize=16)

    cells_subplot.set_title("Cells")
    sgf_subplot.set_title("SGF (Short-range Growth Factor)")
    lgf_subplot.set_title("LGF (Long-range Growth Factor)")

    cell_plot = cells_subplot.imshow(
        cell_grid, origin="lower", cmap=c_map, interpolation="none", vmin=0, vmax=3
    )
    sgf_plot = sgf_subplot.imshow(
        sgf_grid, origin="lower", cmap="Reds", interpolation="none", vmin=0, vmax=0.05
    )
    lgf_plot = lgf_subplot.imshow(
        lgf_grid, origin="lower", cmap="Blues", interpolation="none", vmin=0, vmax=3
    )

    cbar1 = cells_figure.colorbar(
        cell_plot, ax=cells_subplot, ticks=[], orientation="horizontal"
    )
    cells_figure.colorbar(sgf_plot, ax=sgf_subplot, orientation="horizontal")
    cells_figure.colorbar(lgf_plot, ax=lgf_subplot, orientation="horizontal")

    for j, lab in enumerate(["Empty", "Quiet", "Moving", "Dividing"]):
        cbar1.ax.text((2 * j + 1) / 8.0, 0.5, lab, ha="center", va="center")

    for ax in (cells_subplot, sgf_subplot, lgf_subplot):
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    os.makedirs("docs/images", exist_ok=True)
    plt.savefig("docs/images/cell_system_example.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: docs/images/cell_system_example.png")


def generate_network_example() -> None:
    """Generate an example gene regulatory network visualization."""
    print("Generating network example...")

    graph = nx.DiGraph()
    genes = ["G1", "G2", "G3", "G4", "G5", "G6"]
    graph.add_nodes_from(genes)

    edges = [
        ("G1", "G2"),
        ("G1", "G3"),
        ("G2", "G4"),
        ("G3", "G5"),
        ("G4", "G6"),
        ("G5", "G6"),
        ("G6", "G1"),
    ]
    graph.add_edges_from(edges)

    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(graph)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=12,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        edge_color="gray",
        width=2,
    )

    plt.title("Gene Regulatory Network Example", fontsize=16, pad=20)
    plt.tight_layout()

    plt.savefig("docs/images/network_example.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: docs/images/network_example.png")


def generate_evolution_example() -> None:
    """Generate an example evolution fitness plot."""
    print("Generating evolution example...")

    generations = np.arange(1, 51)
    best_fitness = np.log(generations) + np.random.normal(0.0, 0.1, 50)
    avg_fitness = best_fitness * 0.7 + np.random.normal(0.0, 0.05, 50)

    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitness, "b-", linewidth=2, label="Best Fitness")
    plt.plot(generations, avg_fitness, "r--", linewidth=2, label="Average Fitness")

    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness Score", fontsize=12)
    plt.title("Evolutionary Algorithm Progress", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("docs/images/evolution_example.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: docs/images/evolution_example.png")


def main() -> None:
    """Generate all example images for documentation."""
    print("Generating documentation images...")
    os.makedirs("docs/images", exist_ok=True)

    generate_cell_system_example()
    generate_network_example()
    generate_evolution_example()

    print("\nAll images generated successfully!")
    print("Images saved in: docs/images/")
    print("- cell_system_example.png")
    print("- network_example.png")
    print("- evolution_example.png")


if __name__ == "__main__":
    main()
