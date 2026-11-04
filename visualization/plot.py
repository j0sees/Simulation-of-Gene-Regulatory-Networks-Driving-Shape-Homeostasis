"""Matplotlib visualization utilities for the cellular automata and GRN simulation."""

from __future__ import annotations

import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
import numpy as np

import core.tools as tools


class Environment:
    """
    Visualization environment for the cellular automata simulation.

    Provides methods for creating and updating plots showing:
    - Cell states and positions
    - SGF (Short-range Growth Factor) concentrations
    - LGF (Long-range Growth Factor) concentrations
    """

    @staticmethod
    def CellsGridFigure(fieldSize: int, mode: bool):
        """
        Create the main figure with subplots for cell system visualization.

        Parameters
        ----------
        fieldSize : int
            Size of the square field (fieldSize x fieldSize).
        mode : bool
            True: run in headless fitness mode (no display window).
            False: run with interactive display window.

        Returns
        -------
        tuple
            (cellsFigure, cellsSubplot, sgfSubplot, lgfSubplot, cellPlot, sgfPlot, lgfPlot)
        """
        plt.close()

        c_map = ListedColormap(["w", "g", "b", "r"])
        cells_figure, (cells_subplot, sgf_subplot, lgf_subplot) = plt.subplots(
            1, 3, figsize=(15, 5)
        )

        cells_subplot.set_aspect("equal")
        sgf_subplot.set_aspect("equal")
        lgf_subplot.set_aspect("equal")

        cells_figure.suptitle("Cell System")

        cell_grid = np.zeros((fieldSize, fieldSize))
        sgf_grid = np.zeros((fieldSize, fieldSize))
        lgf_grid = np.zeros((fieldSize, fieldSize))

        cells_subplot.set_title("Cells")
        sgf_subplot.set_title("SGF")
        lgf_subplot.set_title("LGF")

        cell_plot = cells_subplot.imshow(
            cell_grid,
            origin="lower",
            cmap=c_map,
            interpolation="none",
            vmin=0,
            vmax=3,
        )
        cbar1 = cells_figure.colorbar(
            cell_plot, ax=cells_subplot, ticks=[], orientation="horizontal"
        )

        cells_subplot.axes.xaxis.set_ticklabels([])
        cells_subplot.axes.yaxis.set_ticklabels([])
        cells_subplot.axes.get_xaxis().set_visible(False)
        cells_subplot.axes.get_yaxis().set_visible(False)

        cbar1.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(["$empty$", "$quiet$", "$moving$", "$divided$"]):
            cbar1.ax.text((2 * j + 1) / 8.0, 0.5, lab, ha="center", va="center")
        cbar1.ax.get_yaxis().labelpad = 15
        cbar1.ax.set_ylabel("states", rotation=270)

        sgf_plot = sgf_subplot.imshow(
            sgf_grid,
            origin="lower",
            cmap="Reds",
            interpolation="none",
            vmin=0,
            vmax=0.05,
        )
        cells_figure.colorbar(sgf_plot, ax=sgf_subplot, orientation="horizontal")

        sgf_plot.axes.xaxis.set_ticklabels([])
        sgf_plot.axes.yaxis.set_ticklabels([])
        sgf_plot.axes.get_xaxis().set_visible(False)
        sgf_plot.axes.get_yaxis().set_visible(False)

        lgf_plot = lgf_subplot.imshow(
            lgf_grid,
            origin="lower",
            cmap="Blues",
            interpolation="none",
            vmin=0,
            vmax=3,
        )
        cells_figure.colorbar(lgf_plot, ax=lgf_subplot, orientation="horizontal")

        lgf_plot.axes.xaxis.set_ticklabels([])
        lgf_plot.axes.yaxis.set_ticklabels([])
        lgf_plot.axes.get_xaxis().set_visible(False)
        lgf_plot.axes.get_yaxis().set_visible(False)

        if not mode:
            plt.show(block=False)

        plt.ion()
        cells_figure.canvas.draw()
        plt.ioff()

        return (
            cells_figure,
            cells_subplot,
            sgf_subplot,
            lgf_subplot,
            cell_plot,
            sgf_plot,
            lgf_plot,
        )

    @staticmethod
    def AntGridPlot(
        cellGrid,
        chemGrid: np.ndarray,
        nLattice: int,
        cellsFigure,
        cellsSubplot,
        sgfSubplot,
        lgfSubplot,
        cellPlot,
        sgfPlot,
        lgfPlot,
        tStep: int,
        mode: bool,
    ) -> None:
        """Extract grid and chemical layers and refresh the Matplotlib figure."""
        cell_data = cellGrid
        sgf_data = chemGrid[:, :, 0]
        lgf_data = chemGrid[:, :, 1]

        Environment.UpdatePlot(
            cellsFigure,
            cellsSubplot,
            sgfSubplot,
            lgfSubplot,
            cellPlot,
            sgfPlot,
            lgfPlot,
            cell_data,
            sgf_data,
            lgf_data,
            tStep,
            mode,
        )

    @staticmethod
    def UpdatePlot(
        cellsFigure,
        cellsSubplot,
        sgfSubplot,
        lgfSubplot,
        cellPlot,
        sgfPlot,
        lgfPlot,
        cell_data,
        sgf_data: np.ndarray,
        lgf_data: np.ndarray,
        tStep: int,
        mode: bool,
    ) -> None:
        """Update image plot artists with the latest grid and chemical arrays."""
        cellPlot.set_data(cell_data)
        sgfPlot.set_data(sgf_data)
        lgfPlot.set_data(lgf_data)

        cellsFigure.canvas.draw_idle()
        cellsFigure.canvas.flush_events()


def FitvsNnodesPlot(statsFile: str) -> None:
    """Plot maximum fitness against the number of neural network nodes."""
    var_space = 5
    n_gen = 10
    with open(f"stats/{statsFile}", "r", encoding="utf-8") as data_file:
        stats_array = np.loadtxt(data_file, delimiter=",").reshape(var_space, n_gen, 2)

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    node_list = [8, 10, 15, 20, 25]
    ax.set_xlabel("number of nodes")
    ax.set_ylabel("fitness of best individual")
    ax.set_xticks(node_list)
    ax.set_ylim([0, 1])
    ax.set_title("Max fitness vs number of nodes")
    ax.plot(node_list, stats_array[:, 0, 0], color="b", linestyle="dashed", label="1st gen")
    ax.plot(node_list, stats_array[:, 9, 0], color="g", linestyle="dashed", label="10th gen")
    ax.legend(loc="best")
    plt.show()


def FitvsnGenPlot(statsFile: str) -> None:
    """Plot average and maximum fitness evolution over generations across node counts."""
    var_space = 5
    n_gen = 10
    with open(f"stats/{statsFile}", "r", encoding="utf-8") as data_file:
        stats_array = np.loadtxt(data_file, delimiter=",").reshape(var_space, n_gen, 2)

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    gen_list = np.arange(1, n_gen + 1)
    ax.set_xlabel("number of generations")
    ax.set_ylabel("fitness")
    ax.set_xticks(gen_list)
    ax.set_title("change in fitness over generations")

    ax.plot(gen_list, stats_array[0, :, 1], "r--", label="average fitness")
    ax.plot(gen_list, stats_array[1, :, 1], "b--", label="20 nodes")
    ax.plot(gen_list, stats_array[2, :, 1], "k--", label="15 nodes")
    ax.plot(gen_list, stats_array[3, :, 1], "y--", label="10 nodes")
    ax.plot(gen_list, stats_array[4, :, 1], "g--", label="8 nodes")

    ax.scatter(gen_list, stats_array[0, :, 0], marker=",", c="r", label="max fitness")
    ax.scatter(gen_list, stats_array[1, :, 0], marker="o", c="b", label="max fitness 20")
    ax.scatter(gen_list, stats_array[2, :, 0], marker="*", c="k", label="max fitness 15")
    ax.scatter(gen_list, stats_array[3, :, 0], marker="+", c="y", label="max fitness 10")
    ax.scatter(gen_list, stats_array[4, :, 0], marker="_", c="g", label="max fitness 8")
    ax.legend(loc="best")
    plt.show()


def benchPlot(benchFile: str) -> None:
    """Plot generation runtime benchmarks."""
    var_space = 5
    with open(f"benchmarks/{benchFile}", "r", encoding="utf-8") as data_file:
        bench_array = np.loadtxt(data_file, delimiter=",")

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    run_list = np.arange(0, var_space)
    x_ticks = np.arange(0, 26, 5)
    width = 0.25

    ax.set_xlabel("number of generations")
    ax.set_ylabel("time (s)")
    ax.set_xticks(run_list + (width / 2.0))
    ax.set_xticklabels(x_ticks)
    ax.set_title("benchmarks for each generation number")

    ax.bar(
        run_list,
        bench_array[run_list, 1],
        width,
        align="center",
        alpha=0.85,
        color="g",
        label="avg time per generation",
    )
    ax.bar(
        run_list + width,
        bench_array[run_list, 0],
        width,
        align="center",
        alpha=0.85,
        color="r",
        label="total time per generation",
    )
    ax.legend(loc="best")
    plt.show()


def ShowNetwork(dataFile: str, ind: int = 460) -> None:
    """Render a directed NetworkX graph from a saved population CSV file."""
    csv_file = f"populations/{dataFile}.csv"
    w_matrix = tools.GetrNN(csv_file, ind)
    graph = nx.from_numpy_array(w_matrix, create_using=nx.DiGraph)

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle("Recurrent Neural Network")
    nx.draw(graph, pos=nx.spring_layout(graph), ax=ax, with_labels=True)
    plt.show()


def Histogram(fitArray: str) -> None:
    """Plot a histogram of population fitness values from a CSV file."""
    with open(fitArray, "r", encoding="utf-8") as data_file:
        fit_data = np.loadtxt(data_file, delimiter=",")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Fitness")
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_title("Fitness distribution")
    ax.hist(fit_data[0], bins=10, range=(0, 1))
    plt.show()


def MapPlot(mapFile: str) -> None:
    """Plot a discrete state map from a CSV file in maps/."""
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    file_name = f"maps/{mapFile}.csv"
    with open(file_name, "r", encoding="utf-8") as data_file:
        network_map = np.loadtxt(data_file, delimiter=",")

    c_map = ListedColormap(["y", "g", "r", "b", "w"])
    map_plot = ax.imshow(
        network_map,
        origin="lower",
        cmap=c_map,
        interpolation="none",
        vmin=0,
        vmax=4,
    )
    fig.colorbar(map_plot, ax=ax, ticks=[], orientation="horizontal")
    plt.show()


if __name__ == "__main__":
    data_file = sys.argv[1]
    MapPlot(data_file)
