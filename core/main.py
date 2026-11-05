"""Main cellular automata simulation engine."""

from __future__ import annotations

import sys
import time
import matplotlib.pyplot as plt
import numpy as np

from core.cell_agent import cell
from core.tools import (
    GenerateIMatrix,
    GenerateTMatrix,
    GetrNN,
    LGFDiffEq,
    SGFDiffEq,
    flatList,
)
from visualization.plot import Environment


def sim(wMatrix: np.ndarray, timeSteps: int, nNodes: int, nLattice: int, mode: bool) -> None:
    """
    Run a cellular automata simulation with gene regulatory networks.

    Parameters
    ----------
    wMatrix : numpy.ndarray
        Weight matrix for the neural network representing the GRN.
    timeSteps : int
        Number of simulation time steps to run.
    nNodes : int
        Number of nodes in the neural network.
    nLattice : int
        Size of the square lattice grid (nLattice x nLattice).
    mode : bool
        True: run in headless fitness mode (no live display).
        False: run with interactive visualization.
    """
    np_cell_grid = np.zeros((nLattice, nLattice))
    semi_flat_grid = [flatList(np_cell_grid[r, :]) for r in range(nLattice)]
    cell_grid = flatList(semi_flat_grid)
    chem_grid = np.zeros((nLattice, nLattice, 2))

    ix = nLattice // 2
    iy = nLattice // 2
    i_time = 0

    cell_list = [cell(ix, iy, wMatrix, nNodes)]
    cell_grid[ix][iy] = 1

    # SGF/LGF reaction-diffusion parameters
    delta_t = 1.0
    delta_r = 1.0
    delta_s = 0.5
    delta_l = 0.1
    diff_const = 1.0
    t_matrix = GenerateTMatrix(nLattice)
    i_matrix = GenerateIMatrix(nLattice)

    tmp_list_loop_avg = 0.0

    (
        cells_figure,
        cells_subplot,
        sgf_subplot,
        lgf_subplot,
        cell_plot,
        sgf_plot,
        lgf_plot,
    ) = Environment.CellsGridFigure(nLattice, mode)

    print("Time running...")
    start_time_main_loop = time.time()

    while i_time < timeSteps:
        sigma_m = np.zeros((nLattice, nLattice))
        lambda_m = np.zeros((nLattice, nLattice))

        tmp_cell_list = list(cell_list)
        start_time_tmp_loop = time.time()

        while len(tmp_cell_list) > 0:
            rnd_cell = np.random.randint(len(tmp_cell_list))
            current_cell = tmp_cell_list[rnd_cell]

            sgf_reading, lgf_reading = current_cell.Sense(chem_grid)
            current_cell.GenerateStatus(sgf_reading, lgf_reading)

            sigma_m[current_cell.yPos, current_cell.xPos] = current_cell.sgfAmount
            lambda_m[current_cell.yPos, current_cell.xPos] = current_cell.lgfAmount

            if current_cell.state == "Quiet":
                current_cell.Quiet(cell_grid)
            elif current_cell.state == "Split":
                current_cell.Split(cell_grid, cell_list)
            elif current_cell.state == "Move":
                current_cell.Move(cell_grid)
            else:
                current_cell.Die(cell_grid)

            del tmp_cell_list[rnd_cell]

        tmp_list_loop_avg += time.time() - start_time_tmp_loop

        for j_cell in range(len(cell_list) - 1, -1, -1):
            if cell_list[j_cell].amidead:
                del cell_list[j_cell]

        chem_grid[:, :, 0] = SGFDiffEq(chem_grid[:, :, 0], sigma_m, delta_s, delta_t)
        chem_grid[:, :, 1] = LGFDiffEq(
            i_matrix,
            t_matrix,
            chem_grid[:, :, 1],
            lambda_m,
            delta_l,
            delta_t,
            delta_r,
            diff_const,
        )

        Environment.AntGridPlot(
            cell_grid,
            chem_grid,
            nLattice,
            cells_figure,
            cells_subplot,
            sgf_subplot,
            lgf_subplot,
            cell_plot,
            sgf_plot,
            lgf_plot,
            i_time,
            mode,
        )
        i_time += 1

    tmp_list_loop_avg += time.time() - start_time_main_loop
    print(
        f"Avg time looping through cells: {tmp_list_loop_avg / timeSteps:.3f} s, "
        f"total: {tmp_list_loop_avg:.3f} s"
    )


if __name__ == "__main__":
    print("System visualization")
    file_name = sys.argv[1]
    n_nodes = int(sys.argv[2])
    individual = int(sys.argv[3])
    time_steps = 200
    n_lattice = 50
    display_mode = False

    w_matrix = GetrNN(file_name, individual)
    sim(w_matrix, time_steps, n_nodes, n_lattice, display_mode)
    plt.close()
