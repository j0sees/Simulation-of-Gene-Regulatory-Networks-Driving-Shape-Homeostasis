"""Parallel Genetic Algorithm for evolving Gene Regulatory Networks driving shape homeostasis."""

from __future__ import annotations

import csv
import ctypes
import multiprocessing as mp
import os
import sys
import time
import numpy as np

from core.cell_agent import cell
from core.tools import (
    GenerateIMatrix,
    GenerateTMatrix,
    GetStructure,
    LGFDiffEq,
    SGFDiffEq,
    flatList,
)
from evolution.tools_ga import Crossover, Mutate
from visualization.plot import Environment


def sim(
    wMatrix: np.ndarray,
    timeSteps: int,
    iGen: int,
    nNodes: int,
    individual: int,
    nLattice: int,
    mode: bool,
) -> np.ndarray | None:
    """
    Run a cellular automata simulation for fitness evaluation or visualization.

    Parameters
    ----------
    wMatrix : numpy.ndarray
        Recurrent neural network weight matrix.
    timeSteps : int
        Number of simulation time steps.
    iGen : int
        Current generation index.
    nNodes : int
        Number of nodes in the neural network.
    individual : int
        Individual index within the population.
    nLattice : int
        Dimension of the square lattice grid.
    mode : bool
        True: evaluate fitness and return the binary structural difference matrix.
        False: display simulation visually.
    """
    np_cell_grid = np.zeros((nLattice, nLattice))
    semi_flat_grid = [flatList(np_cell_grid[r, :]) for r in range(nLattice)]
    cell_grid = flatList(semi_flat_grid)
    chem_grid = np.zeros((nLattice, nLattice, 2))
    halfway_struct = np.zeros((nLattice, nLattice))
    final_struct = np.ones((nLattice, nLattice))

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

    if not mode:
        (
            cells_figure,
            cells_subplot,
            sgf_subplot,
            lgf_subplot,
            cell_plot,
            sgf_plot,
            lgf_plot,
        ) = Environment.CellsGridFigure(nLattice, mode)

    start_time_main_loop = time.time()

    while i_time < timeSteps:
        sigma_m = np.zeros((nLattice, nLattice))
        lambda_m = np.zeros((nLattice, nLattice))

        tmp_cell_list = list(cell_list)
        while len(tmp_cell_list) > 0:
            rnd_cell = np.random.randint(len(tmp_cell_list))
            current_cell = tmp_cell_list[rnd_cell]
            current_cell.border = nLattice

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

        if mode:
            if len(cell_list) == 0:
                halfway_struct = np.zeros((nLattice, nLattice))
                final_struct = np.ones((nLattice, nLattice))
                break
            if i_time == int(timeSteps / 2) - 1:
                if len(cell_list) <= int((nLattice**2) * 0.01):
                    halfway_struct = np.zeros((nLattice, nLattice))
                    final_struct = np.ones((nLattice, nLattice))
                    break
                halfway_struct = np.array(cell_grid)
            elif i_time == timeSteps - 1:
                if len(cell_list) == nLattice**2:
                    halfway_struct = np.zeros((nLattice, nLattice))
                    final_struct = np.ones((nLattice, nLattice))
                else:
                    final_struct = np.array(cell_grid)
        else:
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

    elapsed = time.time() - start_time_main_loop
    print(f"Avg time looping through cells: {elapsed / timeSteps:.3f} s, total: {elapsed:.3f} s")

    halfway_struct = GetStructure(halfway_struct, nLattice)
    final_struct = GetStructure(final_struct, nLattice)
    delta_matrix = (halfway_struct != final_struct).astype(np.float64)

    if mode:
        return delta_matrix
    return None


def EvaluateIndividual(
    individual: int,
    timeSteps: int,
    iGen: int,
    nNodes: int,
    nLattice: int,
    mode: bool,
) -> np.ndarray:
    """Evaluate a single individual's fitness in the shared population array."""
    w_matrix = population[individual, :].reshape(nNodes, nNodes)
    start_time = time.time()
    delta_m = sim(w_matrix, timeSteps, iGen, nNodes, individual, nLattice, mode)
    secs = time.time() - start_time
    print(f"Proc {os.getpid()}: Time taken to run sim: {secs:.3f} s")

    tot_sum = float(np.sum(delta_m))
    fitness[individual] = 1.0 - (tot_sum / (nLattice**2))
    return fitness


if __name__ == "__main__":
    nProcs = 2
    DEFAULT_VALUE = -1
    popSize = 6
    nNodes = 25
    nGenes = nNodes**2
    crossoverProb = 1.0
    mutationProb = 1.0
    crossMutProb = 0.5
    tournamentSize = 4
    eliteNum = 2
    nOfGenerations = 15
    timeSteps = 200
    nLattice = 50
    mode = True
    fileName = sys.argv[1]
    chunkSize = 1

    generationAvg = 0.0
    contestants = np.zeros((tournamentSize, nGenes))
    fitnessInfo = np.zeros((nOfGenerations, 2))

    print(
        f"Parameters:\nFile name: {fileName}\n"
        f"nProcs = {nProcs}, Population size = {popSize}, nNodes = {nNodes}, "
        f"nLattice = {nLattice}, nGen = {nOfGenerations}, "
        f"Crossover Prob = {crossoverProb}, Mutation prob = {mutationProb}"
    )

    population_base = mp.Array(ctypes.c_float, popSize * nGenes, lock=False)
    population = np.frombuffer(population_base, dtype=ctypes.c_float).reshape(popSize, nGenes)
    np.copyto(population, np.random.rand(popSize, nGenes))

    fitness_base = mp.Array(ctypes.c_float, popSize, lock=False)
    fitness = np.frombuffer(fitness_base, dtype=ctypes.c_float)

    for iGen in range(nOfGenerations):
        start_time_generation = time.time()
        print(f"\nGeneration #{iGen + 1}")

        fitness.fill(DEFAULT_VALUE)
        args = [
            (idx, timeSteps, iGen, nNodes, nLattice, mode) for idx in range(popSize)
        ]

        with mp.Pool(processes=nProcs) as pool:
            pool.starmap(EvaluateIndividual, args, chunkSize)

        sorted_fitness = np.argsort(fitness)
        tempPopulation = np.zeros((popSize, nGenes))

        fitnessInfo[iGen, 0] = np.amax(fitness)
        fitnessInfo[iGen, 1] = np.average(fitness)
        print(
            f"Gen: {iGen + 1}, max fit: {fitness[sorted_fitness[popSize - 1]]:.3f}, "
            f"avg fit: {np.average(fitness):.3f}"
        )

        # Elitism: preserve the top eliteNum individuals
        for i_elit in range(1, eliteNum + 1):
            index = sorted_fitness[popSize - i_elit]
            tempPopulation[i_elit - 1, :] = np.array(population[index, :])

        # Tournament selection over the remaining population
        loopCounter = 0
        while len(sorted_fitness) >= tournamentSize:
            selectedInd = np.random.choice(
                range(len(sorted_fitness)), tournamentSize, replace=False
            )
            selectedInd.sort()

            win_index1 = sorted_fitness[selectedInd[tournamentSize - 1]]
            contestants[0, :] = np.array(population[win_index1, :])
            win_index2 = sorted_fitness[selectedInd[tournamentSize - 2]]
            contestants[1, :] = np.array(population[win_index2, :])

            if np.random.random() >= crossMutProb:
                contestants[2, :], contestants[3, :] = Crossover(
                    contestants[0, :], contestants[1, :], crossoverProb
                )
            else:
                contestants[2, :], contestants[3, :] = Mutate(
                    np.array(contestants[0, :]),
                    np.array(contestants[1, :]),
                    mutationProb,
                )

            for offset, ix in enumerate(selectedInd):
                sorted_fitness = np.delete(sorted_fitness, ix - offset)

            for jk in range(tournamentSize):
                index = eliteNum + (loopCounter * tournamentSize) + jk
                tempPopulation[index] = contestants[jk, :]
            loopCounter += 1

        population = np.array(tempPopulation)

        secs = time.time() - start_time_generation
        generationAvg += secs
        minutes = int(secs / 60)
        rem_secs = 60.0 * ((secs / 60.0) - minutes)
        print(f"Time to complete generation: {minutes} m {rem_secs:.3f} s")

    avg_gen_min = int(generationAvg / nOfGenerations / 60)
    avg_gen_sec = 60.0 * ((generationAvg / nOfGenerations / 60.0) - avg_gen_min)
    print(f"Avg time per generation: {avg_gen_min} m {avg_gen_sec:.3f} s")

    with open(f"{fileName}.csv", "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(population)
