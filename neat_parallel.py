"""Parallel NEAT neuroevolution runner for the cellular automata simulation."""

from __future__ import annotations

import os
import neat
import numpy as np

from core.tools import (
    GenerateIMatrix,
    GenerateTMatrix,
    GetStructure,
    LGFDiffEq,
    SGFDiffEq,
    flatList,
)
from neat_cell_agent import cell

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genome(genome, config) -> float:
    """Evaluate a single genome on the 2-input XOR benchmark."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    error = 4.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
    return error


def EvaluateIndividual(genome, config) -> float:
    """Evaluate a NEAT genome using the cellular automata shape homeostasis simulation."""
    n_lattice = 50
    time_steps = 200

    np_cell_grid = np.zeros((n_lattice, n_lattice))
    cell_grid = flatList([flatList(np_cell_grid[r, :]) for r in range(n_lattice)])
    chem_grid = np.zeros((n_lattice, n_lattice, 2))
    halfway_struct = np.zeros((n_lattice, n_lattice))
    final_struct = np.ones((n_lattice, n_lattice))

    ix = n_lattice // 2
    iy = n_lattice // 2
    cell_list = [cell(ix, iy, genome, config)]
    cell_grid[ix][iy] = 1

    delta_t = 1.0
    delta_r = 1.0
    delta_s = 0.5
    delta_l = 0.1
    diff_const = 1.0
    t_matrix = GenerateTMatrix(n_lattice)
    i_matrix = GenerateIMatrix(n_lattice)

    for i_time in range(time_steps):
        sigma_m = np.zeros((n_lattice, n_lattice))
        lambda_m = np.zeros((n_lattice, n_lattice))

        tmp_cell_list = list(cell_list)
        while len(tmp_cell_list) > 0:
            rnd_idx = np.random.randint(len(tmp_cell_list))
            current_cell = tmp_cell_list[rnd_idx]
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

            del tmp_cell_list[rnd_idx]

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

        if len(cell_list) == 0:
            break
        if i_time == (time_steps // 2) - 1:
            if len(cell_list) <= int((n_lattice**2) * 0.01):
                break
            halfway_struct = np.array(cell_grid)
        elif i_time == time_steps - 1 and len(cell_list) < n_lattice**2:
            final_struct = np.array(cell_grid)

    halfway_struct = GetStructure(halfway_struct, n_lattice)
    final_struct = GetStructure(final_struct, n_lattice)
    tot_sum = float(np.sum(halfway_struct != final_struct))
    return 1.0 - (tot_sum / (n_lattice**2))


def run(config_file: str) -> None:
    """Load NEAT configuration and run parallel genome evaluation."""
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    evaluator = neat.ParallelEvaluator(3, EvaluateIndividual)
    winner = population.run(evaluator.evaluate)

    print(f"\nBest genome:\n{winner!s}")


if __name__ == "__main__":
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "config", "config-ca")
    run(config_path)
