"""Evaluate fitness across a population of saved Gene Regulatory Network topologies."""

from __future__ import annotations

from contextlib import contextmanager
import multiprocessing as mp
import sys
import matplotlib.pyplot as plt
import numpy as np

import evolution.main_ga as main_ga


def partialEval(ind: int) -> float:
    """Evaluate a single network index using default simulation parameters."""
    time_steps = 200
    n_nodes = 25
    n_lattice = 50
    return EvaluateIndividual(time_steps, n_nodes, n_lattice, ind)


def EvaluateIndividual(timeSteps: int, nNodes: int, nLattice: int, ind: int) -> float:
    """Run the cellular automata simulation and compute the shape homeostasis fitness."""
    w_matrix = networkArrays[ind]
    delta_m = main_ga.sim(w_matrix, timeSteps, 0, nNodes, ind, nLattice, True)
    tot_sum = float(np.sum(delta_m))
    return 1.0 - (tot_sum / (nLattice**2))


@contextmanager
def poolcontext(*args, **kwargs):
    """Context manager that terminates a multiprocessing pool on exit."""
    pool = mp.Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.terminate()


if __name__ == "__main__":
    file_name = sys.argv[1]
    csv_file = f"populations/{file_name}.csv"
    n_procs = 20

    with open(csv_file, "r", encoding="utf-8") as csvfile:
        network_container = np.loadtxt(csvfile, delimiter=",")
    n_nets, n_genes = network_container.shape
    side = int(np.sqrt(n_genes))
    networkArrays = network_container.reshape(n_nets, side, side)

    index_list = list(range(n_nets))
    with poolcontext(processes=n_procs) as pool:
        fitness_array = pool.map(partialEval, index_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("fitness")
    ax.set_title("histogram")
    ax.hist(fitness_array)
    plt.savefig("test.png")
