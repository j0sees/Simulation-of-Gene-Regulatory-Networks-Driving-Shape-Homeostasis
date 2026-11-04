"""Generate Erdős-Rényi, Watts-Strogatz, and Barabási-Albert graph topologies for GRN studies."""

from __future__ import annotations

import csv
from datetime import datetime as dt
import os
import sys
import networkx as nx
import numpy as np

SUPPORTED_TOPOLOGIES = ("erdos-renyi", "watts-strogatz", "barabasi-albert")


def generate_graph_matrix(
    n_nodes: int,
    topology: str = "erdos-renyi",
    param: float = 0.3,
) -> np.ndarray:
    """
    Generate a single weighted adjacency matrix for the requested graph topology.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network.
    topology : str
        One of 'erdos-renyi', 'watts-strogatz', or 'barabasi-albert'.
    param : float
        Connection/rewiring probability in (0, 1] for Erdős-Rényi and Watts-Strogatz,
        or edge density scaling factor for Barabási-Albert.

    Returns
    -------
    numpy.ndarray
        Weighted n_nodes x n_nodes adjacency matrix with non-zero weights in [-1, 1].
    """
    topo = topology.lower().strip()
    if topo == "erdos-renyi":
        graph = nx.erdos_renyi_graph(n_nodes, float(param), directed=True)
    elif topo == "watts-strogatz":
        k_neighbours = max(2, min(n_nodes - 1, 4))
        graph = nx.watts_strogatz_graph(n_nodes, k_neighbours, float(param))
    elif topo == "barabasi-albert":
        m_edges = max(1, min(n_nodes - 1, int(round(param * (n_nodes - 1)))))
        graph = nx.barabasi_albert_graph(n_nodes, m_edges)
    else:
        raise ValueError(
            f"Unsupported topology '{topology}'. Choose from {SUPPORTED_TOPOLOGIES}."
        )

    w_matrix = nx.to_numpy_array(graph, dtype=np.float64)
    non_zero = w_matrix == 1
    w_matrix[non_zero] = np.random.uniform(-1.0, 1.0, size=int(np.sum(non_zero)))
    return w_matrix


def generate_graph_ensemble(
    comment_string: str,
    n_runs: int,
    topology: str = "erdos-renyi",
    n_nodes: int = 25,
    output_dir: str = ".",
) -> tuple[str, str]:
    """
    Generate and save an ensemble of random graphs across 10 parameter values.

    Returns
    -------
    tuple[str, str]
        Paths to (runs_log_file, population_csv_file).
    """
    node_prob = np.linspace(0.1, 1.0, 10)
    array_container = np.zeros((len(node_prob) * n_runs, n_nodes**2))

    info_file_id = f"{dt.now():%Y%m%d_%H%M%S_%f}"
    runs_dir = os.path.join(output_dir, "runs")
    pops_dir = os.path.join(output_dir, "populations")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(pops_dir, exist_ok=True)

    runs_main_file = os.path.join(runs_dir, f"run_graphs_{info_file_id}.log")

    with open(runs_main_file, "a", encoding="utf-8") as logfile:
        logfile.write(
            f"##################\nRandom graph generator run\n"
            f"Topology: {topology}\n# {comment_string}\n\n"
        )

    graph_counter = 0
    for prob in node_prob:
        for rep in range(n_runs):
            w_matrix = generate_graph_matrix(n_nodes, topology=topology, param=float(prob))
            ind_info = f"n{n_nodes:02d}P{prob:.2f}r{rep:02d}"
            with open(runs_main_file, "a", encoding="utf-8") as logfile:
                logfile.write(f"{graph_counter}: Info {ind_info}\n")
                array_container[graph_counter, :] = w_matrix.ravel()

            graph_counter += 1

    pop_file_id = f"{dt.now():%Y%m%d_%H%M%S_%f}"
    network_file_name = os.path.join(pops_dir, f"{pop_file_id}.csv")

    with open(runs_main_file, "a", encoding="utf-8") as logfile:
        logfile.write(f"network file\n{pop_file_id}\n")

    with open(network_file_name, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(array_container)

    return runs_main_file, network_file_name


if __name__ == "__main__":
    comment_arg = sys.argv[1]
    runs_arg = int(sys.argv[2])
    topology_arg = sys.argv[3] if len(sys.argv) > 3 else "erdos-renyi"
    generate_graph_ensemble(comment_arg, runs_arg, topology=topology_arg)
