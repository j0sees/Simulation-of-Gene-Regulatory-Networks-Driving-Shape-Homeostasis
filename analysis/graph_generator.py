"""Generate random Erdős-Rényi graph topologies for Gene Regulatory Network experiments."""

from __future__ import annotations

import csv
from datetime import datetime as dt
import os
import sys
import networkx as nx
import numpy as np


if __name__ == "__main__":
    comment_string = sys.argv[1]
    n_runs = int(sys.argv[2])

    n_nodes = 25
    node_prob = np.linspace(0.1, 1.0, 10)
    array_container = np.zeros((len(node_prob) * n_runs, n_nodes**2))

    info_file_id = f"{dt.now():%Y%m%d_%H%M%S_%f}"
    os.makedirs("runs", exist_ok=True)
    os.makedirs("populations", exist_ok=True)
    runs_main_file = f"runs/run_graphs_{info_file_id}.log"

    with open(runs_main_file, "a", encoding="utf-8") as logfile:
        logfile.write(
            f"##################\nRandom graph generator run\n"
            f"Erdős-Rényi Graphs\n# {comment_string}\n\n"
        )

    graph_counter = 0
    for prob in node_prob:
        for rep in range(n_runs):
            graph = nx.erdos_renyi_graph(n_nodes, prob, directed=True)
            w_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            non_zero = w_matrix == 1
            w_matrix[non_zero] = np.random.uniform(-1.0, 1.0, size=int(np.sum(non_zero)))

            ind_info = f"n{n_nodes:02d}P{prob:.2f}r{rep:02d}"
            with open(runs_main_file, "a", encoding="utf-8") as logfile:
                logfile.write(f"{graph_counter}: Info {ind_info}\n")
                array_container[graph_counter, :] = w_matrix.ravel()

            graph_counter += 1

    pop_file_id = f"{dt.now():%Y%m%d_%H%M%S_%f}"
    network_file_name = f"populations/{pop_file_id}.csv"

    with open(runs_main_file, "a", encoding="utf-8") as logfile:
        logfile.write(f"network file\n{pop_file_id}\n")

    with open(network_file_name, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(array_container)
