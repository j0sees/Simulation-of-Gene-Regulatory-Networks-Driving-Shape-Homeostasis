"""Unit tests for network topology generators and Plotly figures in visualization/web_app.py."""

from __future__ import annotations

import numpy as np
import pytest

from visualization.web_app import create_cell_grid_plot, generate_weight_matrix


@pytest.mark.parametrize(
    "network_type",
    ["Random", "Erdős-Rényi", "Watts-Strogatz", "Barabási-Albert"],
)
def test_generate_weight_matrix_topologies(network_type: str) -> None:
    """generate_weight_matrix should return an n_nodes x n_nodes matrix in [-1, 1]."""
    n_nodes = 12
    w_mat = generate_weight_matrix(n_nodes, network_type)
    assert w_mat.shape == (n_nodes, n_nodes)
    assert np.all((w_mat >= -1.0) & (w_mat <= 1.0))


def test_create_cell_grid_plot_traces() -> None:
    """create_cell_grid_plot should build a 3-panel Plotly figure (Cells, SGF, LGF)."""
    size = 10
    cell_grid = np.zeros((size, size), dtype=np.float64)
    sgf_grid = np.zeros((size, size), dtype=np.float64)
    lgf_grid = np.zeros((size, size), dtype=np.float64)

    fig = create_cell_grid_plot(cell_grid, sgf_grid, lgf_grid)
    assert len(fig.data) == 3
    assert [trace.name for trace in fig.data] == ["Cells", "SGF", "LGF"]
