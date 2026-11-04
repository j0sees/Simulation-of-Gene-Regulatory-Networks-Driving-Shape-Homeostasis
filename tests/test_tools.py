"""Unit tests for core mathematical utilities and PDE solvers in core/tools.py."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from core.tools import (
    CheckifOccupied,
    CheckifPreferred,
    GenerateIMatrix,
    GenerateTMatrix,
    GetStructure,
    GetrNN,
    LGFDiffEq,
    RecurrentNeuralNetwork,
    SGFDiffEq,
    flatList,
)


def test_flat_list_rejects_negative_indices() -> None:
    """flatList should allow non-negative indices and raise IndexError on negative indices."""
    grid_row = flatList([10, 20, 30])
    assert grid_row[0] == 10
    assert grid_row[2] == 30
    with pytest.raises(IndexError):
        _ = grid_row[-1]


def test_check_if_occupied_and_preferred() -> None:
    """CheckifOccupied and CheckifPreferred should evaluate grid coordinates accurately."""
    grid = [[0, 1], [2, 0]]
    assert not CheckifOccupied(0, 0, grid)
    assert CheckifOccupied(1, 0, grid)
    assert CheckifOccupied(0, 1, grid)

    assert CheckifPreferred(3, 4, 3, 4)
    assert not CheckifPreferred(3, 4, 4, 3)


def test_generate_matrices() -> None:
    """GenerateIMatrix and GenerateTMatrix should construct valid finite-difference matrices."""
    size = 5
    i_mat = GenerateIMatrix(size)
    t_mat = GenerateTMatrix(size)

    assert i_mat.shape == (size, size)
    assert np.allclose(i_mat, np.eye(size))

    assert t_mat.shape == (size, size)
    assert t_mat[0, 0] == -1.0
    assert t_mat[size - 1, size - 1] == -1.0
    assert t_mat[2, 2] == -2.0
    assert t_mat[2, 3] == 1.0
    assert t_mat[3, 2] == 1.0


def test_sgf_and_lgf_diffusion_equations() -> None:
    """SGFDiffEq and LGFDiffEq should update chemical concentrations without NaNs."""
    size = 6
    s_mat = np.zeros((size, size), dtype=np.float64)
    sigma_mat = np.ones((size, size), dtype=np.float64) * 0.2

    s_next = SGFDiffEq(s_mat, sigma_mat, deltaS=0.5, deltaT=1.0)
    assert s_next.shape == (size, size)
    assert np.allclose(s_next, 0.2)

    i_mat = GenerateIMatrix(size)
    t_mat = GenerateTMatrix(size)
    l_mat = np.zeros((size, size), dtype=np.float64)
    lambda_mat = np.zeros((size, size), dtype=np.float64)
    lambda_mat[size // 2, size // 2] = 1.0

    l_next = LGFDiffEq(
        i_mat, t_mat, l_mat, lambda_mat, deltaL=0.1, deltaT=1.0, deltaR=1.0, D=1.0
    )
    assert l_next.shape == (size, size)
    assert not np.isnan(l_next).any()
    assert l_next[size // 2, size // 2] > 0.0


def test_recurrent_neural_network_bounds() -> None:
    """RecurrentNeuralNetwork activations should remain strictly within (0, 1)."""
    n_nodes = 8
    inputs = np.zeros(n_nodes, dtype=np.float64)
    inputs[0] = 0.5
    inputs[1] = 1.2
    w_matrix = np.ones((n_nodes, n_nodes), dtype=np.float64) * 0.25
    v_state = np.zeros(n_nodes, dtype=np.float64)

    updated_v = RecurrentNeuralNetwork(inputs, w_matrix, v_state)
    assert updated_v.shape == (n_nodes,)
    assert np.all((updated_v > 0.0) & (updated_v < 1.0))


def test_get_structure_and_get_rnn(tmp_path: Path) -> None:
    """GetStructure and GetrNN should convert grids and load saved weight matrices."""
    raw_grid = np.array([[0, 2], [3, 0]])
    binary = GetStructure(raw_grid, 2)
    assert np.array_equal(binary, np.array([[0.0, 1.0], [1.0, 0.0]]))

    csv_path = tmp_path / "pop.csv"
    weights = np.arange(16, dtype=np.float64).reshape(1, 16)
    np.savetxt(csv_path, weights, delimiter=",")

    loaded = GetrNN(str(csv_path), 0)
    assert loaded.shape == (4, 4)
    assert np.allclose(loaded, weights.reshape(4, 4))

