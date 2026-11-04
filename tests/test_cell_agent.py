"""Unit tests for the cellular automata cell agent in core/cell_agent.py."""

from __future__ import annotations

import numpy as np

from core.cell_agent import cell
from core.tools import flatList


def _make_empty_grid(size: int) -> flatList:
    """Create a 2D flatList lattice initialized with zeros."""
    arr = np.zeros((size, size), dtype=np.float64)
    return flatList([flatList(arr[r, :]) for r in range(size)])


def test_cell_initialization_and_sense() -> None:
    """Cell agent should initialize at the given coordinates and read local SGF/LGF."""
    n_nodes = 8
    w_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    agent = cell(2, 3, w_matrix, n_nodes)

    assert agent.yPos == 2
    assert agent.xPos == 3
    assert agent.state == "Quiet"
    assert not agent.amidead

    chem_grid = np.zeros((5, 5, 2), dtype=np.float64)
    chem_grid[2, 3, 0] = 0.42
    chem_grid[2, 3, 1] = 1.75

    sgf, lgf = agent.Sense(chem_grid)
    assert np.isclose(sgf, 0.42)
    assert np.isclose(lgf, 1.75)


def test_cell_actions_quiet_move_split_die() -> None:
    """Cell actions (Quiet, Move, Split, Die) should update lattice states and cell list."""
    size = 5
    n_nodes = 8
    w_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    # 1. Quiet action
    grid = _make_empty_grid(size)
    agent = cell(2, 2, w_matrix, n_nodes)
    agent.Quiet(grid)
    assert grid[2][2] == 1
    assert agent.quietCounter == 1

    # 2. Move action to preferred neighbor
    grid = _make_empty_grid(size)
    agent = cell(2, 2, w_matrix, n_nodes)
    grid[2][2] = 1
    agent.orientation = [1, 2]  # North neighbor
    agent.Move(grid)
    assert (agent.yPos, agent.xPos) == (1, 2)
    assert grid[1][2] == 2
    assert grid[2][2] == 0

    # 3. Split action creates a daughter cell
    grid = _make_empty_grid(size)
    agent = cell(2, 2, w_matrix, n_nodes)
    grid[2][2] = 1
    agent.orientation = [2, 3]  # East neighbor
    cell_list = [agent]
    agent.Split(grid, cell_list)
    assert len(cell_list) == 2
    assert grid[2][2] == 3
    assert grid[2][3] == 1

    # 4. Die action marks cell as dead and clears grid spot
    agent.Die(grid)
    assert agent.amidead
    assert grid[2][2] == 0

