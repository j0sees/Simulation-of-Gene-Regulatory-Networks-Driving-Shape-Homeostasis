"""Core mathematical and utility functions for the GRN cellular automata simulation."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func


class flatList(list):
    """
    A list subclass that raises IndexError for negative indices.

    Prevents circular wrap-around indexing on cellular automata grid boundaries.
    """

    def __getitem__(self, index: int):
        if index < 0:
            raise IndexError("list index out of range")
        return super().__getitem__(index)


def CheckifOccupied(xCoord: int, yCoord: int, grid) -> bool:
    """
    Check if a grid position is occupied by a cell.

    Parameters
    ----------
    xCoord, yCoord : int
        Grid coordinates to check.
    grid : list or numpy.ndarray
        2D grid where values > 0 indicate occupied positions.

    Returns
    -------
    bool
        True if the position is occupied, False otherwise.
    """
    return bool(grid[yCoord][xCoord] > 0)


def CheckifPreferred(xOri: int, yOri: int, xCoord: int, yCoord: int) -> bool:
    """
    Check if a position matches the preferred orientation direction.

    Parameters
    ----------
    xOri, yOri : int
        Preferred orientation coordinates.
    xCoord, yCoord : int
        Candidate position coordinates.

    Returns
    -------
    bool
        True if the candidate position matches the preferred direction.
    """
    return xCoord == xOri and yCoord == yOri


@njit
def SGFDiffEq(
    s_matrix: np.ndarray, sigma_matrix: np.ndarray, deltaS: float, deltaT: float
) -> np.ndarray:
    """
    Update SGF (Short-range Growth Factor) concentrations using matrix operations.

    Implements the reaction equation for SGF: ds/dt = sigma - delta_s * s
    where sigma is the production rate and delta_s is the decay rate.
    """
    return s_matrix + deltaT * (sigma_matrix - deltaS * s_matrix)


@njit
def LGFDiffEq(
    i_matrix: np.ndarray,
    t_matrix: np.ndarray,
    l_matrix: np.ndarray,
    lambda_matrix: np.ndarray,
    deltaL: float,
    deltaT: float,
    deltaR: float,
    D: float,
) -> np.ndarray:
    """
    Update LGF (Long-range Growth Factor) concentrations using a Crank-Nicolson scheme.

    Implements the reaction-diffusion PDE:
    dl/dt = D * nabla^2(l) + lambda - delta_l * l
    """
    alpha = D * deltaT / (deltaR**2)
    f = (deltaT / 2.0) * (lambda_matrix - deltaL * l_matrix)
    g = np.linalg.inv(i_matrix - (alpha / 2.0) * t_matrix)
    h = i_matrix + (alpha / 2.0) * t_matrix
    l_half_step = g @ (l_matrix @ h + f)
    f = (deltaT / 2.0) * (lambda_matrix - deltaL * l_half_step)
    l_t_step = ((h @ l_half_step) + f) @ g
    return l_t_step


def GenerateTMatrix(size: int) -> np.ndarray:
    """Generate the discrete 1D Laplacian T-matrix for LGF spatial discretization."""
    t_matrix = np.zeros((size, size), dtype=np.float64)
    for ix in range(size - 1):
        t_matrix[ix, ix] = -2.0
        t_matrix[ix, ix + 1] = 1.0
        t_matrix[ix + 1, ix] = 1.0
    t_matrix[0, 0] = -1.0
    t_matrix[size - 1, size - 1] = -1.0
    return t_matrix


def GenerateIMatrix(size: int) -> np.ndarray:
    """Generate an identity matrix of size x size."""
    return np.eye(size, dtype=np.float64)


@njit
def RecurrentNeuralNetwork(
    inputs: np.ndarray, wMatrix: np.ndarray, V: np.ndarray
) -> np.ndarray:
    """Update Recurrent Neural Network activation values using a sigmoid transfer function."""
    bj = np.dot(wMatrix, V) - inputs
    for ix in range(len(bj)):
        V[ix] = 1.0 / (1.0 + np.exp(-2.0 * bj[ix]))
    return V


def GetStructure(cell_array, nLattice: int) -> np.ndarray:
    """Return a binary occupancy matrix (1 where cell_array != 0, else 0)."""
    return (np.asarray(cell_array) != 0).astype(np.float64)


def GetrNN(csvFile: str, ind: int) -> np.ndarray:
    """Load an individual's recurrent neural network weight matrix from a CSV file."""
    with open(csvFile, "r", encoding="utf-8") as csvfile:
        best_individuals = np.loadtxt(csvfile, delimiter=",")
    row = best_individuals if best_individuals.ndim == 1 else best_individuals[ind, :]
    n_nodes = int(np.sqrt(len(row)))
    return np.array(row.reshape(n_nodes, n_nodes), dtype=np.float64)
