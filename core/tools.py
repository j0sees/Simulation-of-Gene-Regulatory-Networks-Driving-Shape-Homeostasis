import csv
import numpy as np
from scipy import linalg

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func


# List without joint ends
# https://stackoverflow.com/questions/29710249/python-force-list-index-out-of-range-exception
class flatList(list):
    """
    A list subclass that raises IndexError for negative indices.

    This prevents circular indexing behavior that could cause issues
    in the cellular automata grid operations.
    """

    def __getitem__(self, index):
        if index < 0:
            raise IndexError("list index out of range")
        return super().__getitem__(index)


def CheckifOccupied(xCoord, yCoord, grid):
    """
    Check if a grid position is occupied by a cell.

    Parameters:
    -----------
    xCoord, yCoord : int
        Grid coordinates to check
    grid : list or numpy.ndarray
        2D grid where values > 0 indicate occupied positions

    Returns:
    --------
    bool
        True if position is occupied, False otherwise
    """
    return grid[yCoord][xCoord] > 0


def CheckifPreferred(xOri, yOri, xCoord, yCoord):
    """
    Check if a position matches the preferred orientation direction.

    Parameters:
    -----------
    xOri, yOri : int
        Preferred orientation coordinates
    xCoord, yCoord : int
        Position coordinates to check

    Returns:
    --------
    bool
        True if position matches preferred direction, False otherwise
    """
    return xCoord == xOri and yCoord == yOri


@njit
def SGFDiffEq(s_matrix, sigma_matrix, deltaS, deltaT):
    """
    Update SGF (Short-range Growth Factor) concentrations using matrix operations.

    Implements the diffusion equation for SGF: ds/dt = sigma - delta_s * s
    where sigma is production rate and delta_s is decay rate.

    Parameters:
    -----------
    s_matrix : numpy.ndarray
        Current SGF concentration matrix
    sigma_matrix : numpy.ndarray
        SGF production matrix
    deltaS : float
        SGF decay rate
    deltaT : float
        Time step size

    Returns:
    --------
    numpy.ndarray
        Updated SGF concentration matrix
    """
    return s_matrix + deltaT * (sigma_matrix - deltaS * s_matrix)


@njit
def LGFDiffEq(i_matrix, t_matrix, l_matrix, lambda_matrix, deltaL, deltaT, deltaR, D):
    """
    Update LGF (Long-range Growth Factor) concentrations using matrix operations.

    Implements the diffusion equation for LGF with spatial diffusion:
    dl/dt = D * nabla^2(l) + lambda - delta_l * l
    where D is diffusion constant, lambda is production rate, and delta_l is decay rate.

    Uses a semi-implicit Crank-Nicolson scheme for numerical stability.

    Parameters:
    -----------
    i_matrix : numpy.ndarray
        Identity matrix
    t_matrix : numpy.ndarray
        T-matrix for spatial discretization
    l_matrix : numpy.ndarray
        Current LGF concentration matrix
    lambda_matrix : numpy.ndarray
        LGF production matrix
    deltaL : float
        LGF decay rate
    deltaT : float
        Time step size
    deltaR : float
        Spatial step size
    D : float
        Diffusion constant

    Returns:
    --------
    numpy.ndarray
        Updated LGF concentration matrix
    """
    alpha = D * deltaT / (deltaR**2)
    f = (deltaT / 2.0) * (lambda_matrix - deltaL * l_matrix)
    g = np.linalg.inv(i_matrix - (alpha / 2.0) * t_matrix)
    h = i_matrix + (alpha / 2.0) * t_matrix
    l_halftStep = g @ (l_matrix @ h + f)
    f = (deltaT / 2.0) * (lambda_matrix - deltaL * l_halftStep)
    l_tStep = ((h @ l_halftStep) + f) @ g
    return l_tStep


def GenerateTMatrix(size):
    """
    Generate the T-matrix for spatial discretization in LGF diffusion.

    The T-matrix represents the discrete Laplacian operator for finite differences.

    Parameters:
    -----------
    size : int
        Size of the square matrix to generate

    Returns:
    --------
    numpy.ndarray
        T-matrix for LGF diffusion calculations
    """
    t_matrix = np.zeros((size, size), dtype=np.float64)
    for ix in range(size - 1):
        t_matrix[ix, ix] = -2.0
        t_matrix[ix, ix + 1] = 1.0
        t_matrix[ix + 1, ix] = 1.0
    t_matrix[0, 0] = -1.0
    t_matrix[size - 1, size - 1] = -1.0
    return t_matrix


def GenerateIMatrix(size):
    """
    Generate an identity matrix of specified size.

    Parameters:
    -----------
    size : int
        Size of the square matrix to generate

    Returns:
    --------
    numpy.ndarray
        Identity matrix of size x size
    """
    return np.eye(size, dtype=np.float64)


@njit
def RecurrentNeuralNetwork(inputs, wMatrix, V):
    """Update Recurrent Neural Network activation values."""
    bj = np.dot(wMatrix, V) - inputs
    for ix in range(len(bj)):
        V[ix] = 1.0 / (1.0 + np.exp(-2.0 * bj[ix]))
    return V


def GetStructure(cell_array, nLattice):
    """Return a binary occupancy matrix (1 where cell_array != 0, else 0)."""
    return (np.asarray(cell_array) != 0).astype(np.float64)


def GetrNN(csvFile, ind):
    """Load an individual's recurrent neural network weight matrix from a CSV file."""
    with open(csvFile, "r", encoding="utf-8") as csvfile:
        bestIndividuals = np.loadtxt(csvfile, delimiter=",")
    if bestIndividuals.ndim == 1:
        row = bestIndividuals
    else:
        row = bestIndividuals[ind, :]
    nNodes = int(np.sqrt(len(row)))
    return np.array(row.reshape(nNodes, nNodes), dtype=np.float64)
