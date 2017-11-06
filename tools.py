import numpy as np
from scipy import linalg
import csv
#from numba import jit

# Tools
#@jit
#def CheckInBorders(xCoord, yCoord, border):
    #if xCoord < 0 or xCoord > border:     # Check if the neighbour is inbounds on x axis
        #return False
    #elif yCoord < 0 or yCoord > border:    # Check if the neighbour is inbounds on y axis
        #return False
    #else:
        #return True
# CheckInBorders

#@jit
def CheckifOccupied(xCoord, yCoord, grid):
    if grid[xCoord, yCoord][0] > 0:         # if value on grid is 1 (quiet), 2 (moved) or 3 (splitted) then spot is occupied
        return True
    else:                                   # else, value is 0 (empty) or -1 (cell was there before but died) then spot is available
        return False
# CheckifOccupied

#@jit
def CheckifPreferred(xOri, yOri, xCoord, yCoord):
    if xCoord == xOri and yCoord == yOri:
        return True
    else:
        return False
# CheckifPreferred

# SGF dynamics, single value update approach
#@jit
#def SGFDiffEq(s, sigma, deltaS, deltaT):
    #updatedVal = s + deltaT*(sigma - deltaS*s)
    #return updatedVal
# sgfDiffEq

# SGF dynamics with matrix approach
#@jit
def SGFDiffEq(s_matrix, sigma_matrix, deltaS, deltaT):
    updated_matrix = s_matrix + deltaT*(sigma_matrix - deltaS*s_matrix)
    return updated_matrix
# sgfDiffEq

# TODO use linalg solve to make it faster and numerically more stable
# LGF dynamics with matrix approach
#@jit
def LGFDiffEq(i_matrix, t_matrix, l_matrix, lambda_matrix, deltaL, deltaT, deltaR, D):
    alpha = D*deltaT/(deltaR**2)                            # constant
    f = (deltaT/2.)*(lambda_matrix - deltaL*l_matrix)       # term that takes into account LFG production for half time step
    g = linalg.inv(i_matrix - (alpha/2.)*t_matrix)          # inverse of some intermediate matrix
    h = i_matrix + (alpha/2.)*t_matrix                      # some intermediate matrix
    l_halftStep = g@(l_matrix@h + f)                        # half time step calculation for LGF values
    #print('grid after half time step...\n' + str(l_halftStep))
    f = (deltaT/2.)*(lambda_matrix - deltaL*l_halftStep)    # updated term...
    l_tStep = (h@l_halftStep + f)@g                         # final computation
    return l_tStep
# sgfDiffEq

#def LGFDiffEq2(i_matrix, t_matrix, l_matrix, lambda_matrix, deltaL, deltaT, deltaR, D):
    #alpha = D*deltaT/(deltaR**2)                            # constant
    #f = (deltaT/2.)*(lambda_matrix - deltaL*l_matrix)       # term that takes into account LFG production for half time step
    #g = linalg.inv(i_matrix - (alpha/2.)*t_matrix)          # inverse of some intermediate matrix
    #h = i_matrix + (alpha/2.)*t_matrix                      # some intermediate matrix
    #l_halftStep = (l_matrix@h + f)@g                        # half time step calculation for LGF values
    #f = (deltaT/2.)*(lambda_matrix - deltaL*l_halftStep)    # updated term...
    #l_tStep = g@(h@l_halftStep + f)                         # final computation
    #return l_tStep
# sgfDiffEq

# T matrix, used in LGF dynamics
#@jit
def GenerateTMatrix(size):
    t_matrix = np.zeros([size,size])
    for ix in range(size - 1):
        t_matrix[ix,ix] = -2.
        t_matrix[ix,ix + 1] = 1.
        t_matrix[ix + 1,ix] = 1.
    t_matrix[0,0] = -1.
    t_matrix[size - 1, size - 1] = -1.
    return t_matrix
# GenerateTMatrix

# Identity matrix
#@jit
def GenerateIMatrix(size):
    I_matrix = np.zeros([size,size])
    for ix in range(size):
        I_matrix[ix,ix] = 1.
    return I_matrix
# GenerateIMatrix

def NeuralNetwork(inputs, WMatrix, wMatrix, phi, theta):
    #nNodes = 10  # number of nodes
    V = np.zeros([6])
    O = np.zeros([6])
    bj = wMatrix@inputs - theta
    for ix in range(len(bj)):
        V[ix] = TransferFunction(bj[ix],2)

    bi = WMatrix@V - phi
    for ix in range(len(bi)):
        O[ix] = TransferFunction(bi[ix],2)
    return O
# NeuralNetwork

#@jit
#def TransferFunction(x,beta):
    #return 1./(1 + np.exp(-beta*x))
## TransferFunction

#@jit
def RecurrentNeuralNetwork(inputs, wMatrix, V):             # Recurrent Neural Network dynamics
    #beta = 2
    bj = wMatrix@V - inputs
    # might be improved ussing list comprehension...
    for ix in range(len(bj)):
        V[ix] = 1./(1 + np.exp(-2*bj[ix]))   #TransferFunction(bj[ix],2)
    return V
# NeuralNetwork

#@jit
def GetStructure(cell_array, nLattice):
    structure = np.zeros([nLattice,nLattice])
    for ik in range(nLattice):
        for jk in range(nLattice):
            if cell_array[ik,jk] != 0:
                structure[ik,jk] = 1
    return structure
# GetStructure

def GetrNN(csvFile, ind, nNodes):
    #with open('successful_test.csv', 'r') as csvfile:
    with open(csvFile, 'r') as csvfile:
        #reader = csv.reader(csvfile)
        population = np.loadtxt(csvfile,delimiter=',')
    wMatrix = np.array(population[ind,:].reshape(nNodes,nNodes))
    return wMatrix
