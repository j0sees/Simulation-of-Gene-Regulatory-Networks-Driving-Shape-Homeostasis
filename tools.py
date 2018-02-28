import numpy as np
from scipy import linalg
import csv
import neat
import os
import pickle
#from numba import jit

# Tools

# List without joint ends
# https://stackoverflow.com/questions/29710249/python-force-list-index-out-of-range-exception
class flatList(list):
    def __getitem__(self, index):
        if index < 0:
            raise IndexError("list index out of range")
        return super(flatList, self).__getitem__(index)

#@jit
def CheckifOccupied(xCoord, yCoord, grid):
    if grid[yCoord][xCoord] > 0:         # if value on grid is 1 (quiet), 2 (moved) or 3 (splitted) then spot is occupied
        return True
    else:                                   # else, value is 0 (empty)
        return False
# CheckifOccupied

#@jit
def CheckifPreferred(xOri, yOri, xCoord, yCoord):
    if xCoord == xOri and yCoord == yOri:
        return True
    else:
        return False
# CheckifPreferred

# SGF dynamics with matrix approach
#@jit #WARNING ON is good!!
def SGFDiffEq(s_matrix, sigma_matrix, deltaS, deltaT):
    updated_matrix = s_matrix + deltaT*(sigma_matrix - deltaS*s_matrix)
    return updated_matrix
# sgfDiffEq

# TODO use linalg solve to make it faster and numerically more stable
# LGF dynamics with matrix approach
#@jit # WARNING ON is good!!
def LGFDiffEq(i_matrix, t_matrix, l_matrix, lambda_matrix, deltaL, deltaT, deltaR, D):
    alpha = D*deltaT/(deltaR**2)                            # constant
    f = (deltaT/2.)*(lambda_matrix - deltaL*l_matrix)       # term that takes into account LFG production for half time step
    g = linalg.inv(i_matrix - (alpha/2.)*t_matrix)          # inverse of some intermediate matrix
    h = i_matrix + (alpha/2.)*t_matrix                      # some intermediate matrix
    #l_halftStep = g@(l_matrix@h + f)                        # half time step calculation for LGF values
    l_halftStep = np.matmul(g,(np.matmul(l_matrix,h) + f))                        # half time step calculation for LGF values
    #print('grid after half time step...\n' + str(l_halftStep))
    f = (deltaT/2.)*(lambda_matrix - deltaL*l_halftStep)    # updated term...
    l_tStep = np.matmul((np.matmul(h,l_halftStep) + f),g)                         # final computation
    return l_tStep
# sgfDiffEq

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

#@jit #WARNING ON is good!
def RecurrentNeuralNetwork(inputs, wMatrix, V):             # Recurrent Neural Network dynamics
    #beta = 2
    # bj = wMatrix@V - inputs
    bj = np.matmul(wMatrix,V) - inputs
    # might be improved ussing list comprehension...
    for ix in range(len(bj)):
        V[ix] = 1./(1 + np.exp(-2*bj[ix]))   #TransferFunction(bj[ix],2)
    # V = [1./(1 + np.exp(-2*bj[ix])) for ix in range(len(bj))]
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

def GetrNN(csvFile, ind):
    #with open('successful_test.csv', 'r') as csvfile:
    with open(csvFile, 'r') as csvfile:
        #reader = csv.reader(csvfile)
        bestIndividuals = np.loadtxt(csvfile, delimiter = ',')
    # get nNodes from nGenes
    nNodes = int(np.sqrt(len(bestIndividuals[ind,:])))
    wMatrix = np.array(bestIndividuals[ind,:].reshape(nNodes,nNodes))
    return wMatrix

def GetPop(csvFile):
    with open(csvFile, 'r') as csvfile:
        #reader = csv.reader(csvfile)
        networkContainer = np.loadtxt(csvfile, delimiter = ',')
    return networkContainer

def GetNetwork(fileID):#, iGenome):
    """
    Script that quickly loads a rNN created by NEAT
    """
    config_file = 'genomes/{}_config'.format(fileID)
    fileName = 'genomes/{}_best_unique_genomes'.format(fileID)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)

    # Config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # load the winner
    with open(fileName, 'rb') as f:
        genomes = pickle.load(f)#, encoding = 'bytes')

    #return neat.nn.RecurrentNetwork.create(genomes[iGenome], config)
    return genomes, config
    
def GenerateStatus(output):
    """
    Generate cell status out of the network output
    """
    status_data = np.zeros([2], dtype = int)     # [status, polarisation]
    
    # Cellular states
    iStatus = output[0]             # Proliferate: Split
    jStatus = output[1]             # Migrate:     Move
    kStatus = output[2]             # Apoptosis:   Die
    # Values for SGF and LGF
    #status_data[2] = output[3]      # SGF Prod
    #status_data[3] = output[4]      # LGF Prod
    # Polarisation
    compass = output[5]

    xThreshold = 0.5
    yThreshold = 0.001

    # Orientation boundaries:
    nBoundary = 0.25
    sBoundary = 0.5
    eBoundary = 0.75

    # oriented according to numpy order v>, not usual >^
    if abs(compass - sBoundary) <= yThreshold:  # compass == 0.5
        status_data[1] = 0                      # no orientation
        #print('no orientation')
    elif compass < sBoundary:                   # compass != 0.5
        if compass < nBoundary:                 # 0 <= compass < 0.25
            status_data[1] = 1                  # orientation West
            #print('orientation west')
        else:                                   # 0.25 <= compass < 0.5
            status_data[1] = 2                  # orientation North
            #print('orientation north')
    else: 
        if compass <= eBoundary:               # 0.5 < compass <= 0.75
            status_data[1] = 3                  # orientation East
            #print('orientation east')            
        else:                                   # 0.75 < compass <= 1
            status_data[1] = 4                  # orientation South
            #print('orientation south')
    
    if iStatus < xThreshold and jStatus < xThreshold and kStatus < xThreshold:
        status_data[0] = 1          # 'Quiet'
    else:
        for ix in iStatus, jStatus, kStatus:
            if xThreshold < ix:
                xThreshold = ix
        if abs(xThreshold - iStatus) <= yThreshold:
            status_data[0] = 2      # 'Split'
        elif abs(xThreshold - jStatus) <= yThreshold:
            status_data[0] = 3      # 'Move'
        else:
            status_data[0] = 4      # 'Die'
    return status_data
# Generate state
