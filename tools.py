import numpy as np
from scipy import linalg
import csv
import neat
import os
import pickle
import subprocess as sp
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

def GenomicDistanceMatrix(run_file):
    #-------------------------------#
    #       Generate histogram      #
    #-------------------------------#
    listName = 'files_list'                                                             # name of list 
    IDFilesList_command = 'ls plots/{0} | egrep 2018 > {1}'.format(run_file, listName)    # command to generate such file
    sp.call(IDFilesList_command, shell = True)
    fileList = open(listName).read().splitlines()                                       # store names in a python list for later use
    sp.call('rm {}'.format(listName), shell = True)                                     # remove temporary file    

    genomeList = []
    configList = []
    
    for iFile in fileList:                              # Iterate over file names
        genomes, config = GetNetwork(iFile)             # get genomes and config files for a specific folder in the run folder
        for iGenome in genomes:                         # iterate over genomes
            genomeList.append(iGenome)                  # save genomes in a single list
            configList.append(config)                   # save config files as well
    #print('total length of genomes list: {}'.format(len(genomeList)))
    
    nGenomes = len(genomeList)
    GDMatrix = np.zeros([nGenomes, nGenomes], dtype = np.float64)

    for iy in range(nGenomes):
        for ix in range(nGenomes):
            GDMatrix[iy,ix] = genomeList[iy].distance(genomeList[ix], configList[ix].genome_config)
    #print('genomic distance matrix:\n{}'.format(GDMatrix))
    return GDMatrix, 'genomic'

def ReadDigraph(DiGraphFile):
    fileList = open(DiGraphFile).read().splitlines()    # load file as list of strings for each line
    del fileList [-1]                                   # delete last item: '}'
    fileList.reverse()                                  # reverse list to delete items
    for _ in range(10):                                 # iterate and delete
        del fileList [-1]
    fileList.reverse()                                  # turn list to original state
    #print('final list:')
    
    fileList = [string.split('[')[0].strip() for string in fileList]

    # code snippet taken from:
    #https://stackoverflow.com/questions/46458128/generate-adjacency-matrix-in-graphviz

    pairs = [line.replace(" ", "").split("->") for line in fileList]

    keys_inputs = {'SGF':0, 'LGF':1}
    keys_outputs = {'Proliferate':2, 'Migrate':3, 'Apoptosis':4, '"SGFProd"':5, '"LGFProd"':6, 'Polarisation':7}

    adjacency_matrix = np.zeros([8,8], dtype = np.int)

    for iKey, iPos in keys_inputs.iteritems():
        for jKey, jPos in keys_outputs.iteritems():
            for ip in pairs:
                if ip[0] == iKey and ip[1] == jKey:
                    adjacency_matrix[iPos,jPos] = 1
                    adjacency_matrix[jPos,iPos] = 1
                elif ip[0] == jKey:
                    adjacency_matrix[jPos,jPos] = 1

    return adjacency_matrix
    #print('{}'.format(adjacency_matrix))
    #keys_list = ['SGF', 'LGF', 'Proliferate', 'Migrate', 'Apoptosis', '"SGFProd"', '"LGFProd"', 'Polarisation']
    #unique_edges = set(all_edges)
    #matrix = {origin: {dest: 0 for dest in all_edges} for origin in all_edges}
    #for p in pairs:
        #matrix[p[0]][p[1]] += 1
    #import pprint
    #pprint.pprint(matrix)
    ##for ix in range(len(matrix)):
    ##print('{}'.format(matrix))
    #import pandas as pd
    #a = pd.DataFrame(matrix)
    ##print('{}'.format(a.to_string(na_rep='0')))

def GetHammingDistance(matrix_a, matrix_b):
    dim_a, dim_b = np.shape(matrix_a)
    distance = 0.
    scale = 1.#3*(dim_a - 2)
    
    for ix in range(2,dim_a):
        for iy in range(dim_a):
            if matrix_a[iy,ix] != matrix_b[iy,ix]:
                distance += 1
    return distance/scale

def HammingDistanceMatrix(run_file):
    listName = 'files_list'                                                             # name of list 
    IDFilesList_command = 'ls plots/{0} | egrep 2018 > {1}'.format(run_file, listName)    # command to generate such file
    sp.call(IDFilesList_command, shell = True)
    fileList = open(listName).read().splitlines()                                       # store names in a python list for later use
    sp.call('rm {}'.format(listName), shell = True)                                     # remove temporary file    

    adjacencyMatrixList = []
    
    for iFile in fileList:                                  # Iterate over file names
        genomes, _ = GetNetwork(iFile)                      # get genomes and config files for a specific folder in the run folder
        nNetworks = len(genomes)
        for iNet in range(nNetworks):                       # iterate over genomes
            path = 'plots/{0}/{1}/{1}_best_unique_network_{2}'.format(run_file, iFile, iNet + 1)
            adjacencyMatrixList.append(ReadDigraph(path))                        # save genomes in a single list

    #print('total length of genomes list: {}'.format(len(adjacencyMatrixList)))
    
    nMatrices = len(adjacencyMatrixList)
    HDMatrix = np.zeros([nMatrices, nMatrices], dtype = np.float64)

    for iy in range(nMatrices):
        for ix in range(nMatrices):
            HDMatrix[iy,ix] = GetHammingDistance(adjacencyMatrixList[iy], adjacencyMatrixList[ix])
    #print('genomic distance matrix:\n{}'.format(GDMatrix))
    return HDMatrix, 'hamming'
