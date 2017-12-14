# -*- coding: latin_1 -*-
#import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from datetime import datetime as dt
import csv
import sys


if __name__ == '__main__':
    # Parameters
    nNodes = 25#[8, 10, 15, 20, 25]
    nodeProb = np.linspace(0.1,1,10)
    #nNeighb = [5, 10, 15, 20]
    #nEdges = [5,10,15]
    nRuns = int(sys.argv[2])
    nLattice = 50
    tSteps = 100
    arrayContainer = np.zeros([len(nodeProb)*nRuns,nNodes**2])
    infofFileID = '{0:%Y%m%d_%H%M%S_%f}'.format(dt.now())
    runsMainFile = 'runs/run_ graphs_{0}.log'.format(infofFileID)
    commentString = sys.argv[1]
    timeSteps = 200
    nLattice = 50

    # Graph types/ network architectures
    #Graph = nx.erdos_renyi_graph(nNodes, nodeProb, directed = True)        # More appropiate for the kind of network we're using
    #Graph = nx.watts_strogatz_graph(nNodes, nNeighb, nodeProb)
    #Graph = nx.barabasi_albert_graph(nNodes, nEdges)
    #graphlist = [Graph0, Graph1, Graph2]

    # Transform graph as a matrix (numpy array)
    #wMatrix = nx.to_numpy_matrix(Graph)

    # randomise non-zero entries:
    #for ix in range(nNodes):                               
        #for iy in range(nNodes):
            #if wMatrix[ix,iy] == 1:
                #wMatrix[ix,iy] = -1 + 2*np.random.random()
                
    with open(runsMainFile,'a') as csvfile:
        csvfile.write('##################\nRandom graph generator run\nErdṍs-Rényi Graphs\n# {}\n\n'.format(commentString))

    #---------------------------#
    #       Graph generator     #
    #---------------------------#
    # generate graphs for different number of nodes
    #for nN in nNodes:
    # also for different node prob of rewiring
    graphCounter = 0
    for nP in nodeProb:
        # some repetitions of the same settings
        for rep in range(nRuns):
            Graph = nx.erdos_renyi_graph(nNodes, nP, directed = True)
            wMatrix = nx.to_numpy_matrix(Graph)
            for ix in range(nNodes):                               
                for iy in range(nNodes):
                    if wMatrix[ix,iy] == 1:
                        wMatrix[ix,iy] = -1 + 2*np.random.random()
                        #print('wMatrix shape: {}'.format(wMatrix.shape))
            # Run simulation with the selected network architecture     
            #print('Running [{2} of {3}] Barabasi-Albert graph with: {1} nodes and {0} nEdges'.format(nE, nN, x + 1, nRuns))
            #main.sim(wMatrix, tSteps, nN, nLattice, False)
            
            # save network information and location for future use
            indInfo = 'n{0:02d}P{1:.2f}r{2:02d}'.format(nNodes, nP, rep)
            with open(runsMainFile,'a') as csvfile:
                csvfile.write('{0:03d}: Info {1}\n'.format(graphCounter, indInfo))
                #print('container shape: {}, flattened matrix shape: {}'.format(arrayContainer.shape, wMatrix.ravel().shape))
                arrayContainer[graphCounter,:] = wMatrix.ravel()
                
            graphCounter += 1
                
    #---------------------------#
    #       Graph storing       #
    #---------------------------#
    # Create filename: unique, related to current time
    popFileID = '{0:%Y%m%d_%H%M%S_%f}'.format(dt.now())
    networkFileName = 'populations/{0}.csv'.format(popFileID)
    print('Generated network file name:\n{}'.format(popFileID))
    # save network name
    with open(runsMainFile,'a') as csvfile:
        #csvfile.write('{}'.format(fitnessArray))
        csvfile.write('network file\n{}\n'.format(popFileID))

    # Save generated network, unique for each run
    with open(networkFileName, 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in arrayContainer]
