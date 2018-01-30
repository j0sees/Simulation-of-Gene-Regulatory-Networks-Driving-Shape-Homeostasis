#!/usr/bin/env python
import sys
import os
import time
from datetime import datetime as dt
#from datetime import time
import random
import numpy as np
############
# self made classes
from cell_agent import *                    # it is allowed to call from this class because there's an __init__.py file in this directory
from tools import *
from tools_GA import *
############
import multiprocessing as mp
import ctypes
import csv
#import contextlib
#import itertools
from contextlib import contextmanager
from functools import partial
#from numba import jit

#============================================================#
#                                                            #
#                   CELLULAR AUTOMATA                        #
#                                                            #
#============================================================#
#@jit
def sim(wMatrix, timeSteps, nNodes, nLattice):
    """
    Parameters: sim(wMatrix, numberOfTimeSteps, nNodes, nLattice)
    # In ozzy the simulation works solely as a fitness function,
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       PARAMETERS                 #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # TODO: organize in different categories...
    npCellGrid = np.zeros([nLattice,nLattice])    # Initialize empty grid
    semiFlatGrid = [flatList(npCellGrid[r,:]) for r in range(nLattice)]
    cellGrid = flatList(semiFlatGrid)
    chemGrid = np.zeros([nLattice,nLattice,2])  # empty grid
    SGF_read = 0.                               # in the future values will be read from the grid
    LGF_read = 0.
    ix = int(nLattice/2)                        # Initial position for the mother cell
    iy = int(nLattice/2)                        # Initial position for the mother cell
    iTime = 0                                   # time counter

    cellList = []                               # List for cell agents

    # SGF/LGF dynamics parameters
    deltaT = 1.                                 # time step for discretisation [T]
    deltaR = 1.                                 # space step for discretisation [L]
    deltaS = 0.5                                # decay rate for SGF
    deltaL = 0.1                                # decay rate for LGF
    diffConst = 1.#0.05                         # diffusion constant D [dimentionless]
    t_matrix = GenerateTMatrix(nLattice)        # T matrix for LGF operations
    i_matrix = GenerateIMatrix(nLattice)        # I matrix for LGF operations

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       INITIALIZATION             #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # create mother cell and update the grid with its initial location
    cellList.append(cell(ix,iy,wMatrix,nNodes))
    cellGrid[ix][iy] = 1

    while iTime < timeSteps:
        ## decay chemicals in spots where there is some but no cell

        # this matrixes must be updated everytime so that if there's no production in one spot that spot contains a zero
        # but must not lose contained information, i.e. must use it before setting it to zero
        sigma_m = np.zeros([nLattice,nLattice])             # matrix representation of SGF production
        lambda_m = np.zeros([nLattice,nLattice])            # matrix representation of LGF production

        tmpCellList = list(cellList)                        # a copy of the list of current cells is used to iterate over all the cells

        tmpCellListLength = len(tmpCellList)
        quietCounter = 0
        while len(tmpCellList) > 0:                 # while the tmp list of cells is longer than 1
            # 1st step => choose a random cell from the list of existing cells
            rndCell = np.random.randint(len(tmpCellList))

            # 2nd step => read chemicals
            SGF_reading, LGF_reading = tmpCellList[rndCell].Sense(chemGrid)

            # 3rd step => random cell should decide and action
            tmpCellList[rndCell].GenerateStatus(SGF_reading, LGF_reading)     # get status of this cell

            # 4th step => update SGF and LGF amounts on the 'production' matrices sigma & lambda
            # production matrices get updated values
            sigma_m[tmpCellList[rndCell].yPos,tmpCellList[rndCell].xPos] = tmpCellList[rndCell].sgfAmount
            lambda_m[tmpCellList[rndCell].yPos,tmpCellList[rndCell].xPos] = tmpCellList[rndCell].lgfAmount

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            #        Cell Action            #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # according to cell status perform action
            if tmpCellList[rndCell].state == 'Quiet':           # Check the state
                tmpCellList[rndCell].Quiet(cellGrid)            # call method that performs selected action
                quietCounter += 1
                del tmpCellList[rndCell]                        # delete cell from temporal list
                
            elif tmpCellList[rndCell].state == 'Split':
                tmpCellList[rndCell].Split(cellGrid,cellList)
                del tmpCellList[rndCell]

            elif tmpCellList[rndCell].state == 'Move':
                tmpCellList[rndCell].Move(cellGrid)
                del tmpCellList[rndCell]

            else: # Die
                tmpCellList[rndCell].Die(cellGrid)                  # Off the grid, method also changes the "amidead" switch to True
                del tmpCellList[rndCell]
        # while

        # A list of cells that "died" is stored to later actually kill the cells...
        listLength = len(cellList) - 1
        for jCell in range(listLength,-1,-1):                       # checks every cell and if it was set to die then do, in reverse order
            if cellList[jCell].amidead:
                del cellList[jCell]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    SGF/LGF diffusion and/or decay     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        chemGrid[:,:,0] = SGFDiffEq(chemGrid[:,:,0], sigma_m, deltaS, deltaT)
        chemGrid[:,:,1] = LGFDiffEq(i_matrix, t_matrix, chemGrid[:,:,1], lambda_m, deltaL, deltaT, deltaR, diffConst)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    Test grid to discard trivial cases #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(cellList) == 0 or quietCounter == len(cellList):     # if cells die during the simulation resturn two different structs
            halfwayStruct = np.zeros([nLattice,nLattice])
            finalStruct = np.ones([nLattice,nLattice])
            break
        elif iTime == int(timeSteps/2) - 1:                         # special cases get tested halfway through the simulation
            if len(cellList) <= int((nLattice**2)*0.01) or len(cellList) >= int((nLattice**2)*0.9):        # If there are no cells 
                halfwayStruct = np.zeros([nLattice,nLattice])       # return two completely different structure matrices to get 0 fitness
                finalStruct = np.ones([nLattice,nLattice])
                break
            else:
                halfwayStruct = np.array(cellGrid)
        elif iTime == timeSteps - 1:
            if len(cellList) >= int((nLattice**2)*0.9):             # If cells fill space return two completely
                halfwayStruct = np.zeros([nLattice,nLattice])       # different structure matrices to get 0 fitness
                finalStruct = np.ones([nLattice,nLattice])
            else:
                finalStruct = np.array(cellGrid)
        iTime += 1
    # while

    halfwayStruct = GetStructure(halfwayStruct, nLattice)
    finalStruct = GetStructure(finalStruct, nLattice)

    deltaMatrix = np.zeros([nLattice,nLattice])
    for ik in range(nLattice):
        for jk in range(nLattice):
            if halfwayStruct[ik,jk] != finalStruct[ik,jk]:
                deltaMatrix[ik,jk] = 1

    return deltaMatrix
### sim

# workaround used due to the lack of starmap() in python 2.7...
# https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments#5443941
@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

#@jit
def EvaluateIndividual(timeSteps, nNodes, nLattice, individual):
    totSum = 0.
    wMatrix = population[individual,:].reshape(nNodes,nNodes)
    deltaM = sim(wMatrix, timeSteps, nNodes, nLattice)
    deltaMatrix = np.array(deltaM)

    for ix in range(nLattice):
        for jx in range(nLattice):
            totSum += deltaMatrix[ix,jx]
    fit = 1. - (1./(nLattice**2))*totSum
    return fit
# EvaluateIndividual

def partialEval(ind):
    return EvaluateIndividual(timeSteps, nNodes, nLattice, ind)

#============================================================#
#                                                            #
#                   GENETIC ALGORITHM                        #
#                                                            #
#============================================================#
if __name__ == '__main__':
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       PARAMETERS                 #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # nProcs*cycles = 4*int + 2
    # popSize = nProcs*cycles
    nRuns = int(sys.argv[2]) #10
    nProcs = 3 # int(sys.argv[1])                              # multiprocessing will use as many cores as it can see
    popSize = 21 # int(sys.argv[2])                              # Population size. Must have certain 
    nNodes = int(sys.argv[3])
    nGenes = nNodes**2                                          # Number of genes
    crossoverProb = 0.5 #float(sys.argv[2])                                         # Crossover probability
    mutationProb = 1.                                          # Mutation probability
    crossMutProb = 0.5                                          # probability of doing mutation or crossover
    tournamentSize = 4                                          # Tournament size. EVEN
    eliteNum = 1                                                # number of elite solutions to carry to next generation
    nOfGenerations = 10#int(sys.argv[3])                       # 10 is a reasonable number since the GA is so efficient...
    timeSteps = 200
    nLattice = 50
    chunkSize = 1 # int(sys.argv[2])
    #fileHeader = sys.argv[2]
    
    gaInfo = 'p{0:02d}g{1:04d}x{2:0.1f}c{3:0.1f}m{4:0.1f}t{5}e{6}G{7:02d}'.format(popSize,nGenes,crossMutProb,crossoverProb,mutationProb,tournamentSize,eliteNum, nOfGenerations)
    mpInfo = 'P{0:02d}C{1:02d}'.format(nProcs, chunkSize)
    simInfo = 'n{0:02d}T{1:03d}L{2:02d}R{3:02d}'.format(nNodes, timeSteps, nLattice, nRuns)
    
    benchmakingData = np.zeros([nRuns,2])
    statsData = np.zeros([nRuns,nOfGenerations,2])
    populationFiles = []
    infofFileID = sys.argv[1] #'{0:%Y%m%d_%H%M%S}'.format(dt.now())
    runsMainFile = 'runs/run_{0}.log'.format(infofFileID)
    benchmarkFile = 'benchmarks/{0}.csv'.format(infofFileID)
    statsFile = 'stats/{0}.csv'.format(infofFileID)

    # Write run information to a file    
    with open(runsMainFile,'a') as csvfile:
        csvfile.write('#######################################\n')
        csvfile.write('# GA Info: {}\n# mp Info: {}\n# Sim Info: {}\n'.format(gaInfo, mpInfo, simInfo))

    # Run the algorithm nRuns times to generate statistics
    iRun = 0
    while iRun < nRuns:
        print('# Run number {} of {}'.format(iRun + 1, nRuns))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #       INITIALISATION             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        contestants = np.zeros([tournamentSize, nGenes])
        fitnessInfo = np.zeros([nOfGenerations, 2])
        bestIndividuals = np.zeros([nOfGenerations, nGenes])

        # Multiprocessing implementation shared data arrays
        population_base = mp.Array(ctypes.c_float, popSize*nGenes, lock = False)# create mp shared 1D array
        population = np.frombuffer(population_base, dtype = ctypes.c_float)     # convert mp array to np.array
        population = population.reshape(popSize, nGenes)                        # Reshape as popSize x nGenes matrix
        
        # 1st step: Initialise population
        
        # Read population from file
        #csvFile = 'populations/zero_fitness_pop.csv'
        #randPop = GetPop(csvFile)

        # Generate random population         
        np.random.seed()
        randPop = -1. + 2.*np.random.rand(popSize, nGenes)
        population = -1. + 2.*np.random.rand(popSize, nGenes)        
        np.copyto(population, randPop)
        
        # Array to store fitness values
        fitness = np.zeros([popSize])

        # 2nd step: Loop through generations
        for iGen in range(nOfGenerations):
            # 3rd step: Fitness function => Rank idividuals by their fitness, chromosomes get decoded and evaluated

            # arguments to pass to fitness simulation
            #timeSteps_list = [timeSteps for x in range(popSize)]
            #iGen_list = [iGen for x in range(popSize)]
            #nNodes_list = [nNodes for x in range(popSize)]
            #nLattice_list = [nLattice for x in range(popSize)]
            #index_list = list(range(popSize))
            index_list = [ x for x in range(popSize)]
            #args = zip(index_list, timeSteps_list, iGen_list, nNodes_list, nLattice_list)

            # WARNING => To use when running with python 3
            # every pool is a different set of processes        , maxtasksperchild = maxproc
            #with mp.Pool(popSize) as pool:
                #pool.starmap(EvaluateIndividual, args, chunkSize)              # Evaluation of individuals, this runs in parallel!

            # WARNING => To use when running with python 2.7
            with poolcontext(processes = nProcs) as pool:
                fitness = pool.map(partialEval, index_list)

            # 3.1: sort fitness array
            sorted_fitness = np.argsort(fitness)                    # array containing the indexes from less fit to most fit ind
            tempPopulation = np.zeros([popSize, nGenes])            # np.array(population)

            # 3.2: get fittest infividual and mean fitness
            fitnessInfo[iGen, 0] = np.amax(fitness) #sorted_fitness[popSize - 1]                   # np.amax(fitness)
            fitnessInfo[iGen, 1] = np.average(fitness)
            print('Gen: {2}\t=>\tavg fit: {1:.3f}\tmax fit: {0}'.format(fitnessInfo[iGen, 0], fitnessInfo[iGen, 1], iGen + 1))
            string = [float('{:.3f}'.format(x)) for x in fitness]
            print('{}'.format(string))

            # 4th step: Elitism => Save the best individuals for next generation
            iElit = 1                                               # Elite counter: individuals with the best fitness are kept untouched
            # WARNING use list[-1]
            while iElit <= eliteNum:
                index = sorted_fitness[popSize - iElit]             # get the index of the last members of the list, i.e., most fit
                tempPopulation[iElit - 1,:] = np.array(population[index,:])   # store as part of the new generation of individuals
                np.delete(sorted_fitness,popSize - iElit)           # delete last tuple on the list
                iElit += 1
            # while

            # 4.1 => Save best individual por the ages
            bestIndividuals[iGen,:] = np.array(population[sorted_fitness[popSize - 1]])
            
            # 5th step: Tournament selection => Loop over the rest of the population to engage them into a tournament
            loopCounter = 0
            while len(sorted_fitness) >= tournamentSize:            # iterate through all individuals
                # 5.1 => Select random contestants and sort them by index (i.e. by fitness))
                selectedInd = np.random.choice(range(len(sorted_fitness)), tournamentSize, replace = False)
                selectedInd.sort()

                # 5.2 => Get tournament winner individuals from population
                winIndex1 = sorted_fitness[selectedInd[tournamentSize - 1]] # the fittest ind is retrieved from the sorted fitness array
                contestants[0,:] = np.array(population[winIndex1,:])
                winIndex2 = sorted_fitness[selectedInd[tournamentSize - 2]] # the second fittest ind is retrieved from the sorted fitness array
                contestants[1,:] = np.array(population[winIndex2,:])

                # 5.3 => Generate new offsprig by crossover or mutation
                r = np.random.random()
                if r >= crossMutProb:
                    contestants[2,:],contestants[3,:] = Crossover(contestants[0,:], contestants[1,:], crossoverProb)
                else:
                    contestants[2,:],contestants[3,:] = Mutate(contestants[0,:], contestants[1,:], mutationProb)

                # 5.4 => Delete contestants from fitness array
                iCounter = 0
                for ix in selectedInd:
                    index = ix - iCounter
                    sorted_fitness = np.delete(sorted_fitness, index)      # WARNING does this really work?
                    iCounter += 1

                # 5.5 => Save best individuals and offspring for new generation
                for jk in range(tournamentSize):
                    index = eliteNum + (loopCounter*tournamentSize) + jk
                    tempPopulation[index] = contestants[jk,:]

                loopCounter += 1
            # loop over population

            # 6th step: Save new population resulting from last generation
            population = np.array(tempPopulation)
            
            # 7th step: Repeat steps 3 to 7 until iGen = nOfGenerations 
        # Loop over generations

        # 8th step: Save main run data in csv files
        # 8.1 => Generate unique filename related to current time
        popFileID = '{0:%Y%m%d_%H%M%S_%f}'.format(dt.now())
        networkFileName = 'populations/{0}.csv'.format(popFileID)
        
        # 8.2 => Store filenames to associate with main run
        populationFiles.append('{0}.csv'.format(popFileID))
        
        # 8.3 => Save network population unique for each run
        with open(networkFileName, 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in bestIndividuals]

        # 8.4 => Save run statistics
        statsData[iRun,:,:] = np.array(fitnessInfo)    
        
        print('#')
        
        # 9th step: Repeat steps 1 to 8 until iRun = nRuns 
        iRun += 1 
    # Loop over runs
    
    runsBenchAvg = np.mean(benchmakingData, axis = 0, dtype = np.float64)
    runsStatsAvg = np.mean(statsData, axis = 0, dtype = np.float64).reshape(nOfGenerations*2)
    
    # 10th step: Save runs statistics
    # Save time measures, totals per run. One file 
    with open(benchmarkFile, 'a') as bchfile:
        writer = csv.writer(bchfile)
        writer.writerow(runsBenchAvg)
        
    # Save fitness measures, information per generation 
    with open(statsFile, 'a') as statsfile:
        writer = csv.writer(statsfile)
        writer.writerow(runsStatsAvg)
        
    # Write file containing information about runs
    with open(runsMainFile,'a') as csvfile:
        csvfile.write('# Population files:\n')
        [csvfile.write('{}\n'.format(r)) for r in populationFiles]
        csvfile.write('\n')

# End of GA
