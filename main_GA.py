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
from neat_cell_agent import *                    # it is allowed to call from this class because there's an __init__.py file in this directory
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
def sim(genome, config, timeSteps, nLattice):
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

    # timing variables!
    #tmpListLoopAvg = 0
    #chemicalsUpdateAvg = 0

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       INITIALIZATION             #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # create mother cell and update the grid with its initial location
    cellList.append(cell(iy,ix,genome,config))
    cellGrid[ix][iy] = 1
    #cellGrid[0,1,0] = 1
    #print('Initial Grid:\n{}'.format(cellGrid[:,:,0]))

    # DEBUG
    #print('Time running...')
    # Timing!
    start_time_mainLoop = time.time()
    #print('process: {} is running sim!!'.format(os.getpid()))
    while iTime < timeSteps:
        # DEBUG
        #print('\n######### time step #' + str(iTime))

        ## decay chemicals in spots where there is some but no cell

        # this matrixes must be updated everytime so that if there's no production in one spot that spot contains a zero
        # but must not lose contained information, i.e. must use it before setting it to zero
        sigma_m = np.zeros([nLattice,nLattice])             # matrix representation of SGF production
        lambda_m = np.zeros([nLattice,nLattice])            # matrix representation of LGF production

        tmpCellList = list(cellList)                        # a copy of the list of current cells is used to iterate over all the cells

        # Timing!
        start_time_tmpListLoop = time.time()
        tmpCellListLength = len(tmpCellList)
        quietCounter = 0
        while len(tmpCellList) > 0:                 # while the tmp list of cells is longer than 1
            # 1st step => choose a random cell from the list of existing cells
            rndCell = np.random.randint(len(tmpCellList))
            # store lattice size
            #tmpCellList[rndCell].border = nLattice          # TODO rethink this
            #tmpCellList[rndCell].nNodes = nNodes           # WARNING hardcoded

            # 2nd step => read chemicals
            SGF_reading, LGF_reading = tmpCellList[rndCell].Sense(chemGrid)

            # 3rd step => random cell should decide and action
            tmpCellList[rndCell].GenerateStatus(SGF_reading, LGF_reading)     # get status of this cell

            # 4th step => update SGF and LGF amounts on the 'production' matrices sigma & lambda
            # production matrices get updated values
            sigma_m[tmpCellList[rndCell].yPos,tmpCellList[rndCell].xPos] = tmpCellList[rndCell].sgfAmount
            lambda_m[tmpCellList[rndCell].yPos,tmpCellList[rndCell].xPos] = tmpCellList[rndCell].lgfAmount

            # DEBUG
            #print('\ncell number: ' + str(len(cellList)) + '\nCell status: ' + str(tmpCellList[rndCell].state))# + '\n')
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            #        Cell Action            #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # according to cell status perform action: split or stay quiet
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
                tmpCellList[rndCell].Die(cellGrid)              # Off the grid, method also changes the "amidead" switch to True
                del tmpCellList[rndCell]
        # while
        # Timing!
        end_time_tmpListLoop = time.time()
        secs = end_time_tmpListLoop - start_time_tmpListLoop
        #print('time to loop through all cells: {:.3f} number of cells: {}'.format(secs, tmpCellListLength))

        # A list of cells that "died" is stored to later actually kill the cells...
        listLength = len(cellList) - 1
        for jCell in range(listLength,-1,-1):                   # checks every cell and if it was set to die then do, in reverse order
            #print('len(cellList): ' + str(len(cellList)) + '. Current element: ' + str(jCell))
            if cellList[jCell].amidead:
                del cellList[jCell]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    SGF/LGF diffusion and/or decay     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # Timing!
        #start_time_chemicalsUpdate = time.time()
        chemGrid[:,:,0] = SGFDiffEq(chemGrid[:,:,0], sigma_m, deltaS, deltaT)
        chemGrid[:,:,1] = LGFDiffEq(i_matrix, t_matrix, chemGrid[:,:,1], lambda_m, deltaL, deltaT, deltaR, diffConst)
        # Timing!
        #end_time_chemicalsUpdate = time.time()
        #secs = end_time_chemicalsUpdate - start_time_chemicalsUpdate
        #print('time taken to update chemicals:' + str(secs))

        ####################################################################
        ##       IN-PLACE UPDATE
        ####################################################################
        #chemGrid[:,:,0] *=
        #SGFDiffEq(s_matrix, sigma_matrix, deltaS, deltaT):
        #updated_matrix = s_matrix + deltaT*(sigma_matrix - deltaS*s_matrix)
        #return updated_matrix
        ## sgfDiffEq

        ## TODO use linalg solve to make it faster and numerically more stable
        ## LGF dynamics with matrix approach
        ##@jit # WARNING ON is good!!
    #def LGFDiffEq(i_matrix, t_matrix, l_matrix, lambda_matrix, deltaL, deltaT, deltaR, D):
        #alpha = D*deltaT/(deltaR**2)                            # constant
        #f = (deltaT/2.)*(lambda_matrix - deltaL*l_matrix)       # term that takes into account LFG production for half time step
        #g = linalg.inv(i_matrix - (alpha/2.)*t_matrix)          # inverse of some intermediate matrix
        #h = i_matrix + (alpha/2.)*t_matrix                      # some intermediate matrix
        ##l_halftStep = g@(l_matrix@h + f)                        # half time step calculation for LGF values
        #l_halftStep = np.matmul(g,(np.matmul(l_matrix,h) + f))                        # half time step calculation for LGF values
        ##print('grid after half time step...\n' + str(l_halftStep))
        #f = (deltaT/2.)*(lambda_matrix - deltaL*l_halftStep)    # updated term...
        #l_tStep = np.matmul((np.matmul(h,l_halftStep) + f),g)                         # final computation
        #return l_tStep

        ###################################################################
        #       IN-PLACE UPDATE
        ###################################################################

        #print('grid after update...\n' + str(cellGrid[:,:,2]))
        #print('################################LGF total = ' + str(chemsum))
        #print('updated grid:\n' + str(cellGrid[:,:,0]))

        if len(cellList) == 0 or quietCounter == len(cellList):       # if cells die during the simulation resturn two different structs
            halfwayStruct = np.zeros([nLattice,nLattice])
            finalStruct = np.ones([nLattice,nLattice])
            #print('zero cells!')
            break
        elif iTime == int(timeSteps/2) - 1:                           # special cases get tested halfway through the simulation
            if len(cellList) <= int((nLattice**2)*0.01) or len(cellList) >= int((nLattice**2)*0.9):        # If there are no cells 
                halfwayStruct = np.zeros([nLattice,nLattice])       # return two completely different structure matrices to get 0 fitness
                finalStruct = np.ones([nLattice,nLattice])
                #print('too less cells! (halfway)')
                break
            else:
                #print('proc {} halfway'.format(os.getpid()))
                halfwayStruct = np.array(cellGrid)
        elif iTime == timeSteps - 1:
            if len(cellList) >= int((nLattice**2)*0.9):                      # If cells fill space 
                halfwayStruct = np.zeros([nLattice,nLattice])       # return two completely different structure matrices to get 0 fitness
                finalStruct = np.ones([nLattice,nLattice])
                #print('too many cells! (end of run)')
                # break
            else:
                #print('proc {} done!'.format(os.getpid()))
                finalStruct = np.array(cellGrid)
        # print('Grid:\n{}'.format(cellGrid[:,:,0]))
        iTime += 1
    # while
    
    # Timing!
    end_time_mainLoop = time.time()
    secs = end_time_mainLoop - start_time_mainLoop
    #print('\ntime taken in main loop: {:.3f}'.format(secs))

    # DEBUG
    # print(str(timeSteps)+' time steps complete')

    # Timing!
    #start_time_finalFunctions = time.time()

    halfwayStruct = GetStructure(halfwayStruct, nLattice)
    finalStruct = GetStructure(finalStruct, nLattice)

    deltaMatrix = np.zeros([nLattice,nLattice])
    for ik in range(nLattice):
        for jk in range(nLattice):
            if halfwayStruct[ik,jk] != finalStruct[ik,jk]:
                deltaMatrix[ik,jk] = 1
    # Timing!
    #end_time_finalFunctions = time.time()
    #secs = end_time_finalFunctions - start_time_finalFunctions
    #print('\ntime taken to get delta matrix:' + str(secs))


    # DEBUG
    #print('half way structure:\n' + str(halfwayStruct))
    #print('final structure:\n' + str(finalStruct))
    #print('delta matrix:\n' + str(deltaMatrix))
    #print('Final count of cells: {}'.format(len(cellList)))

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
    #print('inside EvaluateIndividual...')
    wMatrix = population[individual,:].reshape(nNodes,nNodes)
    #print('process: {} is running sim with individual: {}!'.format(os.getpid(), individual))
    # Timing!
    start_time_chemicalsUpdate = time.time()
    deltaM = sim(wMatrix, timeSteps, nNodes, nLattice)
    # Timing!
    end_time_chemicalsUpdate = time.time()
    secs = end_time_chemicalsUpdate - start_time_chemicalsUpdate
    #print('Proc: {}, time taken run sim: {:.3f}'.format(os.getpid(), secs))
    #print('process: {} done with individual: {}!'.format(os.getpid(), individual))
    deltaMatrix = np.array(deltaM)

    for ix in range(nLattice):
        for jx in range(nLattice):
            totSum += deltaMatrix[ix,jx]
    # DEBUG
    # print('total sum on delta matrix: ' + str(totSum))
    #if totSum <= int((nLattice**2)*0.1) or totSum == int(nLattice**2):
    #    fitness[individual] = 0.
    #else:
    
    #fitness[individual] = 1. - (1./(nLattice**2))*totSum
    fit = 1. - (1./(nLattice**2))*totSum
    #print('Proc {} computed fitness: {}'.format(os.getpid(), fit))
    return fit
# EvaluateIndividual

#class MultiProcFunc(object):
    #"""Pass a function with fixed parameters and 1 variable to subprocesses."""
    #def __init__(self, func, *params):
        #self.func = func
        #self.params = params
        #self.name = func.__name__
        ## TODO: set nProcsMax as inputvariable/attribute!
        #nProcsMax = 10
        #if nProcsMax is None:
            #self.nProcsMax = mp.cpu_count()
        #else:
            #self.nProcsMax = nProcsMax
            
    #def Evaluate(self, x, **kwargs):
        #partFunc = partial(self.func, *self.params)
        #print('list: {}\n{}, x[0] = {}'.format(x, type(x), type(x[0])))
        #try:
            #for e in x:
                #break
        #except TypeError:
            #return partFunc(x)
        #else:
            #p = mp.Pool(**kwargs)
            #output = p.map(partFunc, x)
            #p.close()
            #p.join()
            #return output

def partialEval(ind):
    # call the target function
    #print('ts {}, nN {}, nL {}, ind {}'.format(timeSteps, nNodes, nLattice, ind))
    return EvaluateIndividual(timeSteps, nNodes, nLattice, ind)

#def YourFitnessFunction(someParam1, param2, x)
    ##do something
    #fitness = x
    
    #return fitness

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
    nProcs = 21 # int(sys.argv[1])                              # multiprocessing will use as many cores as it can see
    #maxproc = 1
    DEFAULT_VALUE = -1                                          # WARNING is this necessary?
    popSize = 21 # int(sys.argv[2])                              # Population size. Must have certain 
    nNodes = int(sys.argv[3])
    nGenes = nNodes**2                                          # Number of genes
    crossoverProb = 0.5 #float(sys.argv[2])                                         # Crossover probability
    mutationProb = 1.                                          # Mutation probability
    crossMutProb = 0.5                                          # probability of doing mutation or crossover
    #tournamentSelParam = 0.75                                  # Tournament selection parameter
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
    
    iRun = 0
    benchmakingData = np.zeros([nRuns,2])
    statsData = np.zeros([nRuns,nOfGenerations,2])
    populationFiles = []
    infofFileID = sys.argv[1] #'{0:%Y%m%d_%H%M%S}'.format(dt.now())
    runsMainFile = 'runs/run_{0}.log'.format(infofFileID)
    benchmarkFile = 'benchmarks/{0}.csv'.format(infofFileID)
    statsFile = 'stats/{0}.csv'.format(infofFileID)

    #benchmarkingFiles = []
    #statsFiles
    
    with open(runsMainFile,'a') as csvfile:
        csvfile.write('#######################################\n')
        csvfile.write('# GA Info: {}\n# mp Info: {}\n# Sim Info: {}\n'.format(gaInfo, mpInfo, simInfo))

    while iRun < nRuns:
        print('# Run number {} of {}'.format(iRun + 1, nRuns))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #       INITIALISATION             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        contestants = np.zeros([tournamentSize, nGenes])
        fitnessInfo = np.zeros([nOfGenerations, 2])
        bestIndividuals = np.zeros([nOfGenerations, nGenes])

        # timing variables!
        generationAvg = 0

        #print('Parameters: \nnProcs = {}, Population size = {}, nNodes = {}, nLattice = {}, nGen = {}, Crossover Prob = {}, Mutation prob = {}\nFile name: {}'.format(nProcs, popSize, nNodes, nLattice, nOfGenerations, crossoverProb, mutationProb, fileName))

        # Multiprocessing implementation
        population_base = mp.Array(ctypes.c_float, popSize*nGenes, lock = False)# create mp shared 1D array
        #print('population_base length: {}'.format(len(population_base)))
        population = np.frombuffer(population_base, dtype = ctypes.c_float)     # convert mp array to np.array
        population = population.reshape(popSize, nGenes)                        # Reshape as popSize x nGenes matrix
        #print('population length: {}'.format(len(population)))
        
        csvFile = 'populations/zero_fitness_pop.csv'
        randPop = GetPop(csvFile)
        
        #np.random.seed()
        #randPop = -1. + 2.*np.random.rand(popSize, nGenes)
        #population = -1. + 2.*np.random.rand(popSize, nGenes)
        
        np.copyto(population, randPop)
        
        #print('population shared array created successfully!')

        #fitness_base = mp.Array(ctypes.c_float, popSize, lock = False)          # create mp shared 1D array
        #fitness = np.frombuffer(fitness_base, dtype = ctypes.c_float)           # convert mp array to np.array
        fitness = np.zeros([popSize])
        #print('fitness shared array created successfully!')

        for iGen in range(nOfGenerations):
            start_time_generation = time.time()
            # DEBUG
            #print('\nGeneration #' + str(iGen + 1))

            # 1st step: Fitness function => Rank idividuals by their fitness
            # chromosomes get decoded and evaluated
            #fitness.fill(DEFAULT_VALUE)
            
            # arguments to pass to fitness simulation
            #timeSteps_list = [timeSteps for x in range(popSize)]
            #iGen_list = [iGen for x in range(popSize)]
            #nNodes_list = [nNodes for x in range(popSize)]
            #nLattice_list = [nLattice for x in range(popSize)]
            #index_list = list(range(popSize))
            index_list = [ x for x in range(popSize)]
            #print('{}'.format(index_list))
            #index_list = np.arange(popSize)
            #args = zip(index_list, timeSteps_list, iGen_list, nNodes_list, nLattice_list)
            
            # every pool is a different set of processes        , maxtasksperchild = maxproc
            #with mp.Pool(popSize) as pool:
                #pool.starmap(EvaluateIndividual, args, chunkSize)              # Evaluation of individuals, this runs in parallel!
            
            #with contextlib.closing(mp.Pool(processes = nProcs)) as pool:
            #    pool.map(EvaluateIndividual, itertools.izip(index_list, itertools.repeat(timeSteps), itertools.repeat(iGen), itertools.repeat(nNodes), itertools.repeat(nLattice)))
                
            #partEvInd = partial(EvaluateIndividual, tStp = timeSteps, nNds = nNodes, nL = nLattice)    
            with poolcontext(processes = nProcs) as pool:
                fitness = pool.map(partialEval, index_list)
                
            #MPFitnessFunc = MultiProcFunc(EvaluateIndividual, timeSteps, nNodes, nLattice)
            #fitness = MPFitnessFunc.Evaluate(index_list)
                
            #pool = mp.Pool(processes = nProcs)                      # Pool of processes
            #print('evaluating pool...')
            # Timing!
            #start_time_fitness = time.time()
            #pool.starmap(EvaluateIndividual, args, chunkSize)              # Evaluation of individuals, this runs in parallel!
            #pool.close()
            #pool.join()
            # Timing!
            #end_time_fitness = time.time()
            #secs = end_time_fitness - start_time_fitness
            # loop over chromosomes
            
            #for ik in range(popSize):
            #    EvaluateIndividual(ik, timeSteps, iGen, nNodes, nLattice)

            # 1.1: sort fitness array
            sorted_fitness = np.argsort(fitness)                    # array containing the indexes from less fit to most fit ind
            tempPopulation = np.zeros([popSize, nGenes])            # np.array(population)
            
            # 1.2: get fittest infividual and mean fitness
            fitnessInfo[iGen, 0] = np.amax(fitness) #sorted_fitness[popSize - 1]                   # np.amax(fitness)
            fitnessInfo[iGen, 1] = np.average(fitness)
            #print('Gen: {2}\t=>\tmax fit: {0:.3f},\tavg fit: {1:.3f}'.format(fitnessInfo[iGen, 0], fitnessInfo[iGen, 1], iGen + 1))
            print('Gen: {2}\t=>\tavg fit: {1:.3f}\tmax fit: {0}'.format(fitnessInfo[iGen, 0], fitnessInfo[iGen, 1], iGen + 1))
            string = [float('{:.3f}'.format(x)) for x in fitness]
            print('{}'.format(string))
            # DEBUG
            #print('sorted fitness array, before deleting:\n' + str(fitness))

            # 2nd step: Elitism => Save the best individuals for next generation
            iElit = 1                                               # Elite counter: individuals with the best fitness are kept untouched
            # WARNING use list[-1]
            while iElit <= eliteNum:
                #index = fitness[popSize - iElit][1]                # get the index of the last members of the list, i.e., most fit
                index = sorted_fitness[popSize - iElit]
                # DEBUG
                #print('=> best fitness: ' + str(fitness[popSize - iElit][0]))
                tempPopulation[iElit - 1,:] = np.array(population[index,:])   # store as part of the new generation of individuals
                #del fitness[popSize - iElit]                       # delete last tuple on the list
                np.delete(sorted_fitness,popSize - iElit)
                iElit += 1
            # while
            # 2.1: => Save best individual por the ages
            bestIndividuals[iGen,:] = np.array(population[sorted_fitness[popSize - 1]])

            # 3rd step: Tousnament selection => Loop over the rest of the population to engage them into a tournament
            loopCounter = 0
            while len(sorted_fitness) >= tournamentSize:            # iterate through all individuals
                #print('fitness array length: ' + str(len(fitness)))
                selectedInd = np.random.choice(range(len(sorted_fitness)), tournamentSize, replace = False)
                selectedInd.sort()                                  # select random contestants and sort them by index (i.e. by fitness))
                # DEBUG
                #print('selected contestants for tournament:\n' + str(selectedInd))

                # General implementation
                #winIndex = np.zeros([int(tournamentSize/2)])
                #for ik in range(int(tournamentSize/2)):
                    #winIndex[ik] = fitness[selectedInd[tournamentSize - 1 - ik]][1]   # the fittest ind are retrieved from the sorted fitness array
                    #contestants[ik,:] = np.array(population[winIndex[ik],:])

                # hardcoded for performance gain
                winIndex1 = sorted_fitness[selectedInd[tournamentSize - 1]] # the fittest ind is retrieved from the sorted fitness array
                contestants[0,:] = np.array(population[winIndex1,:])
                winIndex2 = sorted_fitness[selectedInd[tournamentSize - 2]] # the second fittest ind is retrieved from the sorted fitness array
                contestants[1,:] = np.array(population[winIndex2,:])

                # 3.1 step => Generate new offsprig by Crossover or mutation
                r = np.random.random()
                if r >= crossMutProb:
                    contestants[2,:],contestants[3,:] = Crossover(contestants[0,:], contestants[1,:], crossoverProb)
                else:
                    contestants[2,:],contestants[3,:] = Mutate(contestants[0,:], contestants[1,:], mutationProb)

                # 3.2 => Delete contestants from fitness array
                iCounter = 0
                for ix in selectedInd:
                    index = ix - iCounter
                    # DEBUG
                    #print('deleting ' + str(index) + ' entry:' + str(fitness[index]))
                    sorted_fitness = np.delete(sorted_fitness, index)      # WARNING does this really work?
                    iCounter += 1
                # DEBUG
                #print('sorted fitness array, after deleting:\n' + str(fitness))

                # 3.3 => Save best individuals and offspring for new generation
                for jk in range(tournamentSize):
                    index = eliteNum + (loopCounter*tournamentSize) + jk
                    #print('=> elitNum: ' + str(eliteNum) + ', loopCounter: ' + str(loopCounter) + ', jk: ' + str(jk))
                    tempPopulation[index] = contestants[jk,:]
                loopCounter += 1
            # loop over population

            population = np.array(tempPopulation)

            # Timing!
            end_time_generation = time.time()
            secs = end_time_generation - start_time_generation
            generationAvg += secs
            #print('time to complete generation: {} m {:.3f} s'.format(int(secs/60), 60*((secs/60)-int(secs/60))))
            
        # Loop over generations

        #print('avg time for generation: {} m {:.3f} s'.format(int(generationAvg/nOfGenerations/60), 60*((generationAvg/nOfGenerations/60)-int(generationAvg/nOfGenerations/60))))

        # Create filename: unique, related to current time
        popFileID = '{0:%Y%m%d_%H%M%S_%f}'.format(dt.now())
        networkFileName = 'populations/{0}.csv'.format(popFileID)
        
        # Store filenames to associate with main run
        populationFiles.append('{0}.csv'.format(popFileID))
        
        # Save generated network, unique for each run
        with open(networkFileName, 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in bestIndividuals]
            
        benchmakingData[iRun,:] = np.array([generationAvg, generationAvg/nOfGenerations])
        statsData[iRun,:,:] = np.array(fitnessInfo)    
        
        print('#')
        iRun += 1 
    # loop over runs
    
    runsBenchAvg = np.mean(benchmakingData, axis = 0, dtype = np.float64)
    runsStatsAvg = np.mean(statsData, axis = 0, dtype = np.float64).reshape(nOfGenerations*2)
    
    # Save time measures, totals per run. One file 
    with open(benchmarkFile, 'a') as bchfile:
        writer = csv.writer(bchfile)
        # np.savetxt(bchfile, runsBenchAvg, delimiter=',')
        # csvfile.write('{}\n'.format(runsBenchAvg))
        writer.writerow(runsBenchAvg)
        
    # Save fitness measures, information per generation 
    with open(statsFile, 'a') as statsfile:
        writer = csv.writer(statsfile)
        # np.savetxt(statsfile, runsStatsAvg.transpose, delimiter=',')
        # statsfile.write('{}\n'.format(runsStatsAvg))
        writer.writerow(runsStatsAvg)
        
    # write file containing information about runs
    with open(runsMainFile,'a') as csvfile:
        csvfile.write('# Population files:\n')
        [csvfile.write('{}\n'.format(r)) for r in populationFiles]
        csvfile.write('\n')
