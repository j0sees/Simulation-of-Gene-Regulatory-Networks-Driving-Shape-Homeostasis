import sys                                  # to pass command line args
import time                                 # to get system time
#import random
import numpy as np
#import matplotlib.pyplot as plt
# self made classes
from cell_agent import *                    # it is allowed to call from this class because there's an __init__.py file in this directory
from tools import *
#from plot import *
#import csv
import cProfile
# from numba import jit

#@jit
def sim(wMatrix, timeSteps, iGen, nNodes, individual, nLattice, mode):
    """
    Parameters: sim(wMatrix, numberOfTimeSteps, NumberOfGeneration, nNodes, individual, nLattice, mode)
    # mode = True: cell_system as fitness function
    # mode = False: cell_system as display system
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       PARAMETERS                 #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # TODO: organize in different categories...
    #nLattice = 5                              # TODO change name
    #timeSteps = 40                              # Number of simulation time steps
    cellGrid = np.zeros([nLattice,nLattice,3])  # Initialize empty grid
    SGF_read = 0.                                # in the future values will be read from the grid
    LGF_read = 0.
    ix = int(nLattice/2)                        # Initial position for the mother cell
    iy = int(nLattice/2)                        # Initial position for the mother cell
    iTime = 0                                   # time counter

    cellList = []                               # List for cell agents

    # SGF/LGF dynamics parameters
    deltaT = 1.                                  # time step for discretisation [T]
    deltaR = 1.                                  # space step for discretisation [L]
    deltaS = 0.5                                # decay rate for SGF
    deltaL = 0.1                                 # decay rate for LGF
    diffConst = 1.#0.05                              # diffusion constant D [dimentionless]
    t_matrix = GenerateTMatrix(nLattice)        # T matrix for LGF operations
    i_matrix = GenerateIMatrix(nLattice)        # I matrix for LGF operations
    
    # timing variables!
    tmpListLoopAvg = 0
    chemicalsUpdateAvg = 0
    plotUpdateAvg = 0
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       INITIALIZATION             #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # create mother cell and update the grid with its initial location
    cellList.append(cell(ix,iy,wMatrix))
    cellGrid[ix][iy][0] = 1
    #print('Initial grid:\n' + str(cellGrid[:,:,0]))
    #cellGrid[ix][iy][2] = 400.

    ##### Timing!
    #start_time_figurecall = time.time()
    ## Plot figure and subplots
    #cellsFigure, cellsSubplot, sgfSubplot, lgfSubplot, cellPlot, sgfPlot, lgfPlot = Environment.CellsGridFigure(nLattice, mode)
    #end_time_figurecall = time.time()
    #secs = end_time_figurecall - start_time_figurecall

    #print('time taken to call figures, subplots, plots: {:.5f} s'.format(secs))

    # DEBUG
    #print('Time running...')
    #### Timing!
    start_time_mainLoop = time.time()
    while iTime < timeSteps:
        # DEBUG
        #print('\n######### time step #{}'.format(iTime))

        ## decay chemicals in spots where there is some but no cell

        # this matrixes must be updated everytime so that if there's no production in one spot that spot contains a zero
        # but must not lose contained information, i.e. must use it before setting it to zero
        sigma_m = np.zeros([nLattice,nLattice])     # matrix representation of SGF production
        lambda_m = np.zeros([nLattice,nLattice])    # matrix representation of LGF production

        tmpCellList = list(cellList)                                # a copy of the list of current cells is used to iterate over all the cells

        # Timing!
        start_time_tmpListLoop = time.time()
        tmpCellListLength = len(tmpCellList)
        while len(tmpCellList) > 0:                                 # while  the tmp list of cells is longer than 1
            # 1st step => choose a random cell from the list of existing cells
            rndCell = np.random.randint(len(tmpCellList))
            # store lattice size
            tmpCellList[rndCell].border = nLattice                  # TODO rethink this
            #tmpCellList[rndCell].nNodes = nNodes                   # WARNING hardcoded

            # 2nd step => read chemicals
            SGF_reading, LGF_reading = tmpCellList[rndCell].Sense(cellGrid)

            # 3rd step => random cell should decide and action
            tmpCellList[rndCell].GenerateStatus(SGF_reading, LGF_reading)     # get status of this cell

            # 4th step => update SGF and LGF amounts on the 'production' matrices sigma & lambda
            # production matrices get updated values
            sigma_m[tmpCellList[rndCell].xPos,tmpCellList[rndCell].yPos] = tmpCellList[rndCell].sgfAmount
            lambda_m[tmpCellList[rndCell].xPos,tmpCellList[rndCell].yPos] = tmpCellList[rndCell].lgfAmount

            # DEBUG
            #print('\ncell number: ' + str(len(cellList)) + '\nCell status: ' + str(tmpCellList[rndCell].state))# + '\n')
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            #        Cell Action            #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # according to cell status perform action: split or stay quiet
            if tmpCellList[rndCell].state == 'Quiet':               # Check the state
                tmpCellList[rndCell].Quiet(cellGrid)                # call method that performs selected action
                del tmpCellList[rndCell]                            # delete cell from temporal list

            elif tmpCellList[rndCell].state == 'Split':
                tmpCellList[rndCell].Split2(cellGrid,cellList)
                del tmpCellList[rndCell]

            elif tmpCellList[rndCell].state == 'Move':
                tmpCellList[rndCell].Move2(cellGrid)
                del tmpCellList[rndCell]

            else: # Die
                tmpCellList[rndCell].Die(cellGrid)                  # Off the grid, method also changes the "amidead" switch to True
                del tmpCellList[rndCell]
        # while
        #### Timing!
        end_time_tmpListLoop = time.time()
        secs = end_time_tmpListLoop - start_time_tmpListLoop
        #print('time taken to loop through all living cells: {:.5f}, number of cells: {}'.format(secs, tmpCellListLength))
        tmpListLoopAvg += secs

        # A list of cells that "died" is stored to later actually kill the cells...
        listLength = len(cellList) - 1
        for jCell in range(listLength,-1,-1):                       # checks every cell and if it was set to die then do, in reverse order
            #print('len(cellList): ' + str(len(cellList)) + '. Current element: ' + str(jCell))
            if cellList[jCell].amidead:
                del cellList[jCell]

        ### TEST! equivalent to: cellList[cell].'status'(param_x,param_y)
        #    state = getattr(tmpCellList[rndCell], status)
        #    action = getattr(tmpCellList[rndCell], state)
        #    action(cellGrid, cellList)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    SGF/LGF diffusion and/or decay     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #### Timing!
        #start_time_chemicalsUpdate = time.time()
        #cellGrid[:,:,1] = SGFDiffEq(cellGrid[:,:,1], sigma_m, deltaS, deltaT)
        #cellGrid[:,:,2] = LGFDiffEq(i_matrix, t_matrix, cellGrid[:,:,2], lambda_m, deltaL, deltaT, deltaR, diffConst)
        #### Timing!
        #end_time_chemicalsUpdate = time.time()
        #secs = end_time_chemicalsUpdate - start_time_chemicalsUpdate
        #print('time taken to update chemicals: {:.5f}'.format(secs))
        #chemicalsUpdateAvg += secs

        #chemsum = 0
        #for iPos in range(nLattice):
            #for jPos in range(nLattice):
                #chemsum += cellGrid[iPos,jPos,2]
                #if cellGrid[iPos,jPos,2] < 0.01:
                    #cellGrid[iPos,jPos,2] = 0
        #print('grid after update...\n' + str(cellGrid[:,:,2]))
        #print('################################LGF total = ' + str(chemsum))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #         Plot               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #print('updated grid:\n' + str(cellGrid[:,:,0]))

        #if mode == True:
            #if iTime == int(timeSteps/2) - 1:
                #halfwayStruct = np.array(cellGrid[:,:,0])
                #print('\nHalfway structure:\n {}'.format(halfwayStruct))
                #Environment.AntGridPlot(cellGrid,
                                        #nLattice,
                                        #cellsFigure,
                                        #cellsSubplot,
                                        #sgfSubplot,
                                        #lgfSubplot,
                                        #cellPlot,
                                        #sgfPlot,
                                        #lgfPlot,
                                        #iTime,
                                        #iGen,
                                        #individual,
                                        #mode)

            #elif iTime == timeSteps - 1:
                #finalStruct = np.array(cellGrid[:,:,0])
                #print('\nFinal structure:\n {}'.format(finalStruct))
                #Environment.AntGridPlot(cellGrid,
                                        #nLattice,
                                        #cellsFigure,
                                        #cellsSubplot,
                                        #sgfSubplot,
                                        #lgfSubplot,
                                        #cellPlot,
                                        #sgfPlot,
                                        #lgfPlot,
                                        #iTime,
                                        #iGen,
                                        #individual,
                                        #mode)
        #else:
            ## Timing!
            #start_time_plotUpdate = time.time()
            #Environment.AntGridPlot(    cellGrid,
                                        #nLattice,
                                        #cellsFigure,
                                        #cellsSubplot,
                                        #sgfSubplot,
                                        #lgfSubplot,
                                        #cellPlot,
                                        #sgfPlot,
                                        #lgfPlot,
                                        #iTime,
                                        #iGen,
                                        #individual,
                                        #mode)
            ##### Timing!
            #end_time_plotUpdate = time.time()
            #secs = end_time_plotUpdate - start_time_plotUpdate
            ##print('time taken to update plots: {:.5f}'.format(secs))
            #plotUpdateAvg += secs
            ##time.sleep(0.1)
        iTime += 1

        #if len(cellList) == 0:
            #halfwayStruct = np.zeros([nLattice,nLattice])
            #finalStruct = np.ones([nLattice,nLattice])
        #    break
    # while
    # Timing!
    end_time_mainLoop = time.time()
    secs = end_time_mainLoop - start_time_mainLoop
    #print('\nTime taken in main loop: {:.5f} s'.format(secs))
    #print('Avg time updating chemicals: {:.5f} s'.format(chemicalsUpdateAvg/timeSteps))
    #print('Avg time taken looping through all living cells: {:.5f} s'.format(tmpListLoopAvg/timeSteps))
    #print('Avg time taken to update plots: {:.5f} s'.format(plotUpdateAvg/timeSteps))
    # DEBUG
    # print(str(timeSteps)+' time steps complete')

    # Timing!
    #start_time_finalFunctions = time.time()

    # halfwayStruct = GetStructure(halfwayStruct, nLattice)
    # finalStruct = GetStructure(finalStruct, nLattice)
    #
    # deltaMatrix = np.zeros([nLattice,nLattice])
    # for ik in range(nLattice):
    #     for jk in range(nLattice):
    #         if halfwayStruct[ik,jk] != finalStruct[ik,jk]:
    #             deltaMatrix[ik,jk] = 1
    # # Timing!
    # end_time_finalFunctions = time.time()
    # secs = end_time_finalFunctions - start_time_finalFunctions
    # #print('\ntime taken to get delta matrix:' + str(secs))

    # DEBUG
    #print('half way structure:\n' + str(halfwayStruct))
    #print('final structure:\n' + str(finalStruct))
    #print('delta matrix:\n' + str(deltaMatrix))

    #return deltaMatrix
    #return

if __name__ == '__main__':
    # if executed as main module!
    #print('Test run...')
    individual = int(sys.argv[1])
    nNodes = sys.argv[2]
    timeSteps = 200
    iGen = 0
    nLattice = 50
    mode = False
    fileName = sys.argv[3] #'benchmark_test_ozzy_20171012_crossP0.5a'
    print('Input file: {}\nTime steps: {}, Lattice Size: {}, Individual #{}'.format(fileName, timeSteps, nLattice, individual + 1))
    
    # use: os.path.join()
    wMatrix = GetrNN('populations/' + fileName + '.csv', individual, nNodes)
    #wMatrix = wMatrix.reshape(nNodes,nNodes)
    #cProfile.run('sim(wMatrix,    timeSteps,  iGen, nNodes, individual, nLattice, mode)')
    # parameters
    sim(wMatrix,    timeSteps,  iGen, nNodes, individual, nLattice, mode)
#else:
    # if called from another script
