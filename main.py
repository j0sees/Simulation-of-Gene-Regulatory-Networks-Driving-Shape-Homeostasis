import sys
import time
#import random
import numpy as np
#import matplotlib.pyplot as plt
# self made classes
from neat_cell_agent import *                    # it is allowed to call from this class because there's an __init__.py file in this directory
from tools import *
from plot import *
#import csv
import cProfile
# from numba import jit

#@jit
def sim(genome, config_path, timeSteps, nLattice, mode):
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
    npCellGrid = np.zeros([nLattice,nLattice])    # Initialize empty grid
    semiFlatGrid = [flatList(npCellGrid[r,:]) for r in range(nLattice)]
    cellGrid = flatList(semiFlatGrid)
    chemGrid = np.zeros([nLattice,nLattice,2])
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
    cellList.append(cell(ix,iy,genome,config))
    cellGrid[ix][iy] = 1
    #print('Initial grid:\n' + str(cellGrid[:,:,0]))
    #cellGrid[ix][iy][2] = 400.

    ##### Timing!
    #start_time_figurecall = time.time()
    ## Plot figure and subplots
    cellsFigure, cellsSubplot, sgfSubplot, lgfSubplot, cellPlot, sgfPlot, lgfPlot = Environment.CellsGridFigure(nLattice, mode)
    #end_time_figurecall = time.time()
    #secs = end_time_figurecall - start_time_figurecall

    #print('time taken to call figures, subplots, plots: {:.5f} s'.format(secs))

    # DEBUG
    print('Time running...')
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
            #tmpCellList[rndCell].border = nLattice                  # TODO rethink this
            #tmpCellList[rndCell].nNodes = nNodes                   # WARNING hardcoded

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
            if tmpCellList[rndCell].state == 'Quiet':               # Check the state
                tmpCellList[rndCell].Quiet(cellGrid)                # call method that performs selected action
                del tmpCellList[rndCell]                            # delete cell from temporal list

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

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    SGF/LGF diffusion and/or decay     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #### Timing!
        #start_time_chemicalsUpdate = time.time()
        chemGrid[:,:,0] = SGFDiffEq(chemGrid[:,:,0], sigma_m, deltaS, deltaT)
        chemGrid[:,:,1] = LGFDiffEq(i_matrix, t_matrix, chemGrid[:,:,1], lambda_m, deltaL, deltaT, deltaR, diffConst)
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
        Environment.AntGridPlot(    cellGrid,
                                    chemGrid,
                                    nLattice,
                                    cellsFigure,
                                    cellsSubplot,
                                    sgfSubplot,
                                    lgfSubplot,
                                    cellPlot,
                                    sgfPlot,
                                    lgfPlot,
                                    iTime,
                                    mode)
            ## Timing!
            #end_time_plotUpdate = time.time()
            #secs = end_time_plotUpdate - start_time_plotUpdate
            ##print('time taken to update plots: {:.5f}'.format(secs))
            #plotUpdateAvg += secs
            ##time.sleep(0.1)
        iTime += 1
        # this script is used to see what comes up from the main_GA, doesn't have to check for any conditions on the system, just let it run
        #if len(cellList) == 0:
            ##halfwayStruct = np.zeros([nLattice,nLattice])
            ##finalStruct = np.zeros([nLattice,nLattice])
            #break
    # while
    # Timing!
    end_time_mainLoop = time.time()
    secs = end_time_mainLoop - start_time_mainLoop
    tmpListLoopAvg += secs
    #print('\ntime taken in main loop: {:.5f}'.format(secs))
    #print('Avg time updating chemicals: {:.5f}'.format(chemicalsUpdateAvg/timeSteps))
    print('Avg time looping through cells: {:.3f}, total: {:.3f}'.format(tmpListLoopAvg/timeSteps, tmpListLoopAvg))
    #print('Avg time taken to update plots: {:.5f}'.format(plotUpdateAvg/timeSteps))
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
    print('System visualization')
    timeSteps = 200
    nLattice = 50
    mode = False
    # mode = True: cell_system as fitness function
    # mode = False: cell_system as display system
    #fileName = sys.argv[1]
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ca')
    
    #wMatrix = GetrNN(fileName,individual)
    #wMatrix = wMatrix.reshape(nNodes,nNodes)
    #cProfile.run('sim(wMatrix,    timeSteps,  iGen, nNodes, individual, nLattice, mode)')
    # parameters
    # load the winner
    with open('test_pickled_genome', 'rb') as f:
        genome = pickle.load(f)

    sim(genome, config_path, timeSteps, nLattice, mode)
    plt.close()
#else:
    # if called from another script
