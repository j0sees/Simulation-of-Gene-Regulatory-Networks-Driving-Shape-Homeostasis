import sys
import os
import time
import pickle
import numpy as np
# self made classes
from neat_cell_agent import *                    # it is allowed to call from this class because there's an __init__.py file in this directory
from tools import *
import plot
import stats_plots
import subprocess as sp


#@jit
def sim(network, timeSteps, nLattice, mode, location, iGenome):
    """
    Parameters: sim(wMatrix, numberOfTimeSteps, NumberOfGeneration, nNodes, individual, nLattice, mode)
    # mode = True: cell_system as fitness function
    # mode = False: cell_system as display system
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       PARAMETERS                 #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # TODO: organize in different categories...
    npCellGrid = np.zeros([nLattice,nLattice])    # Initialize empty grid
    semiFlatGrid = [flatList(npCellGrid[r,:]) for r in range(nLattice)]
    cellGrid = flatList(semiFlatGrid)
    chemGrid = np.zeros([nLattice,nLattice,2])
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
    
    # SGF/LGF statistics containers
    SGF_history = np.zeros([nLattice, nLattice, timeSteps], dtype = np.float64)
    LGF_history = np.zeros([nLattice, nLattice, timeSteps], dtype = np.float64)
    
    # Cell count container
    CellCounter = np.zeros([4, timeSteps])      # Number of states x number of generations

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       INITIALIZATION             #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # create mother cell and update the grid with its initial location
    cellList.append(cell(iy,ix,network))
    cellGrid[iy][ix] = 1

    cellsFigure, cellsSubplot, sgfSubplot, lgfSubplot, cellPlot, sgfPlot, lgfPlot = plot.CellsGridFigure(nLattice, mode)

    while iTime < timeSteps:
        # Save SGF/LGF amount to get statistics
        SGF_history[:,:,iTime] = chemGrid[:,:,0]
        LGF_history[:,:,iTime] = chemGrid[:,:,1]
        
        # Cell counters
        migr_counter = 0
        prolif_counter = 0
        quiet_counter = 0
        apopt_counter = 0
        
        # this matrixes must be updated everytime so that if there's no production in one spot that spot contains a zero
        # but must not lose contained information, i.e. must use it before setting it to zero
        sigma_m = np.zeros([nLattice,nLattice])     # matrix representation of SGF production
        lambda_m = np.zeros([nLattice,nLattice])    # matrix representation of LGF production

        tmpCellList = list(cellList)                                # a copy of the list of current cells is used to iterate over all the cells

        while len(tmpCellList) > 0:                                 # while  the tmp list of cells is longer than 1
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
            # according to cell status perform action: split or stay quiet
            if tmpCellList[rndCell].state == 'Quiet':               # Check the state
                tmpCellList[rndCell].Quiet(cellGrid)                # call method that performs selected action
                #quiet_counter += 1
                del tmpCellList[rndCell]                            # delete cell from temporal list

            elif tmpCellList[rndCell].state == 'Split':
                tmpCellList[rndCell].Split(cellGrid,cellList)
                #prolif_counter += 1
                del tmpCellList[rndCell]

            elif tmpCellList[rndCell].state == 'Move':
                tmpCellList[rndCell].Move(cellGrid)
                #migr_counter += 1
                del tmpCellList[rndCell]

            else: # Die
                tmpCellList[rndCell].Die(cellGrid)                  # Off the grid, method also changes the "amidead" switch to True
                apopt_counter += 1
                del tmpCellList[rndCell]
        # while

        # A list of cells that "died" is stored to later actually kill the cells...
        listLength = len(cellList) - 1
        for jCell in range(listLength,-1,-1):                       # checks every cell and if it was set to die then do, in reverse order
            if cellList[jCell].amidead:
                del cellList[jCell]

        # Count stats are saved for current time steop:
        CellCounter[:, iTime] = np.array([migr_counter, prolif_counter, quiet_counter, apopt_counter])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    SGF/LGF diffusion and/or decay     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        chemGrid[:,:,0] = SGFDiffEq(chemGrid[:,:,0], sigma_m, deltaS, deltaT)
        chemGrid[:,:,1] = LGFDiffEq(i_matrix, t_matrix, chemGrid[:,:,1], lambda_m, deltaL, deltaT, deltaR, diffConst)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #         Plot               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        plot.CellGridPlot(  cellGrid,
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
                            mode,
                            location,
                            iGenome)
        iTime += 1
        # this script is used to see what comes up from the main_GA, doesn't have to check for any conditions on the system, just let it run

    # Plot counter stats:
    stats_plots.CounterPlots(CellCounter, location, iGenome)
    
    # Get SGF/LGF statistics
    #SGF_mean = np.mean(SGF_history, axis = 2, dtype = np.float64)
    #LGF_mean = np.mean(LGF_history, axis = 2, dtype = np.float64)
    
    #stats_plots.GF_AverageMap(SGF_mean, LGF_mean, location, iGenome)

def sim(network, timeSteps, nLattice, mode, location, iGenome):
    """
    Parameters: sim(wMatrix, numberOfTimeSteps, NumberOfGeneration, nNodes, individual, nLattice, mode)
    # mode = True: cell_system as fitness function
    # mode = False: cell_system as display system
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       PARAMETERS                 #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # TODO: organize in different categories...
    npCellGrid = np.zeros([nLattice,nLattice])    # Initialize empty grid
    semiFlatGrid = [flatList(npCellGrid[r,:]) for r in range(nLattice)]
    cellGrid = flatList(semiFlatGrid)
    chemGrid = np.zeros([nLattice,nLattice,2])
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
    t_matrix = GeneratePeriodicTMatrix(nLattice)        # T matrix for LGF operations
    i_matrix = GenerateIMatrix(nLattice)        # I matrix for LGF operations
    
    # SGF/LGF statistics containers
    SGF_history = np.zeros([nLattice, nLattice, timeSteps], dtype = np.float64)
    LGF_history = np.zeros([nLattice, nLattice, timeSteps], dtype = np.float64)
    
    # Cell count container
    CellCounter = np.zeros([4, timeSteps])      # Number of states x number of generations

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       INITIALIZATION             #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # create mother cell and update the grid with its initial location
    cellList.append(cell(iy,ix,network))
    cellGrid[iy][ix] = 1

    cellsFigure, cellsSubplot, sgfSubplot, lgfSubplot, cellPlot, sgfPlot, lgfPlot = plot.CellsGridFigure(nLattice, mode)

    while iTime < timeSteps:
        # Save SGF/LGF amount to get statistics
        SGF_history[:,:,iTime] = chemGrid[:,:,0]
        LGF_history[:,:,iTime] = chemGrid[:,:,1]
        
        # Cell counters
        migr_counter = 0
        prolif_counter = 0
        quiet_counter = 0
        apopt_counter = 0
        
        # this matrixes must be updated everytime so that if there's no production in one spot that spot contains a zero
        # but must not lose contained information, i.e. must use it before setting it to zero
        sigma_m = np.zeros([nLattice,nLattice])     # matrix representation of SGF production
        lambda_m = np.zeros([nLattice,nLattice])    # matrix representation of LGF production

        tmpCellList = list(cellList)                                # a copy of the list of current cells is used to iterate over all the cells

        while len(tmpCellList) > 0:                                 # while  the tmp list of cells is longer than 1
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
            # according to cell status perform action: split or stay quiet
            if tmpCellList[rndCell].state == 'Quiet':               # Check the state
                tmpCellList[rndCell].Quiet(cellGrid)                # call method that performs selected action
                #quiet_counter += 1
                del tmpCellList[rndCell]                            # delete cell from temporal list

            elif tmpCellList[rndCell].state == 'Split':
                tmpCellList[rndCell].PeriodicSplit(cellGrid,cellList)
                #prolif_counter += 1
                del tmpCellList[rndCell]

            elif tmpCellList[rndCell].state == 'Move':
                tmpCellList[rndCell].PeriodicMove(cellGrid)
                #migr_counter += 1
                del tmpCellList[rndCell]

            else: # Die
                tmpCellList[rndCell].Die(cellGrid)                  # Off the grid, method also changes the "amidead" switch to True
                apopt_counter += 1
                del tmpCellList[rndCell]
        # while

        # A list of cells that "died" is stored to later actually kill the cells...
        listLength = len(cellList) - 1
        for jCell in range(listLength,-1,-1):                       # checks every cell and if it was set to die then do, in reverse order
            if cellList[jCell].amidead:
                del cellList[jCell]

        # Count stats are saved for current time steop:
        CellCounter[:, iTime] = np.array([migr_counter, prolif_counter, quiet_counter, apopt_counter])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    SGF/LGF diffusion and/or decay     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        chemGrid[:,:,0] = SGFDiffEq(chemGrid[:,:,0], sigma_m, deltaS, deltaT)
        chemGrid[:,:,1] = LGFDiffEq(i_matrix, t_matrix, chemGrid[:,:,1], lambda_m, deltaL, deltaT, deltaR, diffConst)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #         Plot               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        plot.CellGridPlot(  cellGrid,
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
                            mode,
                            location,
                            iGenome)
        iTime += 1
        # this script is used to see what comes up from the main_GA, doesn't have to check for any conditions on the system, just let it run

    # Plot counter stats:
    stats_plots.CounterPlots(CellCounter, location, iGenome)
    
    # Get SGF/LGF statistics
    #SGF_mean = np.mean(SGF_history, axis = 2, dtype = np.float64)
    #LGF_mean = np.mean(LGF_history, axis = 2, dtype = np.float64)
    
    #stats_plots.GF_AverageMap(SGF_mean, LGF_mean, location, iGenome)

if __name__ == '__main__':
    #print('System visualization')
    timeSteps = 200
    nLattice = 50
    mode = False
    loc = sys.argv[1]
    timedateStr = sys.argv[2]
    location = '{}/{}'.format(loc, timedateStr)
    # mode = True: cell_system as fitness function
    # mode = False: cell_system as display system

    #fileName = 'genomes/{}_winner_genome'.format(sys.argv[1])
    fileName = 'genomes/{}_best_unique_genomes'.format(timedateStr)
    config_file = 'genomes/{}_config'.format(timedateStr)
    #config_file = '{}'.format(sys.argv[2])
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)

    # Config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # load the winner
    print('=> Working with folder: {}...'.format(timedateStr))
    with open(fileName, 'rb') as f:
        genomes = pickle.load(f)#, encoding = 'bytes')

    for iGenome in range(len(genomes)):
        #print('=> Running genome #{}'.format(iGenome))
        mkdir = 'mkdir {0}/best_unique_genome_{1}'.format(location, iGenome+1)
        subproc = sp.call(mkdir, shell = True)
        #print('genome file: {0}\nconfig file: {1}'.format(fileName, config_file))
        network = neat.nn.RecurrentNetwork.create(genomes[iGenome], config)
        sim(network, timeSteps, nLattice, mode, location, iGenome)
        #plt.close()
