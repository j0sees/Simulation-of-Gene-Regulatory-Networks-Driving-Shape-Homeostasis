import sys
import os
import time
import pickle
import numpy as np
# self made classes
import neat_cell_agent as nca                    # it is allowed to call from this class because there's an __init__.py file in this directory
import tools
import plot
import stats_plots
import subprocess as sp
import neat

#def CellularSystem(network, timeSteps, nLattice, ):
def CellularSystem(network, periodic_bound_cond, mode, location, iGenome, timeSteps, nLattice):
    """
    Parameters: sim(wMatrix, numberOfTimeSteps, NumberOfGeneration, nNodes, individual, nLattice, mode)
    # mode = True: cell_system as fitness function
    # mode = False: cell_system as display system
    """
    
    # Initialise object that contains the simulation environment
    env = tools.GridEnv(network, nLattice, periodic_bound_cond)
    # Figure objects
    cellsFigure, cellsSubplot, sgfSubplot, lgfSubplot, cellPlot, sgfPlot, lgfPlot = plot.CellsGridFigure(nLattice, mode)
    
    # SGF/LGF statistics containers
    SGF_history = np.zeros([nLattice, nLattice, timeSteps], dtype = np.float64)
    LGF_history = np.zeros([nLattice, nLattice, timeSteps], dtype = np.float64)
    
    # Cell count container
    CellCounter = np.zeros([4, timeSteps])      # Number of states x number of generations

    iTime = 0
    while iTime < timeSteps:
        # Save SGF/LGF amount to get statistics
        SGF_history[:,:,iTime] = env.chemGrid[:,:,0]
        LGF_history[:,:,iTime] = env.chemGrid[:,:,1]
        
        # Cell counters
        migr_counter = 0
        prolif_counter = 0
        quiet_counter = 0
        apopt_counter = 0
        
        # this matrixes must be updated everytime so that if there's no production in one spot that spot contains a zero
        # but must not lose contained information, i.e. must use it before setting it to zero
        sigma_m = np.zeros([nLattice,nLattice])     # matrix representation of SGF production
        lambda_m = np.zeros([nLattice,nLattice])    # matrix representation of LGF production

        tmpCellList = list(env.cellList)            # a copy of the list of current cells is used to iterate over all the cells

        while len(tmpCellList) > 0:                 # while  the tmp list of cells is longer than 1
            # 1st step => choose a random cell from the list of existing cells
            rndCell = np.random.randint(len(tmpCellList))

            # 2nd step => read chemicals
            SGF_reading, LGF_reading = tmpCellList[rndCell].Sense(env.chemGrid)

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
                tmpCellList[rndCell].Quiet(env.cellGrid)                # call method that performs selected action
                #quiet_counter += 1
                del tmpCellList[rndCell]                            # delete cell from temporal list

            elif tmpCellList[rndCell].state == 'Split':
                tmpCellList[rndCell].Split(env.cellGrid,env.cellList)
                #prolif_counter += 1
                del tmpCellList[rndCell]

            elif tmpCellList[rndCell].state == 'Move':
                tmpCellList[rndCell].Move(env.cellGrid)
                #migr_counter += 1
                del tmpCellList[rndCell]

            else: # Die
                tmpCellList[rndCell].Die(env.cellGrid)                  # Off the grid, method also changes the "amidead" switch to True
                apopt_counter += 1
                del tmpCellList[rndCell]
        # while

        # A list of cells that "died" is stored to later actually kill the cells...
        listLength = len(env.cellList) - 1
        for jCell in range(listLength,-1,-1):                       # checks every cell and if it was set to die then do, in reverse order
            if env.cellList[jCell].amidead:
                del env.cellList[jCell]

        # Count stats are saved for current time steop:
        CellCounter[:, iTime] = np.array([migr_counter, prolif_counter, quiet_counter, apopt_counter])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    SGF/LGF diffusion and/or decay     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        env.chemGrid[:,:,0] = tools.SGFDiffEq(env.chemGrid[:,:,0], sigma_m)
        env.chemGrid[:,:,1] = tools.LGFDiffEq(env.i_matrix, env.t_matrix, env.chemGrid[:,:,1], lambda_m)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #         Plot               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        plot.CellGridPlot(  env.cellGrid,
                            env.chemGrid,
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
    periodic_bound_cond = False
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
        CellularSystem(network, periodic_bound_cond, mode, location, iGenome, timeSteps, nLattice)        
        #plt.close()
