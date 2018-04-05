import numpy as np
import neat_cell_agent as nca                    # it is allowed to call from this class because there's an __init__.py file in this directory
import tools

def CellularSystem(network, periodic_bound_cond, timeSteps, nLattice):
    """
    Parameters: sim(wMatrix, numberOfTimeSteps, nNodes, nLattice)
    # In ozzy the simulation works solely as a fitness function,
    """
    
    # Initialise object that contains the simulation environment
    env = tools.GridEnv(network, nLattice, periodic_bound_cond)
    
    iTime = 0                                   # time counter
    while iTime < timeSteps:
        # decay chemicals in spots where there is some but no cell
        # this matrices must be updated everytime so that if there's no production in one spot that spot contains a zero
        # but must not lose contained information, i.e. must use it before setting it to zero
        sigma_m = np.zeros([nLattice,nLattice])             # matrix representation of SGF production
        lambda_m = np.zeros([nLattice,nLattice])            # matrix representation of LGF production

        tmpCellList = list(env.cellList)                    # a copy of the list of current cells is used to iterate over all the cells
        tmpCellListLength = len(tmpCellList)

        quietCounter = 0
        while len(tmpCellList) > 0:                         # while the tmp list of cells is longer than 1
            # 1st step => choose a random cell from the list of existing cells
            rndCell = np.random.randint(len(tmpCellList))

            # 2nd step => read chemicals
            SGF_reading, LGF_reading = tmpCellList[rndCell].Sense(env.chemGrid)

            # 3rd step => random cell should decide and action
            tmpCellList[rndCell].GenerateStatus(SGF_reading, LGF_reading)     # get status of this cell

            # 4th step => update SGF and LGF amounts on the 'production' matrices sigma & lambda
            sigma_m[tmpCellList[rndCell].yPos,tmpCellList[rndCell].xPos] = tmpCellList[rndCell].sgfAmount
            lambda_m[tmpCellList[rndCell].yPos,tmpCellList[rndCell].xPos] = tmpCellList[rndCell].lgfAmount

            # 5th step => according to cell status perform action
            if tmpCellList[rndCell].state == 'Quiet':           # Check the state
                tmpCellList[rndCell].Quiet(env.cellGrid)            # call method that performs selected action
                quietCounter += 1
                del tmpCellList[rndCell]                        # delete cell from temporal list

            elif tmpCellList[rndCell].state == 'Split':
                tmpCellList[rndCell].Split(env.cellGrid, env.cellList)
                del tmpCellList[rndCell]

            elif tmpCellList[rndCell].state == 'Move':
                tmpCellList[rndCell].Move(env.cellGrid)
                del tmpCellList[rndCell]

            else: # Die
                tmpCellList[rndCell].Die(env.cellGrid)              # Off the grid, method also changes the "amidead" switch to True
                del tmpCellList[rndCell]
        # while

        # A list of cells that "died" is stored to later actually kill the cells...
        listLength = len(env.cellList) - 1
        for jCell in range(listLength,-1,-1):                   # checks every cell and if it was set to die then do, in reverse order
            if env.cellList[jCell].amidead:
                del env.cellList[jCell]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    SGF/LGF diffusion and/or decay     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        env.chemGrid[:,:,0] = tools.SGFDiffEq(env.chemGrid[:,:,0], sigma_m)
        env.chemGrid[:,:,1] = tools.LGFDiffEq(env.i_matrix, env.t_matrix, env.chemGrid[:,:,1], lambda_m)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    Test grid to discard trivial cases #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(env.cellList) == 0 or quietCounter == len(env.cellList):       # if cells die during the simulation return two different structs
            halfwayStruct = np.zeros([nLattice,nLattice])
            finalStruct = np.ones([nLattice,nLattice])
            break
        elif iTime == int(timeSteps/2) - 1:                           # special cases get tested halfway through the simulation
            if len(env.cellList) <= int((nLattice**2)*0.01) or len(env.cellList) >= int((nLattice**2)*0.9):        # If there are no cells 
                halfwayStruct = np.zeros([nLattice,nLattice])       # return two completely different structure matrices to get 0 fitness
                finalStruct = np.ones([nLattice,nLattice])
                break
            else:
                halfwayStruct = np.array(env.cellGrid)
        elif iTime == timeSteps - 1:
            if len(env.cellList) >= int((nLattice**2)*0.9):                      # If cells fill space 
                halfwayStruct = np.zeros([nLattice,nLattice])       # return two completely different structure matrices to get 0 fitness
                finalStruct = np.ones([nLattice,nLattice])
            else:
                finalStruct = np.array(env.cellGrid)
        iTime += 1
    # while
    
    halfwayStruct = tools.GetStructure(halfwayStruct, nLattice)
    finalStruct = tools.GetStructure(finalStruct, nLattice)

    deltaMatrix = np.zeros([nLattice,nLattice])
    for ik in range(nLattice):
        for jk in range(nLattice):
            if halfwayStruct[ik,jk] != finalStruct[ik,jk]:
                deltaMatrix[ik,jk] = 1

    return deltaMatrix
### sim
