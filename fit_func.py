import numpy as np
import neat_cell_agent as nca                    # it is allowed to call from this class because there's an __init__.py file in this directory
import tools

def CellularSystem(network, timeSteps, nLattice):
    """
    Parameters: sim(wMatrix, numberOfTimeSteps, nNodes, nLattice)
    # In ozzy the simulation works solely as a fitness function,
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       PARAMETERS                 #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # TODO: organize in different categories...
    npCellGrid = np.zeros([nLattice,nLattice])    # Initialize empty grid
    semiFlatGrid = [tools.flatList(npCellGrid[r,:]) for r in range(nLattice)]
    cellGrid = tools.flatList(semiFlatGrid)
    #cellGrid = np.zeros([nLattice,nLattice])
    chemGrid = np.zeros([nLattice,nLattice,2])  # empty grid
    iTime = 0                                   # time counter

    cellList = []                               # List for cell agents

    t_matrix = tools.GenerateTMatrix(nLattice)        # T matrix for LGF operations
    i_matrix = tools.GenerateIMatrix(nLattice)        # I matrix for LGF operations

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #       INITIALIZATION             #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # create mother cell and update the grid with its initial location
    cellList.append(nca.cell(iy,ix,network))
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

        # A list of cells that "died" is stored to later actually kill the cells...
        listLength = len(cellList) - 1
        for jCell in range(listLength,-1,-1):                   # checks every cell and if it was set to die then do, in reverse order
            if cellList[jCell].amidead:
                del cellList[jCell]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    SGF/LGF diffusion and/or decay     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        chemGrid[:,:,0] = tools.SGFDiffEq(chemGrid[:,:,0], sigma_m)
        chemGrid[:,:,1] = tools.LGFDiffEq(i_matrix, t_matrix, chemGrid[:,:,1], lambda_m)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #    Test grid to discard trivial cases #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(cellList) == 0 or quietCounter == len(cellList):       # if cells die during the simulation return two different structs
            halfwayStruct = np.zeros([nLattice,nLattice])
            finalStruct = np.ones([nLattice,nLattice])
            break
        elif iTime == int(timeSteps/2) - 1:                           # special cases get tested halfway through the simulation
            if len(cellList) <= int((nLattice**2)*0.01) or len(cellList) >= int((nLattice**2)*0.9):        # If there are no cells 
                halfwayStruct = np.zeros([nLattice,nLattice])       # return two completely different structure matrices to get 0 fitness
                finalStruct = np.ones([nLattice,nLattice])
                break
            else:
                halfwayStruct = np.array(cellGrid)
        elif iTime == timeSteps - 1:
            if len(cellList) >= int((nLattice**2)*0.9):                      # If cells fill space 
                halfwayStruct = np.zeros([nLattice,nLattice])       # return two completely different structure matrices to get 0 fitness
                finalStruct = np.ones([nLattice,nLattice])
            else:
                finalStruct = np.array(cellGrid)
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

#def PeriodicCellularSystem(network, timeSteps, nLattice):
    #"""
    #Parameters: sim(wMatrix, numberOfTimeSteps, nNodes, nLattice)
    ## In ozzy the simulation works solely as a fitness function,
    #"""
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ##       PARAMETERS                 #
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ## TODO: organize in different categories...
    #npCellGrid = np.zeros([nLattice,nLattice])    # Initialize empty grid
    #semiFlatGrid = [tools.flatList(npCellGrid[r,:]) for r in range(nLattice)]
    #cellGrid = tools.flatList(semiFlatGrid)
    ##cellGrid = np.zeros([nLattice,nLattice])
    #chemGrid = np.zeros([nLattice,nLattice,2])  # empty grid
    #SGF_read = 0.                               # in the future values will be read from the grid
    #LGF_read = 0.
    #ix = int(nLattice/2)                        # Initial position for the mother cell
    #iy = int(nLattice/2)                        # Initial position for the mother cell
    #iTime = 0                                   # time counter

    #cellList = []                               # List for cell agents

    ## SGF/LGF dynamics parameters
    #deltaT = 1.                                 # time step for discretisation [T]
    #deltaR = 1.                                 # space step for discretisation [L]
    #deltaS = 0.5                                # decay rate for SGF
    #deltaL = 0.1                                # decay rate for LGF
    #diffConst = 1.#0.05                                     # diffusion constant D [dimentionless]
    #t_matrix = tools.GeneratePeriodicTMatrix(nLattice)      # T matrix for LGF operations
    #i_matrix = tools.GenerateIMatrix(nLattice)              # I matrix for LGF operations

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ##       INITIALIZATION             #
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ## create mother cell and update the grid with its initial location
    #cellList.append(nca.cell(iy,ix,network))
    #cellGrid[ix][iy] = 1

    #while iTime < timeSteps:
        ### decay chemicals in spots where there is some but no cell

        ## this matrixes must be updated everytime so that if there's no production in one spot that spot contains a zero
        ## but must not lose contained information, i.e. must use it before setting it to zero
        #sigma_m = np.zeros([nLattice,nLattice])             # matrix representation of SGF production
        #lambda_m = np.zeros([nLattice,nLattice])            # matrix representation of LGF production

        #tmpCellList = list(cellList)                        # a copy of the list of current cells is used to iterate over all the cells

        #tmpCellListLength = len(tmpCellList)
        #quietCounter = 0
        #while len(tmpCellList) > 0:                 # while the tmp list of cells is longer than 1
            ## 1st step => choose a random cell from the list of existing cells
            #rndCell = np.random.randint(len(tmpCellList))

            ## 2nd step => read chemicals
            #SGF_reading, LGF_reading = tmpCellList[rndCell].Sense(chemGrid)

            ## 3rd step => random cell should decide and action
            #tmpCellList[rndCell].GenerateStatus(SGF_reading, LGF_reading)     # get status of this cell

            ## 4th step => update SGF and LGF amounts on the 'production' matrices sigma & lambda
            ## production matrices get updated values
            #sigma_m[tmpCellList[rndCell].yPos,tmpCellList[rndCell].xPos] = tmpCellList[rndCell].sgfAmount
            #lambda_m[tmpCellList[rndCell].yPos,tmpCellList[rndCell].xPos] = tmpCellList[rndCell].lgfAmount

            ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            ##        Cell Action            #
            ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            ## according to cell status perform action: split or stay quiet
            #if tmpCellList[rndCell].state == 'Quiet':           # Check the state
                #tmpCellList[rndCell].Quiet(cellGrid)            # call method that performs selected action
                #quietCounter += 1
                #del tmpCellList[rndCell]                        # delete cell from temporal list

            #elif tmpCellList[rndCell].state == 'Split':
                #tmpCellList[rndCell].PeriodicSplit(cellGrid,cellList)
                #del tmpCellList[rndCell]

            #elif tmpCellList[rndCell].state == 'Move':
                #tmpCellList[rndCell].PeriodicMove(cellGrid)
                #del tmpCellList[rndCell]

            #else: # Die
                #tmpCellList[rndCell].Die(cellGrid)              # Off the grid, method also changes the "amidead" switch to True
                #del tmpCellList[rndCell]
        ## while

        ## A list of cells that "died" is stored to later actually kill the cells...
        #listLength = len(cellList) - 1
        #for jCell in range(listLength,-1,-1):                   # checks every cell and if it was set to die then do, in reverse order
            #if cellList[jCell].amidead:
                #del cellList[jCell]

        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ##    SGF/LGF diffusion and/or decay     #
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #chemGrid[:,:,0] = tools.SGFDiffEq(chemGrid[:,:,0], sigma_m, deltaS, deltaT)
        #chemGrid[:,:,1] = tools.LGFDiffEq(i_matrix, t_matrix, chemGrid[:,:,1], lambda_m, deltaL, deltaT, deltaR, diffConst)

        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ##    Test grid to discard trivial cases #
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #if len(cellList) == 0 or quietCounter == len(cellList):       # if cells die during the simulation return two different structs
            #halfwayStruct = np.zeros([nLattice,nLattice])
            #finalStruct = np.ones([nLattice,nLattice])
            #break
        #elif iTime == int(timeSteps/2) - 1:                           # special cases get tested halfway through the simulation
            #if len(cellList) <= int((nLattice**2)*0.01) or len(cellList) >= int((nLattice**2)*0.9):        # If there are no cells 
                #halfwayStruct = np.zeros([nLattice,nLattice])       # return two completely different structure matrices to get 0 fitness
                #finalStruct = np.ones([nLattice,nLattice])
                #break
            #else:
                #halfwayStruct = np.array(cellGrid)
        #elif iTime == timeSteps - 1:
            #if len(cellList) >= int((nLattice**2)*0.9):                      # If cells fill space 
                #halfwayStruct = np.zeros([nLattice,nLattice])       # return two completely different structure matrices to get 0 fitness
                #finalStruct = np.ones([nLattice,nLattice])
            #else:
                #finalStruct = np.array(cellGrid)
        #iTime += 1
    ## while
    
    #halfwayStruct = tools.GetStructure(halfwayStruct, nLattice)
    #finalStruct = tools.GetStructure(finalStruct, nLattice)

    #deltaMatrix = np.zeros([nLattice,nLattice])
    #for ik in range(nLattice):
        #for jk in range(nLattice):
            #if halfwayStruct[ik,jk] != finalStruct[ik,jk]:
                #deltaMatrix[ik,jk] = 1

    #return deltaMatrix
### sim
