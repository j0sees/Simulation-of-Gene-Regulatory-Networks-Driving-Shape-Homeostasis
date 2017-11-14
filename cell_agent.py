import numpy as np
from tools import *
#from tools import flatList
#from numba import jit

class cell:
    # defines whats needed when a new agent (Cell) of this class is created
    def __init__(self, yPos, xPos, w, nodes):
        self.state = 'Quiet'                        # State of the cell. DEFAULT: quiet
        self.xPos = xPos                            # Initial position on x axis
        self.yPos = yPos                            # Initial position on y axis
        self.compass = True                         # Polarisation: WARNING ON/OFF => True/False
        self.orientation = [self.yPos,self.xPos]    # Preferred direction. DEFAULT: own position
        #self.neighbourList = flatList([flatList([self.yPos - 1, self.xPos]), flatList([self.yPos + 1, self.xPos]), flatList([self.yPos, self.xPos - 1]), flatList([self.yPos, self.xPos + 1])])
        self.neighbourList = [[self.yPos - 1, self.xPos], [self.yPos + 1, self.xPos], [self.yPos, self.xPos - 1], [self.yPos, self.xPos + 1]]
        self.splitCounter = 0                       # Counter for splitting
        self.splitTime = 1                          # Time scale for splitting
        self.deathCounter = 0                       # Countdown to extinction
        self.deathTime = 1                          # Time scale for dying
        self.amidead = False                        # Cell dead or alive
        self.quietCounter = 0                       # Quiet counter
        #self.border = 0                             # size of the lattice
        self.sgfAmount = 0                          # Amount of "pheromone" to deposit in the grid
        self.lgfAmount = 0
        # Neural network stuff
        self.wMatrix = np.array(w)
        #self.WMatrix = W
        #self.phi = phi
        #self.theta = theta
        self.nNodes = nodes                            # WARNING hardcoded!
#        self.nInputs = 2
        self.V = np.zeros([self.nNodes])
    # self

    #   Values stored in grid according to state:
    #       0   =>  spot has been always empty i.e. available
    #       1   =>  quiet cell, or move/split failed/didn't reach the activation value
    #       2   =>  moving cell
    #       3   =>  divided cell

    # WARNING Do I use this method??
    #def GetNeighbours(self, grid, border):
        #neighbourList = []
        ## TODO check if this works if orientation is OFF
        ## if orientation is OFF the method returns the probable neighbours
        #if self.xPos - 1 >= 0: # if coordinate is in-bounds
            #if grid[self.xPos - 1, self.yPos][0] == 0 and self.orientation[0] != self.xPos - 1 and self.orientation[1] != self.yPos:
                ## if is not occupied and not the preferred neighbour
                #neighbourList.append([self.xPos - 1, self.yPos])
        #if self.xPos + 1 <= border:
            #if grid[self.xPos + 1, self.yPos][0] == 0 and self.orientation[0] != self.xPos + 1 and self.orientation[1] != self.yPos:
                #neighbourList.append([self.xPos + 1, self.yPos])
        #if self.yPos - 1 >= 0:
            #if grid[self.xPos, self.yPos - 1][0] == 0 and self.orientation[0] != self.xPos and self.orientation[1] != self.yPos - 1:
                #neighbourList.append([self.xPos, self.yPos - 1])
        #if self.yPos + 1 <= border:
            #if grid[self.xPos, self.yPos + 1][0] == 0 and self.orientation[0] != self.xPos and self.orientation[1] != self.yPos + 1:
                #neighbourList.append([self.xPos, self.yPos + 1])
        ## returns a list of possible neighbours which are not the prefered
        #return neighbourList
    #@jit
    def Sense(self, grid):
        # sense chemicals from the grid
        SGF_reading = grid[self.yPos, self.xPos][1] # grid contains three values on each coordinate:
        LGF_reading = grid[self.yPos, self.xPos][2] # occupation (boolean), SGF level, LGF level
        #reads = [SGF_read, LGF_read]
        return SGF_reading, LGF_reading
    # Sense

    #@jit
    def GenerateStatus(self, SGF_lecture, LGF_lecture):
        # neural network generates a status based on the reads
        #inputs = np.array([SGF_lecture, LGF_lecture])
        # Neural network first implementation
        #O = np.zeros([self.m])
        #O = NeuralNetwork(inputs, self.WMatrix, self.wMatrix, self.phi, self.theta)

        inputs = np.zeros([self.nNodes])
        inputs[0] = SGF_lecture
        inputs[1] = LGF_lecture
        self.V = RecurrentNeuralNetwork(inputs, self.wMatrix, self.V)
        #self.neighbourList = flatList([flatList([self.yPos - 1, self.xPos]), flatList([self.yPos + 1, self.xPos]), flatList([self.yPos, self.xPos - 1]), flatList([self.yPos, self.xPos + 1])])
        self.neighbourList = [[self.yPos - 1, self.xPos], [self.yPos + 1, self.xPos], [self.yPos, self.xPos - 1], [self.yPos, self.xPos + 1]]
        #border = self.border
        # possible states: split, move, die
        iStatus = self.V[2] #np.random.random() #O[0]        # Proliferate:  Split
        jStatus = self.V[3] #np.random.random() #O[1]        # Move:         Move
        kStatus = self.V[4] #np.random.random() #O[2]        # Apoptosis:    Die
        # values for SGF and LGF
        self.sgfAmount = self.V[5] #np.random.randint(5) #O[4]
        self.lgfAmount = self.V[6] #np.random.randint(5) #O[5]

        # ORIENTATION:
        # randomly sets a preferred neighbour (polarisation)
        # if the direction is out of bounds then no preferred direction is stored
        # WARNING This code need to be revisited depending on the implementation of the NN later on
        if self.compass:
            # boundaries for orientation
            nBoundary = 0.25
            sBoundary = 0.5
            eBoundary = 0.75
            #wBoundary = 1
            arrow = self.V[7]  #np.random.random()
            # oriented according to numpy order v>, not usual >^
            if arrow < sBoundary:
                if arrow < nBoundary:
                    #xCoord = self.xPos 
                    #yCoord = self.yPos - 1
                    self.orientation = self.neighbourList[2]
                # orientation North
                else:
                    # orientation South
                    # xCoord = self.xPos 
                    # yCoord = self.yPos + 1
                    self.orientation = self.neighbourList[1]
            else: 
                if arrow < eBoundary:
                    # orientation East
                    #xCoord = self.xPos + 1
                    #yCoord = self.yPos 
                    self.orientation = self.neighbourList[3]
                else:   #arrow < wBoundary:
                    # orientation West
                    #xCoord = self.xPos - 1
                    #yCoord = self.yPos
                    self.orientation = self.neighbourList[2]
            #self.orientation = [yCoord, xCoord]
        else:                                           # update orientation as current position if compass False
            self.orientation = [self.yPos, self.xPos]
        # if
        #print('neighbourList type: {}'.format(type(self.neighbourList)))
        print('Current pos: [{}, {}], arrow: {:.3f}, preferred direction: {}'.format(self.yPos, self.xPos, arrow, self.orientation))

        # Generate state
        maxVal = 0.5
        #tmpVal = 0
        xThreshold = 0.5
        yThreshold = 0.01

        # TODO is the order of this operations the ideal?
        if iStatus < xThreshold and jStatus < xThreshold and kStatus < xThreshold:
            self.state = 'Quiet'
            # DEBUG
            #print('split = ' + str(iStatus) + ', move = ' + str(jStatus) + '\ndie = ' + str(kStatus) + '. Max: quiet\n')
        else:
            for ix in iStatus, jStatus, kStatus:
                if maxVal < ix:
                    maxVal = ix
            if abs(maxVal - iStatus) <= yThreshold:
                self.state = 'Split'
            elif abs(maxVal - jStatus) <= yThreshold:
                self.state = 'Move'
            else:   # abs(maxVal - kStatus) <= yThreshold:
                self.state = 'Die'
            # DEBUG
            #print('split = ' + str(iStatus) + ', move = ' + str(jStatus) + '\ndie = ' + str(kStatus) + '. Max: '+ str(maxVal) + '\n')
    # GenerateStatus

    #@jit
    def Quiet(self,grid):
        grid[self.yPos, self.xPos,0] = 1
        self.quietCounter += 1
    # Quiet

    #@jit
    def Die(self, grid):
        self.deathCounter += 1
        if self.deathCounter == self.deathTime:             # if cell set to die then do
            self.amidead = True
            # DEBUG
            #print('cell died!')
            grid[self.yPos, self.xPos, 0] = 0
        else:                                               # otherwise stay quiet
            # DEBUG
            #print('Death counter = ' + str(self.deathCounter))
            grid[self.yPos, self.xPos, 0] = 1
    # Die

    def Move(self, grid):
        # create a list with the four Von Neumann neighbours
        #finalList = []
        availableSpots = []
        needOtherNeighbours = True
        gridDim = grid.shape
        semiFlatGrid = [flatList(grid[r,:,0]) for r in range(gridDim[0])]
        flatGrid = flatList(semiFlatGrid)
        print('moving... preferred position: {}'.format(self.orientation))
        for neighbr in self.neighbourList:                               # for each possible neighbour:
            try: # the exception IndexError will determine if the neighbour is inbounds
                if CheckifOccupied(neighbr[1], neighbr[0], flatGrid):   # if its occupied
                    # DEBUG
                    print('{}: occupied...'.format(neighbr))                                       
                    continue
                else:   # if neighbour is not occupied
                    xOri = self.orientation[1]
                    yOri = self.orientation[0]
                    if CheckifPreferred(xOri, yOri, neighbr[1], neighbr[0]):    # Check if is preferred
                        grid[yOri, xOri, 0] = 2      # new position gets a 2 value to mark as moving cell
                        grid[self.yPos, self.xPos, 0] = 0        # old position gets a -1 value to indicate that there was a cell there before
                        self.xPos = xOri                                  # update position
                        self.yPos = yOri
                        needOtherNeighbours = False
                        # DEBUG
                        print('{}: moved to preferred position!'.format(neighbr))                                       
                        break
                    else:
                        # DEBUG
                        print('{}: available!'.format(neighbr))                                       
                        availableSpots.append(neighbr)                     # list with other available neighbours
            except IndexError:
                print('{}: out of bounds!!'.format(neighbr))                                       
                continue
        if needOtherNeighbours:
            if len(availableSpots) > 0:
                r = np.random.randint(len(availableSpots))
                print('prefered position not available. Moving to {}'.format(availableSpots[r]))
                grid[availableSpots[r][0], availableSpots[r][1], 0] = 2        # new position gets a 2 value to mark as moving cell
                grid[self.yPos, self.xPos][0] = 0          # old position gets a -1 value to indicate that there was a cell there before
                self.yPos = availableSpots[r][0]                                  # update position
                self.xPos = availableSpots[r][1]
                # DEBUG
                #print('cell moved!')
            else:
                print('moving failed... stay quiet')
                grid[self.yPos, self.xPos, 0] = 1                       # if moving fails then cell is marked as quiet
        print('Grid:\n{}'.format(grid[:,:,0]))
    ## Move

    # OrientedMove, works with orientation ON and OFF
    #@jit
    #def Move2(self, grid):
        ## create a list with the four Von Neumann neighbours
        ##finalList = []
        #tmpList = []
        #movePos = []
        #needOtherNeighbours = True
        #border = self.border - 1
        #for neighbr in self.neighbourList:                               # for each possible neighbour:
            ## TODO its more likely to have neighbours than to be out of bounds... need to swap next two ifs
            #if CheckInBorders(neighbr[0], neighbr[1], border):      # if neighbour is inbunds:
                #if CheckifOccupied(neighbr[0], neighbr[1], grid):   # if its occupied
                    ## DEBUG
                    ##print(str(neighbr) + ': neighbour occupied')
                    #continue
                #else:
                    #xOri = self.orientation[0]
                    #yOri = self.orientation[1]
                    #if CheckifPreferred(xOri, yOri, neighbr[0], neighbr[1]):    # if is preferred
                        #movePos.append(neighbr[0])                  # if available, the return it as available
                        #movePos.append(neighbr[1])
                        #needOtherNeighbours = False                 # if available there's no need to find more spots
                        ## DEBUG
                        ##print(str(neighbr) + ': preferred position available')
                        #break                                       # and stop looking for available spots
                    #else:
                        ## DEBUG
                        ##print(str(neighbr) + ': available neighbour')
                        #tmpList.append(neighbr)                     # list with other available neighbours
            #else:
                ## DEBUG
                ##print(str(neighbr) + ': neighbour not in bounds')
                #continue
        ## if needed and if there's more than one spot available, the move to that spot
        #if needOtherNeighbours and len(tmpList) > 0:
            #r = np.random.randint(len(tmpList))
            #movePos.append(tmpList[r][0])
            #movePos.append(tmpList[r][1])

        #if len(movePos) > 0:
            #grid[movePos[0]][movePos[1]][0] = 2                     # new position gets a 2 value to mark as moving cell
            #grid[self.xPos][self.yPos][0] = 0                      # old position gets a -1 value to indicate that there was a cell there before
            #self.xPos = movePos[0]                                  # update position
            #self.yPos = movePos[1]
            ## DEBUG
            ##print('cell moved!')
        #else:
            #grid[self.xPos][self.yPos][0] = 1                       # if moving fails then cell is marked as quiet
            ## DEBUG
            ##print('moving failed\n')
    # Move2

    def Split(self, grid, cellList):
        self.splitCounter += 1
        if self.splitCounter == self.splitTime:
            print('split counter = {}'.format(self.splitCounter))
            self.splitCounter = 0
            availableSpots = []
            needOtherNeighbours = True
            gridDim = grid.shape
            semiFlatGrid = [flatList(grid[r,:,0]) for r in range(gridDim[0])]
            flatGrid = flatList(semiFlatGrid)
            print('splitting... preferred position: {}'.format(self.orientation))
            for neighbr in self.neighbourList:                               # for each possible neighbour:
                try: # the exception IndexError will determine if the neighbour is inbounds
                    if CheckifOccupied(neighbr[1], neighbr[0], flatGrid):   # if its occupied
                        # DEBUG
                        #print(str(neighbr) + ': neighbour occupied')
                        print('{}: occupied...'.format(neighbr))                                       
                        continue
                    else:   # if neighbour is not occupied
                        yOri = self.orientation[0]
                        xOri = self.orientation[1]
                        if CheckifPreferred(xOri, yOri, neighbr[1], neighbr[0]):    # Check if is preferred
                            grid[yOri, xOri, 0] = 1         # new position gets a 1 value to mark as new quiet cell
                            grid[self.yPos, self.xPos, 0] = 3                               # mark as splitting cell
                            cellList.append(cell(yOri, xOri, self.wMatrix, self.nNodes))
                            needOtherNeighbours = False
                            print('{}: new cell at preferred position!'.format(neighbr))                                       
                            break
                            # DEBUG
                            #print(str(neighbr) + ': preferred position available')                                       
                        else:
                            # DEBUG
                            #print(str(neighbr) + ': available neighbour')
                            print('{}: available!'.format(neighbr))                                       
                            availableSpots.append(neighbr)                     # list with other available neighbours
                except IndexError:
                    print('{}: out of bounds!!'.format(neighbr))                                       
                    continue
            if needOtherNeighbours:
                if len(availableSpots) > 0:
                    r = np.random.randint(len(availableSpots))
                    print('preferred position not available. New cell at {}'.format(availableSpots[r]))
                    grid[availableSpots[r][0], availableSpots[r][1], 0] = 1         # new position gets a 1 value to mark as new quiet cell
                    grid[self.yPos, self.xPos, 0] = 3                               # mark as splitting cell
                    cellList.append(cell(availableSpots[r][0], availableSpots[r][1], self.wMatrix, self.nNodes))
                    # DEBUG
                    #print('cell moved!')
                else:
                    print('moving failed... stay quiet')
                    grid[self.yPos, self.xPos, 0] = 1                       # if splitting fails then cell is marked as quiet
            print('Grid:\n{}'.format(grid[:,:,0]))
    ## Split

    # works with polarisation ON and OFF
    # initial for and then if are the same as in Move2, might be useful to use a single function
    #@jit
    #def Split2(self, grid, cellList):
        #self.splitCounter += 1
        #if self.splitCounter == self.splitTime:
            #self.splitCounter = 0

            ## create a list with the four Von Neumann neighbours
            #neighbourList = [[self.xPos - 1, self.yPos],[self.xPos + 1, self.yPos],[self.xPos, self.yPos - 1],[self.xPos, self.yPos + 1]]
            ##finalList = []
            #tmpList = []
            #movePos = []
            #needOtherNeighbours = True
            #border = self.border - 1

            #for neighbr in neighbourList:                                           # for each possible neighbour:
                #if CheckInBorders(neighbr[0], neighbr[1], border):                  # if neighbour is inbunds:
                    #if CheckifOccupied(neighbr[0], neighbr[1], grid):               # if its occupied
                        ## DEBUG
                        ##print(str(neighbr) + ': neighbour occupied')
                        #continue
                    #else:
                        #xOri = self.orientation[0]
                        #yOri = self.orientation[1]
                        #if CheckifPreferred(xOri, yOri, neighbr[0], neighbr[1]):    # if is preferred
                            #movePos.append(neighbr[0])                              # if available, the return it as available
                            #movePos.append(neighbr[1])
                            #needOtherNeighbours = False                             # if available the no need to find more spots
                            ## DEBUG
                            ##print(str(neighbr) + ': preferred position available')
                            #break                                                   # and stop looking for available spots
                        #else:
                            ## DEBUG
                            ##print(str(neighbr) + ': available neighbour')
                            #tmpList.append(neighbr)                                 # list with other available neighbours
                #else:
                    ## DEBUG
                    ##print(str(neighbr) + ': neighbour not in bounds')
                    #continue
            ## if needed and if there's more than one spot available, then move to that spot
            #if needOtherNeighbours and len(tmpList) > 0:
                #r = np.random.randint(len(tmpList))
                #movePos.append(tmpList[r][0])
                #movePos.append(tmpList[r][1])
                ##finalList.append(tmpList[r])
            #if len(movePos) > 0:
                #grid[self.xPos][self.yPos][0] = 3
                #cellList.append(cell(movePos[0], movePos[1],self.wMatrix,self.nNodes))
                #grid[movePos[0]][movePos[1]][0] = 1
                ## DEBUG
                ##print('new cell created!')
            #else:
                #grid[self.xPos][self.yPos][0] = 1
                ## DEBUG
                ##print('split failed\n')
        #else:
            #grid[self.xPos][self.yPos][0] = 1
            ## DEBUG
            ##print('split counter + 1\n')
    # Split2
# Cell
