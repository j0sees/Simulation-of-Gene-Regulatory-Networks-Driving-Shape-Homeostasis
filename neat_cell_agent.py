import numpy as np
#from tools import flatList, CheckifOccupied, CheckifPreferred, SGFDiffEq, LGFDiffEq, 
from tools import *
import neat
#from tools import flatList as fl
#from numba import jit

class cell:
    # defines whats needed when a new agent (Cell) of this class is created
    def __init__(self, yPos, xPos, network):
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
        #self.wMatrix = np.array(w)
        #self.WMatrix = W
        #self.genome = genome
        #self.config = config
        #self.network = neat.nn.RecurrentNetwork.create(self.genome,self.config)
        self.network = network
#        self.nNodes = nodes                            # WARNING hardcoded!
#        self.nInputs = 2
#        self.V = np.zeros([self.nNodes])
    # self

    #   Values stored in grid according to state:
    #       0   =>  spot has been always empty i.e. available
    #       1   =>  quiet cell, or move/split failed/didn't reach the activation value
    #       2   =>  moving cell
    #       3   =>  divided cell

    def Sense(self, grid):
        # sense chemicals from the grid
        SGF_reading = grid[self.yPos, self.xPos][0] # grid contains two values on each coordinate:
        LGF_reading = grid[self.yPos, self.xPos][1] # occupation (boolean), SGF level, LGF level
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

        #inputs = np.zeros([self.nNodes])
        inputs = [SGF_lecture, LGF_lecture]
        #inputs[0] = SGF_lecture
        #inputs[1] = LGF_lecture
        #self.V = RecurrentNeuralNetwork(inputs, self.wMatrix, self.V)
        self.network.reset()
        outputs = self.network.activate(inputs)
        self.neighbourList = [[self.yPos - 1, self.xPos], [self.yPos + 1, self.xPos], [self.yPos, self.xPos - 1], [self.yPos, self.xPos + 1]]
        #border = self.border
        # possible states: split, move, die
        #iStatus = self.V[2] #np.random.random() #O[0]        # Proliferate:  Split
        #jStatus = self.V[3] #np.random.random() #O[1]        # Move:         Move
        #kStatus = self.V[4] #np.random.random() #O[2]        # Apoptosis:    Die
        iStatus = outputs[0] # Proliferate:  Split
        jStatus = outputs[1] # Move:         Move
        kStatus = outputs[2] # Die
        # values for SGF and LGF
        self.sgfAmount = outputs[3]
        self.lgfAmount = outputs[4]

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
            arrow = outputs[5]
            # oriented according to numpy order v>, not usual >^
            if arrow < sBoundary:
                if arrow < nBoundary:
                    #xCoord = self.xPos 
                    #yCoord = self.yPos - 1
                    self.orientation = self.neighbourList[0]
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
        #print('Current pos: [{}, {}], arrow: {:.3f}, preferred direction: {}'.format(self.yPos, self.xPos, arrow, self.orientation))

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
        grid[self.yPos][self.xPos] = 1
        self.quietCounter += 1
    # Quiet

    #@jit
    def Die(self, grid):
        self.deathCounter += 1
        if self.deathCounter == self.deathTime:             # if cell set to die then do
            self.amidead = True
            # DEBUG
            #print('cell died!')
            grid[self.yPos][self.xPos] = 0
        else:                                               # otherwise stay quiet
            # DEBUG
            #print('Death counter = ' + str(self.deathCounter))
            grid[self.yPos][self.xPos] = 1
    # Die

    def Move(self, grid):
        # create a list with the four Von Neumann neighbours
        #finalList = []
        availableSpots = []
        needOtherNeighbours = True
#        gridDim = grid.shape
#        semiFlatGrid = [flatList(grid[r,:,0]) for r in range(gridDim[0])]
#        flatGrid = flatList(semiFlatGrid)
        #print('moving... preferred position: {}'.format(self.orientation))
        for neighbr in self.neighbourList:                               # for each possible neighbour:
            try: # the exception IndexError will determine if the neighbour is inbounds
                if CheckifOccupied(neighbr[1], neighbr[0], grid):   # if its occupied
                    # DEBUG
                    #print('{}: occupied...'.format(neighbr))                                       
                    continue
                else:   # if neighbour is not occupied
                    xOri = self.orientation[1]
                    yOri = self.orientation[0]
                    if CheckifPreferred(xOri, yOri, neighbr[1], neighbr[0]):    # Check if is preferred
                        grid[yOri][xOri] = 2      # new position gets a 2 value to mark as moving cell
                        grid[self.yPos][self.xPos] = 0        # old position gets a -1 value to indicate that there was a cell there before
                        self.xPos = xOri                                  # update position
                        self.yPos = yOri
                        needOtherNeighbours = False
                        # DEBUG
                        #print('{}: moved to preferred position!'.format(neighbr))                                       
                        break
                    else:
                        # DEBUG
                        #print('{}: available!'.format(neighbr))                                       
                        availableSpots.append(neighbr)                     # list with other available neighbours
            except IndexError:
                #print('{}: out of bounds!!'.format(neighbr))                                       
                continue
        if needOtherNeighbours:
            if len(availableSpots) > 0:
                r = np.random.randint(len(availableSpots))
                #print('prefered position not available. Moving to {}'.format(availableSpots[r]))
                grid[availableSpots[r][0]][availableSpots[r][1]] = 2        # new position gets a 2 value to mark as moving cell
                grid[self.yPos][self.xPos] = 0          # old position gets a -1 value to indicate that there was a cell there before
                self.yPos = availableSpots[r][0]                                  # update position
                self.xPos = availableSpots[r][1]
                # DEBUG
                #print('cell moved!')
            else:
                #print('moving failed... stay quiet')
                grid[self.yPos][self.xPos] = 1                       # if moving fails then cell is marked as quiet
        #print('Grid:\n{}'.format(grid[:,:,0]))
    ## Move

    # OrientedMove, works with orientation ON and OFF

    def Split(self, grid, cellList):
        self.splitCounter += 1
        if self.splitCounter == self.splitTime:
            #print('split counter = {}'.format(self.splitCounter))
            self.splitCounter = 0
            availableSpots = []
            needOtherNeighbours = True
#            gridDim = grid.shape
#            semiFlatGrid = [flatList(grid[r,:,0]) for r in range(gridDim[0])]
#            flatGrid = flatList(semiFlatGrid)
            #print('splitting... preferred position: {}'.format(self.orientation))
            for neighbr in self.neighbourList:                               # for each possible neighbour:
                try: # the exception IndexError will determine if the neighbour is inbounds
                    if CheckifOccupied(neighbr[1], neighbr[0], grid):   # if its occupied
                        # DEBUG
                        #print(str(neighbr) + ': neighbour occupied')
                        #print('{}: occupied...'.format(neighbr))                                       
                        continue
                    else:   # if neighbour is not occupied
                        yOri = self.orientation[0]
                        xOri = self.orientation[1]
                        if CheckifPreferred(xOri, yOri, neighbr[1], neighbr[0]):    # Check if is preferred
                            grid[yOri][xOri] = 1         # new position gets a 1 value to mark as new quiet cell
                            grid[self.yPos][self.xPos] = 3                               # mark as splitting cell
                            cellList.append(cell(yOri, xOri, self.network))
                            needOtherNeighbours = False
                            #print('{}: new cell at preferred position!'.format(neighbr))                                       
                            break
                            # DEBUG
                            #print(str(neighbr) + ': preferred position available')                                       
                        else:
                            # DEBUG
                            #print(str(neighbr) + ': available neighbour')
                            #print('{}: available!'.format(neighbr))                                       
                            availableSpots.append(neighbr)                     # list with other available neighbours
                except IndexError:
                    #print('{}: out of bounds!!'.format(neighbr))                                       
                    continue
            if needOtherNeighbours:
                if len(availableSpots) > 0:
                    r = np.random.randint(len(availableSpots))
                    #print('preferred position not available. New cell at {}'.format(availableSpots[r]))
                    grid[availableSpots[r][0]][availableSpots[r][1]] = 1         # new position gets a 1 value to mark as new quiet cell
                    grid[self.yPos][self.xPos] = 3                               # mark as splitting cell
                    cellList.append(cell(availableSpots[r][0], availableSpots[r][1],  self.network))
                    # DEBUG
                    #print('cell moved!')
                else:
                    #print('moving failed... stay quiet')
                    grid[self.yPos][self.xPos] = 1                       # if splitting fails then cell is marked as quiet
            #print('Grid:\n{}'.format(grid[:,:,0]))
    ## Split

    # works with polarisation ON and OFF
    # initial for and then if are the same as in Move2, might be useful to use a single function

# Cell
