import numpy as np
from tools import *
import neat

class cell:
    # defines whats needed when a new agent (Cell) of this class is created
    def __init__(self, yPos, xPos, network):
        self.state = 'Quiet'                        # State of the cell. DEFAULT: quiet
        self.xPos = xPos                            # Initial position on x axis
        self.yPos = yPos                            # Initial position on y axis
        self.compass = True                         # Polarisation: WARNING ON/OFF => True/False
        self.orientation = [self.yPos,self.xPos]    # Preferred direction. DEFAULT: own position
        self.neighbourList = [[self.yPos - 1, self.xPos], [self.yPos + 1, self.xPos], [self.yPos, self.xPos - 1], [self.yPos, self.xPos + 1]]
        self.splitCounter = 0                       # Counter for splitting
        self.splitTime = 1                          # Time scale for splitting
        self.deathCounter = 0                       # Countdown to extinction
        self.deathTime = 1                          # Time scale for dying
        self.amidead = False                        # Cell dead or alive
        self.quietCounter = 0                       # Quiet counter
        self.sgfAmount = 0                          # Amount of "growth factor" to deposit in the grid
        self.lgfAmount = 0
        self.network = network
        self.network.reset()                        # reset network, only once, when the cell is initialised

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

    def GenerateStatus(self, SGF_lecture, LGF_lecture):
        # neural network generates a status based on the reads
        inputs = [SGF_lecture, LGF_lecture]
        # self.network.reset()
        outputs = self.network.activate(inputs)
        self.neighbourList = [[self.yPos - 1, self.xPos], [self.yPos + 1, self.xPos], [self.yPos, self.xPos - 1], [self.yPos, self.xPos + 1]]
        
        # Cellular states
        iStatus = outputs[0] # Proliferate: Split
        jStatus = outputs[1] # Migrate:     Move
        kStatus = outputs[2] # Apoptosis:   Die
        # values for SGF and LGF
        self.sgfAmount = outputs[3]
        self.lgfAmount = outputs[4]

        xThreshold = 0.5
        yThreshold = 0.01

        # ORIENTATION:
        if self.compass:
            # boundaries for orientation
            nBoundary = 0.25
            sBoundary = 0.5
            eBoundary = 0.75
            #wBoundary = 1
            arrow = outputs[5]
            # oriented according to numpy order v>, not usual >^
            if abs(arrow - sBoundary) <= yThreshold:
                self.orientation = [self.yPos, self.xPos]
            elif arrow < sBoundary:
                if arrow < nBoundary:
                    # orientation North
                    self.orientation = self.neighbourList[0]
                else:
                    # orientation South
                    self.orientation = self.neighbourList[1]
            else: 
                if arrow < eBoundary:   # and arrow > sBoundary:
                    # orientation East
                    self.orientation = self.neighbourList[3]
                else:   #arrow < wBoundary:
                    # orientation West
                    self.orientation = self.neighbourList[2]
        else:   # update orientation as current position if compass False
            self.orientation = [self.yPos, self.xPos]

        # Generate state
        maxVal = 0.5

        # TODO is the order of this operations the ideal?
        if iStatus < xThreshold and jStatus < xThreshold and kStatus < xThreshold:
            self.state = 'Quiet'
        else:
            for ix in iStatus, jStatus, kStatus:
                if maxVal < ix:
                    maxVal = ix
            if abs(maxVal - iStatus) <= yThreshold:
                self.state = 'Split'
            elif abs(maxVal - jStatus) <= yThreshold:
                self.state = 'Move'
            else:
                self.state = 'Die'
    # GenerateStatus

    def Quiet(self,grid):
        grid[self.yPos][self.xPos] = 1
        self.quietCounter += 1
    # Quiet

    def Die(self, grid):
        self.deathCounter += 1
        if self.deathCounter == self.deathTime:             # if cell set to die then do
            self.amidead = True
            grid[self.yPos][self.xPos] = 0
        else:                                               # otherwise stay quiet
            grid[self.yPos][self.xPos] = 1
    # Die

    def Move(self, grid):
        availableSpots = []
        needOtherNeighbours = True
        for neighbr in self.neighbourList:                                  # for each possible neighbour:
            try:                                                            # IndexError exception will determine if the neighbour is inbounds
                if CheckifOccupied(neighbr[1], neighbr[0], grid):           # if its occupied
                    continue
                else:                                                       # if neighbour is not occupied
                    xOri = self.orientation[1]
                    yOri = self.orientation[0]
                    if CheckifPreferred(xOri, yOri, neighbr[1], neighbr[0]):# Check if is preferred
                        grid[yOri][xOri] = 2                                # new position gets a 2 value to mark as moving cell
                        grid[self.yPos][self.xPos] = 0                      # old position gets a 0 as it is empty
                        self.xPos = xOri                                    # update position
                        self.yPos = yOri
                        needOtherNeighbours = False
                        break
                    else:
                        availableSpots.append(neighbr)                      # list with other available neighbours
            except IndexError:
                continue
        if needOtherNeighbours:
            if len(availableSpots) > 0:
                r = np.random.randint(len(availableSpots))
                grid[availableSpots[r][0]][availableSpots[r][1]] = 2        # new position gets a 2 value to mark as moving cell
                grid[self.yPos][self.xPos] = 0                              # old position gets a 0 as it is empty
                self.yPos = availableSpots[r][0]                            # update position
                self.xPos = availableSpots[r][1]
            else:
                grid[self.yPos][self.xPos] = 1                              # if moving fails then cell is marked as quiet
    # Move

    def Split(self, grid, cellList):
        self.splitCounter += 1
        if self.splitCounter == self.splitTime:
            self.splitCounter = 0
            availableSpots = []
            needOtherNeighbours = True
            for neighbr in self.neighbourList:                              # for each possible neighbour:
                try:                                                        # IndexError exception will determine if the neighbour is inbounds
                    if CheckifOccupied(neighbr[1], neighbr[0], grid):       # if its occupied
                        continue
                    else:                                                   # if neighbour is not occupied
                        yOri = self.orientation[0]
                        xOri = self.orientation[1]
                        if CheckifPreferred(xOri, yOri, neighbr[1], neighbr[0]):    # Check if is preferred
                            grid[yOri][xOri] = 1                                    # new position gets a 1 value to mark as new quiet cell
                            grid[self.yPos][self.xPos] = 3                          # mark as splitting cell
                            cellList.append(cell(yOri, xOri, self.network))
                            needOtherNeighbours = False
                            break
                        else:
                            availableSpots.append(neighbr)                  # list with other available neighbours
                except IndexError:
                    continue
            if needOtherNeighbours:
                if len(availableSpots) > 0:
                    r = np.random.randint(len(availableSpots))
                    grid[availableSpots[r][0]][availableSpots[r][1]] = 1    # new position gets a 1 value to mark as new quiet cell
                    grid[self.yPos][self.xPos] = 3                          # mark as splitting cell
                    cellList.append(cell(availableSpots[r][0], availableSpots[r][1],  self.network))
                else:
                    grid[self.yPos][self.xPos] = 1                          # if splitting fails then cell is marked as quiet
    # Split
# Cell
