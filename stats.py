import sys                                  # to get command line args
#import os                                   # to handle paths
#import time                                 # to get system time
import numpy as np  
#import matplotlib as plt
from datetime import datetime as dt
import main_GA
from tools import *
import subprocess as sp
import csv

def ProcInds(timedateStr):
    """
    Test processes vs number of individuals
    """
    procList = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] #list(range(10,21))               # list of number of processes to test
    popList = [1, 2, 3, 4, 5] #list(range(102,150,4))
    for iProc in procList:
        for iPop in popList:
            #main_GA iProc iPop 'test_20171019_{}_procs_{}_inds'.format(iProc, iPop)
            #sp.call('main_GA.py {0} {1} test_20171019_{0}_procs_{1}_inds'.format(iProc, iPop))
            string = './main_GA.py {0} {1} test_20171021_{0}_procs_{1}_inds'.format(iProc, iPop)
            print('Evaluating: {}'.format(string))
            subproc = sp.Popen(string, shell = True)
            print('waiting...')
            subproc.wait()

def ProcChunks(timedateStr):
    """
    Test processes vs chunk size
    """
    procList = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] #list(range(10,21))               # list of number of processes to test
    chunkList = [1, 2, 3, 4, 5] #list(range(102,150,4))
    for iChunk in chunkList:
        for iProc in procList:
            #main_GA iProc iPop 'test_20171019_{}_procs_{}_inds'.format(iProc, iPop)
            #sp.call('main_GA.py {0} {1} test_20171019_{0}_procs_{1}_inds'.format(iProc, iPop))
            string = './main_GA.py {0} {1} test_20171021_{0}_procs_{1}_chunk'.format(iProc, iChunk)
            print('Evaluating: {}'.format(string))
            subproc = sp.Popen(string, shell = True)
            print('waiting...')
            subproc.wait()
        with open('test_results_nProcs_vs_ChunkSize.stats', 'a') as csvfile:
            csvfile.write('\n\n')

def ProcDefaultChunks(timedateStr):
    """
    Test processes vs chunk size
    """
    procList = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] #list(range(10,21))               # list of number of processes to test
    #chunkList = [1, 2, 3, 4, 5] #list(range(102,150,4))
    #for iChunk in chunkList:
    for iProc in procList:
        #main_GA iProc iPop 'test_20171019_{}_procs_{}_inds'.format(iProc, iPop)
        #sp.call('main_GA.py {0} {1} test_20171019_{0}_procs_{1}_inds'.format(iProc, iPop))
        string = './main_GA.py {0} test_20171021_{0}_procs_default_chunk'.format(iProc)
        print('Evaluating: {}'.format(string))
        subproc = sp.Popen(string, shell = True)
        print('waiting...')
        subproc.wait()

def GetfitnessStats(timedateStr):
    """
    Test processes vs default chunksize
    """
    rep = 10 #list(range(10,21))               # list of number of processes to test
    iRep = 0
    while iRep < rep:
        #main_GA iProc iPop 'test_20171019_{}_procs_{}_inds'.format(iProc, iPop)
        #sp.call('main_GA.py {0} {1} test_20171019_{0}_procs_{1}_inds'.format(iProc, iPop))
        string = './main_GA.py test_20171031_{0}_cycle_50_inds_50_gen_1_cs'.format(iRep)
        print('Evaluating: {}'.format(string))
        subproc = sp.Popen(string, shell = True)
        print('waiting...')
        subproc.wait()
        iRep += 1

def NGenvsFitness(timedateStr):
    """
    Number of nodes vs fitness
    """
    nGenList = [5,10,15,20]
    nRuns = 20
    for iGen in nGenList:
        string = './main_GA.py {0} {1} {2}'.format(timedateStr, nRuns, iGen)
        print('Evaluating: {}'.format(string))
        subproc = sp.Popen(string, shell = True)
        print('waiting...')
        subproc.wait()


def NNodesvsFitness(timedateStr):
    """
    Number of nodes vs fitness
    """
    nNodesList = [25, 20, 15, 10, 8]
    nRuns = 20
    for iNode in nNodesList:
        string = './main_GA.py {0} {1} {2}'.format(timedateStr, nRuns, iNode)
        print('Evaluating: {}'.format(string))
        subproc = sp.Popen(string, shell = True)
        print('waiting...')
        subproc.wait()
        
def CrossProbvsFitness(timedateStr):
    """
    Number of nodes vs fitness
    """
    crossList = np.linspace(0, 1, num = 21)
    nRuns = 5
    for iCross in crossList:
        string = './main_GA.py {0} {1} {2}'.format(nRuns, iCross, timedateStr)
        print('Evaluating: {}'.format(string))
        subproc = sp.Popen(string, shell = True)
        print('waiting...')
        subproc.wait()
# CrossProbvsFitness()

def RnnDynamics(timedateStr):
    scale = 100
    delta = 1./scale                                        # increment for input values
    sgfInit = 0.                                         # initial value
    lgfInit = 0.                                         # initial value    
    nNodes = 25                                        # 
    maxChemVal = 1                                          # max value of the pheromones to evaluate
    counters = 4#8                                        # 4 states + 4 directions
    reps = 100                                           # number of repetitions to get to the fixed points
    network = '20171129_113233_571431_sorted'      # network to run
    fileName = '{}'.format(timedateStr)                 # filename
    ind = 386# sys.argv[2]                                  # individual to run
    maxVal = 0.5                                        # min value fo states
    xThreshold = 0.5                                    # threshold value fo states
    yThreshold = 0.01                                   # get state value    
    inputs = np.zeros([nNodes])
    inputs[0] = sgfInit
    inputs[1] = lgfInit
    nBoundary = 0.25
    sBoundary = 0.5
    eBoundary = 0.75

    #data = np.zeros([reps, 4])
    wMatrix = GetrNN('populations/{}.csv'.format(network), ind)
    chemMap = np.zeros([int(maxChemVal/delta), int(maxChemVal/delta)])
    
    #with open('chem_maps/{}.csv'.format(fileName), 'a') as csvfile:
    #    csvfile.write('# {}\n# sgf_amount\tlgf_amount\tquietCount\tsplitCount\tmoveCount\tdieCount\n'.format(network))
        # replace while for a for loop and use np.linspace(0,5,501)
    sgfInit = 0
    while sgfInit <= maxChemVal:
        lgfInit = 0
#        print('testing: SGF = {:.3f}'.format(sgfInit))
        while lgfInit <= maxChemVal:
            V = np.zeros([nNodes])
            tStep = 0
            countsArray = np.zeros(counters)
            # countsArray = [quiet, split, move, die, north, south, east, west]
            while tStep < reps:
                inputs[0] = sgfInit
                inputs[1] = lgfInit
                V = RecurrentNeuralNetwork(inputs, wMatrix, V)
                iStatus = V[2] #np.random.random() #O[0]        # Proliferate:  Split
                jStatus = V[3] #np.random.random() #O[1]        # Move:         Move
                kStatus = V[4] #np.random.random() #O[2]        # Apoptosis:    Die
                if iStatus < xThreshold and jStatus < xThreshold and kStatus < xThreshold:
                    countsArray[0] += 1 #'Quiet'
                else:
                    for ix in iStatus, jStatus, kStatus:
                        if maxVal < ix:
                            maxVal = ix
                    if abs(maxVal - iStatus) <= yThreshold:
                        countsArray[1] += 1 #'Split'
                    elif abs(maxVal - jStatus) <= yThreshold:
                        countsArray[2] += 1 #'Move'
                    else:
                        countsArray[3] += 1 #'Die'
                        # boundaries for orientation
                #arrow = V[7]  #np.random.random()
                #if arrow < sBoundary:
                    #if arrow < nBoundary:
                        ## orientation North
                        #countsArray[4] += 1
                    #else:
                        ## orientation South
                        #countsArray[5] += 1
                #else: 
                    #if arrow < eBoundary:
                        ## orientation East
                        #countsArray[6] += 1
                    #else:   #arrow < wBoundary:
                        ## orientation West
                        #countsArray[7] += 1
                tStep += 1
            
            indexes = np.argsort(countsArray)
            if countsArray[indexes[-1]] - countsArray[indexes[-2]] <= 10:
                chemMap[int(lgfInit*scale), int(sgfInit*scale)] = 0
            else:
                if indexes[-1] == 0:
                    chemMap[int(lgfInit*scale), int(sgfInit*scale)] = 1
                elif indexes[-1] == 1:
                    chemMap[int(lgfInit*scale), int(sgfInit*scale)] = 2
                elif indexes[-1] == 2:
                    chemMap[int(lgfInit*scale), int(sgfInit*scale)] = 3
                else:
                    chemMap[int(lgfInit*scale), int(sgfInit*scale)] = 4
                    
            lgfInit += delta
        #with open('chem_maps/{}.csv'.format(fileName), 'a') as csvfile:
        #    csvfile.write('\n\n')
        sgfInit += delta
    with open('chem_maps/{}.csv'.format(fileName), 'w') as csvfile:
       writer = csv.writer(csvfile)
       [writer.writerow(row) for row in chemMap]
    #with open(networkFileName, 'w') as csvfile:
        #writer = csv.writer(csvfile)
        #[writer.writerow(r) for r in bestIndividuals]
       #csvfile.write('\n\n')
       #csvfile.write('{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}\n'.format(sgfInit, lgfInit, qCount, sCount, mCount, dCount))

# RnnDynamics

#def FitnessGen(nGen):
    #"""
    #Get fitness of best individual after and average fitness after n generations
    #"""

if __name__ == '__main__':
    comment_string = sys.argv[1]
    timedateStr = '{0:%Y%m%d_%H%M%S}'.format(dt.now())
    runsMainFile = 'runs/run_{0}.log'.format(timedateStr)

    # WARNING 
    # it's not allowed to run several test at the same time since the args for the GA have to change for every single test!!
    # also, if they use the same timedate string the same run log file will be used!!
    
    with open(runsMainFile,'a') as csvfile:
        csvfile.write('# Run description:\n')
        csvfile.write('# {}\n\n'.format(comment_string))
    
    # ProcInds(timedateStr)
    # ProcChunks(timedateStr)
    # ProcDefaultChunks(timedateStr)
    # GetfitnessStats(timedateStr)
    # NNodesvsFitness(timedateStr)
    RnnDynamics(timedateStr)
    # CrossProbvsFitness(timedateStr)
    # NGenvsFitness(timedateStr)
