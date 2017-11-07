#import sys                                  # to get command line args
#import os                                   # to handle paths
#import time                                 # to get system time
#import numpy as np  
#import matplotlib as plt
from datetime import datetime as dt
import main_GA
from tools import *
import subprocess as sp

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

def NNodesvsFitness():
    """
    Number of nodes vs fitness
    """
    nNodesList = [25,15]
    nRuns = 2
    for iNode in nNodesList:
        string = './main_GA.py {0} {1}'.format(nRuns, iNode)
        print('Evaluating: {}'.format(string))
        subproc = sp.Popen(string, shell = True)
        print('waiting...')
        subproc.wait()

def RnnDynamics(timedateStr):
    step = 0.01
    nNodes = 16
    sgfInit = 0
    lgfInit = 0
    maxChem = 10
    reps = 100
    network = 'test_20171023_9_cycle_16_nodes_50_inds_20_gen_1_cs'
    fileName = '20171031_testmap_step_0.01'
    individual = 0
    maxVal = 0.5
    xThreshold = 0.5
    yThreshold = 0.01    
    inputs = np.zeros([nNodes])
    inputs[0] = sgfInit
    inputs[1] = lgfInit
    data = np.zeros([reps, 4])
    wMatrix = GetrNN('populations/' + network + '.csv', individual, nNodes)
    with open('chem_maps/' + fileName + '.csv', 'a') as csvfile:
        csvfile.write('# {}\n# sgf_amount\t lgf_amount\tquiet\tsplit\tmove\tdie\n'.format(network))
    while sgfInit <= maxChem:
        lgfInit = 0
        while lgfInit <= maxChem:
            V = np.zeros([nNodes])
            tStep = 0
            qCount = 0
            sCount = 0
            mCount = 0
            dCount = 0
            while tStep < reps:
                V = RecurrentNeuralNetwork(inputs, wMatrix, V)
                iStatus = V[2] #np.random.random() #O[0]        # Proliferate:  Split
                jStatus = V[3] #np.random.random() #O[1]        # Move:         Move
                kStatus = V[4] #np.random.random() #O[2]        # Apoptosis:    Die
                if iStatus < xThreshold and jStatus < xThreshold and kStatus < xThreshold:
                    qCount += 1 #'Quiet'
                else:
                    for ix in iStatus, jStatus, kStatus:
                        if maxVal < ix:
                            maxVal = ix
                    if abs(maxVal - iStatus) <= yThreshold:
                        sCount += 1 #'Split'
                    elif abs(maxVal - jStatus) <= yThreshold:
                        mCount += 1 #'Move'
                    else:
                        dCount += 1 #'Die'
                tStep += 1
            #print('testing: LGF = {:.2f}, SGF = {:.2f}'.format(lgfInit, sgfInit))
            with open('chem_maps/' + fileName + '.csv', 'a') as csvfile:
                #writer = csv.writer(csvfile)
                #[writer.writerow(row) for row in data]
                #csvfile.write('\n\n')
                csvfile.write('{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}\n'.format(sgfInit, lgfInit, qCount, sCount, mCount, dCount))
            lgfInit += step
        with open('chem_maps/' + fileName + '.csv', 'a') as csvfile:
            csvfile.write('\n\n')
        sgfInit += step
# RnnDynamics

#def FitnessGen(nGen):
    #"""
    #Get fitness of best individual after and average fitness after n generations
    #"""

if __name__ == '__main__':
    #timedateStr = '{:%Y%m%d}'.format(dt.now())
    # ProcInds(timedateStr)
    # ProcChunks(timedateStr)
    # ProcDefaultChunks(timedateStr)
    # GetfitnessStats(timedateStr)
    NNodesvsFitness()
    # RnnDynamics(timedateStr)
