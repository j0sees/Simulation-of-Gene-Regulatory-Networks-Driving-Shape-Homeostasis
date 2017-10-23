#import sys                                  # to get command line args
#import os                                   # to handle paths
#import time                                 # to get system time
#import numpy as np  
#import matplotlib as plt
import main_GA
import subprocess as sp

def ProcInds():
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

def ProcChunks():
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

def ProcDefaultChunks():
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

def GetfitnessStats():
    """
    Test processesvs default chunksize
    """
    rep = 10 #list(range(10,21))               # list of number of processes to test
    iRep = 0
    while iRep < rep:
        #main_GA iProc iPop 'test_20171019_{}_procs_{}_inds'.format(iProc, iPop)
        #sp.call('main_GA.py {0} {1} test_20171019_{0}_procs_{1}_inds'.format(iProc, iPop))
        string = './main_GA.py test_20171023_{0}_cicle_50_inds_50_gen_1_cs'.format(iRep)
        print('Evaluating: {}'.format(string))
        subproc = sp.Popen(string, shell = True)
        print('waiting...')
        subproc.wait()
        iRep += 1

def NNodesvsFitness():
    nNodesList = list(range(8,26))
    reps = 10
    for iNode in nNodesList:
        iRep = 0
        while iRep < reps:
            string = './main_GA.py {1} test_20171023_{0}_cycle_{1}_nodes_50_inds_20_gen_1_cs'.format(iRep, iNode)
            print('Evaluating: {}'.format(string))
            subproc = sp.Popen(string, shell = True)
            print('waiting...')
            subproc.wait()
            iRep += 1

#def FitnessGen(nGen):
    #"""
    #Get fitness of best individual after and average fitness after n generations
    #"""

if __name__ == '__main__':
    # ProcInds()
    # ProcChunks()
    # ProcDefaultChunks()
    # GetfitnessStats()
    NNodesvsFitness()
