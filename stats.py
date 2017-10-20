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
    procList = [10, 11, 19, 7, 18] #list(range(10,21))               # list of number of processes to test
    popList = [110, 114, 126] #list(range(102,150,4))
    for iProc in procList:
        for iPop in popList:
            #main_GA iProc iPop 'test_20171019_{}_procs_{}_inds'.format(iProc, iPop)
            #sp.call('main_GA.py {0} {1} test_20171019_{0}_procs_{1}_inds'.format(iProc, iPop))
            string = './main_GA.py {0} {1} test_20171020_{0}_procs_{1}_inds'.format(iProc, iPop)
            print('Evaluating: {}'.format(string))
            subproc = sp.Popen(string, shell = True)
            print('waiting...')
            subproc.wait()

#def FitnessGen(nGen):
    #"""
    #Get fitness of best individual after and average fitness after n generations
    #"""

if __name__ == '__main__':
    ProcInds()
