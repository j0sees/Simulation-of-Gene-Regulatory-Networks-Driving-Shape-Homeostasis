#import sys                                  # to get command line args
#import os                                   # to handle paths
#import time                                 # to get system time
#import numpy as np  
#import matplotlib as plt
import main_GA
import subprocess as sp


if __name__ == '__main__':
    procList = [2, 5] #list(range(2,11))               # list of number of processes to test
    popList = [22, 102] #list(range(20,51))
    for iProc in procList:
        for iPop in popList:
            #main_GA iProc iPop 'test_20171019_{}_procs_{}_inds'.format(iProc, iPop)
            #sp.call('main_GA.py {0} {1} test_20171019_{0}_procs_{1}_inds'.format(iProc, iPop))
            print('Evaluating: ./main_GA.py {0} {1} test_20171019_{0}_procs_{1}_inds'.format(iProc, iPop))
            subproc = sp.Popen('./main_GA.py {0} {1} test_20171019_{0}_procs_{1}_inds'.format(iProc, iPop), shell = True)
            subproc.wait()
