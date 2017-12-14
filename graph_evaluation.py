import sys
import main
import main_GA
import numpy as np
from datetime import datetime as dt
from contextlib import contextmanager
#from functools import partial
#import matplotlib
# Valid strings are ['GTK', 'GTKAgg', 'GTKCairo', 'MacOSX', 'Qt4Agg', 'Qt5Agg', 'TkAgg', 'WX', 'WXAgg', 'GTK3Cairo', 'GTK3Agg', 'WebAgg', 'nbAgg', 'agg', 'cairo', 'gdk', 'pdf', 'pgf', 'ps', 'svg', 'template']
#matplotlib.use('Qt5Agg')
import multiprocessing as mp
import matplotlib.pyplot as plt
import csv

def partialEval(ind):
    # call the target function
    #print('ts {}, nN {}, nL {}, ind {}'.format(timeSteps, nNodes, nLattice, ind))
    timeSteps = 200
    nNodes = 25
    #wMatrix = population[individual,:]
    nLattice = 50
    return EvaluateIndividual(timeSteps, nNodes, nLattice, ind)

def EvaluateIndividual(timeSteps, nNodes, nLattice, ind):
    totSum = 0.
    
    wMatrix = networkArrays[ind]
    #print('process: {} is running sim with individual: {}!'.format(os.getpid(), individual))
    deltaM = main_GA.sim(wMatrix, timeSteps, nNodes, nLattice)
    
    deltaMatrix = np.array(deltaM)

    for ix in range(nLattice):
        for jx in range(nLattice):
            totSum += deltaMatrix[ix,jx]
    
    fit = 1. - (1./(nLattice**2))*totSum
    return fit
# EvaluateIndividual

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

if __name__ == '__main__':
    
    fileName = sys.argv[1]
    csvFile = 'populations/{}.csv'.format(fileName)
    nProcs = 20
    #chunkSize = 10
    #--------------------------#
    #       Fitness            #
    #--------------------------#
    
    with open(csvFile, 'r') as csvfile:
        #reader = csv.reader(csvfile)
        networkContainer = np.loadtxt(csvfile, delimiter = ',')
    nNets, nNodes = networkContainer.shape
    
    networkArrays = networkContainer.reshape(nNets, int(np.sqrt(nNodes)), int(np.sqrt(nNodes)))

    index_list = [ x for x in range(nNets)]
    fitnessArray = np.zeros(nNets)
    with poolcontext(processes = nProcs) as pool:
        fitnessArray = pool.map(partialEval, index_list)
    sortedIndexes = np.argsort(fitnessArray)
    
    arrayFile = '{}_fitness.csv'.format(fileName)
    with open(arrayFile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fitnessArray)
        writer.writerow(sortedIndexes)
        # Save generated network, unique for each run

    sortedArrayFile = 'populations/{}_sorted.csv'.format(fileName)
    sortedArray = np.zeros([nNets, nNodes])
    for r in range(nNets):
        sortedArray[r,:] = networkContainer[sortedIndexes[r],:]
        
    with open(sortedArrayFile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in sortedArray]
