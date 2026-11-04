import multiprocessing as mp
import sys
from contextlib import contextmanager
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import core.main as main
import evolution.main_ga as main_GA


def partialEval(ind):
    timeSteps = 200
    nNodes = 25
    nLattice = 50
    return EvaluateIndividual(timeSteps, nNodes, nLattice, ind)


def EvaluateIndividual(timeSteps, nNodes, nLattice, ind):
    totSum = 0.0
    wMatrix = networkArrays[ind]
    deltaM = main_GA.sim(wMatrix, timeSteps, 0, nNodes, ind, nLattice, True)
    deltaMatrix = np.array(deltaM)

    for ix in range(nLattice):
        for jx in range(nLattice):
            totSum += deltaMatrix[ix, jx]

    fit = 1.0 - (1.0 / (nLattice**2)) * totSum
    return fit


@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


if __name__ == '__main__':
    fileName = sys.argv[1]
    csvFile = 'populations/{}.csv'.format(fileName)
    nProcs = 20

    with open(csvFile, 'r', encoding='utf-8') as csvfile:
        networkContainer = np.loadtxt(csvfile, delimiter=',')
    nNets, nNodes = networkContainer.shape

    networkArrays = networkContainer.reshape(nNets, int(np.sqrt(nNodes)), int(np.sqrt(nNodes)))

    index_list = [x for x in range(nNets)]
    fitnessArray = np.zeros(nNets)
    with poolcontext(processes=nProcs) as pool:
        fitnessArray = pool.map(partialEval, index_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('fitness')
    ax.set_title('histogram')
    ax.hist(fitnessArray)
    plt.savefig('test.png')
