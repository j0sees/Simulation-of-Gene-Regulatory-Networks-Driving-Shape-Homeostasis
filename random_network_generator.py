# -*- coding: latin_1 -*-
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import csv
import sys


if __name__ == '__main__':
    # Parameters
    nNodes = 25#[8, 10, 15, 20, 25]
    popSize = int(sys.argv[1])
    population = np.zeros([popSize,nNodes**2])
    infofFileID = '{0:%Y%m%d_%H%M%S_%f}'.format(dt.now())

    #----------------------------#
    #      Network generator     #
    #----------------------------#
    
    #np.random.seed()
    randPop = -1. + 2.*np.random.rand(popSize, nNodes**2)
    #population = -1. + 2.*np.random.rand(popSize, nGenes)
    np.copyto(population, randPop)
 
    #---------------------------#
    #       Graph storing       #
    #---------------------------#
    # Create filename: unique, related to current time
    popFileID = '{0:%Y%m%d_%H%M%S_%f}'.format(dt.now())
    networkFileName = 'populations/{0}_random.csv'.format(popFileID)
    print('Generated network file name:\n{}'.format(popFileID))
    # Save generated network, unique for each run
    with open(networkFileName, 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in population]
