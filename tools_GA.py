import os
import time
import random
import numpy as np
#from main import *
#from numba import jit

# Evaluate variable using the fitness function

# @jit
#def EvaluateIndividual(individual, timeSteps, iGen, nNodes, nLattice, mode):
    #totSum = 0.
    ##print('generating wMatrix...')
    #wMatrix = population[individual,:].reshape(nNodes,nNodes)
    ##print('process: {} has recovered xMatrix!'.format(os.getpid()))
    #deltaM = sim(wMatrix, timeSteps, iGen, nNodes, individual, nLattice, mode)
    #deltaMatrix = np.array(deltaM)
    ##m, n = deltaMatrix.shape()
    ##m = 50
    #for ix in range(nLattice):
        #for jx in range(nLattice):
            #totSum += deltaMatrix[ix,jx]
    ## DEBUG
    ##print('total sum on delta matrix: ' + str(totSum))
    #if totSum <= int((nLattice**2)*0.1) or totSum == int(nLattice**2):
        #fitness[individual] = 0.
    #else:
        #fitness[individual] = 1. - (1./(nLattice**2))*totSum
    #return fitness
# # EvaluateIndividual

#@jit
#def EvaluateIndividual2(individual, timeSteps, iGen, nNodes, nLattice, mode):
#    print('process: {} here!! individual: {}'.format(os.getpid(), individual))
#    fitness[individual] =
    #print('process: {} here!! array:\n {}'.format(os.getpid(), population[individual,:]))
    #fitness[individual] = 0.
# EvaluateIndividual

# Dirac's delta function
#def DiracDelta(a, b):
    #if a == b:
        #return 0
    #else:
        #return 1
# DiracDelta

# Crossover: length preserving
#TODO implement length change
#@jit
def Crossover(parent1, parent2, crossoverProb):
    r = np.random.random()
    if  r < crossoverProb:
        #someOtherNumber,nGenes = parent1.shape                # Both chromosomes must have the same length!
        crossoverPoint = np.random.randint(1,len(parent1))      # set crossover point
        # DEBUG
        #print('crossover point = ' + str(crossoverPoint))
        offspring1 = np.array(parent1)                          # copy parents
        offspring2 = np.array(parent2)

        for iz in range(len(parent1)):
            if iz < crossoverPoint:                             # exchange elements until crossoverPoint
                val1 = offspring1[iz]
                val2 = offspring2[iz]
                offspring1[iz] = val2
                offspring2[iz] = val1

        return offspring1, offspring2
    else:
        return parent1, parent2
# Crossover

# Mutation
#@jit
def Mutate(parent1, parent2, mutationProb):
    r = np.random.random()
    if  r < mutationProb:
        #someOtherNumber,nGenes = chromosome1.shape
        pos1 = int(np.random.uniform(0,len(parent1)))                      # get random position to apply mutation
        pos2 = int(np.random.uniform(0,len(parent2)))
        mutation1 = np.random.normal(loc = 0.0, scale = np.sqrt(0.7))   # get mutation value from normal dist: mu = 0, sigma squared = 0.7
        mutation2 = np.random.normal(loc = 0.0, scale = np.sqrt(0.7))

        # DEBUG
        #print('Mutations: ' + str(mutation1) + ', ' + str(mutation2))
        mutatedChromosome1 = np.array(parent1)                          # create offspring from parent
        mutatedChromosome1[pos1] += mutation1                           # apply mutation

        mutatedChromosome2 = np.array(parent2)
        mutatedChromosome2[pos2] += mutation2

        return mutatedChromosome1, mutatedChromosome2
    else:
        return parent1, parent2
# Mutation

def DifferentIndex(x1, y1):
    if x1 == y1:
        return False
    else:
        return True
