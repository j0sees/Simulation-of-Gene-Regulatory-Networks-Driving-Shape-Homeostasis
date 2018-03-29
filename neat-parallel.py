# -*- coding: utf-8 -*-
"""
A parallel version of XOR using neat.parallel.

Since XOR is a simple experiment, a parallel version probably won't run any
faster than the single-process version, due to the overhead of
inter-process communication.

If your evaluation function is what's taking up most of your processing time
(and you should check by using a profiler while running single-process),
you should see a significant performance improvement by evaluating in parallel.

This example is only intended to show how to do a parallel experiment
in neat-python.  You can of course roll your own parallelism mechanism
or inherit from ParallelEvaluator if you need to do something more complicated.
"""

from __future__ import print_function
from datetime import datetime as dt
import math
import os
import sys
import time
import main_GA
import neat
import pickle
#import json
import visualize
import subprocess as sp
import numpy as np

def EvaluateIndividual(genome, config):
    totSum = 0.
    nLattice = 50
    timeSteps = 200

    network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    deltaMatrix = main_GA.sim(network, timeSteps, nLattice)

    for ix in range(nLattice):
        for jx in range(nLattice):
            totSum += deltaMatrix[ix,jx]

    fit = 1. - (1./(nLattice**2))*totSum
    return fit
# EvaluateIndividual

def run(config_file, nWorkers, nGen, timedateStr, nUniqueGenomes):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, 
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, 
                         neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    #enc = sys.getdefaultencoding()
    #print('=> before encoding: {}'.format(enc))

    #reload(sys)  
    #sys.setdefaultencoding('utf-8')

    #enc = sys.getdefaultencoding()
    #print('=> after encoding: {}'.format(enc))

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(nWorkers, EvaluateIndividual)
    winner = p.run(pe.evaluate, nGen)

    # Save the winner.
#    filename = 'genomes/{}_winner_genome'.format(timedateStr)
    filename = 'genomes/{}_best_unique_genomes'.format(timedateStr)
#    filename2 = 'genomes/{}_winner_genome1'.format(timedateStr)
#    filename3 = 'genomes/{}_winner_genome2'.format(timedateStr)
    config.save('genomes/{}_config'.format(timedateStr))
    
    # Save best unique genomes 
    unique_genomes = stats.best_unique_genomes(nUniqueGenomes)

#    with open(filename, 'wb') as f:
#        pickle.dump(winner, f, 2)

    with open(filename, 'wb') as f:
        pickle.dump(unique_genomes, f, 2)

    # Log statistics.
    stats.save()

    # Display the winning genome.
    #print('\nBest genome:\n{!s}'.format(winner))
    #print('=> Plots: testing...')
    node_names = {-1:'SGF', -2:'LGF', 0:'Proliferate', 1:'Migrate', 2:'Apoptosis', 3:'SGF Prod', 4:'LGF Prod', 5:'Polarisation'}
    for igen in range(len(unique_genomes)):
        visualize.draw_net(config, unique_genomes[igen], view=False, filename='plots/{0}/{0}_best_unique_network_{1}'.format(timedateStr, igen+1), node_names=node_names)
    #print('\tnetworks... DONE!')
    visualize.plot_stats(stats, ylog=False, view=False, filename='plots/{0}/{0}_avg-fitness.svg'.format(timedateStr))
    #print('\tstats... DONE!')
    visualize.plot_species(stats, view=False, filename='plots/{0}/{0}_speciation.svg'.format(timedateStr))
    print('\tplots... DONE!')
    
    # rename stat files to particular names
    rename1 = 'mv fitness_history.csv stats/{}_fitness_history.csv'.format(timedateStr)
    rename2 = 'mv speciation.csv stats/{}_speciation.csv'.format(timedateStr)
    rename3 = 'mv species_fitness.csv stats/{}_species_fitness.csv'.format(timedateStr)
    subproc = sp.call(rename1, shell = True)
    subproc = sp.call(rename2, shell = True)
    subproc = sp.call(rename3, shell = True)
#    time.sleep(5)
    print('Finished.\n')
    
def SetupConfigFile(configFile, comp_thres, weight_coeff, disj_coeff, p_conn):

    #c_prob = [0.7]#,0.5,0.7]#np.linspace(0.1,1,10)
    #n_prob = [0.0] #np.linspace(0.01,0.1,10)

    current_time = '{0:%Y%m%d_%H%M%S_%f}'.format(dt.now())

    mkdir = 'mkdir plots/{0}'.format(current_time)
    cp = 'cp config.cfg {0}'.format(configFile)
    threshold = 'sed -i "s/compatibility_threshold = NaN/compatibility_threshold = {0}/" {1}'.format(comp_thres, configFile)
    weight = 'sed -i "s/compatibility_weight_coefficient   = NaN/compatibility_weight_coefficient   = {0}/" {1}'.format(weight_coeff, configFile)
    disjoint = 'sed -i "s/compatibility_disjoint_coefficient = NaN/compatibility_disjoint_coefficient = {0}/" {1}'.format(disj_coeff, configFile)
    conn = 'sed -i "s/partial_nodirect NaN/partial_nodirect {0}/" {1}'.format(p_conn, configFile)

    for ix in mkdir, cp, threshold, weight, disjoint, conn:
        sp.call(ix, shell = True)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, configFile)
    
    return config_path


if __name__ == '__main__':
    nWorkers = 10
    nGen = 10
    nUniqueGenomes = 20
    configFile = 'config-ca'
    comp_thres = 101
    weight_coeff = 101
    disj_coeff = 101
    p_conn = 101

    print('=> Running NEAT...\n')
    
    # Run NEAT algorithm
    config_path = SetupConfigFile(configFile, comp_thres, weight_coeff, disj_coeff, p_conn)
    run(config_path, nWorkers, nGen, current_time, nUniqueGenomes)

    rm = 'rm {0}'.format(configFile)
    sp.call(rm, shell = True)

    print('Finished.\n')
