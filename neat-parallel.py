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
#import main_GA
from fit_func import CellularSystem
import neat
import pickle
#import json
import visualize
import subprocess as sp
import numpy as np
import ConfigParser

def WriteConfigFile(fileName, periodic_bound_cond = False, death_cell_presence = False, nWorkers = 10, nGen = 40, nUniqueGenomes = 20, nLattice = 50, timeSteps = 400):
    #configFile = 'config-ca'
    config = ConfigParser.RawConfigParser()

    # When adding sections or items, add them in the reverse order of
    # how you want them to be displayed in the actual file.
    # In addition, please note that using RawConfigParser's and the raw
    # mode of ConfigParser's respective set functions, you can assign
    # non-string values to keys internally, but will receive an error
    # when attempting to write to a file or when you get it in non-raw
    # mode. SafeConfigParser does not allow such assignments to take place.

    config.add_section('Run settings')
    config.set('Run settings', 'periodic_bound_cond', '{}'.format(periodic_bound_cond))
    config.set('Run settings', 'death_cell_presence', '{}'.format(death_cell_presence))
    config.set('Run settings', 'nWorkers', '{}'.format(nWorkers))
    config.set('Run settings', 'nGen', '{}'.format(nGen))
    config.set('Run settings', 'nUniqueGenomes', '{}'.format(nUniqueGenomes))
    config.set('Run settings', 'timeSteps', '{}'.format(timeSteps))
    config.set('Run settings', 'nLattice', '{}'.format(nLattice))

    # Writing our configuration file to 'example.cfg'
    with open(fileName, 'w') as configfile:
        config.write(configfile)

def EvaluateIndividual(genome, config):
    totSum = 0.

    run_config = ConfigParser.RawConfigParser()
    run_config.read('run.cfg')

    periodic_bound_cond = run_config.getboolean('Run settings', 'periodic_bound_cond')
    death_cell_presence = run_config.getboolean('Run settings', 'death_cell_presence')
    timeSteps = run_config.getint('Run settings', 'timeSteps')
    nLattice = run_config.getint('Run settings', 'nLattice')

    #periodic_bound_cond = False
    #death_cell_presence = False

    network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    #deltaMatrix = main_GA.sim(network, timeSteps, nLattice)
    #deltaMatrix = CellularSystem(network, timeSteps, nLattice)
    #deltaMatrix = PeriodicCellularSystem(network, timeSteps, nLattice)    
    deltaMatrix = CellularSystem(network, periodic_bound_cond, death_cell_presence, timeSteps, nLattice)

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
    #print('Finished.\n')
    
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
    #pb = 'sed -i "s/# periodic_bound_cond = bool/# periodic_bound_cond = {0}/" {1}'.format(pb_cond, configFile)
    #dc = 'sed -i "s/# death_cell_presence = bool/# death_cell_presence = {0}/" {1}'.format(dc_cond, configFile)

    for ix in mkdir, cp, threshold, weight, disjoint, conn:#, pb, dc:
        sp.call(ix, shell = True)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, configFile)
    
    return config_path, current_time

if __name__ == '__main__':
    nWorkers = 10
    nGen = 40
    nUniqueGenomes = 20

    configFile = 'config-ca'
    comp_thres = 0.75
    weight_coeff = 1.
    disj_coeff = 0.5
    p_conn = 0.7
    periodic_bound_cond = False
    death_cell_presence = False
    run_config = 'run.cfg'

    print('\n=> Running NEAT...')

    for iBool in [[False, False],[True, False],[False, True],[True, True]]:
        periodic_bound_cond = iBool[0]
        death_cell_presence = iBool[1]

        if periodic_bound_cond:
            if death_cell_presence:
                simType_string = 'PB_DC_runs'
                print('=> Periodic boundaries and death cell presence')
            else:
                simType_string = 'PB_nDC_runs'
                print('=> Periodic boundaries and no death cell presence')
        else:
            if death_cell_presence:
                simType_string = 'nPB_DC_runs'
                print('=> No periodic boundaries and death cell presence')
            else:
                simType_string = 'nPB_nDC_runs'
                print('=> No periodic boundaries and no death cell presence')

        WriteConfigFile(run_config, periodic_bound_cond, death_cell_presence, nWorkers, nGen, nUniqueGenomes)

        # Run NEAT algorithm
        config_path, current_time = SetupConfigFile(configFile, comp_thres, weight_coeff, disj_coeff, p_conn)
        run(config_path, nWorkers, nGen, current_time, nUniqueGenomes)

        mv = 'mv {0} plots/{1}/'.format(configFile, current_time)
        mv1 = 'mv {0} plots/{1}/'.format(run_config, current_time)
        mv_dir = 'mv -t plots/{0}/ plots/{1}'.format(simType_string, current_time)
        
        for iComm in mv, mv1, mv_dir:
            sp.call('{}'.format(iComm), shell = True)

        print('Finished.\n')
