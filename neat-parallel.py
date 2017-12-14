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
from main_GA import sim
import neat
import pickle
from importlib import reload

#import visualize

def eval_genome(genome, config):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).

    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last few lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    error = 4.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
    return error

def EvaluateIndividual(genome, config):
    totSum = 0.
    nLattice = 50
    timeSteps = 200
    #wMatrix = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    #print('{}'.format(wMatrix))
    #wMatrix = # Get matrix from genome. population[individual,:].reshape(nNodes,nNodes)
    #nNodes = wMatrix.shape# int(np.sqrt(len(bestIndividuals[ind,:])))

    # Timing!
    start_time_chemicalsUpdate = time.time()
    network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    deltaM = sim(network, timeSteps, nLattice)
    # Timing!
    end_time_chemicalsUpdate = time.time()
    secs = end_time_chemicalsUpdate - start_time_chemicalsUpdate

    #print('Proc: {}, time taken run sim: {:.3f}'.format(os.getpid(), secs))
    #print('process: {} done with individual: {}!'.format(os.getpid(), individual))
    deltaMatrix = deltaM

    for ix in range(nLattice):
        for jx in range(nLattice):
            totSum += deltaMatrix[ix,jx]
    # DEBUG
    # print('total sum on delta matrix: ' + str(totSum))
    #if totSum <= int((nLattice**2)*0.1) or totSum == int(nLattice**2):
    #    fitness[individual] = 0.
    #else:
    
    #fitness[individual] = 1. - (1./(nLattice**2))*totSum
    fit = 1. - (1./(nLattice**2))*totSum
    #print('Proc {} computed fitness: {}'.format(os.getpid(), fit))
    return fit
# EvaluateIndividual

def run(config_file):
    # Load configuration.
    nWorkers = 10
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    enc = sys.getdefaultencoding()
    print('=> before encoding: {}'.format(enc))

    reload(sys)  
    sys.setdefaultencoding('UTF-8')

    enc = sys.getdefaultencoding()
    print('=> after encoding: {}'.format(enc))


    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(nWorkers, EvaluateIndividual)
    winner = p.run(pe.evaluate)

    # Save the winner.
    timedateStr = '{0:%Y%m%d_%H%M%S_%f}'.format(dt.now())
    filename = 'genomes/{}_winner_genome'.format(timedateStr)
    config.save('genomes/{}_config'.format(timedateStr))
    
    print('=> Encoding: {}'.format(enc))
    with open(filename, 'wb') as f:
        pickle.dump(winner, f)

    # Log statistics.
    stats.save()

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #for xi, xo in zip(xor_inputs, xor_outputs):
    #    output = winner_net.activate(xi)
    #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True)#, node_names = node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ca')
    run(config_path)
