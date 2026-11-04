#!/usr/bin/env python3
"""
Genetic algorithm example.

This script demonstrates how to run the genetic algorithm to evolve
gene regulatory networks.
"""

import os
import sys
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution import main_ga
from evolution import tools_ga


def run_evolution_example():
    """Run a simple genetic algorithm example."""
    # Small-scale parameters for demonstration
    popSize = 4
    nNodes = 10
    nGenes = nNodes**2
    nOfGenerations = 5
    timeSteps = 30
    nLattice = 15
    mode = True  # Fitness mode (no visualization)

    print("Running genetic algorithm with:")
    print(f"  Population size: {popSize}")
    print(f"  Nodes: {nNodes}")
    print(f"  Generations: {nOfGenerations}")
    print(f"  Lattice size: {nLattice}x{nLattice}")

    # Create an initial random population with weights in [-1, 1]
    population = np.random.rand(popSize, nGenes) * 2.0 - 1.0

    for gen in range(nOfGenerations):
        print(f"\nGeneration {gen + 1}/{nOfGenerations}")

        fitness = np.zeros(popSize)
        for i in range(popSize):
            wMatrix = population[i].reshape(nNodes, nNodes)
            delta_matrix = main_ga.sim(wMatrix, timeSteps, gen, nNodes, i, nLattice, mode)
            tot_sum = float(np.sum(delta_matrix))
            fitness[i] = 1.0 - (tot_sum / (nLattice**2))

        sorted_idx = np.argsort(fitness)
        best_idx = sorted_idx[-1]
        second_idx = sorted_idx[-2]

        print(f"  Best fitness: {fitness[best_idx]:.3f}")
        print(f"  Average fitness: {np.mean(fitness):.3f}")

        # Evolve next generation via elitism, crossover, and mutation
        next_pop = np.zeros_like(population)
        next_pop[0] = population[best_idx]
        next_pop[1] = population[second_idx]
        child1, child2 = tools_ga.Crossover(population[best_idx], population[second_idx], 0.8)
        child1, child2 = tools_ga.Mutate(child1, child2, 0.5)
        next_pop[2] = child1
        next_pop[3] = child2
        population = next_pop

    print("\nEvolution completed!")


if __name__ == "__main__":
    run_evolution_example()
