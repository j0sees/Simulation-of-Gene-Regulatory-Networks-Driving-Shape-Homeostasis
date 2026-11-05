"""Genetic algorithm operators for evolving Gene Regulatory Networks."""

from __future__ import annotations

import numpy as np


def DiracDelta(a: float, b: float) -> int:
    """Return 0 if a == b, otherwise 1 (discrete difference indicator)."""
    return 0 if a == b else 1


def Crossover(
    parent1: np.ndarray, parent2: np.ndarray, crossoverProb: float
) -> tuple[np.ndarray, np.ndarray]:
    """Perform single-point length-preserving crossover between two parent chromosomes."""
    if np.random.random() < crossoverProb:
        crossover_point = np.random.randint(1, len(parent1))
        offspring1 = np.array(parent1, copy=True)
        offspring2 = np.array(parent2, copy=True)

        offspring1[:crossover_point] = parent2[:crossover_point]
        offspring2[:crossover_point] = parent1[:crossover_point]
        return offspring1, offspring2
    return parent1, parent2


def Mutate(
    parent1: np.ndarray, parent2: np.ndarray, mutationProb: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Gaussian point mutation (mu=0, sigma^2=0.7) to two chromosomes."""
    if np.random.random() < mutationProb:
        pos1 = int(np.random.uniform(0, len(parent1)))
        pos2 = int(np.random.uniform(0, len(parent2)))
        mutation1 = np.random.normal(loc=0.0, scale=np.sqrt(0.7))
        mutation2 = np.random.normal(loc=0.0, scale=np.sqrt(0.7))

        mutated_chromosome1 = np.array(parent1, copy=True)
        mutated_chromosome1[pos1] += mutation1

        mutated_chromosome2 = np.array(parent2, copy=True)
        mutated_chromosome2[pos2] += mutation2

        return mutated_chromosome1, mutated_chromosome2
    return parent1, parent2


def DifferentIndex(x1: int, y1: int) -> bool:
    """Return True if indices x1 and y1 are distinct."""
    return x1 != y1
