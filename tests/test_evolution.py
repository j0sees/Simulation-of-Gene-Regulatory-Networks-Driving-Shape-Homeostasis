"""Unit tests for evolutionary operators and headless simulation evaluation."""

from __future__ import annotations

import numpy as np

from evolution import main_ga, tools_ga


def test_dirac_delta_and_different_index() -> None:
    """DiracDelta and DifferentIndex should return expected boolean/integer indicators."""
    assert tools_ga.DiracDelta(1.0, 1.0) == 0
    assert tools_ga.DiracDelta(1.0, 0.0) == 1
    assert not tools_ga.DifferentIndex(4, 4)
    assert tools_ga.DifferentIndex(4, 5)


def test_crossover_and_mutate_preserve_shape() -> None:
    """Crossover and Mutate should preserve chromosome lengths and modify genes when p=1."""
    parent1 = np.zeros(16, dtype=np.float64)
    parent2 = np.ones(16, dtype=np.float64)

    # With probability 0, parents are returned unchanged
    c1_no, c2_no = tools_ga.Crossover(parent1, parent2, crossoverProb=0.0)
    assert np.array_equal(c1_no, parent1)
    assert np.array_equal(c2_no, parent2)

    # With probability 1, single-point crossover exchanges a non-empty prefix
    c1, c2 = tools_ga.Crossover(parent1, parent2, crossoverProb=1.0)
    assert c1.shape == (16,)
    assert c2.shape == (16,)
    assert np.allclose(c1 + c2, np.ones(16))

    # Mutation with probability 1 alters one locus per chromosome
    m1, m2 = tools_ga.Mutate(parent1, parent2, mutationProb=1.0)
    assert m1.shape == (16,)
    assert m2.shape == (16,)
    assert np.count_nonzero(m1 != parent1) == 1
    assert np.count_nonzero(m2 != parent2) == 1


def test_headless_simulation_returns_delta_matrix() -> None:
    """main_ga.sim in fitness mode (mode=True) should return a binary delta matrix."""
    np.random.seed(42)
    n_nodes = 8
    n_lattice = 10
    time_steps = 10
    w_matrix = np.random.uniform(-1.0, 1.0, size=(n_nodes, n_nodes))

    delta_matrix = main_ga.sim(
        w_matrix,
        timeSteps=time_steps,
        iGen=0,
        nNodes=n_nodes,
        individual=0,
        nLattice=n_lattice,
        mode=True,
    )
    assert delta_matrix is not None
    assert delta_matrix.shape == (n_lattice, n_lattice)
    assert set(np.unique(delta_matrix)).issubset({0.0, 1.0})

