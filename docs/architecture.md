# Scientific Model and Software Architecture

## 1. Scientific Background

This project implements the computational framework developed in the master's thesis **"Classification of Gene Regulatory Networks Driving Shape Homeostasis"** by **M.Sc. J. Esteban Pérez-Hidalgo** (supervised by **Ph.D. Philip Gerlee**), building upon:

> Gerlee, P., Basanta, D., & Anderson, A. R. A. (2015). *The influence of cellular characteristics on the evolution of shape homeostasis*. Journal of the Royal Society Interface, 12(103), 20141222.

The model investigates how individual cellular capabilities—intracellular gene regulation, spatial polarization, and chemical communication—give rise to self-organizing, homeostatic multicellular structures on a two-dimensional lattice.

---

## 2. Mathematical Model

### Cellular Automata Lattice
Cells inhabit a discrete 2D square lattice of size $N \times N$. At each discrete time step, every living cell is updated in random asynchronous order and adopts one of four behavioral states:

- **Quiet (`1`)**: Remains stationary at its current lattice site.
- **Move (`2`)**: Migrates to its preferred Von Neumann neighbor (if unoccupied), or to a random available Von Neumann neighbor.
- **Split (`3`)**: Divides and places a daughter cell with an identical genotype into its preferred (or another available) Von Neumann neighbor.
- **Die (`0`)**: Undergoes apoptosis and vacates its lattice site.

![Cell System Simulation](./images/cell_system_example.png)

*Simulation snapshot showing discrete cell states alongside Short-range Growth Factor (SGF) and Long-range Growth Factor (LGF) concentration fields.*

### Chemical Signaling (SGF and LGF)
Intercellular communication is mediated by two chemical fields over the lattice:

1. **Short-range Growth Factor (SGF, $s$)**: Produced locally at rate $\sigma$ and decays at rate $\delta_s$ without spatial diffusion:
   $$\frac{\partial s}{\partial t} = \sigma - \delta_s s$$

2. **Long-range Growth Factor (LGF, $l$)**: Produced locally at rate $\lambda$, decays at rate $\delta_l$, and diffuses across the lattice with diffusion constant $D$:
   $$\frac{\partial l}{\partial t} = D \nabla^2 l + \lambda - \delta_l l$$
   Integrated numerically in `core/tools.py` (`LGFDiffEq`) via an alternating-direction Crank-Nicolson scheme using the discrete 1D Laplacian matrix $T$.

### Intracellular Gene Regulatory Network (RNN)
Each cell carries an intracellular Recurrent Neural Network (RNN) of $n$ nodes (`core/tools.py:RecurrentNeuralNetwork`) with weight matrix $W \in [-1, 1]^{n \times n}$. At each time step:
- Nodes `0` and `1` receive the local SGF and LGF concentrations sensed from the grid.
- Node activations update via a steep sigmoid transfer function:
  $$V_i(t+1) = \frac{1}{1 + \exp(-2 (\sum_j W_{ij} V_j(t) - I_i))}$$
- Output nodes encode the cell's action probabilities (`V[2]` Split, `V[3]` Move, `V[4]` Die), chemical secretion rates (`V[5]` SGF, `V[6]` LGF), and polarization compass direction (`V[7]`).

![Gene Regulatory Network](./images/network_example.png)

*Directed graph representation of a gene regulatory network with regulatory feedback loops.*

### Evolutionary Optimization
To evolve networks that achieve **shape homeostasis** (a spatial structure that stabilizes and persists over time), the genetic algorithm (`evolution/main_ga.py`) evaluates each candidate genotype over $T$ time steps and compares the binary cell occupancy structure halfway through the simulation ($S_{T/2}$) against the final structure ($S_T$):

$$F = 1 - \frac{1}{N^2} \sum_{i,j} |S_{T/2}(i, j) - S_T(i, j)|$$

Trivial outcomes (extinction, $<1\%$ lattice occupancy at $T/2$, or complete lattice saturation) receive $F = 0$.

![Evolutionary Progress](./images/evolution_example.png)

*Best and average population fitness across generations during evolutionary optimization.*

---

## 3. Software Architecture

The codebase is organized into modular Python 3.11+ namespace packages:

| Directory / File | Responsibility |
|---|---|
| `core/cell_agent.py` | Defines the `cell` agent class (`Sense`, `GenerateStatus`, `Quiet`, `Move`, `Split`, `Die`). |
| `core/tools.py` | Implements `flatList`, Numba `@njit`-accelerated `SGFDiffEq`, `LGFDiffEq`, `RecurrentNeuralNetwork`, and matrix utilities. |
| `core/main.py` | Standalone simulation loop combining cellular automata updates, PDE integration, and Matplotlib rendering. |
| `evolution/main_ga.py` | Multiprocessing genetic algorithm with elitism and tournament selection. |
| `evolution/tools_ga.py` | Genetic operators (`Crossover`, `Mutate`, `DiracDelta`, `DifferentIndex`). |
| `evolution/neat_cell_agent.py` | Cell agent variant driven by a `neat-python` recurrent network genome. |
| `evolution/neat_parallel.py` | Parallel NEAT neuroevolution runner using `config/config-ca`. |
| `analysis/graph_generator.py` | Generates Erdős-Rényi random graph ensembles for GRN topology studies. |
| `analysis/graph_evaluation.py` | Evaluates saved network populations in parallel and plots fitness distributions. |
| `visualization/plot.py` | Matplotlib `Environment` class and statistical plotting helpers. |
| `visualization/web_app.py` | Interactive Streamlit and Plotly web interface for real-time browser simulation. |
| `examples/` | Self-contained scripts (`run_simulation.py`, `run_evolution.py`, `generate_documentation_images.py`). |
| `tests/` | `pytest` unit test suite covering PDE solvers, cell transitions, genetic operators, and network topologies. |

