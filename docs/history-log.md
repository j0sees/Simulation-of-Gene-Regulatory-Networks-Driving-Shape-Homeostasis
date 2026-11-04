# Gene Regulatory Networks Driving Shape Homeostasis — Development History

## Project Overview

This document records the chronological development history of the Gene Regulatory Networks (GRN) simulation project, from the original 2017 master's thesis research logs to the repository modernization and containerization.

## Useful Links

- Diffusion equation with Python: https://hinderedsettling.com/2015/02/06/exploring-the-diffusion-equation-with-python/
- Streamlit documentation: https://docs.streamlit.io/
- Docker documentation: https://docs.docker.com/

## Useful Commands

- Generate video from simulation snapshots:
  ```bash
  ffmpeg -f image2 -pattern_type glob -framerate 24 -i 'cell_system-*.png' -s 1024x1024 cell_system.avi
  ```
- Start containerized web interface: `./scripts/run_docker.sh`
- Start local web interface: `./scripts/run_web_app.sh`
- Web interface URL: `http://localhost:8501`

---

## Original Thesis Log (2017)

### 2017-09-01: Started Working
- Implemented initial cell on a grid capable of reproducing.

### 2017-09-07
- Plotting works, though initial implementation needs performance improvements.
- Added states: `move`, `split`, `die`. Started working on cell dynamics before integrating the neural network.

### 2017-09-11
- Added functions for moving and splitting. In principle they work regardless of the state of the compass (ON/OFF); requires testing.
- Pending design question: what happens if the orientation direction is out of bounds or occupied (fail action vs. choose a random available neighbor)?

### 2017-09-13
- Started implementing a more stable and faster plotting scheme.
- Reviewed literature on neural networks:
  - Evaluate which NN architecture is best suited for the cellular model.
  - Compare original model NN against alternative architectures.
  - Decide between external libraries and custom NN implementation.
  - Plan timelapse video recording of developing cell structures.

### 2017-09-14
- Fixed plotting; now runs smoothly and quickly.
- Fixed `move2` and `split2` functions for grid borders and orientation.
- Pending fix for the `die` method.

### 2017-09-20
#### Done
- Fixed `die` method.
- Implemented dynamics for Short-range Growth Factor (SGF).
- Cleaned up codebase.
- Standardized grid state encoding:
  - `-1`: Spot previously occupied by a cell that moved away or died
  - `0`: Spot has always been empty
  - `1`: Quiet cell
  - `2`: Cell moved to this spot
  - `3`: Cell just divided

#### To Do
- Double-check action functions: `quiet`, `move`, `die`, `split`, `generateStatus`, and orientation system.
- Test decaying dynamics of SGF.
- Implement dynamics for Long-range Growth Factor (LGF).
- Implement custom neural network.

### 2017-09-21
#### To Do
- Add detailed code comments for clarity.
- Update SGF and LGF via matrix operations.
- Draw a flow diagram of the main simulation loop.

### 2017-09-25
#### Done
- Cleaned repository and code.
- Implemented matrix scheme for SGF and LGF dynamics (pending verification).
- Drafted initial flow diagrams.

#### To Do
- Test every cell action and SGF/LGF dynamics.
- Start neural network implementation.
- Create digital version of flow charts.
- Review update order in cell dynamics.

### 2017-09-26
#### Done
- Implemented neural network.
- Corrected bugs in `split`, `move`, and `die` actions.

#### To Do
- Test neural network with a working example or begin Evolutionary Algorithm (EA) implementation.
- Standardize `(x, y)` coordinate ordering.
- Adopt a colorblind-friendly palette for cell states.

### 2017-10-04
#### Done
- Implemented recurrent neural network (RNN) dynamics.
- Completed initial Genetic Algorithm (GA) implementation.
- Configured `main.py` to be callable from `main_ga.py`.
- Added GA operators: fitness, crossover, mutation, elitism, and tournament selection.
- Added serialization helpers to store and reload RNN weight matrices.

#### To Do
- Run end-to-end integration tests.
- Separate configurable parameters from constants.
- Profile and improve runtime performance.
- Write batch execution scripts for parameter sweeps.
- Add boolean toggle for plotting snapshots vs. every time step.

### 2017-10-05
#### Done
- Initial test runs yielded several observations:
  - Offspring generation via crossover converges prematurely to local minima.
  - GA evaluation can be parallelized across CPU cores by distributing individuals to worker processes.
  - Matrix operations in NumPy provide sufficient speed without requiring custom CUDA kernels.
  - Fitness evaluation compares system structure halfway through and at the end of the simulation.
- Cleaned code and enabled `main.py` to run both as an imported module and as a standalone script.

#### To Do
- Refine GA operators to avoid premature convergence.
- Profile the simulation loop.

### 2017-10-09
#### Done
- Parallelized evaluation using Python's `multiprocessing` library.
- Adapted code for cluster execution (`ozzy`).
- Updated fitness sorting to return index permutations.
- Created `ozzy-branch` for headless parallel cluster runs without plotting.

#### To Do
- Refactor shared array handling between CA and GA modules.
- Benchmark serial vs. parallel configurations.
- Decouple plotting completely from CA/GA core logic.
- Export structured CSV logs for cluster runs.

### 2017-11-01
#### Done
- Benchmarked processor counts, population sizes, chunk sizes, and generation counts.
- Updated `main_ga.py` to accept CLI arguments for automated parameter sweeps.
- Exported simulation metrics as CSV files with accompanying `gnuplot` scripts.
- Fixed fitness normalization so maximum fitness reaches `1.0`.
- Corrected polarization compass index (`V[5]` -> `V[7]`).
- Recorded chemical concentration maps for steady-state analysis.

#### To Do
- Evaluate structured storage formats (`pickle` or database).
- Explore alternative parallelization strategies.
- Investigate NEAT (NeuroEvolution of Augmenting Topologies) integration.
- Improve exception handling with `try`/`except` blocks.

### 2017-11-10
#### Done
- Assigned timestamp-based unique run IDs to associate parameter files, populations, and benchmarks automatically.
- Added multi-run averaging inside `main_ga.py` for max/mean fitness and runtime benchmarks.
- Improved boundary handling during cell movement using exception handling.
- Added Matplotlib statistical plotting methods to `plot.py`.

#### To Do
- Review RNN activation dynamics against reference literature (Haykin).
- Generate final thesis figures and write model documentation.
- Prevent toroidal wrap-around artifacts at lattice boundaries.
- Investigate high initial fitness values in random populations.

### 2017-11-16
#### Done
- Adapted parallel `multiprocessing` execution for Python 2.7 compatibility (`map` wrapper).
- Updated cell grid container to enforce boundary constraints cleanly.

#### To Do
- Evaluate JIT/Cython acceleration options.
- Finalize data plotting routines.

### 2017-12-15
#### Done
- Added network topology plotting scripts.
- Added graph generators for Erdős-Rényi and related topologies mapped to GRNs.
- Presented research poster at the 9th Swedish Meeting on Mathematical Biology (Västerås, Sweden, Dec 7–8, 2017).
- Integrated `neat-python` to evolve cell agent networks (`neat_cell_agent.py`, `neat_parallel.py`).

#### To Do
- Tune `neat-python` hyperparameters for production runs.
- Visualize cluster-evolved NEAT genomes locally.

---

## Project Modernization (2024–2025)

### Summary of Modernization Changes
- **Modular Package Structure**: Reorganized source files into `core/`, `evolution/`, `visualization/`, `analysis/`, `config/`, `examples/`, `scripts/`, `tests/`, and `docs/` using Python 3 PEP 420 namespace packages.
- **Modern Tooling (`uv` & `pyproject.toml`)**: Migrated dependency management to `uv` and `pyproject.toml`, targeting Python 3.11+ and modern scientific stack releases.
- **Containerization**: Added `Dockerfile` (`python:3.12-slim` + `uv`), `docker-compose.yml`, `.dockerignore`, and `scripts/run_docker.sh` for reproducible deployment.
- **Interactive Web Application**: Created `visualization/web_app.py` and `scripts/run_web_app.sh` using Streamlit and Plotly for real-time browser-based simulation and visualization.
- **Continuous Integration & Unit Tests**: Added GitHub Actions CI (`.github/workflows/ci.yml`) and a `pytest` unit test suite in `tests/`.
- **Documentation Suite**: Added structured guides in `docs/` (`architecture.md`, `web-interface.md`, `docker.md`, `history-log.md`) and automated figure generation (`examples/generate_documentation_images.py`).
