#!/usr/bin/env python3
"""
Streamlit Web Application for Gene Regulatory Network Simulation.

This web application provides an interactive interface for running cellular
automata simulations driven by recurrent gene regulatory networks and
visualizing cell states and chemical signaling fields in real time.
"""

from __future__ import annotations

import os
import sys
import time
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cell_agent import cell
from core.tools import (
    GenerateIMatrix,
    GenerateTMatrix,
    LGFDiffEq,
    SGFDiffEq,
    flatList,
)
from evolution import main_ga, tools_ga

# Page configuration
st.set_page_config(
    page_title="Gene Regulatory Network Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)


def generate_weight_matrix(n_nodes: int, network_type: str) -> np.ndarray:
    """Generate a GRN weight matrix for the requested network topology."""
    if network_type == "Erdős-Rényi":
        graph = nx.erdos_renyi_graph(n_nodes, p=0.3, directed=True)
        adj = nx.to_numpy_array(graph, dtype=np.float64)
        weights = np.random.uniform(-1.0, 1.0, size=(n_nodes, n_nodes))
        return adj * weights
    if network_type == "Watts-Strogatz":
        k = max(2, min(n_nodes - 1, 4))
        graph = nx.watts_strogatz_graph(n_nodes, k=k, p=0.2)
        adj = nx.to_numpy_array(graph, dtype=np.float64)
        weights = np.random.uniform(-1.0, 1.0, size=(n_nodes, n_nodes))
        return adj * weights
    if network_type == "Barabási-Albert":
        m = max(1, min(n_nodes - 1, 2))
        graph = nx.barabasi_albert_graph(n_nodes, m=m)
        adj = nx.to_numpy_array(graph, dtype=np.float64)
        weights = np.random.uniform(-1.0, 1.0, size=(n_nodes, n_nodes))
        return adj * weights
    return np.random.uniform(-1.0, 1.0, size=(n_nodes, n_nodes))


def init_simulation_state(n_nodes: int, n_lattice: int, network_type: str) -> None:
    """Initialize simulation grids, matrices, and mother cell in session state."""
    w_matrix = generate_weight_matrix(n_nodes, network_type)
    np_cell_grid = np.zeros((n_lattice, n_lattice), dtype=np.float64)
    cell_grid = flatList([flatList(np_cell_grid[r, :]) for r in range(n_lattice)])
    chem_grid = np.zeros((n_lattice, n_lattice, 2), dtype=np.float64)

    ix = n_lattice // 2
    iy = n_lattice // 2
    cell_grid[ix][iy] = 1

    st.session_state.w_matrix = w_matrix
    st.session_state.cell_grid = cell_grid
    st.session_state.chem_grid = chem_grid
    st.session_state.cell_list = [cell(ix, iy, w_matrix, n_nodes)]
    st.session_state.t_matrix = GenerateTMatrix(n_lattice)
    st.session_state.i_matrix = GenerateIMatrix(n_lattice)
    st.session_state.sim_config = (n_nodes, n_lattice, network_type)


def run_simulation_step(n_lattice: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance the cellular automata and chemical diffusion by one time step."""
    cell_grid = st.session_state.cell_grid
    chem_grid = st.session_state.chem_grid
    cell_list = st.session_state.cell_list
    t_matrix = st.session_state.t_matrix
    i_matrix = st.session_state.i_matrix

    delta_t = 1.0
    delta_r = 1.0
    delta_s = 0.5
    delta_l = 0.1
    diff_const = 1.0

    sigma_m = np.zeros((n_lattice, n_lattice), dtype=np.float64)
    lambda_m = np.zeros((n_lattice, n_lattice), dtype=np.float64)

    tmp_cell_list = list(cell_list)
    while len(tmp_cell_list) > 0:
        rnd_idx = np.random.randint(len(tmp_cell_list))
        current_cell = tmp_cell_list[rnd_idx]

        sgf_reading, lgf_reading = current_cell.Sense(chem_grid)
        current_cell.GenerateStatus(sgf_reading, lgf_reading)

        sigma_m[current_cell.yPos, current_cell.xPos] = current_cell.sgfAmount
        lambda_m[current_cell.yPos, current_cell.xPos] = current_cell.lgfAmount

        if current_cell.state == "Quiet":
            current_cell.Quiet(cell_grid)
        elif current_cell.state == "Split":
            current_cell.Split(cell_grid, cell_list)
        elif current_cell.state == "Move":
            current_cell.Move(cell_grid)
        else:
            current_cell.Die(cell_grid)

        del tmp_cell_list[rnd_idx]

    for j_cell in range(len(cell_list) - 1, -1, -1):
        if cell_list[j_cell].amidead:
            del cell_list[j_cell]

    chem_grid[:, :, 0] = SGFDiffEq(chem_grid[:, :, 0], sigma_m, delta_s, delta_t)
    chem_grid[:, :, 1] = LGFDiffEq(
        i_matrix, t_matrix, chem_grid[:, :, 1], lambda_m, delta_l, delta_t, delta_r, diff_const
    )

    return np.asarray(cell_grid, dtype=np.float64), chem_grid[:, :, 0], chem_grid[:, :, 1]


def create_cell_grid_plot(
    cell_grid: np.ndarray, sgf_grid: np.ndarray, lgf_grid: np.ndarray
) -> go.Figure:
    """Create an interactive Plotly visualization of the cell system."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Cells", "SGF Concentration", "LGF Concentration"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]],
    )

    cell_labels = ["Empty", "Quiet", "Moving", "Divided"]

    fig.add_trace(
        go.Heatmap(
            z=cell_grid,
            colorscale=[
                [0.0, "white"],
                [0.25, "green"],
                [0.5, "blue"],
                [0.75, "red"],
                [1.0, "red"],
            ],
            zmin=0,
            zmax=3,
            showscale=True,
            colorbar=dict(title="Cell States", ticktext=cell_labels, tickvals=[0, 1, 2, 3]),
            name="Cells",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=sgf_grid,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="SGF Level"),
            name="SGF",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=lgf_grid,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="LGF Level"),
            name="LGF",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title="Gene Regulatory Network Simulation",
        height=420,
        showlegend=False,
    )

    for i in range(1, 4):
        fig.update_xaxes(title_text="X Position", row=1, col=i)
        fig.update_yaxes(title_text="Y Position", row=1, col=i)

    return fig


def main() -> None:
    """Main Streamlit application."""
    st.title("Gene Regulatory Network Simulation")
    st.markdown("Interactive simulation of cellular automata with gene regulatory networks")

    st.sidebar.header("Simulation Parameters")

    n_nodes = st.sidebar.slider("Number of Nodes", 8, 50, 25, help="Size of the neural network")
    n_lattice = st.sidebar.slider("Lattice Size", 10, 100, 50, help="Size of the simulation grid")
    time_steps = st.sidebar.slider("Time Steps", 10, 500, 100, help="Number of simulation steps")

    network_type = st.sidebar.selectbox(
        "Network Type",
        ["Random", "Erdős-Rényi", "Watts-Strogatz", "Barabási-Albert"],
        help="Type of network architecture",
    )

    if "running" not in st.session_state:
        st.session_state.running = False
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if (
        "sim_config" not in st.session_state
        or st.session_state.sim_config != (n_nodes, n_lattice, network_type)
    ):
        init_simulation_state(n_nodes, n_lattice, network_type)
        st.session_state.current_step = 0
        st.session_state.running = False

    tab1, tab2, tab3 = st.tabs(["Simulation", "Analysis", "About"])

    with tab1:
        st.header("Live Simulation")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Start Simulation", type="primary"):
                at_end = st.session_state.current_step >= time_steps
                if st.session_state.current_step == 0 or at_end:
                    init_simulation_state(n_nodes, n_lattice, network_type)
                    st.session_state.current_step = 0
                st.session_state.running = True

        with col2:
            if st.button("Pause"):
                st.session_state.running = False

        with col3:
            if st.button("Reset"):
                st.session_state.running = False
                st.session_state.current_step = 0
                init_simulation_state(n_nodes, n_lattice, network_type)

        plot_placeholder = st.empty()

        if st.session_state.running and st.session_state.current_step < time_steps:
            cell_grid, sgf_grid, lgf_grid = run_simulation_step(n_lattice)
            fig = create_cell_grid_plot(cell_grid, sgf_grid, lgf_grid)
            plot_placeholder.plotly_chart(fig, use_container_width=True)

            st.session_state.current_step += 1
            st.progress(st.session_state.current_step / time_steps)
            st.write(
                f"Step {st.session_state.current_step} of {time_steps} "
                f"(Living cells: {len(st.session_state.cell_list)})"
            )

            time.sleep(0.05)
            st.rerun()
        else:
            cell_grid = np.asarray(st.session_state.cell_grid, dtype=np.float64)
            sgf_grid = st.session_state.chem_grid[:, :, 0]
            lgf_grid = st.session_state.chem_grid[:, :, 1]
            fig = create_cell_grid_plot(cell_grid, sgf_grid, lgf_grid)
            plot_placeholder.plotly_chart(fig, use_container_width=True)

            if st.session_state.current_step >= time_steps:
                st.success("Simulation completed!")
                st.session_state.running = False

    with tab2:
        st.header("Analysis Tools")
        st.subheader("Fitness Analysis")

        if st.button("Generate Fitness Analysis"):
            generations = list(range(1, 11))
            pop_size = 4
            eval_nodes = min(max(n_nodes, 8), 15)
            eval_lattice = 15
            eval_steps = 20
            n_genes = eval_nodes**2
            population = np.random.uniform(-1.0, 1.0, size=(pop_size, n_genes))

            best_fitness: list[float] = []
            avg_fitness: list[float] = []

            progress_bar = st.progress(0.0)
            for idx, gen in enumerate(generations):
                scores = np.zeros(pop_size)
                for i in range(pop_size):
                    w_mat = population[i].reshape(eval_nodes, eval_nodes)
                    delta_m = main_ga.sim(w_mat, eval_steps, gen, eval_nodes, i, eval_lattice, True)
                    scores[i] = 1.0 - float(np.sum(delta_m)) / (eval_lattice**2)

                best_fitness.append(float(np.max(scores)))
                avg_fitness.append(float(np.mean(scores)))

                order = np.argsort(scores)
                c1, c2 = tools_ga.Crossover(population[order[-1]], population[order[-2]], 0.8)
                c1, c2 = tools_ga.Mutate(c1, c2, 0.5)
                population = np.array([population[order[-1]], population[order[-2]], c1, c2])
                progress_bar.progress((idx + 1) / len(generations))

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=generations, y=best_fitness, mode="lines+markers", name="Best Fitness")
            )
            fig.add_trace(
                go.Scatter(
                    x=generations, y=avg_fitness, mode="lines+markers", name="Average Fitness"
                )
            )

            fig.update_layout(
                title="Fitness Evolution Over Generations",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("About This Simulation")

        st.markdown(
            """
        ### Gene Regulatory Networks Driving Shape Homeostasis

        This simulation explores how gene regulatory networks (GRNs) drive shape homeostasis
        in multicellular systems. The model combines:

        - **Cellular Automata**: Spatial modeling of cell dynamics
        - **Neural Networks**: Gene regulatory networks within each cell
        - **Evolutionary Algorithms**: Optimization of network parameters

        ### Key Components

        1. **Cell States**: Empty, Quiet, Moving, Divided
        2. **Chemical Signaling**: SGF (Short-range) and LGF (Long-range) Growth Factors
        3. **Neural Networks**: Control cell behavior based on local conditions

        ### Research Applications

        - Developmental biology
        - Artificial life
        - Emergent behavior studies
        - Homeostasis mechanisms

        ### Technical Details

        - **Language**: Python 3.11+
        - **Package Manager**: uv
        - **Dependencies**: NumPy, SciPy, Matplotlib, NetworkX, Numba, NEAT-Python
        - **Web Interface**: Streamlit + Plotly
        - **Containerization**: Docker for reproducible environments
        """
        )

        st.info(
            """
        **Interactive Web Interface**: This web application provides real-time browser-based
        simulation controls and visualization for the cellular automata and GRN model.
        """
        )

        st.markdown(
            """
        ### Research Project Information

        This software is based on the master's thesis
        *"Classification of Gene Regulatory Networks Driving Shape Homeostasis"*.

        - **Original Research & Model (`v0.1`)**: M.Sc. J. Esteban Pérez-Hidalgo
          ([@j0sees](https://github.com/j0sees), Costa Rica Institute of Technology)
        - **Software Architecture & Lead Maintainer (`v0.2.0`–`v0.3.0`)**: Reymer Vargas
          ([@userravs](https://github.com/userravs), Independent Research Software Engineer)
        """
        )


if __name__ == "__main__":
    main()

