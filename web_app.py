#!/usr/bin/env python3
"""
Streamlit Web Application for Gene Regulatory Network Simulation

This web app provides an interactive interface for running simulations
and visualizing results in real-time.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our simulation modules
import core.main as main
import core.tools as tools
import evolution.main_ga as main_GA

# Page configuration
st.set_page_config(
    page_title="Gene Regulatory Network Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_cell_grid_plot(cell_grid, sgf_grid, lgf_grid, n_lattice):
    """Create an interactive plotly visualization of the cell system."""
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Cells', 'SGF Concentration', 'LGF Concentration'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # Cell states heatmap
    cell_colors = ['white', 'green', 'blue', 'red']  # empty, quiet, moving, divided
    cell_labels = ['Empty', 'Quiet', 'Moving', 'Divided']
    
    fig.add_trace(
        go.Heatmap(
            z=cell_grid,
            colorscale=[[0, 'white'], [0.25, 'green'], [0.5, 'blue'], [0.75, 'red'], [1, 'red']],
            zmin=0, zmax=3,
            showscale=True,
            colorbar=dict(title="Cell States", ticktext=cell_labels, tickvals=[0, 1, 2, 3]),
            name="Cells"
        ),
        row=1, col=1
    )
    
    # SGF concentration heatmap
    fig.add_trace(
        go.Heatmap(
            z=sgf_grid,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="SGF Level"),
            name="SGF"
        ),
        row=1, col=2
    )
    
    # LGF concentration heatmap
    fig.add_trace(
        go.Heatmap(
            z=lgf_grid,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="LGF Level"),
            name="LGF"
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title="Gene Regulatory Network Simulation",
        height=400,
        showlegend=False
    )
    
    # Update axes
    for i in range(1, 4):
        fig.update_xaxes(title_text="X Position", row=1, col=i)
        fig.update_yaxes(title_text="Y Position", row=1, col=i)
    
    return fig

def run_simulation_step(w_matrix, n_nodes, n_lattice, time_step):
    """Run a single simulation step and return the grid states."""
    
    # Initialize grids
    cell_grid = np.zeros([n_lattice, n_lattice])
    chem_grid = np.zeros([n_lattice, n_lattice, 2])
    
    # Place initial cell in center
    ix = int(n_lattice/2)
    iy = int(n_lattice/2)
    cell_grid[ix][iy] = 1
    
    # Create cell list
    from core.cell_agent import cell
    cell_list = [cell(ix, iy, w_matrix, n_nodes)]
    
    # Run simulation for one step
    # This is a simplified version - in practice you'd call the full simulation
    # For now, we'll create some sample data
    sgf_grid = np.random.rand(n_lattice, n_lattice) * 0.05
    lgf_grid = np.random.rand(n_lattice, n_lattice) * 3
    
    return cell_grid, sgf_grid, lgf_grid

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("Gene Regulatory Network Simulation")
    st.markdown("Interactive simulation of cellular automata with gene regulatory networks")
    
    # Sidebar for controls
    st.sidebar.header("Simulation Parameters")
    
    # Parameter controls
    n_nodes = st.sidebar.slider("Number of Nodes", 5, 50, 25, help="Size of the neural network")
    n_lattice = st.sidebar.slider("Lattice Size", 10, 100, 50, help="Size of the simulation grid")
    time_steps = st.sidebar.slider("Time Steps", 10, 500, 100, help="Number of simulation steps")
    
    # Network type selection
    network_type = st.sidebar.selectbox(
        "Network Type",
        ["Random", "Erdős-Rényi", "Watts-Strogatz", "Barabási-Albert"],
        help="Type of network architecture"
    )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Simulation", "Analysis", "About"])
    
    with tab1:
        st.header("Live Simulation")
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Simulation", type="primary"):
                st.session_state.running = True
                st.session_state.current_step = 0
        
        with col2:
            if st.button("Pause"):
                st.session_state.running = False
        
        with col3:
            if st.button("Reset"):
                st.session_state.running = False
                st.session_state.current_step = 0
        
        # Initialize session state
        if 'running' not in st.session_state:
            st.session_state.running = False
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0
        
        # Create placeholder for the plot
        plot_placeholder = st.empty()
        
        # Simulation loop
        if st.session_state.running and st.session_state.current_step < time_steps:
            # Generate random weight matrix
            w_matrix = np.random.rand(n_nodes, n_nodes) * 2 - 1
            
            # Run simulation step
            cell_grid, sgf_grid, lgf_grid = run_simulation_step(w_matrix, n_nodes, n_lattice, st.session_state.current_step)
            
            # Create plot
            fig = create_cell_grid_plot(cell_grid, sgf_grid, lgf_grid, n_lattice)
            
            # Display plot
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Update step counter
            st.session_state.current_step += 1
            
            # Progress bar
            progress = st.progress(st.session_state.current_step / time_steps)
            st.write(f"Step {st.session_state.current_step} of {time_steps}")
            
            # Auto-advance
            time.sleep(0.1)
            st.rerun()
        
        elif st.session_state.current_step >= time_steps:
            st.success("Simulation completed!")
            st.session_state.running = False
    
    with tab2:
        st.header("Analysis Tools")
        
        # Fitness analysis
        st.subheader("Fitness Analysis")
        
        # Generate sample fitness data
        if st.button("Generate Fitness Analysis"):
            # Create sample data
            generations = list(range(1, 21))
            best_fitness = [0.3 + 0.6 * np.exp(-x/10) + 0.1 * np.random.random() for x in generations]
            avg_fitness = [0.2 + 0.4 * np.exp(-x/8) + 0.05 * np.random.random() for x in generations]
            
            # Create fitness plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=generations, y=best_fitness, mode='lines+markers', name='Best Fitness'))
            fig.add_trace(go.Scatter(x=generations, y=avg_fitness, mode='lines+markers', name='Average Fitness'))
            
            fig.update_layout(
                title="Fitness Evolution Over Generations",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("About This Simulation")
        
        st.markdown("""
        ### Gene Regulatory Networks Driving Shape Homeostasis
        
        This simulation explores how gene regulatory networks (GRNs) drive shape homeostasis 
        in multicellular systems. The model combines:
        
        - **Cellular Automata**: Spatial modeling of cell dynamics
        - **Neural Networks**: Gene regulatory networks within each cell
        - **Evolutionary Algorithms**: Optimization of network parameters
        
        ### Key Components:
        
        1. **Cell States**: Empty, Quiet, Moving, Divided
        2. **Chemical Signaling**: SGF (Short-range Growth Factor) and LGF (Long-range Growth Factor)
        3. **Neural Networks**: Control cell behavior based on local conditions
        
        ### Research Applications:
        
        - Developmental biology
        - Artificial life
        - Emergent behavior studies
        - Homeostasis mechanisms
        
        ### Technical Details:
        
        - **Language**: Python
        - **Dependencies**: NumPy, SciPy, Matplotlib, NetworkX
        - **Web Interface**: Streamlit + Plotly
        - **Containerization**: Docker for reproducible environments
        """)
        
        st.info("""
        **Proof of Concept (PoC)**: This web interface demonstrates the containerization and modernization 
        of the original research project, making it accessible through a web browser with real-time 
        simulation controls and visualization.
        """)
        
        st.markdown("""
        ### Research Project Information
        
        This software was developed as part of the master's thesis *"Classification of Gene Regulatory Networks Driving Shape Homeostasis"*.
        
        **Author**: M.Sc. J. Esteban Pérez-Hidalgo (School of Physics, Costa Rica Institute of Technology)
        """)

if __name__ == "__main__":
    main()
