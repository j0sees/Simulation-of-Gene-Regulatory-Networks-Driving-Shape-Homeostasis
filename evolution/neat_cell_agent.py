"""NEAT-based cell agent implementation for evolving network topologies."""

from __future__ import annotations

import neat
import numpy as np

from core.tools import (
    CheckifOccupied,
    CheckifPreferred,
)


class cell:
    """
    A cell agent controlled by a NEAT recurrent neural network genome.

    Grid State Encoding
    -------------------
    - 0: Empty (available) grid spot
    - 1: Quiet cell (or failed move/split action)
    - 2: Moving cell
    - 3: Newly divided cell
    """

    def __init__(self, yPos: int, xPos: int, genome, config) -> None:
        """Initialize a NEAT-controlled cell agent at (yPos, xPos)."""
        self.state = "Quiet"
        self.xPos = xPos
        self.yPos = yPos
        self.compass = True
        self.orientation = [self.yPos, self.xPos]
        self.neighbourList = [
            [self.yPos - 1, self.xPos],
            [self.yPos + 1, self.xPos],
            [self.yPos, self.xPos - 1],
            [self.yPos, self.xPos + 1],
        ]
        self.splitCounter = 0
        self.splitTime = 1
        self.deathCounter = 0
        self.deathTime = 1
        self.amidead = False
        self.quietCounter = 0
        self.sgfAmount = 0.0
        self.lgfAmount = 0.0
        self.genome = genome
        self.config = config
        self.network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        self.V = np.zeros(8, dtype=np.float64)

    def Sense(self, grid: np.ndarray) -> tuple[float, float]:
        """Sense SGF and LGF concentrations at the cell's current grid coordinate."""
        sgf_reading = float(grid[self.yPos, self.xPos][0])
        lgf_reading = float(grid[self.yPos, self.xPos][1])
        return sgf_reading, lgf_reading

    def GenerateStatus(self, SGF_lecture: float, LGF_lecture: float) -> None:
        """Activate the NEAT recurrent network to determine cell state and chemical output."""
        outputs = self.network.activate((SGF_lecture, LGF_lecture))
        for idx, val in enumerate(outputs[:6]):
            self.V[idx + 2] = val

        self.neighbourList = [
            [self.yPos - 1, self.xPos],
            [self.yPos + 1, self.xPos],
            [self.yPos, self.xPos - 1],
            [self.yPos, self.xPos + 1],
        ]

        i_status = self.V[2]  # Proliferate: Split
        j_status = self.V[3]  # Migrate: Move
        k_status = self.V[4]  # Apoptosis: Die

        self.sgfAmount = float(self.V[5])
        self.lgfAmount = float(self.V[6])

        if self.compass:
            n_boundary = 0.25
            s_boundary = 0.5
            e_boundary = 0.75
            arrow = self.V[7]
            if arrow < s_boundary:
                if arrow < n_boundary:
                    self.orientation = self.neighbourList[0]  # North
                else:
                    self.orientation = self.neighbourList[1]  # South
            else:
                if arrow < e_boundary:
                    self.orientation = self.neighbourList[3]  # East
                else:
                    self.orientation = self.neighbourList[2]  # West
        else:
            self.orientation = [self.yPos, self.xPos]

        max_val = 0.5
        x_threshold = 0.5
        y_threshold = 0.01

        if i_status < x_threshold and j_status < x_threshold and k_status < x_threshold:
            self.state = "Quiet"
        else:
            for val in (i_status, j_status, k_status):
                if max_val < val:
                    max_val = val
            if abs(max_val - i_status) <= y_threshold:
                self.state = "Split"
            elif abs(max_val - j_status) <= y_threshold:
                self.state = "Move"
            else:
                self.state = "Die"

    def Quiet(self, grid) -> None:
        """Keep the cell in the quiet state at its current position."""
        grid[self.yPos][self.xPos] = 1
        self.quietCounter += 1

    def Die(self, grid) -> None:
        """Advance the death counter and remove the cell when deathTime is reached."""
        self.deathCounter += 1
        if self.deathCounter == self.deathTime:
            self.amidead = True
            grid[self.yPos][self.xPos] = 0
        else:
            grid[self.yPos][self.xPos] = 1

    def Move(self, grid) -> None:
        """Move the cell to its preferred or another available Von Neumann neighbor."""
        available_spots = []
        need_other_neighbours = True

        for neighbr in self.neighbourList:
            try:
                if CheckifOccupied(neighbr[1], neighbr[0], grid):
                    continue
                x_ori = self.orientation[1]
                y_ori = self.orientation[0]
                if CheckifPreferred(x_ori, y_ori, neighbr[1], neighbr[0]):
                    grid[y_ori][x_ori] = 2
                    grid[self.yPos][self.xPos] = 0
                    self.xPos = x_ori
                    self.yPos = y_ori
                    need_other_neighbours = False
                    break
                available_spots.append(neighbr)
            except IndexError:
                continue

        if need_other_neighbours:
            if len(available_spots) > 0:
                idx = np.random.randint(len(available_spots))
                target_y, target_x = available_spots[idx]
                grid[target_y][target_x] = 2
                grid[self.yPos][self.xPos] = 0
                self.yPos = target_y
                self.xPos = target_x
            else:
                grid[self.yPos][self.xPos] = 1

    def Split(self, grid, cellList: list) -> None:
        """Divide the cell, placing a daughter cell in an available neighboring spot."""
        self.splitCounter += 1
        if self.splitCounter == self.splitTime:
            self.splitCounter = 0
            available_spots = []
            need_other_neighbours = True

            for neighbr in self.neighbourList:
                try:
                    if CheckifOccupied(neighbr[1], neighbr[0], grid):
                        continue
                    y_ori = self.orientation[0]
                    x_ori = self.orientation[1]
                    if CheckifPreferred(x_ori, y_ori, neighbr[1], neighbr[0]):
                        grid[y_ori][x_ori] = 1
                        grid[self.yPos][self.xPos] = 3
                        cellList.append(cell(y_ori, x_ori, self.genome, self.config))
                        need_other_neighbours = False
                        break
                    available_spots.append(neighbr)
                except IndexError:
                    continue

            if need_other_neighbours:
                if len(available_spots) > 0:
                    idx = np.random.randint(len(available_spots))
                    target_y, target_x = available_spots[idx]
                    grid[target_y][target_x] = 1
                    grid[self.yPos][self.xPos] = 3
                    cellList.append(
                        cell(target_y, target_x, self.genome, self.config)
                    )
                else:
                    grid[self.yPos][self.xPos] = 1

