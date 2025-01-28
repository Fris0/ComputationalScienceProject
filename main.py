# Program: main.py
# Authors: Mark, Riemer and Hans.
# Location: Programmed at school on single computer.
#
# Summary:
# main.py simulates multiple angles on the 2d rocket model
# that was built after validating the 1d model. The goal is
# to find the optimal angle to travel the greatest distance.
#
# Usage:
# - Download the python3 libraries with the requirements file.
# - Run this python file in an up-to-date interpreter > 3.10.

import numpy as np
import pandas as pd
from solver import Solver
from physics import Rocket
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    la = np.linspace(0, 90, 10)  # Testing angles.
    fig, ax = plt.subplots(1,1, figsize=(16,8))
    for angle in la:
        # Resetting / Starting simulation.
        solver = Solver(tend=400)

        # Simulating with new angle.
        result = solver.solve_rocket2d(Rocket(la=angle))

        # Extracting data.
        distance = np.nan_to_num(result[0][:, 0].reshape(1, -1)[0], 0)
        altitude = np.nan_to_num(result[0][:, 1].reshape(1, -1)[0], 0)

        # x-axis
        T = solver.tend - solver.tbegin
        x = np.linspace(0, T, len(altitude) - 1)

        # Plotting the different angles for analysis
        ax.plot(distance, altitude, label=str(angle))
        ax.set_xlabel("Distance")
        ax.set_ylabel("Altitude")
    ax.set_ylim([0, None])
    ax.legend()
    plt.show()