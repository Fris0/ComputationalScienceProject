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
import sys
from solver import Solver
from physics import Rocket
import cv2
import matplotlib.pyplot as plt

def plot_while_processing():
    la = np.linspace(0, 90, 5)  # Testing angles.
    fig, ax = plt.subplots(1,4, figsize=(16,8))
    for angle in la:
        # Resetting / Starting simulation.
        solver = Solver(tend=40)

        # Simulating with new angle.
        result = solver.solve_rocketNike(Rocket(la=angle))

        # Extracting data.
        distance = np.nan_to_num(result[0][:, 0].reshape(1, -1)[0], 0)
        altitude = np.nan_to_num(result[0][:, 1].reshape(1, -1)[0], 0)
        velocity = result[0][:, 2].reshape(1, -1)[0]
        mass = result[0][:, 3].reshape(1, -1)[0]

        # x-axis
        T = solver.tend - solver.tbegin

        # Plotting the different angles for analysis
        ax[0].plot(distance, altitude, label=str(angle))
        ax[0].set_xlabel("Distance (ft)")
        ax[0].set_ylabel("Altitude (ft)")
        ax[1].plot(np.linspace(solver.tbegin, solver.tend, len(mass)), mass, label=str(angle))
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Mass (lbs)")
        ax[2].plot(np.linspace(solver.tbegin, solver.tend, len(mass)), velocity, label=str(angle))
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("Velocity (ft/s)")
        ax[3].plot(velocity, altitude, label=str(angle))
        ax[3].set_xlabel("Velocity (ft/s)")
        ax[3].set_ylabel("Altitude (ft)")

    ax[0].set_ylim([-1, None])
    ax[0].set_xlim([-1, None])
    ax[0].legend()
    plt.show()

def obtain_data_from_simulation():
    la = np.linspace(0, 90, 10)  # Testing angles.

    with open("data.txt", "w") as f:
        for angle in la:
            # Resetting / Starting simulation.
            solver = Solver(tend=400)

            # Simulating with new angle.
            result = solver.solve_rocket2d(Rocket(la=angle))

            # Extracting data.
            distance = np.nan_to_num(result[0][:, 0].reshape(1, -1)[0], 0)
            altitude = np.nan_to_num(result[0][:, 1].reshape(1, -1)[0], 0)

            # Writing to file
            f.write(f"{angle} {np.max(distance)} {np.max(altitude)}\n")

def read_data_and_plot():
    with open("data.txt", "r") as f:
        while True:
            line = f.readline()
            if line != "":
                res = np.array(list(map(float, line.split(sep=" "))))
                print(res)
                plt.bar(res[0], res[1])
            else:
                break
    plt.show()

if __name__ == "__main__":
    option = None
    try:
        option = int(sys.argv[1])
    except:
        print(f"No option given {option}.")

    if option == 1:
        plot_while_processing()
    elif option == 2:
        obtain_data_from_simulation()
    else:
        read_data_and_plot()