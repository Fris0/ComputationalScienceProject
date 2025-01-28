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

        # Plotting the different angles for analysis
        ax.plot(distance, altitude, label=str(angle))
        ax.set_xlabel("Distance")
        ax.set_ylabel("Altitude")

    ax.set_ylim([0, None])
    ax.legend()
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
<<<<<<< HEAD
            
            # Writing to file
=======

>>>>>>> 31c563eb2b6d20ccf3cb5a2167ef4389ee078408
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

