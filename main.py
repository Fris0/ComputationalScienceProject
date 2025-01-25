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
from solver import Solver
from physics import Rocket
import matplotlib.pyplot as plt

if __name__ == "__main__":
    la = np.linspace(0, 90, 5)  # Testing angles.

    for angle in la:
        rocket = Rocket(la=angle)
        solver = Solver()
        result = solver.solve_rocket2d(rocket)
        plt.plot(result[0], result[1])
    plt.show()