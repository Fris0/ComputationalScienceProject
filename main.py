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
import physics, solver
import matplotlib.pyplot as plt

if __name__ == "__main__":
    la = np.linspace(0, 90, 90)  # Testing angles.

    ax, fig = plt.subplots(1, 1, figsize=(10,12))  # The Figure we plot in all trajectories.

    for angle in la:
        
        ax[0][0] = 
