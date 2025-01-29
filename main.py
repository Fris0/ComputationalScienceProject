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
        solver = Solver(tend=1000)

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

def obtain_angle_data_from_simulation():
    la = np.linspace(0, 90, 10)  # Testing angles.

    with open("data.txt", "w") as f:
        for angle in la:
            # Resetting / Starting simulation.
            solver = Solver(tend=1000)

            # Simulating with new angle.
            result = solver.solve_rocket2d(Rocket(la=angle))

            # Extracting data.
            distance = np.nan_to_num(result[0][:, 0].reshape(1, -1)[0], 0)
            altitude = np.nan_to_num(result[0][:, 1].reshape(1, -1)[0], 0)

            # Writing to file
            f.write(f"{angle} {np.max(distance)} {np.max(altitude)}\n")

def obtain_thrust_data_from_simulation():
    thrust_samples = np.random.normal(5130, 0.05*5130, 1000)
    with open("data.txt", "w") as f:
        for idx, thrust in enumerate(thrust_samples):
            # Resetting / Starting simulation.
            solver = Solver(tend=1000)

            # Simulating with new angle.
            result = solver.solve_rocketNike(Rocket(T= thrust * 32.174))

            # Extracting data.
            distance = np.nan_to_num(result[0][:, 0].reshape(1, -1)[0], 0)
            altitude = np.nan_to_num(result[0][:, 1].reshape(1, -1)[0], 0)
            velocity = np.nan_to_num(result[0][:, 2].reshape(1, -1)[0], 0)
            mass = np.nan_to_num(result[0][:, 3].reshape(1, -1)[0], 0)
            time = np.linspace(solver.tbegin, solver.tend, len(distance))

            # Writing to file
            d = {'run': idx, 'distance': distance, 'altitude': altitude, 'velocity': velocity, 'mass': mass, 'time': time}
            data = pd.DataFrame(data=d)
            data.to_csv('thrust_data.csv', mode='a', index=False)

def plot_thrust_data_from_simulation():
    df = pd.read_csv("thrust_data.csv", low_memory=False)
    df["run"] = pd.to_numeric(df["run"], errors="coerce")

    start = df.run.min()
    end = df.run.max()

    fig, ax = plt.subplots(1,4, figsize=(16,8))

    for run in range(int(start), int(end) + 1):

        df_run = df[df["run"] == run]

        # Extracting data.
        distance = df_run["distance"].to_numpy(dtype=np.float64)
        altitude = df_run["altitude"].to_numpy(dtype=np.float64)
        velocity = df_run["velocity"].to_numpy(dtype=np.float64)
        mass = df_run["mass"].to_numpy(dtype=np.float64)

        # Plotting the different angles for analysis
        ax[0].plot(distance, altitude)
        ax[0].set_xlabel("Distance (ft)")
        ax[0].set_ylabel("Altitude (ft)")
        ax[1].plot(np.linspace(0, 1000, len(mass)), mass)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Mass (lbs)")
        ax[2].plot(np.linspace(0, 1000, len(mass)), velocity)
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("Velocity (ft/s)")
        ax[3].plot(velocity, altitude)
        ax[3].set_xlabel("Velocity (ft/s)")
        ax[3].set_ylabel("Altitude (ft)")

    ax[0].set_ylim([-1, None])
    ax[0].set_xlim([-1, None])
    ax[0].legend()

def obtain_data_from_launchangle_simulation(startlaunchangle, endlaunchangle, N):
    for launchangle in np.linspace(startlaunchangle, endlaunchangle, N):
        # Resetting / Starting simulation.
        solver = Solver(tend=1000)

        # Simulating with new angle.
        result = solver.solve_rocketNike(Rocket(la=launchangle))

        # Extracting data.
        distance = np.nan_to_num(result[0][:, 0].reshape(1, -1)[0], 0)
        altitude = np.nan_to_num(result[0][:, 1].reshape(1, -1)[0], 0)
        velocity = np.nan_to_num(result[0][:, 2].reshape(1, -1)[0], 0)
        mass = np.nan_to_num(result[0][:, 3].reshape(1, -1)[0], 0)
        time = np.linspace(solver.tbegin, solver.tend, len(distance))
        plt.plot(distance, altitude)

        # Writing to file
        d = {'distance': distance, 'altitude': altitude, 'velocity': velocity, 'mass': mass, 'time': time}
        data = pd.DataFrame(data=d)
        data.to_csv("Modeldata/AngleData/LAis" + str(int(launchangle)) + ".csv")
    plt.show()

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
        startlaunch = int(sys.argv[2])
        endlaunch = int(sys.argv[3])
        N = int(sys.argv[4])
    except:
        print(f"No option given {option}.")

    if option == 1:
        plot_while_processing()
    elif option == 2:
        obtain_angle_data_from_simulation()
    elif option == 3:
        obtain_thrust_data_from_simulation()
    elif option == 4:
        obtain_data_from_launchangle_simulation(startlaunch, endlaunch, N)
    else:
        plot_thrust_data_from_simulation()

