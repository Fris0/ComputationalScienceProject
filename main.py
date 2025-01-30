# Program: main.py
# Authors: Mark, Riemer and Hans and Rik.
#
# Summary:
# main.py simulates multiple angles on the 2d rocket model
# called Nike Apache Physics that was built after validating the 1d model
# and 2d model. The goal is to find the optimal angle to
# travel the greatest distance.
#
# Usage:
# - Download the python3 libraries imported below.
# - Run this python file in an up-to-date interpreter > 3.10.
# - Run by python3 main.py <option number>

import numpy as np  # Used for the numpy arrays and it's functions.
import pandas as pd  # Used for storing the numpy arrays of the simulation.
import sys  # For reading out the std-in from terminal input.
from solver import Solver
from physics import Rocket
import matplotlib.pyplot as plt  # Used for plotting the simulation data.
from os import listdir  # Used for showing directories.
from os.path import isfile, join  # Used for storing data in a file.


def plot_while_processing() -> None:
    """
    Simulate the Nike rocket, obtain the data (distance, altitude, velocity,
    and mass)
    and plot the results in a 1 by 4 subplot.

    Sidenote: Use this function when you need to test
    general functionality because the tolerance is set to 1e4.

    Input: None

    Output: None

    Side-effects:
    - Plots the 4 subplots using matplotlib.pyplot and shows them to the
    user.
    """
    la = np.linspace(0, 90, 5)  # Testing angles.
    fig, ax = plt.subplots(1, 4, figsize=(16, 8))
    for angle in la:
        # Resetting / Starting simulation.
        solver = Solver(tend=1000, tolerance=1e4)

        # Simulating with new angle.
        result = solver.solve_rocketNike(Rocket(la=angle))

        # Extracting data.
        distance = np.nan_to_num(result[1][:, 0].reshape(1, -1)[0], 0)
        altitude = np.nan_to_num(result[1][:, 1].reshape(1, -1)[0], 0)
        velocity = result[0][:, 2].reshape(1, -1)[0]
        mass = result[0][:, 3].reshape(1, -1)[0]

        # Plotting the different angles for analysis
        ax[0].plot(distance, altitude, label=str(angle))
        ax[0].set_xlabel("Distance (ft)")
        ax[0].set_ylabel("Altitude (ft)")
        ax[1].plot(np.linspace(solver.tbegin, solver.tend, len(mass)),
                   mass, label=str(angle))
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Mass (lbs)")
        ax[2].plot(np.linspace(solver.tbegin, solver.tend, len(mass)),
                   velocity, label=str(angle))
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("Velocity (ft/s)")
        ax[3].plot(velocity, altitude, label=str(angle))
        ax[3].set_xlabel("Velocity (ft/s)")
        ax[3].set_ylabel("Altitude (ft)")

    ax[0].set_ylim([-1, None])
    ax[0].set_xlim([-1, None])
    ax[0].legend()
    plt.show()


def obtain_thrust_data_from_simulation() -> None:
    """
    Obtain the distance of the Nike rocket by using the angle
    of 45 degrees and using thrust samples from a normal
    distribution.

    Input: None
    Output: None

    Side-effects:
    - Write the dictionaries as (Panda) dataframes to the
    thrust_data.csv file.
    """
    thrust_samples = np.random.normal(5130, 0.2*5130, 20)

    for idx, thrust in enumerate(thrust_samples):
        # Resetting / Starting simulation.
        solver = Solver(tend=1000)
        # Simulating with new angle.
        result = solver.solve_rocketNike(Rocket(T_a=thrust * 32.174, la=45))
        # Extracting data.
        distance = np.nan_to_num(result[0][:, 0].reshape(1, -1)[0], 0)
        # Writing to file
        d = {'run': idx, 'distance': distance, 'thrust': thrust}
        data = pd.DataFrame(data=d)
        data.to_csv('thrust_data.csv', mode='a', index=False)


def plot_thrust_data_from_simulation() -> None:
    """
    Plot the thrust data from the simulation ran in
    the obtain_thrust_data_from_simulation() function.

    Input: None
    Output: None

    Side-effects:
    - Plotting the distance by thrust for in a barplot
    and showing the result to the user.
    """
    df = pd.read_csv("thrust_data.csv", low_memory=False)
    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["thrust"] = pd.to_numeric(df["thrust"], errors="coerce")

    start = df.run.min()
    end = df.run.max()

    for run in range(int(start), int(end) + 1):

        df_run = df[df["run"] == run]

        # Extracting data.
        distance = df_run["distance"].to_numpy(dtype=np.float64)
        thrust = df_run["thrust"].to_numpy(dtype=np.float64)

        # Plotting the different angles for analysis
        plt.bar(thrust, distance)

    plt.xlabel("distance")
    plt.ylabel("altitude")
    plt.show()


def obtain_data_from_launchangle_simulation(startlaunchangle,
                                            endlaunchangle, N) -> None:
    """
    Obtain the distance of the Nike rocket for different angles,
    store the data in a csv file and plot the data.

    Input: None
    Output: None

    Side-effects:
    - Write the dictionaries as (Panda) dataframes to the
    thrust_data.csv file.
    - Plot the data in a line graph.
    """

    # for launchangle in np.linspace(startlaunchangle, endlaunchangle, N):
    for launchangle in [51, 52, 53, 54, 55]:
        # Resetting / Starting simulation.
        solver = Solver(tend=1000)

        # Simulating with new angle.
        result = solver.solve_rocketNike(Rocket(la=launchangle))

        # Extracting data.
        distance = np.nan_to_num(result[1][:, 0].reshape(1, -1)[0], 0)
        altitude = np.nan_to_num(result[1][:, 1].reshape(1, -1)[0], 0)
        velocity = np.nan_to_num(result[1][:, 2].reshape(1, -1)[0], 0)
        mass = np.nan_to_num(result[0][:, 3].reshape(1, -1)[0], 0)
        time = np.linspace(solver.tbegin, solver.tend, len(distance))
        plt.plot(distance, altitude, label=str(launchangle) + str('°'))

        # Writing to file
        d = {'distance': distance, 'altitude': altitude, 'velocity': velocity,
             'mass': mass, 'time': time}
        data = pd.DataFrame(data=d)
        data.to_csv("Modeldata/AngleData/LAis" +
                    str(launchangle).replace('.', '_') + ".csv")
    plt.xlim([4275000, 4303500])
    plt.ylim([0, 10000])
    plt.xlabel("Distance (ft)")
    plt.ylabel("Altitude (ft)")
    plt.legend()
    plt.savefig("Modeldata/AngleData/bestangleplot3.png", dpi=600)
    plt.show()


def obtain_optimal_angle() -> None:
    """
    Run the Nike physics simulation with distinct angles and store the results
    in a csv file with a tolerance of 1e2.

    Side note: Used for the data analysis and requires long
    cpu time.

    Input: None
    Output: None

    Side-effect:
    - Writing the results of the simulation to the csv file.
    """
    files = [f for f in listdir("Modeldata/AngleData")
             if isfile(join("Modeldata/AngleData", f))]
    mymax = 0
    bestangle = 0

    for i, file in enumerate(files):
        if ".png" in file:
            continue
        data = pd.read_csv(join("Modeldata/AngleData", file))
        distance = data['distance']
        # print(distance)
        curmax = np.max(distance)
        # print(curmax)
        if mymax < curmax:
            print(curmax, mymax, file, np.argmax(distance))
            mymax = curmax
            str_angle = str(file).removeprefix("LAis").removesuffix(".csv")
            bestangle = float(str_angle)
    curangle = bestangle
    bestangle = 0

    for launchangle in np.linspace(curangle-1, curangle+1, 201):
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

        if mymax < np.max(distance):
            mymax = np.max(distance)
            bestangle = launchangle
            print(bestangle)

        # Writing to file
        d = {'distance': distance, 'altitude': altitude, 'velocity': velocity,
             'mass': mass, 'time': time}
        data = pd.DataFrame(data=d)
        data.to_csv("Modeldata/AngleData/LAis" +
                    str(launchangle).replace('.', '_') + ".csv")
    print(bestangle)


if __name__ == "__main__":
    """
    Read the input from standard in and run the
    corresponding function to the number.

    Input: None
    Output: None:

    Side-effects:
    - Read the input from stdin
    - Run the Function corresponding to the number.
    """
    option = None
    try:
        option = int(sys.argv[1])
        startlaunch = float(sys.argv[2])
        endlaunch = float(sys.argv[3])
        N = int(sys.argv[4])
    except (ValueError, IndexError) as e:
        print(f"Error: {e}. Please provide valid numeric arguments.")
        sys.exit(1)  # Exit to indicate an error

    if option == 1:
        plot_while_processing()
    elif option == 3:
        obtain_thrust_data_from_simulation()
    elif option == 4:
        obtain_data_from_launchangle_simulation(startlaunch, endlaunch, N)
    elif option == 5:
        obtain_optimal_angle()
    else:
        plot_thrust_data_from_simulation()
