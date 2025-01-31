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

# Data manipulation
import numpy as np  # Used for the numpy arrays and it's functions.
# self-made libraries
from solver import Solver
from physics import Rocket
# Plots
import matplotlib.pyplot as plt  # Used for plotting the simulation data.
# IO stuff
import sys  # For reading out the std-in from terminal input.
import pandas as pd  # Used for storing the numpy arrays of the simulation.
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

def obtain_data_from_launchangle_simulation(startlaunchangle: float,
                                            endlaunchangle: float,
                                            N: int) -> None:
    """
    Obtain the distance of the Nike rocket for different angles,
    store the data in a csv file and plot the data.
    Tip: this is useful for "multithreading" the data obtaining specifically
         one can run in bash: >>> python3 main.py 4 0 15 16
         and then in another terminal: >>>python3 main.py 4 15 30 15
         This significantly speeds up the code.


    Input:
        - startlaunchangle, the lowest value for the linspace;
        - endlaunchangle, the highest value for the linspace;
        - N, the number of launch angles to test;
    Output: None

    Side-effects:
    - Write the dictionaries as (Panda) dataframes to the
    thrust_data.csv file;
    - Plot the data in a line graph;
    """

    for launchangle in np.linspace(startlaunchangle, endlaunchangle, N):
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
        data.to_csv("ModelData/AngleData/LAis" +
                    str(launchangle).replace('.', '_') + ".csv")
    plt.xlabel("Distance (ft)")
    plt.ylabel("Altitude (ft)")
    plt.legend()
    plt.show()

def obtain_poster_launchangle_pic() -> None:
    """
    Obtains the pictures of the launchangles for the launch angles on the
    poster.

    Input: None
    Output: None

    Side-effects:
    - Write the dictionaries as (Panda) dataframes to the
    thrust_data.csv file.
    - Plot the data in a line graph.
    - Write the plot to "ModelData/AngleData/bestangleplot3.png"
    """
    for launchangle in [0, 22.5, 45, 51, 52, 53, 54, 55, 67.5, 90]:
        # Resetting / Starting simulation.
        solver = Solver(tend=1000, tolerance=1e4)

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
        data.to_csv("ModelData/AngleData/LAis" +
                    str(launchangle).replace('.', '_') + ".csv")
    plt.xlim([4275000, 4303500])
    plt.ylim([0, 10000])
    plt.xlabel("Distance (ft)")
    plt.ylabel("Altitude (ft)")
    plt.legend()
    plt.savefig("ModelData/AngleData/bestangleplot2.png", dpi=600)
    plt.show()

def obtain_zoomed_in_launchangle_pic() -> None:
    """
    Obtains the zoomed in pictures of the launchangles for the almost optimal
    launch angles.

    Input: None
    Output: None

    Side-effects:
    - Write the dictionaries as (Panda) dataframes to the
    thrust_data.csv file.
    - Plot the data in a line graph.
    - Write the plot to "ModelData/AngleData/bestangleplot3.png"
    """

    # for launchangle in np.linspace(startlaunchangle, endlaunchangle, N):
    for launchangle in [51, 52, 53, 54, 55]:
        # Resetting / Starting simulation.
        solver = Solver(tend=1000, tolerance=1e2)

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
        data.to_csv("ModelData/AngleData/LAis" +
                    str(launchangle).replace('.', '_') + ".csv")
    plt.xlim([4275000, 4303500])
    plt.ylim([0, 10000])
    plt.xlabel("Distance (ft)")
    plt.ylabel("Altitude (ft)")
    plt.legend()
    plt.savefig("ModelData/AngleData/bestangleplot3.png", dpi=600)
    plt.show()


def obtain_optimal_angle() -> None:
    """
    Goes through the list of csv's stored in ModelData/AngleData and then
    figures out which one reaches the highest distance.

    Input: None
    Output: None

    Side effects:
        - Prints the best angle to stdout
    """
    files = [f for f in listdir("ModelData/AngleData")
             if isfile(join("ModelData/AngleData", f))]
    mymax = 0
    bestangle = 0

    for file in files:
        if ".png" in file:
            continue
        data = pd.read_csv(join("ModelData/AngleData", file))
        distance = data['distance']
        curmax = np.max(distance)
        if mymax < curmax:
            mymax = curmax
            str_angle = str(file).removeprefix("LAis").removesuffix(".csv")
            bestangle = float(str_angle)
    print(bestangle)

def obtain_angle_plot() -> None:
    """
    Goes through the list of csv's stored in ModelData/AngleData and then
    figures out which one reaches the highest distance.

    Input: None
    Output: None

    Side effects:
        - Writes the plot describing all angles to
          "ModelData/AngleData/bestangleplot5.png"
    """
    files = [f for f in listdir("ModelData/AngleData")
             if isfile(join("ModelData/AngleData", f))]
    angle_dist_list = []

    for file in files:
        if ".png" in file:
            continue
        data = pd.read_csv(join("ModelData/AngleData", file))
        str_angle = str(file).removeprefix("LAis").removesuffix(".csv").replace('_', '.')
        distance = data['distance']
        curdist = np.max(distance)
        angle_dist_list.append([float(str_angle), curdist])
    angle_dist_list.sort(key=lambda x: x[0])
    angle_dist_list = np.transpose(angle_dist_list)
    plt.plot(angle_dist_list[0], angle_dist_list[1])
    plt.xlim([0, 90])
    plt.ylim([0,5e6])
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Distance (ft)")
    plt.axvline(53, color='red', label='53°')
    plt.savefig("ModelData/AngleData/bestangleplot5.png", dpi=600)
    plt.show()


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
        if (option == None):
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
    elif option == 6:
        obtain_poster_launchangle_pic()
    elif option == 7:
        obtain_zoomed_in_launchangle_pic()
    elif option == 8:
        obtain_angle_plot()
    else:
        plot_thrust_data_from_simulation()
