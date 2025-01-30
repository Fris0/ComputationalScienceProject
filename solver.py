# Program: solver.py
# Authors: Mark, George, Hans, Riemer, Rik
#
# Summary:
#    - Contains the code to solve for the rockets launches.
# Functions:
#    - changedmax: a function that determines the maximum value of a continuous
#                  list, ignoring discontinuities.
#
# Classes:
#    - Solver: class containing the actual solvers for different types of
#       rocket launches and the general settings for those solver functions.
# Usage:
#    - Used in the main folder for solving rocket trajectories.

# The rocket object to solve
from physics import Rocket

# General numerical handling of data
import numpy as np


def changedmax(arr1: list[list[float]],
               arr2: list[list[float]],
               arr3: list[list[float]]) -> float:
    """
    Custom maximum value function, because there are points where
    we suddenly have a big drop in value (crash and detachment).
    Thus we can't just simply use np.max for error finding.
    In:
        - arr1, the 2d array containing the absolute errors between arr2
                and arr3;
        - arr2, the 2d array containing the first simulation's points;
        - arr3, the 2d array containing the second simulation's points.

    Out:
        - mymax, the maximum absolute error disregarding discontinuous
                    points.

    Side effects:
        - N/A
    """
    assert (len(arr1) == len(arr2)) and (len(arr1) == len(arr3)), \
        f"All arrays need to be of the samelength, currently they \
        are: arr1: {len(arr1)}, arr2: {len(arr2)}, arr3: {len(arr3)} "
    mymax = 0
    for i, item in enumerate(arr1):
        # Check for discontinuities
        if ((arr2[i][2] < 1e-10  # The rocket has crashed.
                and arr2[i][3] < 1e-10)
                or (arr3[i][2] < 1e-10
                    and arr3[i][3] < 1e-10)
                or (np.abs(arr3[i][4] - 217) < 1e-10  # The rocket detaches.
                    and np.abs(arr2[i][4] - 217) > 1e-10)
                or (np.abs(arr3[i][4] - 217) > 1e-10
                    and np.abs(arr2[i][4] - 217) < 1e-10)):
            continue

        if mymax < np.max(item):
            mymax = np.max(item)
    return mymax


class Solver():
    """
    A solver class implementing the solvers for specific rocket models.
    And the settings for the solver.

    Parameters:
        - tolerance: the absolute tolerance of the error of the solution;
        - tbegin: the start time of the simulation;
        - tend: the end time of the simulation;
        - min_its: the minimum number of iterations for the solver;
        - max_its: the maximum number of iterations for the solver;
        - stage_dropped: boolean describing whether it has ejected the first
                         stage;
        - Nike: boolean describing wheter it is solving the Nike rocket,
                required for correctly changing the rocket mass.

    Functions:
        - solve_singlestep: Implements the fourth order Runge-Kutta;
        - solve_general: Solves the Runge-Kutta for longer timestamps;
        - solve_rocket1d: Solves the 1 dimensional rocket solver;
        - solve_rocket1d: Solves the 1 staged 2 dimensional rocket solver;
        - solve_rocket1d: Solves the 2 stages 2 dimensional rocket solver;
    """
    def __init__(self, tolerance: float = 1e2,
                 tbegin: float = 0,
                 tend: float = 5,
                 min_its: float = 1e4,
                 max_its: float = 1e7) -> None:
        """
        Input:
            - tolerance: the absolute tolerance of the error of the solution;
            - tbegin: the start time of the simulation;
            - tend: the end time of the simulation;
            - min_its: the minimum number of iterations for the solver;
            - max_its: the maximum number of iterations for the solver.
        """
        self.eps = tolerance
        self.tbegin = tbegin
        self.tend = tend
        self.min_its = min_its
        self.max_its = max_its
        self.stage_dropped = False
        self.Nike = False

    def solve_singlestep(self, f: callable,
                         tn: float,
                         un: list[float],
                         h: float) -> list[float]:
        """
        Perform one step of the fourth order Runge-Kutta

        Arguments:
        - f: the function that takes arguments (tn, un)
        - tn (a number): the time
        - un (a vector): the state vector of the system

        Output:
        A vector of the same dimension as un with one step of
            the Runge-Kutta performed

        Side effects:
            - None
        """

        k1 = f(tn, un)
        if np.all(k1 == 0):
            # Set vx and vy to zero
            return np.array([un[0], un[1], 0, 0, un[4]])
        k2 = f(tn + h/2, un + k1*h/2)
        k3 = f(tn + h/2, un + k2*h/2)
        k4 = f(tn + h, un + k3*h)
        un1 = un + h/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Rocket stage dropped
        if (tn > Rocket().t_b_n and not self.stage_dropped and self.Nike):
            un1[4] -= (un[4] - Rocket().Mp_a)
            self.stage_dropped = True
        return un1

    def solve_general(self, u_0: list[float],
                      f: callable,
                      T: float,
                      N: int) -> np.matrix[float]:
        """
        Solve the ODE with a fourth order Runge-Kutta method.
        Arguments:
        - u_0: the initial state of size d.
        - f: the right-hand side of the ODE, as a *function*
            taking as arguments (t, u(t)).
        - T: the maximum timestep.
        - N: the number of grid points to simulate on.

        Output:
        - (u_n): a matrix (np.array) of size (N+1, d).
        """
        self.stage_dropped = False
        h = T/N
        d = u_0.shape[0]
        u = np.zeros((N, d))
        u[0] = u_0

        for n in range(N - 1):
            u[n+1] = self.solve_singlestep(f, n*h, u[n], h)
            if u[n+1][2] == 0 and u[n+1][3] == 0:
                # Rocked stopped on ground
                for i in range(n+1, N):
                    u[i] = np.array([u[n+1][0], u[n+1][1], 0, 0, u[n+1][4]])
                return u

        return u

    def solve_rocket1d(self, rocket: Rocket) -> list[list[list[float]]]:
        """
        Solves the system of equations for a specific rocket.

        Input:
            - rocket, A rocket class object, contains the rocket we want to launch
        Output:
            - sol2, a (N + 1, 3) array containing the [altitude, velocity, mass]
              at various timesteps.
        Side effects:
            - None
        """

        # Starting conditions
        mass = rocket.Mr
        altitude = 0
        velocity = 0
        args = np.array([altitude, velocity, mass])

        # Print initial values to stdout
        print(f"Intial conditions are: {altitude, velocity, mass}")
        T = self.tend-self.tbegin  # Total run time.
        N = int(self.min_its)  # Steps

        result = self.solve_general(args, rocket.rocket_1d_dynamics, T, N//2)
        result_2 = self.solve_general(args, rocket.rocket_1d_dynamics, T, N)

        mymax = np.max(np.abs(np.repeat(result, repeats=2, axis=0) - result_2))

        while (mymax >= self.eps and 2 * N < self.max_its):
            print(f"Desired tolerance not reached, increasing number of \
                  interpolation points to {N} and current maximum error is {mymax}")
            N *= 2
            result = result_2
            result_2 = self.solve_general(args, rocket.rocket_1d_dynamics, T, N)
            mymax = np.max(np.abs(np.repeat(result, repeats=2, axis=0) - result_2))

        return np.asarray((np.repeat(result, repeats=2, axis=0), result_2))

    def solve_rocket2d(self, rocket: Rocket) -> list[list[list[float]]]:
        """
        Solves the system of equations for a specific rocket.

        Input:
            - rocket, A rocket class object, contains the rocket we want to launch
        Output:
            - sol2, a (N + 1, 4) array containing the [distance, altitude, velocity, mass]
              at various timesteps.
        Side effects:
            - None
        """

        # Starting conditions
        mass = rocket.Mr
        posx = 0
        posy = 0
        velx = 0
        vely = 0
        args = np.array([posx, posy, velx, vely, mass])

        # Print initial values to stdout
        print(f"Intial conditions are: {posx, posy, velx, vely, mass}")
        T = self.tend-self.tbegin  # Total run time.
        N = int(self.min_its)  # Steps

        # Obtain results and determine the actual error.
        result = self.solve_general(args,
                                    Rocket(np.rad2deg(rocket.la)).rocket_2d_dynamics,
                                    T, N//2)
        result_2 = self.solve_general(args,
                                      Rocket(np.rad2deg(rocket.la)).rocket_2d_dynamics,
                                      T, N)
        absolute_errors = np.abs(np.repeat(result, repeats=2, axis=0) - result_2)
        mymax = changedmax(absolute_errors, np.repeat(result, repeats=2, axis=0), result_2)

        while (mymax >= self.eps and 2 * N < self.max_its):
            print(f"Desired tolerance not reached, increasing number of \
                  interpolation points to {N} and current maximum error is {mymax}")
            N *= 2
            result = result_2
            result_2 = self.solve_general(args,
                                          Rocket(la=np.rad2deg(rocket.la)).rocket_2d_dynamics,
                                          T, N)
            absolute_errors = np.abs(np.repeat(result, repeats=2, axis=0) - result_2)
            mymax = changedmax(absolute_errors, np.repeat(result, repeats=2, axis=0), result_2)

        return_list = np.asarray((np.repeat(result, repeats=2, axis=0), result_2))

        return np.asarray([[(item[0], item[1],
                             np.sqrt(item[2]**2 + item[3]**2), item[4])
                            for item in result]
                           for result in return_list])

    def solve_rocketNike(self, rocket: Rocket) -> list[list[list[float]]]:
        """
        Solves the system of equations for a specific rocket.

        Input:
            - rocket, A rocket class object, contains the rocket we want to launch
        Output:
            - sol2, a (N + 1, 4) array containing the [distance, altitude, velocity, mass]
            at various timesteps.
        Side effects:
            - None
        """

        def x_y_to_curvature(x: float, y: float) -> tuple[float, float]:
            """'
            We solve the following trigonomic problem to convert x,y to
            altitude and distance on a curved earth.

                b
            start--__
            |        x_rocketpos
            |        /
            | r     /
            |      / a
            |     /
            |  e /
            earth

            We can calculate a, b.
            After that we use the cosine rule to calculate e (earthangle).
            https://en.wikipedia.org/wiki/Law_of_cosines#Use_in_solving_triangles
            Then we calculate distance using alpha*r

            In:
                - x, the x position in the plane;
                - y, the y position in the plane.

            Out:
                - distance, altitude, the actual distance and altitude of the
                  spherical earth.
            Side effects:
                - N/A
            """
            r = Rocket().Re
            altitude = np.sqrt((x-0)**2 + (y+r)**2) - r
            a = altitude + r
            b = np.sqrt((x-0)**2 + (y-0)**2)
            if (b**2 < 1e-10):
                # Removes an invalid value encountered error.
                earthangle = 0
            else:
                # + 1 is added due to floating point rounding errors.
                earthangle = np.arccos((r**2 + a**2 - b**2)/(2*a*r+1))

            # For small angle it doesn't really matter whether we use cos.
            # But this is more numerically stable.
            if (np.abs(earthangle) < 1e-3 or np.isnan(earthangle)):
                earthintersectionpos = r/(r+altitude)
                return (np.sqrt((x*earthintersectionpos-0)**2
                                + ((y+r)*earthintersectionpos-r)**2), altitude)

            distance = earthangle*r
            return distance, altitude

        # Starting conditions
        mass = rocket.Mr
        posx = 0
        posy = 0
        velx = 0
        vely = 0
        args = np.array([posx, posy, velx, vely, mass])

        # Print initial values to stdout
        print(f"Intial conditions are: {posx, posy, velx, vely, mass}")
        T = self.tend-self.tbegin  # Total run time.
        N = int(self.min_its)  # Steps
        self.Nike = True

        # Obtain results and determine the actual error.
        result = self.solve_general(args,
                                    Rocket(la=np.rad2deg(rocket.la)).Nike_Apache_physics,
                                    T, N//2)
        result_2 = self.solve_general(args,
                                      Rocket(la=np.rad2deg(rocket.la)).Nike_Apache_physics,
                                      T, N)
        absolute_errors = np.abs(np.repeat(result, repeats=2, axis=0) - result_2)
        mymax = changedmax(absolute_errors, np.repeat(result, repeats=2, axis=0), result_2)

        while (mymax >= self.eps and 2 * N < self.max_its):
            print(f"Desired tolerance not reached, increasing number of \
                interpolation points to {N} and current maximum error is {mymax}")
            N *= 2
            result = result_2
            result_2 = self.solve_general(args,
                                          Rocket(la=np.rad2deg(rocket.la)).Nike_Apache_physics,
                                          T, N)
            absolute_errors = np.abs(np.repeat(result, repeats=2, axis=0) - result_2)
            mymax = changedmax(absolute_errors, np.repeat(result, repeats=2, axis=0), result_2)

        return_list = np.asarray((np.repeat(result, repeats=2, axis=0), result_2))

        return np.asarray([[(x_y_to_curvature(item[0], item[1])[0],
                            x_y_to_curvature(item[0], item[1])[1],
                            np.sqrt(item[2]**2 + item[3]**2), item[4])
                            for item in result]
                          for result in return_list])
