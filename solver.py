# Here the numerical solver, probably using Runge-Kutta
# Responsible person: Mark, Riemer

from physics import Rocket
import numpy as np
import matplotlib.pyplot as plt
# import copy


class Solver():
    def __init__(self, tolerance=1e4, tbegin=0, tend=5, min_its=1e4, max_its=1e7):  # General settings for the solver.
        """
        Parameters:
            - min_its, should be divisible by 2.
        """
        self.eps = tolerance
        self.tbegin = tbegin
        self.tend = tend
        self.min_its = min_its
        self.max_its = max_its
        self.rocket_dropped = False
        self.Nike = False


    def solve_singlestep(self, f, tn, un, h):
        """
        Perform one step of the fourth order Runge-Kutta

        Arguments:
        - f: the function that takes arguments (tn, un)
        - tn (a number): the time
        - un (a vector): the state vector of the system

        Output:
        A vector of the same dimension as un with one step of
            the Runge-Kutta performed
        """

        k1 = f(tn, un)
        if np.all(k1 == 0):
            return np.array([un[0], un[1], 0, 0, un[4]]) #Set vx and vy to zero
        k2 = f(tn + h/2, un + k1*h/2)
        k3 = f(tn + h/2, un + k2*h/2)
        k4 = f(tn + h, un + k3*h)
        un1 = un + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        if (tn > Rocket().t_b_n and not self.rocket_dropped and self.Nike):
            un1[4] -= (un[4] - Rocket().Mp_a)
            self.rocket_dropped = True
        return un1

    def solve_general(self, u_0, f, T, N):
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
        self.rocket_dropped = False
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

    def solve_rocket1d(self, rocket):
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
                  interpolation points to {N} and current maximum error is {mymax} \
                  at {np.argmax(np.abs(np.repeat(result, repeats=2, axis=0) - result_2), axis=0)}")
            N *= 2
            result = result_2
            result_2 = self.solve_general(args, rocket.rocket_1d_dynamics, T, N)
            mymax = np.max(np.abs(np.repeat(result, repeats=2, axis=0) - result_2))

        return np.asarray((np.repeat(result, repeats=2, axis=0), result_2))

    def solve_rocket2d(self, rocket):
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

        result = self.solve_general(args, Rocket(np.rad2deg(rocket.la)).rocket_2d_dynamics, T, N//2)
        result_2 = self.solve_general(args, Rocket(np.rad2deg(rocket.la)).rocket_2d_dynamics, T, N)
        mymax = np.max(np.abs(np.repeat(result, repeats=2, axis=0) - result_2))

        while (mymax >= self.eps and 2 * N < self.max_its):
            print(f"Desired tolerance not reached, increasing number of \
                  interpolation points to {N} and current maximum error is {mymax}, \
                  at {np.argmax(np.abs(np.repeat(result, repeats=2, axis=0) - result_2), axis=0)}")
            for val in np.argmax(np.abs(np.repeat(result, repeats=2, axis=0) - result_2), axis=0):
                print(np.abs(np.repeat(result, repeats=2, axis=0) - result_2)[val], result[val], np.repeat(result, repeats=2, axis=0)[val])
            N *= 2
            result = result_2
            result_2 = self.solve_general(args, Rocket(la=np.rad2deg(rocket.la)).rocket_2d_dynamics, T, N)
            mymax = np.max(np.abs(np.repeat(result, repeats=2, axis=0) - result_2))

        return_list = np.asarray((np.repeat(result, repeats=2, axis=0), result_2))

        return np.asarray([[(item[0], item[1],
                             np.sqrt(item[2]**2 + item[3]**2), item[4])
                            for item in result]
                           for result in return_list])

    def solve_rocketNike(self, rocket):
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

        def x_y_to_a_d(x, y):
            """
                b
            start--__
            |        x_rocketpos
            |        /
            | r     /
            |      / a
            |     /
            |  $ /
            earth
            We can calculate a, b.
            After that we use the cosine rule to calculate $ (alpha).
            https://en.wikipedia.org/wiki/Law_of_cosines#Use_in_solving_triangles
            Then we calculate distance using (2*pi)/(alpha)*r
            """
            r = Rocket().Re
            altitude = np.sqrt((x-0)**2 + (y+r)**2) - r
            a = altitude + r
            b = np.sqrt((x-0)**2 + (y-0)**2)
            earthangle = np.arccos((r**2 + a**2 - b**2)/(2*a*r))
            if (np.abs(earthangle) < 0.001 or np.isnan(earthangle)): # For small angle it doesn't really matter
                earthintersectionpos = r/(r+altitude)
                # print(earthintersectionpos)
                return (np.sqrt((x*earthintersectionpos-0)**2
                                + ((y+r)*earthintersectionpos-r)**2),
                                 altitude)
            distance = earthangle*r
            # print(f"x: {x}, y: {y}, a: {a}, b: {b}, r: {r}, distance: {distance}, altitude: {altitude}, earthangle: {earthangle}")
            return distance, altitude

        def changedmax(arr1):
            mymax = 0
            for item in arr1:
                if item[3] != 0 and item[4] != 0:
                    if mymax < np.max(item):
                        mymax = np.max(item)
            return mymax

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

        result = self.solve_general(args, Rocket(la=np.rad2deg(rocket.la)).Nike_Apache_physics, T, N//2)
        result_2 = self.solve_general(args, Rocket(la=np.rad2deg(rocket.la)).Nike_Apache_physics, T, N)
        mymax = changedmax(np.abs(np.repeat(result, repeats=2, axis=0) - result_2))
        # mymax = np.max(np.abs(np.repeat(result, repeats=2, axis=0) - result_2))

        while (mymax >= self.eps and 2 * N < self.max_its):
            print(f"Desired tolerance not reached, increasing number of \
                interpolation points to {N} and current maximum error is {mymax}")
            N *= 2
            result = result_2
            result_2 = self.solve_general(args, Rocket(la=np.rad2deg(rocket.la)).Nike_Apache_physics, T, N)
            mymax = changedmax(np.abs(np.repeat(result, repeats=2, axis=0) - result_2))

        return_list = np.asarray((np.repeat(result, repeats=2, axis=0), result_2))

        return np.asarray([[(x_y_to_a_d(item[0], item[1])[0],
                                x_y_to_a_d(item[0], item[1])[1],
                            np.sqrt(item[2]**2 + item[3]**2), item[4])
                            for item in result]
                            for result in return_list])
