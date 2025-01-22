# Here the numerical solver, probably using Runge-Kutta
# Responsible person: Mark, Riemer

from physics import Rocket
import numpy as np
import matplotlib.pyplot as plt

class Solver():
    def __init__(self, tolerance=1e4, tbegin=0, tend=5, min_its=10e4, max_its=10e7):  # General settings for the solver.
        self.eps = tolerance
        self.tbegin = tbegin
        self.tend = tend
        self.min_its = min_its
        self.max_its = max_its

    def solve_singlestep(f, tn, un, h):
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

        k1 = (f(tn, un))
        k2 = f(tn + h/2, un + k1*h/2)
        k3 = f(tn + h/2, un + k2*h/2)
        k4 = f(tn + h, un + k3*h)
        un1 = un + h/6 * (k1 + 2*k2 + 2*k3 + k4)
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

        h = T/N
        d = u_0.shape[0]
        u = np.zeros((N+1, d))
        u[0] = u_0
        for n in range(N):
            u[n+1] = Solver.solve_singlestep(f, n*h, u[n], h)

        return u

    def solve_rocket(self, rocket):
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
        N = int(self.min_its) # Steps

        result = self.solve_general(args, rocket.rocket_1d_dynamics, T, N)

        return result
    
if __name__ == "__main__":
    rocket = Rocket()
    solver = Solver()
    result = solver.solve_rocket(rocket)

    print(result[0], result[1])

    altitude = result[:, 0].reshape(1, -1)[0]
    velocity = result[:, 1].reshape(1, -1)[0]
    mass = result[:, 2].reshape(1, -1)[0]


    T = solver.tend - solver.tbegin
    x = np.linspace(0, T, int(solver.min_its))

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x, altitude[:-1], label="Altitude")
    ax[0].plot(x, velocity[:-1], label="Velocity")
    ax[1].plot(x, mass[:-1], label="Mass")
    ax[0].legend()
    ax[1].legend()
    plt.show()