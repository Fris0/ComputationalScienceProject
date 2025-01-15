# Here the numerical solver, probably using Runge-Kutta
# Responsible person: Mark, Riemer

from physics import Rocket, PhysicsFormulas

class Solver():
    def __init__(self): # General settings for the solver.
        pass

    def solve_singlestep(f, tn, un, h):
        """
        Perform one step of the fourth order Runge-Kutta
        
        Arguments:
        - f: the function that takes arguments (tn, un)
        - tn (a number): the time
        - un (a vector): the state vector of the system
        
        Output:
        A vector of the same dimension as un with one step of the Runge-Kutta performed
        """
        
        k1 = (f(tn, un))
        k2 = f(tn + h/2, un + k1*h/2)
        k3 = f(tn + h/2, un + k2*h/2)
        k4 = f(tn + h, un + k3*h)
        un1 = un + h/6 *(k1 + 2*k2 + 2*k3 + k4)
        return un1

    def solve_general(u0, f, T, N):
        """
        Solve the ODE with a fourth order Runge-Kutta method.
        Arguments:
        - u_0: the initial state of size d.
        - f: the right-hand side of the ODE, as a *function* taking as arguments (t, u(t)).
        - T: the maximum timestep.
        - N: the number of grid points to simulate on.

        Output:
        - (u_n): a matrix (np.array) of size (N+1, d).
        """
        
        h = T/N
        d = u0.shape[0]
        u = np.zeros((N+1, d))
        u[0] = u_0
        for n in range(N):
            u[n+1] = solve_singlestep(f, n*h, u[n], h)
            
        return u
        
