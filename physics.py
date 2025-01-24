# Here the general physics.
# Responsible people: Hans, George, Riemer, Mark, Rik
import numpy as np
from fps_to_cd import fps_to_Cd


def rho(h):
    """
    Calculates the altitude specific atmospheric density in slug/ft^3

    First calculates the density in kg/m^3 and then converts it to slug/ft^3

    h: height in feet

    returns: rho in slug/ft^3
    """

    R = 8.3144598       # Universal gas constant (N*m/(mol*K))
    g_0 = 9.80665       # gravitational accelaration (m/s^2)
    M = 0.0289644       # molar mass of Earth's air (kg/mol)

    boundaries = [0, 11000, 20000, 32000, 47000, 51000, 71000]
    values = {
        0: (1.2250, 288.15, 0.0065),
        1: (0.36391, 216.65, 0),
        2: (0.08803, 216.65, -0.001),
        3: (0.01322, 228.65, -0.0028),
        4: (0.00143, 270.65, 0),
        5: (0.00086, 270.65, 0.0028),
        6: (0.000064, 214.65, 0.002)
    }
    h = h * 0.3048
    if h in boundaries:
        level = boundaries.index(h)
    else:
        boundaries.append(h)
        boundaries.sort()
        level = boundaries.index(h) - 1

    rho_b, T_b, L_b = values[level]
    h_b = boundaries[level]

    case = 2 if (level == 1 or level == 4) else 1

    if case == 1:
        rho_metric = rho_b * ((T_b - (h-h_b)*L_b)/(T_b))**((g_0*M)/(R*L_b)-1)
    elif case == 2:
        rho_metric = rho_b * np.e**((-1*g_0*M*(h-h_b))/(R*T_b))

    return 0.0019 * rho_metric


class Rocket():
    def __init__(self, g=32.174, rho=0.0023769, Cd=0.5, A=1.67,
                 T=42500*32.174, M=1534.0, Mp=886.0, t_b=3.5, I=32800.0, la=90):

        self.g = g              # gravitational acceleration (ft/s^2)
        self.rho = rho          # air density (slug/ft^3)
        self.Cd = Cd            # drag coefficient
        self.A = A              # cross-sectional area (ft^2), largest diameter occurs at nike connection to apache
        self.t_b = t_b          # burn time (sec) (Nike + Apache)
        self.T = T              # thrust of Nike (lbf)
        self.I = T * t_b        # Total impulse of Nike (lb-sec)
        self.Mr = M             # total rocket mass with propellant (lbs)
        self.Mp = Mp            # propellant mass (Nike + Apache) (lbs)
        self.Mr_n = 755.0       # Total propellant mass in Nike (lbs)
        self.mass_flow_nike = self.Mr_n / t_b  # Nike Mass flow rate (lbs/sec)
        self.la = np.deg2rad(la)  # launch angle (radians)
        self.Re = 2.09e7        # approx. earth radius in feet

        # Two stage additions
        self.Launcher = 0.25    # Launcher rod of Nike Apache (ft) (data on 21 or 0.25ft)
        self.t_b_a = 6.36       # burn time (sec) (Apache)
        self.t_b_n = 3.5        # burn time (sec) (Nike)
        self.t_int = 16.5       # time between Nike burnout and Apache ignition (sec)
        self.Mr_a = 131.0      # total propellant mass in Apache after Nike burnout (lbs)
        self.Mp_a = 217.0      # rocket mass after Nike detaches (lbs)
        self.A_a = 0.239      # cross-sectional area (ft^2) after Nike detaches
        self.T_n = 42500.0      # thrust of Nike (lbf)
        self.I_n = self.T_n * self.t_b_n  # Total impulse of Nike (lb-sec)
        self.T_a = 5130.0       # Thrust of Apache (lbf)
        self.I_a = 32800.0      # Total impulse of Apache (lb-sec)
        self.mass_flow_apache = self.Mr_a / self.t_b_a  # Apache Mass flow rate (lbs/sec)

    def rocket_1d_dynamics(self, t, state):
        """
        Computes the time derivatives (dy/dt, dv/dt, dm/dt)
        for a 1D rocket under constant gravity and drag.

        t = time of type float
        state = a tuple of size 3 containing altitude, velocity and mass.

        Resulting ODEs after one time step:
        y' = v (Altitude changes at the rate of current velocity)
        v' = Fnet / m (Velocity changes at the rate of net force divided by
            current mass equivalent to acceleration)
        m' = Mr / t_b or 0 (mass decreases at a constant rate while the
            engine is burning, then remains unchanged after burnout)
        """

        # Unpack state
        y, v, m = state

        # Decide if rocket is still burning
        if t < self.t_b:
            # Thrust is constant T
            thrust = self.T
            # Constant mass flow rate
            mdot = self.mass_flow_nike
        else:
            # No more thrust
            thrust = 0.0
            mdot = 0.0

        # Forces
        F_weight = -m * self.g
        F_drag = -0.5 * rho(y) * fps_to_Cd(v) * self.A * v**2
        F_thrust = thrust
        F_net = F_weight + F_drag + F_thrust

        # Acceleration
        a = F_net / m

        if (y <= 0 and v < 0):
            v = 0

        return np.array([v, a, -mdot])

    def rocket_2d_dynamics(self, t, state):
        """
        Computes the time derivatives (dx/dt, dy/dt, dvx/dt, dvy/dt, dm/dt)
        for a 2D rocket under nonconstant gravity.

        There is no rail or launch constraint phase in the simulation.
        """

        # Unpack state
        x, y, vx, vy, m = state

        h = np.sqrt(x**2 + y**2)

        # Decide if rocket is still burning
        if t < self.t_b:
            # Thrust is constant T
            thrust = self.T
            mdot = self.mass_flow_nike
        else:
            # No more thrust
            thrust = 0.0
            mdot = 0.0

        # Inverse-square law: g(y) = g0 * (Re / (Re + y))^2
        g_local = self.g * (self.Re / (self.Re + y))**2 if (self.Re + y) > 0 else self.g

        # Flight angle from vertical
        dist = np.sqrt(x**2 + y**2)
        if dist < 2:
            fa = self.la
        else:
            fa = np.arctan2(vy, vx)

        speed = np.sqrt(vx**2 + vy**2)
        if speed > 1e-12:
            vx_hat = vx / speed
            vy_hat = vy / speed
        else:
            vx_hat = 0
            vy_hat = 0

        # Drag magnitude
        D = 0.5 * rho(h) * fps_to_Cd(speed) * self.A * speed**2

        # Drag forces(opposite to velocity)
        Fx_drag = -D * vx_hat
        Fy_drag = -D * vy_hat

        # Thrust forces
        Fx_thrust = thrust * np.cos(fa)
        Fy_thrust = thrust * np.sin(fa)

        # Net forces
        Fx_net = Fx_drag + Fx_thrust
        Fy_net = Fy_drag + Fy_thrust + (-m * g_local)

        # Accelerations
        ax = Fx_net / m
        ay = Fy_net / m

        return np.array([vx, vy, ax, ay, -mdot])

    def Nike_Apache_physics(self, t, state):
        """
        Computes the time derivatives (dx/dt, dy/dt, dvx/dt, dvy/dt, dm/dt)
        for a two stage Nike Apache Rocket under nonconstant gravity.
        1) Stage 1 burn: 0 <= t < t_b_n
        2) Coast (interstage): t_b_n <= t < t_b_n + t_int
        3) Stage 2 burn: t_b_n + t_int <= t < t_b_n + t_int + t_b_a
        4) Post-burn coast: t >= t_b_n + t_int + t_b_a
        """

        # Unpack state
        x, y, vx, vy, m = state

        h = np.sqrt(x**2 + y**2)

        # Stage 1 Nike Burn and Detach
        if t < self.t_b_n:
            thrust = self.T_n
            mdot = self.mass_flow_nike
            area = self.A

        # Nike dropped, no thrust
        elif t < self.t_b_n + self.t_int:
            thrust = 0.0
            mdot = 0.0
            area = self.A_a

        # Stage 2 Apache burn
        elif t < self.t_b_n + self.t_int + self.t_b_a:
            thrust = self.T_a
            mdot = self.mass_flow_apache
            area = self.A_a

        # Post Apache burnout
        else:
            thrust = 0.0
            mdot = 0.0
            area = self.A_a

        # Inverse-square law: g(y) = g0 * (Re / (Re + y))^2
        g_local = self.g * (self.Re / (self.Re + h)**2) if (self.Re + h) > 0 else self.g

        # Flight angle from vertical
        fa = np.arctan2(vx, vy)

        speed = np.sqrt(vx**2 + vy**2)
        if speed > 1e-12:
            vx_hat = vx / speed
            vy_hat = vy / speed
        else:
            vx_hat = 0
            vy_hat = 0

        # Drag magnitude
        D = 0.5 * rho(h) * fps_to_Cd(speed) * area * speed**2

        # Drag forces(opposite to velocity)
        Fx_drag = -D * vx_hat
        Fy_drag = -D * vy_hat

        # Thrust forces
        Fx_thrust = thrust * np.cos(fa)
        Fy_thrust = thrust * np.sin(fa)

        # Net forces
        Fx_net = Fx_drag + Fx_thrust
        Fy_net = Fy_drag + Fy_thrust + (-m * g_local)

        # Accelerations
        ax = Fx_net / m
        ay = Fy_net / m

        return np.array([vx, vy, ax, ay, -mdot])


#if __name__ == "__main__":
#    h = np.linspace(0, 300000, 1000000)
#    rhos = []
#    for height in h:
#        rhos.append(rho(height))
#    import matplotlib.pyplot as plt
#    plt.plot(h, rhos)
#    plt.rcParams["text.usetex"]=True
#    plt.xlabel("height(feet)")
#    plt.ylabel("Air density(slug/ft$^3$)")
#    plt.show()
