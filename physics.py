# Here the general physics.
# Responsible people: Hans, George
import numpy as np

class Rocket():
    def __init__(self, g=32.174, rho=0.0023769, Cd=0.5, A=1.67,
        T=42500, M=1534.0, Mp=886.0, t_b=3.5, I=32800.0, la=85):

        self.g = g              # gravitational acceleration (ft/s^2)
        self.rho = rho          # air density (slug/ft^3)
        self.Cd = Cd            # drag coefficient
        self.A = A              # cross-sectional area (ft^2), largest diameter occurs at nike nozzle
        self.T = T              # thrust of Nike (lbf)
        self.I = T * t_b        # Total impulse of Nike (lb-sec)
        self.mass_flow_nike = (Mp-131) / t_b # We know that we use up 755 lbs of propellant.
        self.Mp = Mp            # propellant mass (Nike + Apache) (lbs)
        self.Mr = M             # total rocket mass with propellant (lbs)
        self.t_b = t_b          # burn time (sec) (Nike + Apache)
        self.la=np.deg2rad(la)  # launch angle (radians)
        self.Re = 2.09e7        # approx. earth radius in feet

        #Two stage addiitons
        self.Launcher = 0.25    # Launcher rod of Nike Apache (ft) (data on 21 or 0.25ft)
        self.t_b_a = 6.36       # burn time (sec) (Apache)
        self.t_b_n = 3.5        # burn time (sec) (Nike)
        self.t_int = 16.5       # time between Nike burnout and Apache ignition (sec)
        self.Mr_a  = 131        # total propellant mass after Nike burnout (lbs)
        self.Mp_a  = 217        # rocket mass after Nike detaches (lbs)
        self.A_a   = 0.239      # cross-sectional area (ft^2) after Nike detaches
        self.T_n = 42500        # thrust of Nike (lbf)
        self.I_n = self.T_n * self.t_b_n  # Total impulse of Nike (lb-sec)
        self.T_a = 5130.0       # Thrust of Apache (lbf)
        self.I_a = 32800.0      # Total impulse of Apache (lb-sec)

    def rocket_1d_dynamics(self, t, state):
        """
        Computes the time derivatives (dy/dt, dv/dt, dm/dt)
        for a 1D rocket under constant gravity and drag.


        Resulting ODEs after one time step:
        y' = v (Altitude changes at the rate of current velocity)
        v' = Fnet / m (Velocity changes at the rate of net force divided by
            current mass equivalent to acceleration)
        m' = -I/(t_burn * g) or 0 (mass decreases at a constant rate while the
            engine is burning, then remains unchanged after burnout)
        """

        #Unpack state
        y,v,m = state

        # Decide if rocket is still burning
        if t < self.t_b:
            # Thrust is constant T
            thrust = self.T
            # Constant mass flow rate
            # mdot = self.I / (self.t_b * self.g) # Note that this doesn't work because Impulse =/= Specific Impulse
            mdot = self.mass_flow_nike
        else:
            # No more thrust
            thrust = 0.0
            mdot = 0.0

        # Forces
        F_weight = -m * self.g
        F_drag = -0.5 * self.rho * self.Cd * self.A * v * abs(v)
        F_thrust = thrust
        F_net = F_weight + F_drag + F_thrust

        # Acceleration
        a = F_net / m

        return np.array([v, a, -mdot])

    def rocket_2d_dynamics(self, t, state):
        """
        Computes the time derivatives (dx/dt, dy/dt, dvx/dt, dvy/dt, dm/dt)
        for a 2D rocket under nonconstant gravity.

        There is no rail or launch constraint phase in the simulation.
        """

        #Unpack state
        x, y, vx, vy, m = state

        # Decide if rocket is still burning
        if t < self.t_b:
            # Thrust is constant T
            thrust = self.T
            mdot = self.I / (self.t_b * self.g)  # Mass flow rate
        else:
            # No more thrust
            thrust = 0.0
            mdot = 0.0

        #Inverse-square law: g(y) = g0 * (Re / (Re + y))^2
        g_local = self.g * (self.Re / (self.Re + y))**2 if (self.Re + y) > 0 else self.g

        #Flight angle from vertical
        fa = np.arctan2(vx, vy)

        speed = np.sqrt(vx**2 + vy**2)
        if speed > 1e-12:
            vx_hat = vx / speed
            vy_hat = vy / speed
        else:
            vx_hat = 0
            vy_hat = 0

        #Drag magnitude
        D = 0.5 * self.rho * self.Cd * self.A * speed**2

        #Drag forces(opposite to velocity)
        Fx_drag = -D * vx_hat
        Fy_drag = -D * vy_hat

        #Thrust forces
        Fx_thrust = thrust * np.sin(fa)
        Fy_thrust = thrust * np.cos(fa)

        #Net forces
        Fx_net = Fx_drag + Fx_thrust
        Fy_net = Fy_drag + Fy_thrust + (-m * g_local)

        #Accelerations
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

        #Unpack state
        x, y, vx, vy, m = state

        #Stage 1 Nike Burn and Detach
        if t < self.t_b_n:
            thrust = self.T_n
            mdot = self.I_n / (self.t_b_n * self.g)
            area = self.A

        #Nike dropped, no thrust
        elif t < self.t_b_n + self.t_int:
            thrust = 0.0
            mdot = 0.0
            area = self.A_a

        #Stage 2 Apache burn
        elif t < self.t_b_n + self.t_int + self.t_b_a:
            thrust = self.T_a
            mdot = self.I_a / (self.t_b_a * self.g)
            area = self.A_a

        #Post Apache burnout
        else:
            thrust = 0.0
            mdot = 0.0
            area = self.A_a

        #Inverse-square law: g(y) = g0 * (Re / (Re + y))^2
        g_local = self.g * (self.Re / (self.Re + y))**2 if (self.Re + y) > 0 else self.g

        #Flight angle from vertical
        fa = np.arctan2(vx, vy)

        speed = np.sqrt(vx**2 + vy**2)
        if speed > 1e-12:
            vx_hat = vx / speed
            vy_hat = vy / speed
        else:
            vx_hat = 0
            vy_hat = 0

        #Drag magnitude
        D = 0.5 * self.rho * self.Cd * area * speed**2

        #Drag forces(opposite to velocity)
        Fx_drag = -D * vx_hat
        Fy_drag = -D * vy_hat

        #Thrust forces
        Fx_thrust = thrust * np.sin(fa)
        Fy_thrust = thrust * np.cos(fa)

        #Net forces
        Fx_net = Fx_drag + Fx_thrust
        Fy_net = Fy_drag + Fy_thrust + (-m * g_local)

        #Accelerations
        ax = Fx_net / m
        ay = Fy_net / m

        return np.array([vx, vy, ax, ay, -mdot])

