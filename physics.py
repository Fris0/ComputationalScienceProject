# Here the general physics.
# Responsible people: Hans, George
class Rocket():
    def __init__(self,
                 g=9.81,
                 rho=1.225,
                 Cd=0.5,
                 A=0.1,
                 I_sp=440.0,
                 T=66000.0,
                 Mp=13600.0,
                 t_b=100.0):
        self.g = g       #gravitational acceleration (m/s^2)
        self.rho = rho     #air density (kg/m^3)
        self.Cd = Cd      #drag coefficient
        self.A = A       #cross-sectional area (m^2)
        self.I_sp = I_sp    #specific impulse (s)
        self.T = T       #thrust (N)
        self.Mp = Mp      #propellant mass (kg)
        self.t_b = t_b     #burn time (s)

class PhysicsFormulas():
    def rocket_1d_dynamics(t, state):
        """
        Computes the time derivatives (dy/dt, dv/dt, dm/dt)
        for a 1D rocket under constant gravity and drag.
        """

        #Unpack state
        y, v, m = state

        #Decide if rocket is still burning
        if t < t_b:
            #Thrust is constant T
            thrust = T
            #Constant mass flow rate
            mdot = T / (I_sp * g)
        else:
            #No more thrust
            thrust = 0.0
            mdot = 0.0

        #Forces
        F_weight = -m * g
        F_drag = -0.5 * rho * Cd * A * v * abs(v)
        F_thrust = thrust
        F_net = F_weight + F_drag + F_thrust

        #Acceleration
        a = F_net / m

        dm_dt = -mdot #Mass rate of change
        dy_dt = v     #Altitude changes at rate = velocity
        dv_dt = a     #Velocity changes at rate = acceleration

        return np.array([dy_dt, dv_dt, dm_dt])
