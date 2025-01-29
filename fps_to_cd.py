# In your code, use from fps_to_cd import fps_to_Cd,
# other variables are only used to calculate it.


import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate


Machs1 = [
    0.00,
    0.38,
    0.46,
    0.50,
    0.54,
    0.60,
    0.66,
    0.70,
    0.76,
    0.79,
    0.84,
    0.90,
    0.96,
    0.99,
    1.02,
    1.04,
    1.06,
    1.10,
    1.16,
    1.20,
    1.26,
    1.38,
    1.40,
    1.60,
    1.80,
    2.00,
    2.20,
    2.40,
    2.60,
    2.80,
    3.00,
    6.00,
    30.00
]

Cds_nike = [
    0.47,
    0.59,
    0.63,
    0.66,
    0.71,
    0.80,
    0.95,
    1.08,
    1.27,
    1.40,
    1.60,
    1.90,
    2.20,
    2.23,
    2.20,
    2.00,
    1.68,
    1.53,
    1.46,
    1.44,
    1.40,
    1.35,
    1.34,
    1.255,
    1.18,
    1.12,
    1.07,
    1.02,
    0.99,
    0.96,
    0.93,
    0.48,
    0.48,
]

Machs2 = [
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
    3.50,
    4.00,
    4.50,
    5.00,
    5.50,
    6.00,
    6.50,
    7.00,
    7.50,
    8.00,
    30.00
]

Cds_apache_coasting = [
    0.930,
    0.841,
    0.785,
    0.740,
    0.704,
    0.643,
    0.590,
    0.544,
    0.507,
    0.479,
    0.454,
    0.432,
    0.412,
    0.396,
    0.388,
    0.384,
    0.384,
    0.384
]

Cds_apache_thrusting = [
    0.780,
    0.707,
    0.665,
    0.634,
    0.607,
    0.564,
    0.527,
    0.496,
    0.467,
    0.444,
    0.423,
    0.402,
    0.384,
    0.374,
    0.368,
    0.364,
    0.364,
    0.364
]

feet_per_secs1 = [1125.33 * mach for mach in Machs1]
feet_per_secs2 = [1125.33 * mach for mach in Machs2]


def create_f(fps, Cds):
    xs = np.array(fps)
    ys = np.array(Cds)
    linear_function = scipy.interpolate.interp1d(xs, ys, kind='linear')
    return linear_function


Cd_nike_thrusting = create_f(feet_per_secs1, Cds_nike)
Cd_apache_coasting = create_f(feet_per_secs2, Cds_apache_coasting)
Cd_apache_thrusting = create_f(feet_per_secs2, Cds_apache_thrusting)

if __name__ == "__main__":
    plt.scatter(feet_per_secs1[:], Cds_nike[:], label="data")
    t = np.linspace(0, 11000, 3500)
    plt.plot(t, fps_to_Cd(t), color="red", label="interpolated")
    plt.xlabel("feet per second")
    plt.ylabel("Cd")
    plt.legend()
    plt.savefig("figure.png")
    plt.show()
