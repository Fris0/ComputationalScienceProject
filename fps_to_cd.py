# In your code, use from fps_to_cd import fps_to_Cd,
# other variables are only used to calculate it.


import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate


Machs = [
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

Cds = [
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

feet_per_secs = [1125.33 * mach for mach in Machs]


def create_f():
    xs = np.array(feet_per_secs)
    ys = np.array(Cds)
    linear_function = scipy.interpolate.interp1d(xs, ys, kind='linear')
    return linear_function


fps_to_Cd = create_f()

if __name__ == "__main__":
    plt.scatter(feet_per_secs[:], Cds[:], label="data")
    t = np.linspace(0, 11000, 3500)
    plt.plot(t, fps_to_Cd(t), color="red", label="interpolated")
    plt.xlabel("feet per second")
    plt.ylabel("Cd")
    plt.legend()
    plt.savefig("figure.png")
    plt.show()
