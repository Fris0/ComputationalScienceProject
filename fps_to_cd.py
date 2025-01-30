# Only last three functions are relevant,
# the rest of the variables just calucalte them

# Data comes from pages 65 and 66 of the nasa document

import numpy as np
import scipy


Machs1, Cds_nike = [], []
with open('ValidationSets/ftps_to_cd_data/Nike.txt') as f:
    for line in f:
        line = line.split(' ')
        Machs1.append(float(line[0].strip()))
        Cds_nike.append(float(line[1].strip()))


Machs2, Cds_apache_coasting, Cds_apache_thrusting = [], [], []
with open('ValidationSets/ftps_to_cd_data/Apache.txt') as f:
    for line in f:
        line = line.split(' ')
        Machs2.append(float(line[0].strip()))
        Cds_apache_coasting.append(float(line[1].strip()))
        Cds_apache_thrusting.append(float(line[2].strip()))

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
