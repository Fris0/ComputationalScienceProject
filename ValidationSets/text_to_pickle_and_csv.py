import fileinput
import pickle
import csv

# filenames = ["Wallops, 1, %d" % i for i in [75, 80, 85, 90]]

offsets = {
    50: 0,
    60: 1,
    70: 2,
    80: 3,
    90: 4,
    100: 5,
    'Nike Burnout': 0,
    'Apache Ignition': 6,
    'Apache Burnout': 12,
    'Apogee': 18,
    'Impact': 24
}

pickle_dict = {}
for la in [75, 80, 85, 90]:  # launch angle
    data = [[], [], [], [], []]
    n = 0
    with fileinput.input(files=('Wallops, 1, %d' % la)) as f:
        for line in f:
            line = line.strip()
            if not line:
                n += 1
            elif n < 5:
                data[n].append(float(line))

    # times, speeds, altitudes, horizontal_ranges, flight_path_angles
    t, s, a, hr, fpa = data

    pickle_dict[la] = {}
    for plw in [50, 60, 70, 80, 90, 100]:  # payload weight
        pickle_dict[la][plw] = {}
        for tsn in ['Nike Burnout',  # time step name
                    'Apache Ignition',
                    'Apache Burnout',
                    'Apogee', 'Impact']:
            pickle_dict[la][plw][tsn] = {}
            pickle_dict[la][plw][tsn]['t'] = t[offsets[plw] + offsets[tsn]]
            pickle_dict[la][plw][tsn]['s'] = s[offsets[plw] + offsets[tsn]]
            pickle_dict[la][plw][tsn]['a'] = a[offsets[plw] + offsets[tsn]]
            pickle_dict[la][plw][tsn]['hr'] = hr[offsets[plw] + offsets[tsn]]
            pickle_dict[la][plw][tsn]['fpa'] = fpa[offsets[plw] + offsets[tsn]]


with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    field = ['launch angle',
             'payload weight',
             'timestep name',
             'time',
             'speed',
             'altitude',
             'horizontal range',
             'flight path angle']

    writer.writerow(field)
    for la in [75, 80, 85, 90]:  # launch angle
        for plw in [50, 60, 70, 80, 90, 100]:  # payload weight
            for tsn in ['Nike Burnout',  # time step name
                        'Apache Ignition',
                        'Apache Burnout',
                        'Apogee', 'Impact']:
                writer.writerow([str(la),
                                 str(plw),
                                 tsn,
                                 str(pickle_dict[la][plw][tsn]['t']),
                                 str(pickle_dict[la][plw][tsn]['s']),
                                 str(pickle_dict[la][plw][tsn]['a']),
                                 str(pickle_dict[la][plw][tsn]['hr']),
                                 str(pickle_dict[la][plw][tsn]['fpa'])])

file = open('data_pickle', 'wb')
pickle.dump(pickle_dict, file)
