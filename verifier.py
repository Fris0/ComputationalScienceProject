# Program: verifier.py
# Authors: Mark, George, Hans, Riemer, Rik
#
# Summary:
#    - Contains the code to verify the rocket launch.
# Functions:
#    - N/A
# Classes:
#    - Verifier, class which creates data from images, plots some data.
# Usage:
#    - Used as a standalone file to verify how good our model is compared to
#      the actual model. We didn't do a statistical analysis, because our model
#      is deterministic and it is clear from the plot that our model needs some
#      more improvements.
#    - To run the file simply input: python3 verifier.py

# Data manipulation
import cv2 as cv
import numpy as np

# IO
import pandas as pd

# Plots
import matplotlib.pyplot as plt

# self-made libraries
from solver import Solver
from physics import Rocket


class Verifier():
    """
    Class which creates data from images, plots some data.

    Parameters:
        - N/A

    Functions:
        - PlotData, a function which plots some verification data versus model
                    data.
        - DataCreater, a function that has been made to convert plots into the
                    original data as a csv file.
    """

    def PlotData(modeldata: str, verificationdata: str, xlabel: str ="x",
                 ylabel: str ="y", titel: str ="Verification data",
                 savefig: bool =False, savepos: str =""):
        """
        Function that plots the models data versus the data of the actual solve.
        Input:
            - modeldata, a 2D numpy array of shape (n,2), with n describing the
                        length and furthermore containing an x and y to plot
                        against eachother;
            - verificationdata, a 2D numpy array of shape (n,2), with n
                        describing the length and furthermore containing an x
                        and y to plot against eachother;
            - xlabel, a string describing the label for the x axis;
            - ylabel, a string describing the label for the y axis;
            - titel, a string describint the titel for the plot;
            - savefig, a bool describing whether to save the figure to disc,.
        Output:
            - A plt.figure object.
        Side effects:
            - If savefig is True writes a figure to the disc.
        """
        plt.plot(modeldata[0], modeldata[1], label="Model data")
        plt.plot(verificationdata[0], verificationdata[1], label="Verification data")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titel)
        plt.legend()
        if (savefig):
            assert savepos != "", "Please insert a valid string to save to."
            plt.savefig(savepos)
        plt.show()

    def DataCreator(sourceimg: str, outputcsv: str, xaxis_map: dict,
                    yaxis_map: dict, xlabel: str="x", ylabel: str="y"):
        """
        Function that reads the data of a plot and then turns it into a csv.
        Input:
            - sourceimg, a string describing the img to
                turn into a csv of the data;
            - outputcsv, a string describing the name and
                location to save the csv at;
            - xaxis_map, a dict containing which number
                corresponds to which datapoint on the x axis;
            - yaxis_map, a dict containing which number corresponds
                to which datapoint on the y axis;
            - xlabel, a string describing what name to use in the csv
                file for the x values;
            - ylabel, a string describing what name to use in the csv
                file for the y values.
        Output:
            - None.
        Side effects:
            - Writes the data of a plot into a csv.

        >>> Verifier.DataCreator("ValidationSets/NASA data/Flight path data.png",\
                            "ValidationSets/NASA data/Flight path data2.csv",\
                            {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},\
                            {0: 200, 1: 150, 2: 100, 3: 50, 4: 0},\
                            "Speed (thousands of feet per second)",\
                            "Altitude (thousands of feet)")\
        """
        img = cv.imread(sourceimg)
        assert img is not None, \
            "file could not be read, check with os.path.exists()."

        # Read dots positions into array
        reds_arr = []
        blues_arr = []
        greens_arr = []
        for y in range(len(img)):
            for x in range(len(img[0])):
                # cv2 has BGR notation therefore the weird order.
                if all(a == b for a, b in zip(img[y, x], (255, 0, 0))):
                    blues_arr.append(x)
                elif all(a == b for a, b in zip(img[y, x], (0, 255, 0))):
                    greens_arr.append(y)
                elif all(a == b for a, b in zip(img[y, x], (0, 0, 255))):
                    reds_arr.append((x, y))

        # Obtain the correct axis ticks.
        greens_arr.sort()
        blues_arr.sort()
        blues_arr = [(x, xaxis_map[i]) for i, x in enumerate(blues_arr)]
        greens_arr = [(y, yaxis_map[i]) for i, y in enumerate(greens_arr)]

        # Obtain the bounds of the image
        upperleftpos = (min(blues_arr, key=lambda x: x[0])[0],
                        min(greens_arr, key=lambda x: x[0])[0])
        lowerrightpos = (max(blues_arr, key=lambda x: x[0])[0],
                         max(greens_arr, key=lambda x: x[0])[0])

        # Turn position of value in the image to a value in the plot
        crudedata = []
        for point in reds_arr:

            # Check if data in bounds
            if (upperleftpos[0] <= point[0]
                    and point[0] <= lowerrightpos[0]
                    and upperleftpos[1] <= point[1]
                    and point[1] <= lowerrightpos[1]):

                xbetween = len(blues_arr)-2
                ybetween = len(greens_arr)-2
                for i in range(len(blues_arr)-1):
                    if (blues_arr[i][0] <= point[0] and
                            point[0] <= blues_arr[i+1][0]):
                        xbetween = i
                for i in range(len(greens_arr)-1):
                    if (greens_arr[i][0] <= point[1] and
                            point[1] <= greens_arr[i + 1][0]):
                        ybetween = i

                # Interpolate position, simple linear will do
                xinterval = (blues_arr[xbetween+1][0]-blues_arr[xbetween][0])
                yinterval = (greens_arr[ybetween+1][0]-greens_arr[ybetween][0])

                # percentage to take of left one
                p_leftx = 1-(point[0]-blues_arr[xbetween][0])/xinterval

                # percentage to take of lower y one
                p_lowery = 1-(point[1]-greens_arr[ybetween][0])/yinterval
                xdata = p_leftx*blues_arr[xbetween][1] + \
                    (1-p_leftx)*blues_arr[xbetween+1][1]
                ydata = p_lowery*greens_arr[ybetween][1] + \
                    (1-p_lowery)*greens_arr[ybetween+1][1]
                crudedata.append((point[0], point[1], xdata, ydata))

        # Clean up datapoints; We only want one datapoint per y value.
        data = []
        datadict = {}
        for i in range(len(crudedata)):
            if crudedata[i][1] not in datadict:
                datadict[crudedata[i][1]] = []
            datadict[crudedata[i][1]].append((crudedata[i][2],
                                              crudedata[i][3]))
        for key in datadict.keys():
            data.append(np.average(datadict[key], axis=0))

        # Save it
        data = np.array(data).transpose()
        mydataframe = pd.DataFrame()
        mydataframe[xlabel] = data[0]
        mydataframe[ylabel] = data[1]
        mydataframe.to_csv(outputcsv)


if __name__ == "__main__":
    """
    Verifies the model selected in mysolver.solve_* to the actual data.
    On sight, because so far we haven't reached the point to numerically
    validate the data.

    If wanting to recreate the current validation data from the plot of
     p. 38 of the NASA pdf.
    insert:
    - Verifier.DataCreator("ValidationSets/NASA data/Flight path data.png",
                            "ValidationSets/NASA data/Flight path data2.csv",
                            {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
                            {0: 200, 1: 150, 2: 100, 3: 50, 4: 0},
                            "Speed (thousands of feet per second)",
                            "Altitude (thousands of feet)")
    """
    # Initialize
    currocket = Rocket(la = 80, M = 1534 + 80, Mp_a=217+80)

    mysolver = Solver()
    mysolver.tbegin = 0
    mysolver.tend = 36

    # Prepare data
    sol = mysolver.solve_rocketNike(currocket)

    # Rotate the two models around their axes.
    modeldata = np.transpose(sol, [0, 2, 1])
    plotdata = (modeldata[1][2], modeldata[1][1])

    # Plot
    verificationdata = (pd.read_csv("ValidationSets/NASA data/Flight path data.csv").
                        sort_values(by=["Altitude (thousands of feet)"])).to_numpy()
    verificationdata = (verificationdata.T)[1:]
    Verifier.PlotData(np.array(plotdata)/1000, verificationdata,
                      "Speed (thousands of feet per second)",
                      "Altitude (thousands of feet)",
                      savefig=False, savepos="verifier.png")
