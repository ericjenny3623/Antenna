#This file is basically a wrapper to run the firefly function.
#Use this function to call many ensembles and generate data.
#DO NOT do extensive analysis in this file.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import shutil
import os


import functions
from minspacings import Firefly
import settings


if __name__ == "__main__":
    firefly = Firefly(n=settings.N,nff=settings.Nff,alpha=settings.Alpha,gamma=settings.Gamma,t=settings.T)
    firefly.response_model(k=settings.k*np.pi, angle_increment=settings.Nphi,peakWidth=np.deg2rad(settings.Pwdes),sidelobeHeight=settings.Slhdes)

    time = datetime.__str__(datetime.today())
    # time = time[:-10]
    time = time[0:10] + "_" + time[11:13] + "." + time[14:16]
    filepath = "data/" + time + "/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    shutil.copyfile("./settings.py", filepath + "settings.txt")

    for ensemble in range(0, settings.ENSEMBLES):
        firefly.generate_initials()
        print "ensemble: %s" % ensemble

        x, R, fitness, avgGS, avgF, avgP = firefly.run()

        if ensemble == 0:
            with open(filepath + "x.csv", "w") as xfile:
                writer = csv.writer(xfile)
                writer.writerows(x)
            with open(filepath + "R.csv", "w") as Rfile:
                writer = csv.writer(Rfile)
                writer.writerows(R)
            with open(filepath + "fitness.csv", "w") as fitnessfile:
                writer = csv.writer(fitnessfile)
                writer.writerow(fitness)
            with open(filepath + "avgFitness.csv", "w") as avgFitnessFile:
                writer = csv.writer(avgFitnessFile)
                writer.writerow(avgF)
            with open(filepath + "avgGoodSpacings.csv", "w") as avgSpacingsFile:
                writer = csv.writer(avgSpacingsFile)
                writer.writerow(avgGS)
            with open(filepath + "positions.csv", "w") as positionFile:
                writer = csv.writer(positionFile)
                writer.writerows(avgP)
            with open(filepath + "settings.txt", "a") as settingsFile:
                settingsFile.write("Min Spacing = %.3f" % (firefly.spacing_model.MIN_DISTANCE))

        else:
            with open(filepath + "x.csv", "a") as xfile:
                writer = csv.writer(xfile)
                writer.writerows(x)
            with open(filepath + "R.csv", "a") as Rfile:
                writer = csv.writer(Rfile)
                writer.writerows(R)
            with open(filepath + "fitness.csv", "a") as fitnessfile:
                writer = csv.writer(fitnessfile)
                writer.writerow(fitness)
            with open(filepath + "avgFitness.csv", "a") as avgFitnessFile:
                writer = csv.writer(avgFitnessFile)
                writer.writerow(avgF)
            with open(filepath + "avgGoodSpacings.csv", "a") as avgSpacingsFile:
                writer = csv.writer(avgSpacingsFile)
                writer.writerow(avgGS)
            with open(filepath + "positions.csv", "a") as positionFile:
                writer = csv.writer(positionFile)
                writer.writerows(avgP)

    constraint = firefly.response_constraint
    plt.figure()
    plt.plot(20 * np.log10(np.absolute(np.mean(R, axis=0))))
    plt.plot(constraint)
    plt.show()

    count, bins = np.histogram(x, bins=128, density=True)
    plt.figure()
    plt.plot(bins[1:], count)
    plt.show()
