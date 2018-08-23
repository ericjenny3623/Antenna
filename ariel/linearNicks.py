import numpy as np
import pandas as pd
#import scikit-lear

N = 10

x = pd.read_csv('../data/x.txt', sep = ",", header = None)
d = x.copy()
d[0] = x[0]
for i in range(1, N):
    d[i] = x[i] - x[i - 1]

#Fit regression models
regression = [np.zeros((N - 1, 2)), np.zeros((N - 1, 2))]
#regression[0][i] - forward. predicting i + 1 given i
#regression[1][i] - backward, predicting i given i + 1
#regression[0][0][0] - slope
#regression[0][0][1] - intercept
for i in range(0, N - 1):
    regression[0][i] = np.polyfit(d[i], d[i + 1], 1)
    regression[1][i] = np.polyfit(d[i + 1], d[i], 1)

#Let's make some predictions sampling uniformly from spacing range
ENSEMBLES = 1000
output_d = np.zeros((ENSEMBLES, N))
output_x = np.zeros((ENSEMBLES, N))
for ensemble in range(0, ENSEMBLES):
    start_index = 5
    start_spacing = np.random.uniform(min(d[start_index]), max(d[start_index]))
    output_d[ensemble][start_index] = start_spacing
    #Going backward
    for i in range(start_index - 1, -1, -1):
        output_d[ensemble][i] = output_d[ensemble][i + 1] * regression[1][i][0] + regression[1][i][1]
    #Going forward
    for i in range(start_index + 1, N):
        output_d[ensemble][i] = output_d[ensemble][i - 1] * regression[0][i - 1][0] + regression[0][i - 1][1]
    #Make x array
    for i in range(0, N):
        output_x[ensemble][i] += output_d[ensemble][i]


