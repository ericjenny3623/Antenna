#This file contains generalized functions.
#Don't depend on global variables for these functions!


import numpy as np
from math import cos
pi = np.pi

#Function to generate random positions of antennas for full array
def compute_x(N):
    x = [0 for i in range(0, N)]
    for i in range(0, N):
        x[i] = np.random.uniform()
    return x

#Function to compute the response with array of elements as input
def compute_R(KHAT, N_phi, N, x):
    R = [0 for i in range(0, N_phi)]
    for i in range(0, N_phi):
        for j in range(0, N):
            R[i] += cos(KHAT * x[j] * cos(i * pi / 128))
        R[i] /= N
    return R

#Compute the error by comparing the response with an input function
#This error function wants obj < constraint
#Error = (obj - constraint)**2 only if obj > constraint
def compute_error(N, obj = [], constraint = []):
    error = 0.0
    for i in range(0, N):
        if(obj[i] > constraint[i]):
            error = error + (obj[i] - constraint[i])**2
    return error / N

#Compute the pedestal function
def compute_constraint(N_PHI, BW_DES, SLL_DES):
    constraint = [0 for i in range(0, N_PHI)]
    for i in range(0, N_PHI):
        if ((i * pi / N_PHI) < ((90 + BW_DES) * pi / 180.0) and (i * pi / N_PHI) > ((90 - BW_DES) * pi / 180.0)):
            constraint[i] = 0
        else:
            constraint[i] = SLL_DES
    return constraint

#Computes the Cartesian distance between two points
def compute_cartesian_distance(N, a = [], b = []):
    distance = 0
    for i in range(0, N):
        distance = distance + (a[i] - b[i])**2
    return np.sqrt(distance)

if __name__ == '__main__':
    print compute_cartesian_distance(10, [0,0,0,0,0,0,0,0,0,0], [0.23,0.54,0.54,2,45.5,0,2,0.0,3,4])
