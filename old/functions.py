import numpy as np
import matplotlib.pyplot as plt
from math import cos
import random as rand
#import functions2


class ResponseModel:

    def __init__(self, k=5*np.pi, angle_increment=128, peakWidth=28.0/180.0*np.pi, sidelobeHeight=-24):
        self.K = k
        self.N_phi = angle_increment
        self.Pwdes = peakWidth
        self.Shdes = sidelobeHeight
        self.rt = self.desiredResponse(self.Pwdes, self.Shdes)

    def calculateIndividual(self, x_n, phi):
        r = cos(self.K*x_n*cos(phi))
        return r

    def calculateSum(self, phi, x=[]):
        sum = 0
        for x_n in x:
            sum += self.calculateIndividual(x_n, phi)
        return sum

    def angleSpectrum(self, x=[]):
        angle = 0
        index = 0
        r = [0 for i in range (0, self.N_phi)]
        while index < self.N_phi:
            r[index] = self.calculateSum(angle,x)/len(x)
            angle += np.pi/self.N_phi
            index += 1

        return r

    def desiredResponse(self, peakWidth, sidelobeHeight):
        angle = 0
        rt = [0 for i in range (0, self.N_phi)]
        for i in range (0, self.N_phi):
            if np.abs(angle-(np.pi/2)) < peakWidth*1.0/2.0:
                rt[i] = 1
            else:
                rt[i] = sidelobeHeight
            angle += np.pi/self.N_phi
        return rt

    def calculateFitness(self, r=[]):
        sum = 0
        for i in range (0, self.N_phi):
            dif = r[i] - self.rt[i]
            if dif > 0:
                sum += dif ** 2

        return sum/self.N_phi

    def getRt(self):
        return self.rt



def distance(dimensions = 0, array1 = [], array2 = []):
    sum = 0
    for i in range (0, dimensions):
        dif = array1[i] - array2[i]
        square = dif ** 2
        sum += square

    distance = np.sqrt(sum)
    return distance

def convertDb(input):
    return 20*np.log(np.absolute(input))


if __name__ == '__main__':
    # antenna = ResponseModel()
    # antenna.sample()
    print distance(10, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.23, 0.54, 0.54, 2, 45.5, 0, 2, 0.0, 3, 4])
