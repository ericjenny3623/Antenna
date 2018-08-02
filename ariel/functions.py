import numpy as np
import matplotlib.pyplot as plt
from math import cos


class ResponseModel:

    def __init__(self, k=5*np.pi, n=10, angle_increment=128, peakWidth=28.0/180.0*np.pi, sidelobeHeight=-24):
        self.K = k
        self.N = n
        self.N_phi = angle_increment
        self.Pwdes = peakWidth
        self.Shdes = sidelobeHeight
        self.rt = self.desiredResponse(self.Pwdes, self.Shdes)
        self.cos_spectrum = self.generate_cos_spectrum(self.N_phi)

    def calculateIndividual(self, x_n, phi):
        r = cos(self.K*x_n*cos(phi))
        return r

    def calculateNonSymmetric(self, cos_phi, x):
        r = np.exp(self.K*x*cos_phi)
        sum = np.sum(r, axis=1)
        return sum

    def calculateSum(self,cos_phi, x=[]):
        r = np.cos(self.K*x*cos_phi)
        sum = np.sum(r, axis=1)
        return sum

    def generate_cos_spectrum(self, n_phi):
        spectrum = np.cos(np.array([np.ones(self.N)*(np.pi/n_phi*i) for i in range (0, n_phi)]))
        return spectrum

    def angleSpectrum(self, x=[]):
        r = self.calculateSum(self.cos_spectrum, x)/len(x)
        return r

    def desiredResponse(self, peakWidth, sidelobeHeight):
        angle = 0
        rt = np.empty(self.N_phi)
        for i in range (0, self.N_phi):
            if np.abs(angle-(np.pi/2)) < peakWidth*1.0/2.0:
                rt[i] = 1
            else:
                rt[i] = sidelobeHeight
            angle += np.pi/self.N_phi
        return rt

    def calculateFitness(self, r=[]):
        dif = r-self.rt
        dif[dif <= 0] = 0
        sum = np.sum(dif**2)
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
    return 20*np.log10(np.absolute(input))


# def calculateSumOld(self, phi, x=[]):
#     r = np.cos(self.K*x*np.cos(phi))
#     sum = np.sum(r)
#     return sum
# def angleSpectrumOld(self, x=[]):
#     angle = 0
#     index = 0
#     r = np.empty(self.N_phi)
#     for i in range (0, self.N_phi):
#         angle = self.angle_spectrum[i]
#         r[i] = self.calculateSumOld(angle,x)/len(x)
#
#     return r

if __name__ == '__main__':
    None
