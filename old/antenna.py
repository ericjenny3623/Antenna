import numpy as np
import matplotlib.pyplot as plt
from math import cos
import random as rand
import functions2

class ResponseModel:

    def __init__(self, k=5*np.pi, n=10, angle_increment=128, sample_size=100):
        self.K = k
        self.N = n
        self.N_phi = angle_increment
        self.SAMPLES = sample_size
        rand.seed()

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
            r[index] = self.calculateSum(angle,x)/self.N
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

    def calculateFitness(self, r=[], rt=[]):
        sum = 0
        for i in range (0, self.N_phi):
            dif = r[i] - rt[i]
            if dif > 0:
                sum += dif ** 2

        return sum/self.N_phi

    def runSample(self):
        x = [rand.random() for i in range (0, self.N)]
        r = self.angleSpectrum(x)
        rt = self.desiredResponse(28.0/180.0*np.pi, -24)
        fit = self.calculateFitness(20*np.log(np.absolute(r)), rt)
        fit2 = functions2.compute_error(self.N_phi, 20 * np.log(np.absolute(r)), rt)
        # print fit, fit2
        return r, fit

    def sample(self):
        samples = [[0 for i in range (0, self.N_phi)] for j in range (0, self.SAMPLES)]
        fit = [0 for l in range (0, self.N_phi)]

        for k in range (0, self.SAMPLES):
            samples[k], fit[k] = self.runSample()

        averages = 20 * np.log10(np.absolute(np.mean(samples, axis=0)))
        variances = 20 * np.log10(np.absolute(np.var(samples, axis=0)))
        average_fit = np.mean(fit)
        print average_fit

        fig = plt.figure()
        plt.plot(averages)
        plt.plot(self.desiredResponse(28.0/180.0*np.pi, -24))
        plt.show()
        plt.savefig('averages.png')
        plt.plot(variances)
        plt.show()
        plt.savefig('variances.png')

if __name__ == '__main__':
    antenna_model = ResponseModel()
    antenna_model.sample()


