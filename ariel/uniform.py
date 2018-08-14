import numpy as np
import matplotlib.pyplot as plt
import functions
from functions import ResponseModel
import pandas as pd
from scipy import stats


class RandomSamplingModel:

    def __init__(self, k=5*np.pi, n=10, angle_increment=128, sample_size=100, alpha=0.1):
        self.K = k
        self.N = n
        self.N_phi = angle_increment
        self.SAMPLES = sample_size
        self.response_model()
        self.Alpha = alpha

    def response_model(self, k=5*np.pi, n=10, angle_increment=128, peakWidth=28.0/180.0*np.pi, sidelobeHeight=-24):
        self.response_model = ResponseModel(k, n, angle_increment, peakWidth, sidelobeHeight)

    def modeledResponse(self, phi):
        u = self.phi2u(phi)
        mu = np.sin(u) / u
        return mu

    def modelededResponseVariance(self, phi, n):
        u = self.phi2u(phi)
        # sinc = np.sin(u) / u
        sigma = (0.5 + (np.sin(2*u) / (4*u)) - ((np.sin(u) / u) ** 2)) / n
        return sigma

    def phi2u(self, phi):
        u = 5 * np.pi * np.cos(phi)
        return u

    def generateModeledResponse(self):
        angle = 0
        index = 0
        mu = np.zeros(self.N_phi)
        while index < self.N_phi:
            mu[index] = self.modeledResponse(angle)
            angle += np.pi / self.N_phi
            index += 1

        return mu

    def generateModeledVariance(self):
        angle = 0
        index = 0
        sigma = [0 for i in range(0, self.N_phi)]
        while index < self.N_phi:
            sigma[index] = self.modelededResponseVariance(angle, self.N)
            angle += np.pi / float(self.N_phi)
            index += 1

        return sigma


    def get_dataset(self):
        filename = "data/2018-07-19_09.42/"
        dfx = pd.read_csv(filename + "x.csv", header=None)
        return dfx

    def generateCDF(self, dataset):
        count, bins = np.histogram(dataset, bins=128, range=[0, 1], density=True)
        cdf = np.cumsum(count) / self.N_phi
        return cdf

    def sampleCDF(self, cdf):
        rand = np.random.rand()
        index = 0
        current = cdf[index]
        while current < rand:
            index += 1
            current = cdf[index]
        return current

    def samplePercentileRange(self, min, max, length):
        dif = max - min
        rand = np.random.rand(length)
        scaled = rand * dif
        finished = scaled + min
        return finished

    def checkRandomUniformSampling(self):
        fireflies = np.array([np.random.rand(self.N) for i in range (0, self.SAMPLES)])

        count, bins = np.histogram(fireflies, bins=128, range=[0, 1], density=True)
        plt.plot(bins[:-1], count)
        plt.xlim([0,1])
        plt.show()

        responses = np.empty([self.SAMPLES,self.N_phi])
        fitnesses = np.empty(self.SAMPLES)

        for index in range (0, self.SAMPLES):
            firefly = fireflies[index]
            responses[index] = self.response_model.angleSpectrum(firefly)
            fitnesses[index] = self.response_model.calculateFitness(functions.convertDb(responses[index]))

        averageResponse = np.mean(responses, axis=0)
        variance = np.var(responses, axis=0)

        modeledAverage = self.generateModeledResponse()
        modeledVariance = self.generateModeledVariance()

        meanFitness = np.mean(fitnesses)

        mse = ((averageResponse - modeledAverage) ** 2).mean(axis=None)

        DbAverageResponse = functions.convertDb(np.mean(responses, axis=0))
        DbModeledAverage = functions.convertDb(self.generateModeledResponse())

        fig = plt.figure()
        angleSpectrum = [(i*1.0/self.N_phi*180) for i in range (0, self.N_phi)]
        plt.plot(angleSpectrum, DbAverageResponse, label="Simulation")
        plt.plot(angleSpectrum, DbModeledAverage, label="Analytical Model")
        plt.plot(angleSpectrum, self.response_model.getRt(), label="Desired")
        plt.title("Mean Response Generated from\nRandom Uniform Sampling for Antenna Element Positions")
        plt.legend()
        plt.ylabel("Response (dB)")
        plt.xlabel("Angle (degrees)")
        plt.ylim([-65,5])
        plt.show()

        plt.plot(variance, label="Simulation")
        plt.plot(modeledVariance, label="Analytical Model")
        plt.legend()

        plt.show()
        return mse, DbAverageResponse

    def sampleUsingPercentileRanges(self):
        dfx = self.get_dataset()
        mins = np.percentile(dfx, 10.0, axis=0)
        medians = np.percentile(dfx, 50.0, axis=0)
        maxs = np.percentile(dfx, 90.0, axis=0)

        fireflies = np.array([self.samplePercentileRange(mins, maxs, self.N) for i in range (0, self.SAMPLES)])

        for i in range (0, 10):
            x = fireflies[:,i]
            count, bins = np.histogram(x, bins=128, range=[0, 1], density=True)
            plt.plot(bins[:-1], count)

        plt.xlim([0,1])
        plt.show()

        responses = np.empty([self.SAMPLES,self.N_phi])
        fitnesses = np.empty(self.SAMPLES)

        for index in range (0, self.SAMPLES):
            firefly = fireflies[index]
            responses[index] = self.response_model.angleSpectrum(firefly)
            fitnesses[index] = self.response_model.calculateFitness(functions.convertDb(responses[index]))

        averageResponse = functions.convertDb(np.mean(responses, axis=0))
        variance= np.var(responses, axis=0)

        medianResponse = functions.convertDb(self.response_model.angleSpectrum(medians))

        meanFitness = np.mean(fitnesses)

        fig = plt.figure()
        plt.plot(averageResponse, label="Mean Sampling from Range 10th to 90th Percentile")
        plt.plot(medianResponse, label="Response From Median")
        plt.plot(self.response_model.getRt(), label="Desired")
        plt.legend()
        plt.ylim([-65,5])
        plt.show()

        plt.plot(variance, label="Simulation")
        plt.legend()
        plt.show()
        return meanFitness, averageResponse

    def sampleFromCDF(self):
        dfx = self.get_dataset()
        cdf = self.generateCDF(dfx)

        fireflies = np.array([np.array([self.sampleCDF(cdf) for j in range (0, self.N)]) for i in range (0, self.SAMPLES)])

        count, bins = np.histogram(fireflies, bins=128, range=[0, 1], density=True)
        plt.plot(bins[:-1], count)
        plt.xlim([0,1])
        plt.show()

        responses = np.empty([self.SAMPLES,self.N_phi])
        fitnesses = np.empty(self.SAMPLES)

        for index in range (0, self.SAMPLES):
            firefly = fireflies[index]
            responses[index] = self.response_model.angleSpectrum(firefly)
            fitnesses[index] = self.response_model.calculateFitness(functions.convertDb(responses[index]))

        averageResponse = np.mean(responses, axis=0)
        variance = np.var(responses, axis=0)

        modeledAverage = self.generateModeledResponse()
        modeledVariance = self.generateModeledVariance()

        meanFitness = np.mean(fitnesses)

        mse = ((averageResponse - modeledAverage) ** 2).mean(axis=None)

        DbAverageResponse = functions.convertDb(np.mean(responses, axis=0))
        DbModeledAverage = functions.convertDb(self.generateModeledResponse())

        fig = plt.figure()
        plt.plot(DbAverageResponse, label="Mean from CDF Sampled Antenna")
        plt.plot(DbModeledAverage, label="Random Uniform Sampling")
        plt.plot(self.response_model.getRt(), label="Desired")
        plt.legend()
        plt.ylim([-65,5])
        plt.show()

        plt.plot(variance, label="Simulation")
        plt.plot(modeledVariance, label="Random Uniform Sampling")
        plt.legend()
        plt.show()
        return mse, DbAverageResponse


if __name__ == '__main__':

    model = RandomSamplingModel(sample_size=10000, alpha=0.0)
    print model.checkRandomUniformSampling()
    # print model.sampleUsingPercentileRanges()
    # print model.sampleFromCDF()
