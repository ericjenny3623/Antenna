import numpy as np
import matplotlib.pyplot as plt
import functions
from functions import ResponseModel
import pandas as pd
from scipy import stats



class LinearRegressionModel:
    def __init__(self, k=5*np.pi, n=10, angle_increment=128, sample_size=100, stdev = 0.0):
        self.K = k
        self.N = n
        self.N_phi = angle_increment
        self.SAMPLES = sample_size
        self.response_model()
        self.Stdev = stdev
        self.filename = "data/2018-08-02_16.32/"


    def response_model(self, k=5*np.pi, n=10, angle_increment=128, peakWidth=28.0/180.0*np.pi, sidelobeHeight=-24):
        self.response_model = ResponseModel(k, n, angle_increment, peakWidth, sidelobeHeight)

    def get_fireflies_positions(self):
        dfx = pd.read_csv(self.filename + "xNew.csv", header=None)
        return dfx

    def get_fireflies_responses(self):
        dfr = pd.read_csv(self.filename + "R.csv", header=None)
        return dfr

    def samplePercentileRange(self, min, max, length):
        dif = max - min
        rand = np.random.rand(length)
        scaled = rand * dif
        finished = scaled + min
        return finished


    def calculateDifs(self, dfx):
        dif = dfx.diff(axis=1)
        dif[0] = dfx[0]
        dif[10] = 1 - dfx[9]
        return dif

    def linearRegOld(self, dfx):
        dif = dfx.diff(axis=1)
        dif[10] = 1 - dfx[9]
        num = len(dif.columns)

        slope = np.empty(num - 2)
        intercept = np.empty(num - 2)
        r_value = np.empty(num - 2)
        p_value = np.empty(num - 2)
        std_err = np.empty(num - 2)

        for i in range(1, num - 1):
            y = dif[i]
            x = dif[i + 1]
            slope[i - 1], intercept[i - 1], r_value[i - 1], p_value[i - 1], std_err[i - 1] = stats.linregress(x, y)
        return slope, intercept

    def linearReg(self, dif):
        num = len(dif.columns)

        slope = np.empty(num - 1)
        intercept = np.empty(num - 1)
        r_value = np.empty(num - 1)
        p_value = np.empty(num - 1)
        std_err = np.empty(num - 1)

        for i in range(0, num - 1):
            x = dif[i]
            y = dif[i + 1]
            slope[i], intercept[i], r_value[i], p_value[i], std_err[i] = stats.linregress(x, y)
        return slope, intercept

    def linearRegInverse(self, dif):
        num = len(dif.columns)

        slope = np.empty(num - 1)
        intercept = np.empty(num - 1)
        r_value = np.empty(num - 1)
        p_value = np.empty(num - 1)
        std_err = np.empty(num - 1)

        for i in range(0, num - 1):
            y = dif[i]
            x = dif[i + 1]
            slope[i], intercept[i], r_value[i], p_value[i], std_err[i] = stats.linregress(x, y)
        return slope, intercept

    def generateLastElement(self, last_point, slopes, intercepts):
        rslopes = np.flip(slopes,0)
        rintercepts = np.flip(intercepts,0)
        antenna = np.empty(len(rslopes)+1)
        antenna[0] = last_point
        for i in range (1, self.N):
            if (i==1):
                dif = 1- antenna[i-1]
            else:
                dif = antenna[i-2]-antenna[i-1]
            antenna[i] = antenna[i-1]-((rslopes[i-1]*dif)+rintercepts[i-1])
        return np.flip(antenna,0)

    def generateAny(self, separation, index, slopes, intercepts, slopesr, interceptsr):
        differences = np.empty(self.N)
        differences[index] = separation
        for i in range (index, (self.N-1)):
            differences[i+1] = (slopes[i] * differences[i]) + intercepts[i]
        for i in range (0, index):
            j = index-i
            differences[j-1] = (differences[j] * slopesr[j-1]) +  interceptsr[j-1]

        firefly = np.cumsum(differences)
        return firefly

    def generateAnyNew(self, separation, index, slopes, intercepts, slopesr, interceptsr):
        differences = np.empty(self.N+1)
        differences[index] = separation
        for i in range (index, (self.N)):
            differences[i+1] = (slopes[i] * differences[i]) + intercepts[i]
        for i in range (0, index-1):
            j = index-i
            differences[j-1] = (differences[j] * slopesr[j-1]) +  interceptsr[j-1]

        differencesr = np.flip(differences,0)
        positionsr = 1 - np.cumsum(differencesr)
        positions = np.flip(positionsr, 0)[1:]
        # print differences
        # print differencesr
        # print positionsr
        # print positions
        return positions

    def randomMovement(self, x, stdev):
        if (np.random.rand(1) > 0.5):
            if x + stdev > 1:
                movement = (1-x) * np.random.normal(0,stdev)
            else:
                movement =  np.random.normal(0,stdev)
        else:
            if x - stdev < 0:
                movement = (0-x) * np.random.normal(0,stdev)
            else:
                movement = stdev * np.random.normal(0,stdev) * -1.0
        return x + movement

    def randomMovements(self, x, stdev):
        for i in range (0, self.N):
            x[i] = self.randomMovement(x[i], stdev)
        return x

    def generateFromLinRegressLast(self):
        dfx = self.get_fireflies_positions()
        tenth = np.percentile(dfx, 10.0, axis=0)
        ninetieth = np.percentile(dfx, 90.0, axis=0)

        slopes, intercepts = self.linearRegOld(dfx)
        fireflies = np.array([np.empty(self.N) for i in range (0,self.SAMPLES)])

        for i in range (0, self.SAMPLES):
            index = 9
            initials = self.samplePercentileRange(tenth[index],ninetieth[index],1)
            initials = 0.95
            generated = self.generateLastElement(initials, slopes, intercepts)
            fireflies[i] = self.randomMovements(generated, self.Stdev)

        return fireflies

    def generateFromLinRegressAny(self):
        dfx = self.get_fireflies_positions()
        dif = self.calculateDifs(dfx)
        tenth = np.percentile(dif, 05.0, axis=0)
        ninetieth = np.percentile(dif, 95.0, axis=0)

        slopes, intercepts = self.linearReg(dif)
        slopesr, interceptsr = self.linearRegInverse(dif)
        fireflies = np.array([np.empty(self.N) for i in range (0,self.SAMPLES)])

        for i in range (0, self.SAMPLES):
            index = np.random.randint(0,10) + 1
            # index = 7
            initials = self.samplePercentileRange(tenth[index],ninetieth[index],1)
            # initials = 0.05
            generated = self.generateAnyNew(initials, index, slopes, intercepts, slopesr, interceptsr)
            fireflies[i] = self.randomMovements(generated, self.Stdev)

        return fireflies



    def analyze(self, fireflies, show=False):
        responses = np.empty([self.SAMPLES,self.N_phi])
        fitnesses = np.empty(self.SAMPLES)

        for index in range (0, self.SAMPLES):
            firefly = fireflies[index]
            responses[index] = self.response_model.angleSpectrum(firefly)
            fitnesses[index] = self.response_model.calculateFitness(functions.convertDb(responses[index]))

        averageResponse = functions.convertDb(np.mean(responses, axis=0))
        variance = np.var(responses, axis=0)

        meanFitness = np.mean(fitnesses)


        if show==True:
            for i in range (0, 10):
                x = fireflies[:,i]
                count, bins = np.histogram(x, bins=50, density=True)
                plt.plot(bins[:-1], count)
            plt.xlim([0,1])
            plt.title("Positions Generated from Linear Regression")
            plt.show()

            fig = plt.figure()
            plt.plot(averageResponse, label="Mean from model")
            # plt.plot(average, label="Theoretical")
            plt.plot(self.response_model.getRt(), label="Desired")
            plt.legend()
            plt.ylim([-65,5])
            plt.show()

            plt.plot(variance, label="Experimental")
            plt.legend()
            plt.show()

        print meanFitness
        return averageResponse, variance, meanFitness


    def samples(self):
        # fits = np.empty(30)
        # stdev = np.empty(30)
        # for i in range (0,30):
        #     r_, v_, fits[i] = self.analyze(self.generateFromLinRegressAny())
        #     stdev[i] = self.Stdev
        #     self.Stdev += 0.01

        # plt.plot(stdev, fits, label="From Random Separation")

        self.Stdev = 0.0
        fits = np.empty(30)
        stdev = np.empty(30)
        for i in range (0,30):
            r_, v_, fits[i] = self.analyze(self.generateFromLinRegressLast())
            stdev[i] = self.Stdev
            self.Stdev += 0.01

        plt.plot(stdev, fits, label="From Last Separation")


        # self.Stdev = 0.0
        # fits = np.empty(30)
        # stdev = np.empty(30)
        # for i in range (0,30):
        #     fireflies = self.get_fireflies_positions
        #     r_, v_, fits[i] = self.analyze(self.randomMovements(fireflies, self.tdev))
        #     stdev[i] = self.Stdev
        #     self.Stdev += 0.01

        # plt.plot(stdev, fits, label="From Firefly")


        plt.xlabel('Standard Deviation')
        plt.ylabel('Average Fitness')
        plt.title("Standard Deviation of Noise vs. Average Fitness")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    model = LinearRegressionModel(sample_size=1000, stdev=0.0)
    # model.samples()

    # rLast, vLast, fLast = model.analyze(model.generateFromLinRegressLast(), show=False)
    rAny, vAny, fAny = model.analyze(model.generateFromLinRegressAny(), show=True)
    # print model.generateFromLinRegressLast()
    # print model.generateFromLinRegressAny()

    # rFirefly = model.get_fireflies_responses()
    # vFirefly = np.var(rFirefly, axis=0)
    # rFirefly = np.mean(rFirefly, axis=0)
    # rFirefly = functions.convertDb(rFirefly)

    # fig = plt.figure()
    # angleSpectrum = [(i*1.0/model.N_phi*180) for i in range (0, model.N_phi)]
    # plt.plot(angleSpectrum, rLast, label="From Last Position")
    # # plt.plot(angleSpectrum, rAny, label="From Random Spacing")
    # plt.plot(angleSpectrum, rFirefly, label="Firefly")
    # plt.plot(angleSpectrum, model.response_model.getRt(), label="Desired")
    # plt.title("Mean Response Generated from\nLinear Regression for Antenna Element Positions")
    # plt.legend()
    # plt.ylabel("Response (dB)")
    # plt.xlabel("Angle (degrees)")
    # plt.ylim([-65,5])
    # plt.show()
    # print vFirefly
    # plt.plot(angleSpectrum, vLast, label="From Last Position")
    # # plt.plot(angleSpectrum, vAny, label="From Random Sppacing")
    # plt.plot(angleSpectrum, vFirefly, label="Firefly")
    # # plt.plot(angleSpectrum, model.response_model.getRt(), label="Desired")
    # plt.title("Variance of Response Generated from\nLinear Regression for Antenna Element Positions")
    # plt.legend()
    # plt.ylabel("Variance")
    # plt.xlabel("Angle (degrees)")
    # # plt.ylim([-65,5])
    # plt.show()