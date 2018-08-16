import numpy as np
import matplotlib.pyplot as plt
import functions
from functions import ResponseModel
import csv
from spacings import SpacingModel

class Firefly:

    def __init__(self, n=10, nff=100, alpha=1.0/80.0, gamma=10.0, t=100):
        self.N = n
        self.Nff = nff
        self.Alpha = alpha
        self.Gamma = gamma
        self.T = t
        np.random.seed()

    def response_model(self, k=5*np.pi, n=10, angle_increment=128, peakWidth=28.0/180.0*np.pi, sidelobeHeight=-24):
        self.response_model = ResponseModel(k, n, angle_increment, peakWidth, sidelobeHeight)
        self.spacing_model = SpacingModel((1.0 / k * np.pi / 2.0), self.N)


    def generate_initials(self):
        # self.fireflies = [np.sort(np.random.rand(self.N)) for i in range (0, self.Nff)]
        self.generateFireflies()
        self.responses = [self.response_model.angleSpectrum(self.fireflies[j]) for j in range (0, self.Nff)]
        self.response_constraint = self.response_model.getRt()
        self.fitnesses = [self.response_model.calculateFitness(functions.convertDb(self.responses[k])) for k in range (0, self.Nff)]
        self.averageFitnessOverTime = [0 for l in range (0, self.T)]
        self.averageFitnessOverTime[0] = np.mean(self.fitnesses)
        self.separations = [self.spacing_model.calculateSpacings(self.fireflies[i]) for i in range (0, self.Nff)]
        self.goodSeparationCount = [self.spacing_model.checkSpacings(self.separations[i]) for i in range (0, self.Nff)]
        self.averageGoodSeparationOverTime = np.empty(self.T)
        self.averageGoodSeparationOverTime[0] = np.mean(self.goodSeparationCount)
        self.averagePositionsOverTime = np.array([np.empty(self.N) for i in range (0, self.T)])
        self.averagePositionsOverTime[0] = np.mean(self.fireflies, axis=0)

    def generateFireflies(self):
        top = self.spacing_model.MAX_CONSTRAINT
        bottom = self.spacing_model.MIN_CONSTRAINT
        scale = top-bottom
        # print top, bottom, scale
        self.fireflies = [np.fromfunction(lambda i: bottom+top*i, (self.N,)) + (scale*np.random.rand(self.N))for j in range (0, self.Nff)]
        # print self.fireflies

    def calculate_convergence(self, xi=[], xt=[]):
        distance = functions.distance(self.N,xi,xt)
        xn = (np.exp(-self.Gamma*(distance**2)))* (xt-xi)
        return xn

    def total_convergence(self, index, fireflies, fitnesses):
        fitness = fitnesses[index]
        xi = fireflies[index]
        goodSeparationCount = self.goodSeparationCount[index]
        for i in range (0, self.Nff):
            if (fitnesses[i] < fitness and i != index and goodSeparationCount < self.goodSeparationCount[i]):
                xi += self.calculate_convergence(xi, fireflies[i])
        return xi

    def randomMovementMagnitude(self, t, T):
        max_movement = self.Alpha*(1-(t/float(T)))*0.5
        return max_movement


    def random_movement(self, maxDown, maxUp, magnitude):
        if (np.random.rand(1) > 0.5):
            if magnitude > maxUp:
                movement = maxUp * np.random.rand(1)
            else:
                movement = magnitude * np.random.rand(1)
        else:
            if magnitude > maxDown:
                movement = maxDown * np.random.rand(1) * -1.0
            else:
                movement = magnitude * np.random.rand(1) * -1.0
        return movement

    def random_movements(self, x, spacings, t, T):
        magnitude = self.randomMovementMagnitude(t, T)
        # print x
        # print spacings
        for i in range (0, self.N):
            maxDown = max(spacings[i] - self.spacing_model.MIN_DISTANCE, 0.0)
            maxUp = max(spacings[i+1] - self.spacing_model.MIN_DISTANCE, 0.0)
            if i == self.N-1:
                maxUp = spacings[i+1]
            # print i, maxDown, maxUp
            x[i] = x[i] + self.random_movement(maxDown, maxUp, magnitude)
        return x


    def run(self):
        self.totalList = [[] for i in range(0, self.T)]

        for t in range (1, self.T):

            # for i in self.fireflies:
            #     for j in i:
            #         if j < 0 or j > 1:
            #             print "Bounds error"

            for i in range(0, self.Nff):
                self.fireflies[i] = self.total_convergence(i, self.fireflies, self.fitnesses)

            for i in range(0, self.Nff):
                self.separations[i] = self.spacing_model.calculateSpacings(self.fireflies[i])

                self.fireflies[i] = self.random_movements(self.fireflies[i], self.separations[i], t, self.T)

                self.fireflies[i] = np.sort(self.fireflies[i])
                self.responses[i] = self.response_model.angleSpectrum(self.fireflies[i])
                self.fitnesses[i] = self.response_model.calculateFitness(functions.convertDb(self.responses[i]))
                self.separations[i] = self.spacing_model.calculateSpacings(self.fireflies[i])
                self.goodSeparationCount[i] = self.spacing_model.checkSpacings(self.separations[i])

            self.averageGoodSeparationOverTime[t] = np.mean(self.goodSeparationCount)
            self.averageFitnessOverTime[t] = np.mean(self.fitnesses)
            self.averagePositionsOverTime[t] = np.mean(self.fireflies, axis=0)


        return self.fireflies, self.responses, self.fitnesses, self.goodSeparationCount, self.averageGoodSeparationOverTime, self.averageFitnessOverTime, self.averagePositionsOverTime

    # time = datetime.__str__(datetime.today())
    # self.filename = "debug/debug_" + time + ".csv"
    # self.imagename = time + ".png"

    # for sublist in self.fireflies:
    #     for item in sublist:
    #         self.totalList[t].append(item)
    # self.totalList[t].extend(self.fitnesses)
    # self.totalList[t].append(self.averageFitnessOverTime[t])
    # self.totalList[t] = [round(elem, 4) for elem in self.totalList[t]]
    #
    # with open(self.filename, "wb") as file:
    #     self.writer = csv.writer(file, delimiter=',')
    #     self.writer.writerows(self.totalList)


if __name__ == '__main__':
    firefly = Firefly(n=6, t=300)
    firefly.response_model(n=6)
    firefly.generate_initials()

    firefly.run()
    fig = plt.figure()

    plt.plot(firefly.averageGoodSeparationOverTime, label='Good Spacings')
    plt.plot(firefly.averageFitnessOverTime, label='Fitness')
    plt.legend()
    # plt.ylim(0,15)
    plt.show()

    indexsY = [np.full(firefly.N, i) for i in range (0, firefly.Nff)]
    plt.scatter(firefly.fireflies, indexsY, c='mediumspringgreen', edgecolors='mediumaquamarine')
    plt.show()

    indexsY = [np.full(firefly.N, i) for i in range (0, firefly.T)]
    plt.scatter(firefly.averagePositionsOverTime, indexsY, c='goldenrod', edgecolors='darkgoldenrod')
    plt.show()
