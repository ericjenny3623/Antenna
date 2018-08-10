import numpy as np
import matplotlib.pyplot as plt
import functions
from functions import ResponseModel
import csv
from constraint import ConstraintModel

class Firefly:

    def __init__(self, n=10, nff=100, alpha=1.0/80.0, gamma=10.0, t=100):
        self.N = n
        self.Nff = nff
        self.Alpha = alpha
        self.Gamma = gamma
        self.T = t
        np.random.seed()
        self.separation_model = ConstraintModel(0.1, self.N)

    def response_model(self, k=5*np.pi, n=10, angle_increment=128, peakWidth=28.0/180.0*np.pi, sidelobeHeight=-24):
        self.response_model = ResponseModel(k, n, angle_increment, peakWidth, sidelobeHeight)

    def generate_initials(self):
        self.fireflies = [np.sort(np.random.rand(self.N)) for i in range (0, self.Nff)]
        self.responses = [self.response_model.angleSpectrum(self.fireflies[j]) for j in range (0, self.Nff)]
        self.response_constraint = self.response_model.getRt()
        self.fitnesses = [self.response_model.calculateFitness(functions.convertDb(self.responses[k])) for k in range (0, self.Nff)]
        self.averageFitnessOverTime = [0 for l in range (0, self.T)]
        self.averageFitnessOverTime[0] = np.mean(self.fitnesses)
        self.separations = [self.separation_model.calculateSpacings(self.fireflies[i]) for i in range (0, self.Nff)]
        self.goodSeparationCount = [self.separation_model.checkSpacings(self.separations[i]) for i in range (0, self.Nff)]
        self.averageGoodSeparationOverTime = np.empty(self.T)
        self.averageGoodSeparationOverTime[0] = np.mean(self.goodSeparationCount)


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

    def random_movement(self, x, t, T):
        max_movement = self.Alpha*(1-(t/float(T)))*0.5
        if (np.random.rand(1) > 0.5):
            if x + max_movement > 1:
                movement = (1-x) * np.random.rand(1)
            else:
                movement = max_movement * np.random.rand(1)
        else:
            if x - max_movement < 0:
                movement = (0-x) * np.random.rand(1)
            else:
                movement = max_movement * np.random.rand(1) * -1.0
        return x + movement

    def random_movements(self, x, t, T):
        for i in range (0, self.N):
            x[i] = self.random_movement(x[i], t, T)
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
                self.fireflies[i] = self.random_movements(self.fireflies[i], t, self.T)

                self.fireflies[i] = np.sort(self.fireflies[i])
                self.responses[i] = self.response_model.angleSpectrum(self.fireflies[i])
                self.fitnesses[i] = self.response_model.calculateFitness(functions.convertDb(self.responses[i]))
                self.separations[i] = self.separation_model.calculateSpacings(self.fireflies[i])
                self.goodSeparationCount[i] = self.separation_model.checkSpacings(self.separations[i])

            self.averageGoodSeparationOverTime[t] = np.mean(self.goodSeparationCount)
            self.averageFitnessOverTime[t] = np.mean(self.fitnesses)

        return self.fireflies, self.responses, self.fitnesses

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
    firefly = Firefly()
    firefly.response_model()
    firefly.generate_initials()
    firefly.run()
    fig = plt.figure()
    print np.sum(firefly.separations[0])

    plt.plot(firefly.averageGoodSeparationOverTime)
    plt.show()

    indexsY = [np.full(firefly.N, i) for i in range (0, firefly.Nff)]
    plt.scatter(firefly.fireflies, indexsY, cmap='inferno')
    plt.show()

