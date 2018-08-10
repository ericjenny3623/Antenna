import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats
from functions import ResponseModel
import functions


class SpacingModel:

	def __init__(self, min, n):
		# self.MIN_DISTANCE = wavelength/4
		self.MIN_DISTANCE = min
		self.N = n
		self.MIN_DISTANCE_ARRAY = self.MIN_DISTANCE*(np.ones(self.N+1))

	def calculateSpacings(self, x):
		xNew = np.concatenate(([-x[0]], x, [1.0]))
		dif = np.diff(xNew)
		return dif

	def checkSpacings(self, dif):
		greater = np.greater(dif, self.MIN_DISTANCE_ARRAY)
		greater[self.N] = False
		count = np.sum(greater)
		weight = np.sum(dif*greater)
		# print dif
		# print greater
		return count#, weight



if __name__ == "__main__":
	model = ConstraintModel(0.1, 10)
	array = np.sort(np.random.rand(10))
	print array
	spacings = model.calculateSpacings(array)
	print spacings
	print spacings - model.MIN_DISTANCE
	print model.checkSpacings()

