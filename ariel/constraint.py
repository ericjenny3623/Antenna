import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats
from functions import ResponseModel
import functions


class ConstraintModel:

	def __init__(self, min, n):
		# self.MIN_DISTANCE = wavelength/4
		self.MIN_DISTANCE = min
		self.N = n
		self.MIN_DISTANCE_ARRAY = self.MIN_DISTANCE*(np.ones(self.N))


	def checkPositions(self, x):
		xNew = np.concatenate(([-x[0]], x))
		dif = np.diff(xNew)
		greater = np.greater(dif, self.MIN_DISTANCE_ARRAY)
		count = np.sum(greater)
		weight = np.sum(dif*greater)
		print dif, self.MIN_DISTANCE_ARRAY
		return count, weight

if __name__ == "__main__":
	model = ConstraintModel(4.0/5.0, 4)
	print model.checkPositions(np.array([0.11,0.25,0.4,0.61]))


