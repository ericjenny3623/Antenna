import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats
from functions import ResponseModel
import functions

if __name__ == "__main__":
	filename = "data/2018-07-19_09.42/"
	# filename = "data/2018-07-27_16.40/"
	dfx = pd.read_csv(filename + "xNew.csv", header=None)

	model = ResponseModel()
	count = 0
	totalCount = 0
	for index, row in dfx.iterrows():
		if row[0] > 0.2:
			totalCount += 1
			row = row.as_matrix()
			response = model.angleSpectrum(row)
			fitness = model.calculateFitness(functions.convertDb(response))
			print fitness
			if fitness > 0.005:
				count += 1
				print row
	print count, totalCount