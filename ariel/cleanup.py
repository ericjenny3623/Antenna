import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats
from functions import ResponseModel
import functions


if __name__ == "__main__":
	model = ResponseModel()

	filename = "data/2018-07-19_09.42/"
	# filename = "data/2018-07-27_16.40/"
	filename = "data/2018-08-02_16.32/"

	dfx = pd.read_csv(filename + "x.csv", header=None)
	dff = pd.read_csv(filename+ "fitness.csv", header=None)

	dfff = dff.values.flatten()
	dropIndexs = []
	fig = plt.figure()
	count = 0

	for i, val in enumerate(dfff):
		antenna = dfx.iloc[i]
		antenna = antenna.as_matrix()
		response = model.angleSpectrum(antenna)
		fitness = model.calculateFitness(functions.convertDb(response))
		if fitness > 0.01:
			dropIndexs.append(i)
			count += 1
			# plt.plot(functions.convertDb(response))
			# plt.show()
			# print antenna

	print count

	# print dropIndexs, len(dropIndexs)
	dfx = dfx.drop(dropIndexs)
	dfx.to_csv(filename + "xNew.csv", index=False, header=False)

	dffNew = np.delete(dfff, dropIndexs)
	np.savetxt(filename + "fNew.csv", dffNew)
