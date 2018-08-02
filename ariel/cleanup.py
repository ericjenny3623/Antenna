import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats

if __name__ == "__main__":
	filename = "data/2018-07-19_09.42/"
	dfx = pd.read_csv(filename + "x.csv", header=None)
	dff = pd.read_csv(filename+ "fitness.csv")
	dfff = dff.values.flatten()
	dropIndexs = []

	for index, row in dff.iterrows():
		for i, val in enumerate(row):
			if val > 0.005:
				transIndex = (index*100)+i
				dropIndexs.append(transIndex)

    			# print val, i, index
    			# print dfff[transIndex]
    			# print dfx.iloc[transIndex]

	# print dropIndexs
	dfxNew = dfx.drop(dropIndexs)
	# print dfxNew
	dfxNew.to_csv(filename + "xNew.csv", index=False)
	dffNew = np.delete(dfff, dropIndexs)
	# print dffNew
	np.savetxt(filename + "fNew.csv", dffNew)
