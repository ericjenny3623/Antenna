import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats

if __name__ == "__main__":
    # filename = "data/nicks/"
    # dfx = pd.read_csv(filename + "x.csv", header=None, delim_whitespace=True)

    filename = "data/2018-07-12_14.51/"
    filename = "data/2018-07-19_09.42/"
    filename = "data/2018-07-27_16.40/"
    filename = "data/2018-08-02_16.32/"
    dfx = pd.read_csv(filename + "xNew.csv", header=None)
    # print dfx

    # for i in range (0, 10):
    #     x = dfx[i]
    #     count, bins = np.histogram(x, bins=50, density=True)
    #     plt.plot(bins[:-1], count)
    # plt.xlim([0,1])
    # title = "Histogram of Firefly Algorithm Generated Antenna Arrays"
    # plt.title(title)
    # plt.savefig("graphs/" + title)
    # plt.show()

    # # plotting.scatter_matrix(dfx, alpha=0.2, figsize=(6, 6))
    # # plt.show()

    # count, bins = np.histogram(dfx, bins=128, range=[0, 1], density=True)
    # cdf = np.cumsum(count)/128
    # plt.plot(bins[:-1],cdf)
    # plt.show()

    # dfx.hist(color='aqua', alpha=0.5, bins=50)
    # plt.title("Positions Histogram")
    # plt.xlim(0.0,1.0)
    # plt.show()

    # print dfx.quantile([0.25, 0.5, 0.75])
    # dfx.boxplot()
    # plt.show()

    # cor = dfx.corr()
    # print cor
    # plt.matshow(cor)
    # plt.colorbar()
    # plt.show()

    dif = dfx.diff(axis=1)
    dif[10] = 1 - dfx[9]
    # dif.drop(dif.columns[0],axis=1,inplace=True)
    dif[0] = dfx[0]
    # print dif

    # dif.hist(color='aqua', alpha=0.5, bins=50)
    # # plt.title("Separation Histogram")
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    difcor = dif.corr()
    print difcor
    cax = ax.matshow(difcor, interpolation='nearest')
    fig.colorbar(cax, label="Correlation")
    plt.title("Correlation between Antenna Element Separations")
    alpha = ['(E1-0)', 'S1','S2','S3','S4','S5','S6','S7','S8','S9', '(1-E10)']
    ax.set_xticks(np.arange(len(alpha)))
    ax.set_yticks(np.arange(len(alpha)))
    ax.set_xticklabels(alpha)
    ax.set_yticklabels(alpha)
    ax.axis('image')
    print np.arange(len(alpha))

    plt.show()


    num = len(dif.columns)
    # # print num
    # fig, axs = plt.subplots(ncols=num, nrows=num, sharey=True, sharex=True, figsize=(6, 6))
    # # fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
    # for i in range (0, num):
    #     for j in range (0, num):
    #         ax = axs[i][j]
    #         hb = ax.hexbin(dif.iloc[:,i], dif.iloc[:,j], gridsize=20, cmap='inferno', extent=[0.0, 0.3, 0.0, 0.3])
    #         ax.axis([0.0, 0.3, 0.0, 0.3])
    #         ax.axis('off')
    #         # ax.set_title("Hexagon binning")
    #         # cb = fig.colorbar(hb, ax=ax)
    #         # cb.set_label('counts')
    # plt.show()


    # num = len(dif.columns)
    # for i in range (1, num):
    #     x = dif[i]
    #     y = dif[i+1]
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    #     hb = plt.hexbin(x, y, gridsize=25, cmap='viridis', extent=[0.0, 0.3, 0.0, 0.3])
    #     plt.plot(np.unique(x), (np.unique(x)*slope)+intercept, label="Std Error: %.3f" %(std_err), color='mediumvioletred')
    #     plt.axis([0.0, 0.3, 0.0, 0.3])
    #     title = "Separations of %d & %d and %d & %d with fit" % (i , i + 1, i + 1, i + 2)

    #     plt.title(title)
    #     plt.legend()
    #     plt.ylabel("%d & %d" % (i + 1, i + 2))
    #     plt.xlabel("%d & %d" % (i, i + 1))
    #     cb = plt.colorbar(hb)
    #     plt.savefig(filename + "graphs/" + title)
    #     # plt.show()
    #     plt.close()
    #     # cb.set_label('Counts')
    #     print i

    # num = len(dif.columns)
    # for i in range (0, num):
    #     for j in range (0, num):
    #         hb = plt.hexbin(dif.iloc[:,i], dif.iloc[:,j], gridsize=25, cmap='inferno', extent=[0.0, 0.3, 0.0, 0.3])
    #         plt.axis([0.0, 0.3, 0.0, 0.3])
    #         title = "Relationship between Separations of %d & %d and %d & %d" % (i+1, i+2, j+1, j+2)
    #         plt.title(title)
    #         plt.ylabel("%d & %d" %(j+1, j+2))
    #         plt.xlabel("%d & %d" %(i+1, i+2))
    #         cb = plt.colorbar(hb)
    #         plt.savefig(filename + "graphs/" + title)
    #         # plt.show()
    #         plt.close()
    #         # cb.set_label('Counts')

    # plt.show()

