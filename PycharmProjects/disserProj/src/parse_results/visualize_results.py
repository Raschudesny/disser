import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from os import walk
from core.find_rings import *




def visualize_results(results_directory):
    files = []
    for (dirpath, dirnames, filenames) in walk(results_directory):
        files.extend(filenames)
        break
    files.sort()
    print(files)

    data = []
    for file in files:
        full_name = os.path.join(results_directory, file)
        file_in = open(full_name, 'r')
        x = []
        for y in file_in.read().split('\n'):
            if y:
                str = y[1:-1]
                x.append(str)
        values = []
        temp = []
        prevThresh = -1
        for str in x:
            floatArr = str.split()
            currentThresh = float(floatArr[0])
            if currentThresh != prevThresh and prevThresh != -1:
                values.append(temp)
                temp = []

            currentHeight = float(floatArr[1])
            currentCenter = float(floatArr[2])
            currentValue = float(floatArr[3])
            print(currentThresh, currentHeight, currentCenter, currentValue)
            temp.append(currentValue)
            prevThresh = currentThresh

        values = np.asarray(values)
        print(values)
        print(values.shape)

        ox = np.arange(40, 170, 10)
        oy = np.arange(2500, 6600, 100)

        # cmap="YlGnBu_r"
        ax = sns.heatmap(values, xticklabels=ox, yticklabels=oy, linewidth=0.3)
        data.append(values)
        plt.title(file)
        plt.show()

    grid = sns.FacetGrid(data)
    grid.map(sns.heatmap, data)
    plt.show()
    return

if __name__ == "__main__":
    visualize_results("../../results/params_results/jac_2500_6600_and_40_170")
