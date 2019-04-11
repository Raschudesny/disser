import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from os import walk
from core.find_rings import *




def visualize_results(results_directory, reverse=True):
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
        thresh_set = set()
        height_set = set()
        for str in x:
            floatArr = str.split()
            currentThresh = float(floatArr[0])
            if currentThresh != prevThresh and prevThresh != -1:
                if reverse == True:
                    temp = list(reversed(temp))
                values.append(temp)
                temp = []

            currentHeight = float(floatArr[1])
            currentCenter = float(floatArr[2])
            currentValue = float(floatArr[3])
            thresh_set.add(currentThresh)
            height_set.add(currentHeight)
            #print(currentThresh, currentHeight, currentCenter, currentValue)
            temp.append(currentValue)
            prevThresh = currentThresh

        if reverse == True:
            temp = list(reversed(temp))
        values.append(temp)
        temp = []

        if reverse == True:
            values = list(reversed(values))
        values = np.asarray(values)
        print(values)
        print(values.shape)

        min_thresh = min(thresh_set)
        max_thresh = max(thresh_set)
        thresh_set.remove(min_thresh)
        thresh_step = min(thresh_set) - min_thresh

        min_height = min(height_set)
        max_height = max(height_set)
        height_set.remove(min_height)
        height_step = min(height_set) - min_height

        ox = np.arange(min_height, max_height + height_step, height_step)
        oy = np.arange(min_thresh, max_thresh + thresh_step, thresh_step)

        # cmap="YlGnBu_r"
        ax = sns.heatmap(values, xticklabels=ox, yticklabels=oy, linewidth=0.3)
        data.append(values)
        plt.title('Значение суммы метрик для интервалов параметров')
        plt.xlabel('Высота линии')
        plt.ylabel('Значение порога бинаризации')
        plt.show()

    #grid = sns.FacetGrid(data)
    #grid.map(sns.heatmap, data)
    #plt.show()
    return

if __name__ == "__main__":
    #visualize_results("../../results/params_results/jac_2500_6600_and_40_170", reverse = False)
    visualize_results("../../results/params_results/all_rings", reverse=True)
