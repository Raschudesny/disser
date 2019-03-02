from find_rings import *
from metrics.test import calculate

import multiprocessing
from multiprocessing import Pool
import os


def runnableCalculate(input):
    return calculate(input[0], input[1], input[2], input[3], input[4])


# MAIN FUNCTION
if __name__ == "__main__":

    start_time = time.time()


    imagePath = "../papka/AllRings/rings6.png"
    truthPath = "../papka/AllRings/marked6.png"
    p = multiprocessing.Pool(processes = 2)

    inputs = [[imagePath, truthPath, 0, 5000, 120],[imagePath, truthPath, 0, 5500, 120]]
    res = p.map(runnableCalculate, inputs)
    print(res)

    #print(caluclate(imagePath, truthPath, 0, 5000, 120))

    print("--- %s seconds ---" % (time.time(p) - start_time))