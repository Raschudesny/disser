from core.find_rings import *

import multiprocessing


def runnableCalculate(input):
    return calculate(input[0], input[1], input[2], input[3], input[4])




def calc_params1(imgPath, truthPath):
    results = []
    max = max_i = max_j = 0
    for i in range(2000, 3500, 100):
        temp_res = []
        for j in range(50, 150, 30):
            new_res = calculate(imgPath, truthPath, 0, i, j)
            print("new result =  " + str(new_res) + " i =  " + str(i) + " j = " + str(j))
            temp_res.append(new_res)
            if (new_res > max):
                max = new_res
                max_i = i
                max_j = j
        results.append(temp_res)
        print("################")

    print("max = " + str(max) + " i = " + str(max_i) + " j = " + str(max_j))
    return results



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

    print("--- %s seconds ---" % (time.time() - start_time))