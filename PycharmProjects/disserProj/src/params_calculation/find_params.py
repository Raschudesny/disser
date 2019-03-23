from core.find_rings import *

import multiprocessing


def runnableCalculate(input):
    JAC, RWM, PRED = calculate(input[0], input[1], input[2], input[3], input[4], input[5], input[6])
    PRED = None
    RWM = None
    thresh = input[3]
    height = input[4]
    center_height = input[5]
    return (thresh, height, center_height, JAC)

#imagePath, truthPath, info, thresh, height, center_height, OnlyJac
def create_params(imgPath, truthPath, info, onlyJac):

    results = []

    for i in range(2500, 6600, 100):
        for j in range(40, 170, 10):
            temp_res = [imgPath, truthPath, info]
            #thresh
            temp_res.append(i)
            #height
            temp_res.append(j)
            #center_height
            temp_res.append(40)
            #only jac?
            temp_res.append(True)
            results.append(temp_res)
    return results


# MAIN FUNCTION
if __name__ == "__main__":

    global_start_time = time.time()

    imagePath = "../papka/AllRings/rings8.png"
    truthPath = "../papka/AllRings/marked8.png"

    p = multiprocessing.Pool(processes = 8)
    
    inputs = create_params(imagePath, truthPath, 0, True)
    print(inputs)
    print(len(inputs))



    res = p.map(runnableCalculate, inputs)
    res = np.asarray(res)

    with open('../results/params_results/jac_2500_6600_and_40_170/find_params_res_8_100_10.txt', 'w') as outfile:
        for i in res:
            outfile.write(np.array2string(i, formatter={'float': lambda x: '%0.8f' % x}) + '\n')

    #print(calculate(imagePath, truthPath, 0, 5500, 100, 40, True))
    print("--- %s seconds ---" % (time.time() - global_start_time))