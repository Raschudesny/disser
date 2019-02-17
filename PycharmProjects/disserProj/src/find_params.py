from find_rings import *

import multiprocessing
from multiprocessing import Pool
import os

MARKING_COLOR = np.array([0, 0, 255])

top = np.array(MARKING_COLOR)
bot = np.array(MARKING_COLOR)




def caluclate(imagePath, truthPath, info, thresh = 2000, height = 2000):
    img = cv.imread(imagePath, 0)
    truth = cv.imread(truthPath, 1)

    truth = cv.inRange(truth, bot, top)

    res = findRingsConnected(img, showInfo = info, thresh_bound = thresh, height_bound = height)
    pred = cv.inRange(res, bot, top)
    if info == 1:
        cv.imwrite("results/RAW.png", img)
        cv.imwrite("results/temp.png", truth)
        cv.imwrite("results/temp1.png", pred)


    pred = pred.ravel()
    truth = truth.ravel()

    start_time = time.time()

    pred1 = pred[(pred == 255) | (truth == 255)]
    truth1 = truth[(pred == 255) | (truth == 255)]

    print("--- %s seconds 255 | 255 ---" % (time.time() - start_time))


    res = jaccard_similarity_score(truth1, pred1)
    return res




def calc_params1(imgPath, truthPath):

    results = []
    for i in range(2000 , 3000, 100):
        old_x = 500
        old_y = 0

        step = -10
        new_x = old_x + step
        new_y = caluclate(imgPath, truthPath, 0, i, new_x)
        print("old_x = " + str(old_x) + " old_y = " + str(old_y) + " new_x = " + str(new_x) + " new_y = " + str(new_y))
        while (new_y >= old_y) :
            old_x = new_x
            old_y = new_y
            new_x = old_x + step
            new_y = caluclate(imgPath, truthPath, 0, i, new_x)
            print("old_x = " + str(old_x) + " old_y = " + str(old_y) + " new_x = " + str(new_x) + " new_y = " + str(
                new_y))
        print("result: ")
        print("old_x = " + str(old_x) + " old_y = " + str(old_y))
        results.append(old_x)
    return results









def doubler(data):
    print(data[0])
    print(data[1])
    print(data[2])
    print(data[3])
    print(data[4])
    return caluclate(data[0], data[1], data[2], data[3], data[4])



if __name__ == '__main__':
    a = np.array([[0,0,0], [0, 1, 1], [1 , 1 , 1]])
    b = np.array([[0,0,0], [0,0,0],[0,0,1]])
    c= cv.bitwise_and(a,b);
    area = cv.countNonZero(c);

    print(c)
    print(area)


# do some other stuff in the main process
#print(async_result)




#jaccard = caluclate(imagePath, truthPath,  0, 2800, 110)
#print(jaccard)

#res = calc_params1(imagePath, truthPath)
#count = 2000
#for r in res:
#    print('j = ' + str(r) + 'i = ' + str(count))
#    count += 100

