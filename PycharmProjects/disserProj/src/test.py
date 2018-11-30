from find_rings import *

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


    pred1 = pred[(pred == 255) | (truth == 255)]
    truth1 = truth[(pred == 255) | (truth == 255)]


    res = jaccard_similarity_score(truth1, pred1)
    return res




def calc_params1(imgPath, truthPath):

    results = []
    max = max_i = max_j = 0
    for i in range(3500, 3700, 20):
        temp_res = []
        for j in range(90, 121, 10):
            new_res = caluclate(imgPath, truthPath, 0, i, j)
            print("new result =  " + str(new_res) + " i =  " + str(i) + " j = " + str(j))
            temp_res.append(new_res)
            if (new_res > max) :
                max = new_res
                max_i = i
                max_j = j
        results.append(temp_res)
        print("################")

    print("max = " + str(max) + " i = " + str(max_i) + " j = " + str(max_j))
    return results







#******* FUNCTION MAIN =) *************





imagePath = "papka/AllRings/rings8.png"
truthPath = "papka/AllRings/marked8.png"


start_time = time.time()


#jaccard = caluclate(imagePath, truthPath,  1, 2892, 101)
#print(jaccard)

#jaccard = caluclate(imagePath, truthPath,  1, 2870, 95)
#print(jaccard)

#jaccard = caluclate(imagePath, truthPath,  1, 2700, 93)
#print(jaccard)

jaccard = caluclate(imagePath, truthPath,  1, 3640, 90)
print(jaccard)

#res = calc_params1(imagePath, truthPath)

#count = 2000
#for r in res:
#    print('j = ' + str(r) + 'i = ' + str(count))
#    count += 100

print("--- %s seconds ---" % (time.time() - start_time))



