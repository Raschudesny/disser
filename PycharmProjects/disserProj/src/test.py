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
    old_x = 100
    old_y = 0

    step = 10
    new_x = old_x + step
    new_y = caluclate(imgPath, truthPath, 0, 2000, new_x)
    print("old_x = " + str(old_x) + " old_y = " + str(old_y) + " new_x = " + str(new_x) + " new_y = " + str(new_y))
    while new_y >= old_y :
        old_x = new_x
        old_y = new_y
        new_x = old_x + step
        new_y = caluclate(imgPath, truthPath, 0, 2000, new_x)
        print("old_x = " + str(old_x) + " old_y = " + str(old_y) + " new_x = " + str(new_x) + " new_y = " + str(new_y))

    print("result: ")
    print("new_x = " + str(new_x) + " new_y = " + str(new_y))







#******* FUNCTION MAIN =) *************





imagePath = "papka/AllRings/rings11.png"
truthPath = "papka/AllRings/marked11.png"


start_time = time.time()

jaccard = caluclate(imagePath, truthPath,  1, 2000, 110)
print(jaccard)

#calc_params1(imagePath, truthPath)

print("--- %s seconds ---" % (time.time() - start_time))



#start_time = time.time()

#print("--- %s seconds ---" % (time.time() - start_time))

