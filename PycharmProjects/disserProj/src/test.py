from find_rings import *

MARKING_COLOR = np.array([0, 0, 255])

top = np.array(MARKING_COLOR)
bot = np.array(MARKING_COLOR)

def jaccard_metric(pred, truth):
    pred = pred.ravel()
    truth = truth.ravel()

    pred1 = pred[(pred == 255) | (truth == 255)]
    truth1 = truth[(pred == 255) | (truth == 255)]

    res = jaccard_similarity_score(truth1, pred1)
    return res

def more_cool_metric(pred, truth):
    # CONNECTED COMPONENTS DETECTION

    connectivity = 8
    output1 = cv.connectedComponentsWithStats(pred, connectivity, cv.CV_32S)
    num_labels1 = output1[0]
    labels1 = output1[1]
    stats1 = output1[2]
    areas1 = stats1[:, cv.CC_STAT_AREA]
    print(areas1)

    output2 = cv.connectedComponentsWithStats(truth, connectivity, cv.CV_32S)
    num_labels2 = output2[0]
    labels2 = output2[1]
    stats2 = output2[2]
    areas2 = stats2[:, cv.CC_STAT_AREA]
    print(areas2)

    N = 0
    N1 = 0
    N2 = 0
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    w4 = 1.0
    w5 = 1.0
    w6 = 1.0

    for label in range(1, num_labels1):
        temp = cv.bitwise_and(cv.inRange(labels1, label, label), truth)
        areas = cv.connectedComponentsWithStats(temp)[2][:, cv.CC_STAT_AREA][1:]
        if any(areas >= int(areas1[label] * 0.8)):
            N += 1
        elif areas.shape[0] > 0:
            N1 += 1
        areas = None

    for label in range(1, num_labels2):
        temp = cv.bitwise_and(cv.inRange(labels2, label, label), pred)
        areas = cv.connectedComponentsWithStats(temp)[2][:, cv.CC_STAT_AREA][1:]
        if any(areas >= int(areas2[label] * 0.8)):
            continue
        elif areas.shape[0] > 0:
            N2 += 1
        areas = None

    output1 = None
    output2 = None
    print('N =', N)
    print('N1 =', N1)
    print('N2 =', N2)
    num_labels1 -= 1
    num_labels2 -= 1
    print('num_labels1 =', num_labels1)
    print('num_labels2 =', num_labels2)

    p1 = w1 * (N / num_labels1) + w2 * (N1 / num_labels1) + w3 * (N2 / num_labels1);
    p2 = w4 * (N / num_labels2) + w5 * (N1 / num_labels2) + w6 * (N2 / num_labels2);
    print(p1)
    print(p2)
    Q = (2.0 * p1 * p2) / (p1 + p2)
    print("Q= ", Q)
    return Q

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

    result = jaccard_metric(pred, truth)

    more_cool_metric(pred, truth)
    return result




def calc_params1(imgPath, truthPath):

    results = []
    max = max_i = max_j = 0
    for i in range(2000, 3500, 100):
        temp_res = []
        for j in range(50, 150, 30):
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





imagePath = "papka/AllRings/rings6.png"
truthPath = "papka/AllRings/marked6.png"


start_time = time.time()


#jaccard = caluclate(imagePath, truthPath,  1, 2892, 101)
#print(jaccard)

#jaccard = caluclate(imagePath, truthPath,  1, 2870, 95)
#print(jaccard)

#jaccard = caluclate(imagePath, truthPath,  1, 2700, 93)
#print(jaccard)

#jaccard = caluclate(imagePath, truthPath,  1, 6000, 120)

#jaccard = caluclate(imagePath, truthPath,  0, 5700, 95)
#print(jaccard)

img1 = cv.imread("kek.png", 0)
ret, img1 = cv.threshold(img1, thresh = 50, maxval = 255, type = cv.THRESH_BINARY)
img2 = cv.imread("lal.png", 0)
ret, img2 = cv.threshold(img2, thresh = 50, maxval = 255, type = cv.THRESH_BINARY)


img2 = cv.inRange(img2, -3, -3)
img2[180:220, 0:20] = 255
img2[0:20,180:220] = 255
cv.imwrite("lal.png", img2)



print("result = ", more_cool_metric(img2, img1))

#res = calc_params1(imagePath, truthPath)

#count = 2000
#for r in res:
#    print('j = ' + str(r) + 'i = ' + str(count))
#    count += 100

print("--- %s seconds ---" % (time.time() - start_time))



