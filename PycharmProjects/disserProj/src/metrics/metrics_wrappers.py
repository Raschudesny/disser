from core.find_rings import *
from metrics.region_wise_metric import region_metrics
from metrics.region_wise_metric import jaccard_index

MARKING_COLOR = np.array([0, 0, 255])

top = np.array(MARKING_COLOR)
bot = np.array(MARKING_COLOR)


def jaccard_metric_my(pred, truth):
    pred = pred.ravel()
    truth = truth.ravel()

    pred1 = pred[(pred == 255) | (truth == 255)]
    truth1 = truth[(pred == 255) | (truth == 255)]

    res = jaccard_similarity_score(truth1, pred1)
    return res


def more_cool_metric_my(pred, truth):
    THRESHOLD_COEFFICIENT = 0.4
    pred = np.uint8(pred)
    truth = np.uint8(truth)

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

    OO = 0
    OM = 0
    MO = 0
    w1 = 1.0
    w2 = 0.75
    w3 = 0.75
    w4 = 1.0
    w5 = 0.75
    w6 = 0.75

    for label in range(1, num_labels1):
        temp = cv.bitwise_and(cv.inRange(labels1, label, label), truth)
        components_num = cv.connectedComponentsWithStats(temp, connectivity, cv.CV_32S)[0]
        components_num -= 1
        area1 = cv.countNonZero(temp)
        area2 = cv.countNonZero(cv.bitwise_or(temp, cv.inRange(labels1, label, label)))
        DIOU = (float(area1) / float(area2))
        print('label =', label, 'DIOU = ', DIOU)
        if DIOU >= THRESHOLD_COEFFICIENT:
            if components_num == 1:
                OO += 1
            elif components_num > 1:
                OM += 1

    for label in range(1, num_labels2):
        temp = cv.bitwise_and(cv.inRange(labels2, label, label), pred)
        components_num = cv.connectedComponentsWithStats(temp, connectivity, cv.CV_32S)[0]
        components_num -= 1
        area1 = cv.countNonZero(temp)
        area2 = cv.countNonZero(cv.bitwise_or(temp, cv.inRange(labels2, label, label)))
        GIOU = (float(area1) / float(area2))
        print('label =', label, 'GIOU = ', GIOU)
        if (components_num > 1) and (GIOU >= THRESHOLD_COEFFICIENT):
            MO += 1

    print('OO =', OO)
    print('OM =', OM)
    print('MO =', MO)

    num_labels1 -= 1
    num_labels2 -= 1
    print('num_labels1 =', num_labels1)
    print('num_labels2 =', num_labels2)

    num_labels1 = float(num_labels1)
    num_labels2 = float(num_labels2)
    OO = float(OO)
    OM = float(OM)
    MO = float(MO)

    DR = w1 * (OO / num_labels1) + w2 * (OM / num_labels1) + w3 * (MO / num_labels1)
    RA = w4 * (OO / num_labels2) + w5 * (OM / num_labels2) + w6 * (MO / num_labels2)

    print('DR = ', DR)
    print('RA = ', RA)
    RM = float(2.0 * float(DR) * float(RA)) / float(DR + RA)
    print("RM = ", RM)
    return RM


def jaccard_metric(pred, truth):
    return jaccard_index(pred, truth)


def more_cool_metric(pred, truth):
    return region_metrics(truth, pred)


def temp_calculate(imagePath, truthPath, info, thresh=2000, height=2000):
    img = cv.imread(imagePath, 0)
    truth = cv.imread(truthPath, 1)
    truth = cv.inRange(truth, bot, top)

    pred = findRingsConnected(img, showInfo=info, thresh_bound=thresh, height_bound=height)

    pred = np.uint8(pred)

    if info == 1:
        cv.imwrite("../results/RAW.png", img)
        cv.imwrite("../results/temp.png", truth)
        cv.imwrite("../results/temp1.png", pred)

    result1 = jaccard_index(pred, truth)
    result2 = jaccard_metric_my(pred, truth)
    result3 = more_cool_metric(pred, truth)
    result4 = more_cool_metric_my(pred, truth)

    return result1, result2, result3, result4
