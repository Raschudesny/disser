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


def get_label(img, label):
    result = np.array(img)
    result[result != label] = 0
    return result


def get_all_labels_which_have_not_null_intersect(labeled_img, labeled_region):
    res = np.array(labeled_img)
    result_labeled_areas = np.array(labeled_img)
    res[labeled_region == 0] = 0
    intersected_labels_nums = np.unique(res)
    intersected_labels_nums = intersected_labels_nums > 0
    result_labeled_areas[np.logical_not(np.isin(result_labeled_areas, intersected_labels_nums))] = 0
    res = None
    return result_labeled_areas


def DIOU_t(truth, dt, Gdt):
    up = np.logical_and(truth.astype(bool), dt.astype(bool)).astype(int)
    down = np.logical_or(Gdt.astype(bool), dt.astype(bool)).astype(int)
    S_up = cv.countNonZero(up)
    S_down = cv.countNonZero(down)

    up = None
    down = None
    return S_up / S_down

def GIOU_k(pred, gk , Dgk):
    up = np.logical_and(pred.astype(bool), gk.astype(bool)).astype(int)
    down = np.logical_or(Dgk.astype(bool), gk.astype(bool)).astype(int)
    S_up = cv.countNonZero(up)
    S_down = cv.countNonZero(down)

    up = None
    down = None
    return S_up / S_down


def more_cool_metric_my(pred, truth):
    THRESHOLD_COEFFICIENT = 0.1
    OO = 0
    OM = 0
    MO = 0
    w1 = 1.0
    w2 = 0.75
    w3 = 0.75
    w4 = 1.0
    w5 = 0.75
    w6 = 0.75

    pred = np.uint8(pred)
    truth = np.uint8(truth)

    # CONNECTED COMPONENTS DETECTION
    connectivity = 8
    output1 = cv.connectedComponentsWithStats(truth, connectivity, cv.CV_32S)
    N = output1[0]
    G = output1[1]
    stats1 = output1[2]
    areas1 = stats1[:, cv.CC_STAT_AREA]


    output2 = cv.connectedComponentsWithStats(pred, connectivity, cv.CV_32S)
    M = output2[0]
    D = output2[1]
    stats2 = output2[2]
    areas2 = stats2[:, cv.CC_STAT_AREA]


    G = np.uint8(G)
    D = np.uint8(D)



    for label_dt in range(1, M):
        dt = get_label(D, label_dt)
        tempG = np.array(G)
        tempG[dt == 0] = 0
        num_of_intersections = cv.countNonZero(np.unique(tempG))

        if num_of_intersections == 1:
            Gdt = get_all_labels_which_have_not_null_intersect(G, dt)
            diou_t = DIOU_t(truth, dt, Gdt)
            print(label_dt, diou_t)
            if diou_t > THRESHOLD_COEFFICIENT :
                OO = OO + 1
        elif num_of_intersections > 1:
            Gdt = get_all_labels_which_have_not_null_intersect(G, dt)
            diou_t = DIOU_t(truth, dt, Gdt)
            print(label_dt, diou_t)
            if diou_t > THRESHOLD_COEFFICIENT:
                OM = OM + 1

        dt = None
        tempG = None
        Gdt = None

    for label_gk in range(1, N):
        gk = get_label(G, label_gk)
        tempD = np.array(D)
        tempD[gk == 0] = 0
        num_of_intersections = cv.countNonZero(np.unique(tempD))
        if num_of_intersections > 1:
            Dgk = get_all_labels_which_have_not_null_intersect(D, gk)
            giou_k = GIOU_k(pred, gk, Dgk)
            print(label_gk, giou_k)
            if giou_k > THRESHOLD_COEFFICIENT:
                MO = MO + 1

        gk = None
        tempD = None
        Dgk = None

    N = N - 1
    M = M - 1

    print('OO =', OO)
    print('OM =', OM)
    print('MO =', MO)

    weights1 = np.asarray((w1, w2, w3))
    weights2 = np.asarray((w4, w5, w6))
    values = np.asarray((OO, OM, MO))

    DR = (np.sum(weights1 * values) / N)
    RA = (np.sum(weights2 * values) / M)

    print('DR = ', DR)
    print('RA = ', RA)

    RWM = (2 * DR * RA) / (DR + RA)
    print("RWM = ", RWM)
    return RWM


def jaccard_metric(pred, truth):
    return jaccard_index(pred, truth)


def more_cool_metric(pred, truth):
    return more_cool_metric_my(pred, truth)
    #return region_metrics(truth, pred)


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
