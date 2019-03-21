import numpy as np
from scipy import ndimage
import cv2 as cv
import math
from matplotlib import pyplot as plt
from metrics.metrics_wrappers import jaccard_metric
from metrics.metrics_wrappers import more_cool_metric

from sklearn.metrics import jaccard_similarity_score
import time

MARKING_COLOR = np.array([0, 0, 255])

top = np.array(MARKING_COLOR)
bot = np.array(MARKING_COLOR)

# можно
# бол-во работ пытаются устранить в рамках процедуры восстановления из боковых проекциях
# даже после такой процедуры могло что-то остаться
# мы же пытаемся устранить с помощью постпроцессинга уже восстановлненных данных(пост процессинг реконструированного изображения)
# подобный артефакт мог возникнуть как ошибка в боковых проекциях
# возниклв как ошибки в боковых проекцич
# нал ичие артефактов приводит к искажению последующих моделируемых по ним результатов


# увеличить размер фильтра до 21
CONV_KERNEL_SMALL = np.array([
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1]])

CONV_KERNEL_BIG = np.array([
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
    [-1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1]])

MORPH_KERNEL = np.uint8(np.array([
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0]]))

MORPH_KERNEL_BIG = np.uint8(np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
]))


def equal_contrast(img):
    height = img.shape[0]
    width = img.shape[1]
    masked_img = img[int(height / 4): int(height * 3 / 4), int(width / 4): int(width * 3 / 4)]
    ma = np.percentile(masked_img, 99)
    mi = np.percentile(masked_img, 1)
    img = ((img - mi) / (ma - mi)) * 255
    img[img < 0] = 0
    img[img > 255] = 255
    img = np.uint8(np.round(img, 0))
    return img


def drawGrayHist(img, maxValue=256, title='Gray Histogram'):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.hist(img.ravel(), maxValue, [0, maxValue])
    plt.xlim([0, maxValue])
    plt.grid()
    plt.show()
    return


def drawGrayHistMasked(img, maxValue=256, title='Masked Gray Histogram'):
    # create a mask
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[int(height / 4): int(height * 3 / 4), int(width / 4): int(width * 3 / 4)] = 255
    # masked_img = cv.bitwise_and(img, img, mask = mask)
    hist_mask = cv.calcHist([img], [0], mask, [maxValue], [0, maxValue])
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(hist_mask, color='b')
    plt.xlim([0, maxValue])
    plt.grid()
    plt.show()


def findRingsConnected(img, showInfo=0, thresh_bound=2000, height_bound=0, res_img_prefix=''):
    contr_img = equal_contrast(img)

    # преобразуем в полярную систему координат
    polar = cv.linearPolar(contr_img, (contr_img.shape[1] // 2, contr_img.shape[0] // 2), contr_img.shape[1] // 2,
                           cv.WARP_FILL_OUTLIERS)

    convolved_polar = ndimage.convolve(np.float32(polar), CONV_KERNEL_BIG)
    abs_polar = np.abs(convolved_polar)

    if showInfo == 1:
        cv.imwrite("../../results/" + res_img_prefix + "equaled.png", contr_img)
        cv.imwrite("../../results/" + res_img_prefix + "polar.png", polar)
        cv.imwrite("../../results/" + res_img_prefix + "convolved_polar.png", convolved_polar)
        cv.imwrite("../../results/" + res_img_prefix + "abs_polar.png", abs_polar)
        drawGrayHist(contr_img, title='raw contrasted image histogram')
        drawGrayHistMasked(contr_img, title='raw contrasted image masked histogram')
        drawGrayHist(polar, title='polar image histogram')
        # drawGrayHist(abs_polar, int(np.max(abs_polar) + 1), title='abs_polar histogram')

    convolved_polar = None
    # *********************************************************

    # THRESHOLDING
    THRESH = thresh_bound
    ret, bin_result = cv.threshold(abs_polar, THRESH, 255, cv.THRESH_BINARY)
    bin_result = np.uint8(bin_result)
    if showInfo == 1:
        cv.imwrite("../../results/" + res_img_prefix + "bin_result.png", bin_result)

    # MORPHOLOGY
    morph = cv.morphologyEx(bin_result, cv.MORPH_CLOSE, MORPH_KERNEL)
    bin_result = np.uint8(morph)

    if showInfo == 1:
        cv.imwrite("../../results/" + res_img_prefix + "morph_close1.png", morph)

    morph = None
    abs_polar = None

    # *********************************************************

    # next rings detection without center
    bin_result[:, 0: bin_result.shape[1] // 10] = 0

    # CONNECTED COMPONENTS DETECTION
    connectivity = 8
    output = cv.connectedComponentsWithStats(bin_result, connectivity, cv.CV_32S)

    bin_result = None

    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    heights = stats[:, cv.CC_STAT_HEIGHT]

    if showInfo == 1:
        print("Num labels = " + str(num_labels))
        print("Labels\n", labels)
        print("Stats\n", stats)
        print("Centroids\n", centroids)
        print('Heights equals = ', heights)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    heights = heights[1:]
    idx = np.where(heights >= height_bound)
    idx = idx[0] + 1

    outputImg = np.zeros(labels.shape)
    outputImg[np.isin(labels, idx)] = 255

    # MORPHOLOGY TRY AGAIN

    morph = cv.morphologyEx(outputImg, cv.MORPH_CLOSE, MORPH_KERNEL_BIG)
    if showInfo == 1:
        cv.imwrite("../../results/" + res_img_prefix + "before_morph_close2.png", outputImg)
        cv.imwrite("../../results/" + res_img_prefix + "morph_close2.png", morph)
    outputImg = np.uint8(morph)

    morph = None

    # reverse linear polar transformation
    result = outputImg
    result = cv.linearPolar(result, (result.shape[1] // 2, result.shape[0] // 2), result.shape[1] // 2,
                            cv.WARP_INVERSE_MAP + cv.WARP_FILL_OUTLIERS)

    if showInfo == 1:
        cv.imwrite("../../results/" + res_img_prefix + "result.png", result)

    polar = None
    output = None
    num_labels = None
    labels = None
    stats = None
    centroids = None
    heights = None
    return result


def calculate(imagePath, truthPath=None, info=0, thresh=2000, height=2000, center_height=2000):
    start_time1 = time.time()
    img = cv.imread(imagePath, 0)
    if truthPath != None:
        truth = cv.imread(truthPath, 1)
        truth = cv.inRange(truth, bot, top)

    pred = findRingsConnected(img, showInfo=info, thresh_bound=thresh, height_bound=height, res_img_prefix='big_')
    end_time1 = time.time() - start_time1

    # center rings finding
    start_time2 = time.time()
    height = img.shape[0]
    width = img.shape[1]
    center_img = img[height // 2 - 200: height // 2 + 200, width // 2 - 200: width // 2 + 200]
    pred_center = findRingsConnected(center_img, showInfo=info, thresh_bound=thresh, height_bound=center_height,
                                     res_img_prefix='small_')
    pred[height // 2 - 200: height // 2 + 200, width // 2 - 200: width // 2 + 200] = pred_center
    end_time2 = time.time() - start_time2

    # result metrics finding
    start_time3 = time.time()
    if truthPath != None:
        jac_res = jaccard_metric(pred, truth)
        rwm_res = more_cool_metric(pred, truth)
    else:
        jac_res = -1
        rwm_res = -1
    end_time3 = time.time() - start_time3

    if info == 1:
        cv.imwrite("../../results/raw_input.png", img)
        cv.imwrite("../../results/truth_image.png", truth)
        cv.imwrite("../../results/predicted_image.png", pred)
        print("--- Rings detection %s seconds ---" % end_time1)
        print("--- Center rings detection  takes: %s seconds ---" % end_time2)
        print("--- Metrics calculation takes: %s seconds ---" % end_time3)

    return jac_res, rwm_res, pred
