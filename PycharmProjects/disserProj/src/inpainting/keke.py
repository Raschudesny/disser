from core.find_rings import *
from skimage.measure import *
import random

def print_img(img, filename):
    print(cv.imwrite(filename, img))
    print(filename)

if __name__ == "__main__":
    imagePath = "../../papka/NotRings/not_rings3.png"
    outputMask1 = "../../src/inpainting/metrics/mask1.png"
    outputMask2 = "../../src/inpainting/metrics/mask2.png"
    outputAnd = "../../src/inpainting/metrics/and.png"
    outputOr = "../../src/inpainting/metrics/or.png"
    outputSquare = "../../src/inpainting/metrics/square.png"

    img = cv.imread(imagePath, 0)

    mask1 = np.zeros(img.shape)
    mask1[0:300, 0:300] = 255
    mask1 = np.uint8(mask1)

    mask2 = np.zeros(img.shape)
    mask2[500: 800, 500:800] = 255
    mask2 = np.uint8(mask2)
    mask1 = np.bitwise_or(mask1, mask2)

    print_img(mask1, outputMask1)


    BIG_SQUARE = np.zeros(img.shape)
    BIG_SQUARE[0:900, 0:900] = 255
    BIG_SQUARE = np.uint8(BIG_SQUARE)
    print_img(BIG_SQUARE, outputSquare)

    print(more_cool_metric( mask1, BIG_SQUARE))

#[(3.25, 0.29545454545454547, 0.5416666666666667), (2.75, 0.14473684210526316, 0.27499999999999997), (1.25, 1.25, 1.25), (1.125, 0.225, 0.37499999999999994), (1.75, 0.11666666666666667, 0.21875), (0.875, 0.010869565217391304, 0.02147239263803681), (2.5, 0.0625, 0.12195121951219512), (1.5, 0.5, 0.75), (1.75, 0.0625, 0.1206896551724138), (0.6388888888888888, 0.16911764705882354, 0.2674418604651163)]
