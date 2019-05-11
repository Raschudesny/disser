from core.find_rings import *
from skimage.measure import *
import random

def compare_images(img1, img2):
    MSE = compare_mse(img1, img2)
    SSIM = compare_ssim(img1, img2)
    PSNR = compare_psnr(img1, img2)
    print("MSE = ", MSE)
    print("SSIM = ", SSIM)
    print("PSNR = ", PSNR)
    print("****************")
    return MSE, SSIM, PSNR

if __name__ == "__main__":
    imagePath = "../../papka/NotRings/not_rings3.png"
    outputMask = "../../src/inpainting/mask.png"
    outputPath1 = "../../src/inpainting/NS.png"
    outputPath2 = "../../src/inpainting/Telea.png"
    img = cv.imread(imagePath, 0)

    for radius in range(1, 15):

        for i in range(0, 100):
            max = img.shape[0] // 2
            min = img.shape[0] // 10
            radius = 4

            pos_x = int(random.random() * (max - min)) + int(min)
            width_x = int(random.random() * (max - min)) + int(min)
            pos_y = int(random.random() * (max - min)) + int(min)
            height_y = int(random.random() * (40)) + 1

            print(pos_x)
            print(pos_x + width_x)
            print(pos_y)
            print(pos_y + height_y)


            mask = np.zeros(img.shape)
            mask[pos_x: pos_x+width_x, pos_y: pos_y + height_y] = 255
            mask = cv.linearPolar(mask, (mask.shape[1] // 2, mask.shape[0] // 2), mask.shape[1] // 2,
                                   cv.WARP_INVERSE_MAP + cv.WARP_FILL_OUTLIERS)
            mask = np.uint8(mask)

            print(cv.imwrite(outputMask, mask))
            print(outputMask)

            img[np.where(mask)] = 0
            print(cv.imwrite("../../src/inpainting/img.png", img))

            resNS = cv.inpaint(img, mask, radius, cv.INPAINT_NS)
            resTelea = cv.inpaint(img, mask, radius, cv.INPAINT_TELEA)


            print(cv.imwrite(outputPath1, resNS))
            compare_images(resNS, img)
            print(cv.imwrite(outputPath2, resTelea))
            compare_images(resTelea, img)
            print(outputPath1)
            print(outputPath2)

            mask = None
            img = None
            resNS = None
            resTelea = None


