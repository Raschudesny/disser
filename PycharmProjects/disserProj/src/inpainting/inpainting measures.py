from core.find_rings import *
from skimage.measure import *
import random



def find_all_files_in_dir(directory_name):
    files = []
    for (dirpath, dirnames, filenames) in walk(directory_name):
        files.extend(filenames)
        break
    files.sort()
    return files


def compare_images(img1, img2):
    MSE = compare_mse(img1, img2)
    SSIM = compare_ssim(img1, img2)
    PSNR = compare_psnr(img1, img2)
    #print("MSE = ", MSE)
    #print("SSIM = ", SSIM)
    #print("PSNR = ", PSNR)
    #print("****************")
    return MSE, SSIM, PSNR

if __name__ == "__main__":
    images_directory = "../../papka/NotRings"
    images_files = find_all_files_in_dir(images_directory)
    images_files.sort()
    print(images_files)
    # imagePath = "../../papka/NotRings/not_rings3.png"
    #outputMask = "../../src/inpainting/mask.png"
    #outputPath1 = "../../src/inpainting/NS.png"
    #outputPath2 = "../../src/inpainting/Telea.png"


    for imagePath in images_files:
        image_full_name = os.path.join(images_directory, imagePath)
        img = cv.imread(image_full_name, 0)

        mse_results = []
        ssim_results = []
        psnr_results = []

        for radius in range(3, 8):
            ns_ssim_sum = 0
            tl_ssim_sum = 0
            ns_mse_sum = 0
            tl_mse_sum = 0
            ns_psnr_sum = 0
            tl_psnr_sum = 0
            number_of_iters = 10

            for i in range(0, number_of_iters):

                max = img.shape[0] // 2
                min = img.shape[0] // 10
                radius = 4

                pos_x = int(random.random() * (max - min)) + int(min)
                width_x = int(random.random() * (max - min)) + int(min)
                pos_y = int(random.random() * (max - min)) + int(min)
                height_y = int(random.random() * (40)) + 1

                mask = np.zeros(img.shape)
                mask[pos_x: pos_x+width_x, pos_y: pos_y + height_y] = 255
                mask = cv.linearPolar(mask, (mask.shape[1] // 2, mask.shape[0] // 2), mask.shape[1] // 2,
                                       cv.WARP_INVERSE_MAP + cv.WARP_FILL_OUTLIERS)
                mask = np.uint8(mask)



                img[np.where(mask)] = 0


                resNS = cv.inpaint(img, mask, radius, cv.INPAINT_NS)
                resTelea = cv.inpaint(img, mask, radius, cv.INPAINT_TELEA)


                #print(cv.imwrite(outputPath1, resNS))
                ns_mse, ns_ssim, ns_psnr = compare_images(resNS, img)
                #print(cv.imwrite(outputPath2, resTelea))
                tl_mse, tl_ssim, tl_psnr = compare_images(resTelea, img)

                mask = None
                resNS = None
                resTelea = None

                ns_mse_sum += ns_mse
                tl_mse_sum += tl_mse
                ns_ssim_sum += ns_ssim
                tl_ssim_sum += tl_ssim
                ns_psnr_sum += ns_psnr
                tl_psnr_sum += tl_psnr

            ns_mse_sum /= number_of_iters
            tl_mse_sum /= number_of_iters
            ns_ssim_sum /= number_of_iters
            tl_ssim_sum /= number_of_iters
            ns_psnr_sum /= number_of_iters
            tl_psnr_sum /= number_of_iters

            mse_results.append((ns_mse_sum, tl_mse_sum))
            ssim_results.append((ns_ssim_sum, tl_ssim_sum))
            psnr_results.append((ns_psnr_sum, tl_psnr_sum))
        print(str(image_full_name) + ' results:')
        print(mse_results)
        print(ssim_results)
        print(psnr_results)
