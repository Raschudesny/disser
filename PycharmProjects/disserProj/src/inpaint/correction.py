from core.find_rings import *

if __name__ == "__main__":
    imagePath = "../../papka/AllRings/rings1.png"
    truthPath = "../../papka/AllRings/marked1.png"

    start_time = time.time()

    input = cv.imread(imagePath, 0)
    JAC, RWM, pred = calculate(imagePath, truthPath, 0, 5000, 105)
    pred = np.uint8(pred)
    print(JAC)
    print(RWM)

    corrected_img1 = cv.inpaint(input, pred, 4, cv.INPAINT_NS)
    corrected_img2 = cv.inpaint(input, pred, 4, cv.INPAINT_TELEA)

    cv.imwrite('../../corrected_img1.png', corrected_img1)
    cv.imwrite('../../corrected_img2.png', corrected_img2)



    print("--- %s seconds ---" % (time.time() - start_time))