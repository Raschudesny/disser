from core.find_rings import *

if __name__ == "__main__":
    imagePath = "../../papka/AllRings/rings1.png"
    truthPath = "../../papka/AllRings/marked1.png"

    start_time = time.time()

    input = cv.imread(imagePath, 0)

    img = cv.imread(imagePath, 0)
    truth = cv.imread(truthPath, 1)
    truth = cv.inRange(truth, bot, top)

    pred = findRingsConnected(img, showInfo=0, thresh_bound=5000, height_bound=105)



    print("--- %s seconds ---" % (time.time() - start_time))
