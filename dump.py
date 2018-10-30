# **************** MAIN PROGRAM ********************
def main_prog():

    img = cv.imread("papka/krygi/part.png", 0) # Загружаем изображение сразу в Grayscale
    cv.imwrite("results/rawImg.png", img)
    #преобразуем в полярную систему координат
    polar = cv.linearPolar(img, (img.shape[1]//2 , img.shape[0]//2), img.shape[1]//2,  cv.WARP_FILL_OUTLIERS)
    cv.imwrite("results/polar.png", polar)


    drawGrayHist(img, title = 'raw image histogram')
    drawGrayHistMasked(img, title = 'raw image masked histogram')
    drawGrayHist(polar, title = 'polar image histogram')


    #FREE IMG
    img = None

    kernel = np.array([
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
    convolved_polar = ndimage.convolve(np.float32(polar),kernel)
    cv.imwrite("results/convolved_polar.png",convolved_polar)
    abs_polar = np.abs(convolved_polar)
    cv.imwrite("results/abs_polar.png",abs_polar)


    #FREE CONVOLVED
    convolved_polar = None




    drawGrayHist(abs_polar, int(np.max(abs_polar) + 1), title='abs_polar histogram')



    #THRESH = 3000
    THRESH = np.percentile(abs_polar, 95)
    ret,bin_result = cv.threshold(abs_polar, THRESH , 255, cv.THRESH_BINARY)
    cv.imwrite("results/bin_result.png",bin_result)
    bin_result = np.uint8(bin_result)




    my_kernel = np.uint8(np.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]))



    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT,(5, 5))
    morph = cv.morphologyEx(bin_result, cv.MORPH_CLOSE, my_kernel)
    cv.imwrite("results/morph_close.png", morph)
    bin_result = np.uint8(morph)

    #FREE MORPH
    morph = None



    parampam  = 1

    if parampam == 0 :
        print(morph)
        AccArray = (np.sum(bin_result, axis = 0) / 255)
        print(AccArray)

        polar = cv.imread("results/polar.png", 0)
        polar = cv.cvtColor(polar, cv.COLOR_GRAY2RGB)
        percent = 0.5
        for i in range(0, len(AccArray)):
            if ((AccArray[i] / 4000.0) > percent):
                pt1 = (i, 0)
                pt2 = (i, polar.shape[0])
                cv.line(polar, pt1, pt2, (0, 0, 255), 1)

        cv.imwrite("results/AccumulativeArrayDetect.png", polar)
        print(np.max(AccArray))



        #inverse
        result = cv.linearPolar(polar, (polar.shape[1] // 2, polar.shape[0] // 2), polar.shape[1] // 2,
                                cv.WARP_INVERSE_MAP + cv.WARP_FILL_OUTLIERS)
        cv.imwrite("results/result.png", result)
    else:
        # You need to choose 4 or 8 for connectivity type
        connectivity = 8
        # Perform the operation
        output = cv.connectedComponentsWithStats(bin_result, connectivity, cv.CV_32S)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]

        print("Num labels = " + str(num_labels))
        print("Labels\n", labels)
        print("Stats\n", stats)
        print("Centroids\n", centroids)

        lblareas = stats[:, cv.CC_STAT_AREA]
        print(lblareas)
        print(polar.shape)

        print(labels.shape[1])
        print(lblareas[1:])
        print(num_labels)


        outputImg = np.zeros(labels.shape)
        # markedLabels = []
        upperBound = int(np.percentile(lblareas[1:], 99))
        print(upperBound)
        for i in range(1, num_labels):
            if (lblareas[i] >= upperBound):
                outputImg[labels == i] = 255
                # markedLabels.append(i)

        # print(markedLabels)
        print("done")

        # for i in range(0, labels.shape[0]) :
        #    if i % 100 == 0:
        #        print(i)
        #    for j in range(0, labels.shape[1]) :
        #        if labels[i][j] in markedLabels :
        #            outputImg[i][j] = 255

        # outputImg[labels in markedLabels] = 255
        plt.imshow(outputImg)
        plt.colorbar()
        plt.show()

        polar = cv.imread("results/polar.png", 0)
        polar = cv.cvtColor(polar, cv.COLOR_GRAY2RGB)
        for i in range(0, outputImg.shape[0]):
            for j in range(0, outputImg.shape[1]):
                if outputImg[i][j] == 255 :
                    polar[i][j] =  (0, 0, 255)

        result = cv.linearPolar(polar, (polar.shape[1] // 2, polar.shape[0] // 2), polar.shape[1] // 2,
                                cv.WARP_INVERSE_MAP + cv.WARP_FILL_OUTLIERS)
        cv.imwrite("results/result.png", result)
