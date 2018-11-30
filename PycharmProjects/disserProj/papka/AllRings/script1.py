import sys
import cv2 as cv

file = sys.argv[1]
img = cv.imread(file, 1) # Загружаем изображение сразу в Grayscale
result = cv.linearPolar(img, (img.shape[1] // 2, img.shape[0] // 2), img.shape[1] // 2,cv.WARP_INVERSE_MAP + cv.WARP_FILL_OUTLIERS)
cv.imwrite("newimg1.png", result)