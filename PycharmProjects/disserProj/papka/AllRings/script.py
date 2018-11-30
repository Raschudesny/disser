import sys
import cv2 as cv

file = sys.argv[1]
img = cv.imread(file, 1) # Загружаем изображение сразу в Grayscale
polar = cv.linearPolar(img, (img.shape[1] // 2, img.shape[0] // 2), img.shape[1] // 2, cv.WARP_FILL_OUTLIERS)
cv.imwrite("newimg.png", polar)