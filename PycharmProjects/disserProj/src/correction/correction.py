from core.find_rings import *


def image_correction(imagePath, truthPath=None, info=0, thresh=5000, height=100, center_height=10):
    start_time = time.time()
    # detection stage

    JAC, RWM, detectedRings = calculate(imagePath, truthPath, info, thresh, height, center_height)

    # correction stage
    start_time = time.time()
    detectedRings = np.uint8(detectedRings)
    print('jaccard =', JAC)
    print('RWM =', RWM)

    input = cv.imread(imagePath, 0)
    corrected_img = cv.inpaint(input, detectedRings, 3, cv.INPAINT_TELEA)
    if info == 1:
        cv.imwrite('../../results/corrected_img.png', corrected_img)

    print("--- Correction takes: %s seconds ---" % (time.time() - start_time))
    return


if __name__ == "__main__":
    imagePath = "../../papka/AllRings/rings6.png"
    truthPath = "../../papka/AllRings/marked6.png"
    image_correction(imagePath, truthPath, info=1, thresh=5500, height=100, center_height=40)



"""
    TASKS:
    
    все собрать вместе
    
    поробовать провести сравнение в результатах, чтобы было видно, что мы, оптимизируя параметры, увеличиваем метрику
    
    берется образец, делается множество теневых проекций, пропуская через него лучи и получая теневые проекции
    далее из этих проекций восстанавливается трехмерное изображение объекта
    
    на этом изображении могут присутствовать кольцевые артефакты, в их общем понимании, в том случае если на стадии 
    реконструкции не применялись никакие алгоритмы по их подавлению
    
    В случае же, если таковые алгоритмы применялись, мы получаем наши стандартные кольцевые артефакты 
    
    
    причины возникновения рингов в целом: 
    битые пиксели, небезупречность аппаратуры, ослабление сигнала на любой из стадий + различные устройства 
"""
