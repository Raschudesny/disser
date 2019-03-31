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
    corrected_img = cv.inpaint(input, detectedRings, 4, cv.INPAINT_TELEA)
    if info == 1:
        cv.imwrite('../../results/corrected_img.png', corrected_img)

    print("--- Correction takes: %s seconds ---" % (time.time() - start_time))
    return


if __name__ == "__main__":
    imagePath = "../../papka/AllRings/rings/rings1.png"
    truthPath = "../../papka/AllRings/marked/marked1.png"
    #[4337.00000000 131.00000000 40.00000000 0.35580709]
    #truth = cv.imread(truthPath, 1)
    #truth = cv.inRange(truth, bot, top)
    #print(more_cool_metric(truth, truth))
    image_correction(imagePath, truthPath, info=1, thresh=4337, height=131, center_height=40)



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
    
    
    метрики на каждом этапе алгоритма
    
    cross validation
    
"""
