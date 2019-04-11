from core.find_rings import *


def image_correction(imagePath, truthPath=None, info=0, thresh=5000, height=100, center_height=10):
    start_time = time.time()
    # detection stage

    JAC, RWM, detectedRings = calculate(imagePath, truthPath, info, thresh, height, center_height, only_jaccard_metrics=True)

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
    #5
    imagePath = "../../papka/AllRings/rings/rings4.png"
    truthPath = "../../papka/AllRings/marked/marked4.png"
    #[4337.00000000 131.00000000 40.00000000 0.35580709]
    #truth = cv.imread(truthPath, 1)
    #truth = cv.inRange(truth, bot, top)
    #print(more_cool_metric(truth, truth))

    # 0.35580708    0.4269034   0.42199225  0.175015    0.445079    0.648879    0.069764    0.111756    0.004874    0.429239    0.142271
    # 0.5           1.15384     1.07407     1.325806    1.483754    1.5         1.2         1.39908     1.45        0.857142    1.0
    # NAN           0.115384    0.1296296   NaN




    #4749 151
    image_correction(imagePath, truthPath, info=0, thresh=4749, height=151, center_height=40)

    #jac,r , p = calculate(imagePath, truthPath, info=0, thresh=5000, height=100, center_height=40, only_jaccard_metrics=True)
    #print(jac)
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
