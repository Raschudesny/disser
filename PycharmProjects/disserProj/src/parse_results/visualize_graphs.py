import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # data to plot
    n_groups = 4
    #means_frank = (1.560821, 3.255070, 2.515439, 3.343634)
    #means_guido = (3.343634, 3.343634, 3.343634, 3.392296)

    means_frank = (1.95467, 2.639492, 2.38128, 2.846141)
    means_guido = (2.846141, 2.846141, 2.846141, 5.67904)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.3
    opacity = 0.6

    rects1 = plt.bar(index, means_frank, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Без')

    rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                     alpha=opacity,
                     color='g',
                     label='С')

    plt.xlabel('Этапы алгоритма')
    plt.ylabel('Сумма значений метрик для изображений')
    plt.xticks(index + bar_width, ('Контрастирование', 'Обработка центра', 'Морфология', 'Оптимальные параметры'))
    plt.legend(loc = 9)

    plt.tight_layout()
    plt.show()