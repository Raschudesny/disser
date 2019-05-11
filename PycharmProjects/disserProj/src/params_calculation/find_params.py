from core.find_rings import *
from multiprocessing import Pool


def runnableCalculate(input):
    JAC, RWM, PRED = calculate(input[0], input[1], input[2], input[3], input[4], input[5], input[6])
    PRED = None
    thresh = input[3]
    height = input[4]
    center_height = input[5]
    return (thresh, height, center_height, RWM)


# imagePath, truthPath, info, thresh, height, center_height, OnlyJac
def create_params(imgPath, truthPath, info, thresh_start, thresh_end, thresh_step, heigth_start, height_end,
                  height_step):
    results = []
    for i in reversed(range(thresh_start, thresh_end, thresh_step)):
        for j in reversed(range(heigth_start, height_end, height_step)):
            temp_res = [imgPath, truthPath, info]
            # thresh
            temp_res.append(i)
            # height
            temp_res.append(j)
            # center_height
            temp_res.append(40)
            # only jac
            temp_res.append(False)
            results.append(temp_res)
    return results


def find_all_files_in_dir(directory_name):
    files = []
    for (dirpath, dirnames, filenames) in walk(directory_name):
        files.extend(filenames)
        break
    files.sort()
    return files


def find_params_separately(images_directory, marked_directory, images_files, marked_files):
    count = 0
    for image in images_files:
        global_start_time = time.time()

        image_full_name = os.path.join(images_directory, image)
        truth_full_name = os.path.join(marked_directory, marked_files[count])
        print("Current image is: ", image_full_name)
        count += 1

        center_thresh = 5000
        center_height = 150

        thresh_interval = 3000
        height_interval = 200

        p = Pool(processes=8)
        while thresh_interval != 0:
            if int(thresh_interval / 10) == 0:
                thresh_step = 1
            else:
                thresh_step = int(thresh_interval / 10)

            if int(height_interval / 10) == 0:
                height_step = int(1)
            else:
                height_step = int(height_interval / 10)

            print("thresh step = ", thresh_step)
            print("height step = ", height_step)

            thresh_start = center_thresh - int(thresh_interval / 2)
            thresh_end = center_thresh + int(thresh_interval / 2) + thresh_step
            heigth_start = center_height - int(height_interval / 2)
            height_end = center_height + int(height_interval / 2) + height_step
            params = create_params(image_full_name, truth_full_name, 0,
                                   thresh_start=int(thresh_start),
                                   thresh_end=int(thresh_end),
                                   thresh_step=int(thresh_step),
                                   heigth_start=int(heigth_start),
                                   height_end=int(height_end),
                                   height_step=int(height_step))
            print(params)
            print(len(params))
            print('*********')

            res = p.map(runnableCalculate, params)
            print("iteration :" + str(center_thresh) + "passed")

            res = np.asarray(res)
            ind = image.find('.')
            image_str = image[0:ind]
            with open('../results/params_results/' + image_str + '_' + "jac_" + str(thresh_start) + '_' + str(
                    thresh_end) + '_and_'
                      + str(heigth_start) + "_" + str(height_end) + "_steps_" + str(thresh_step) + "_" + str(
                height_step) + ".txt", 'w') as outfile:
                for i in res:
                    outfile.write(np.array2string(i, formatter={'float': lambda x: '%0.8f' % x}) + '\n')

            # res[:] = res[::-1]
            max_res = max(res, key=lambda item: item[3])
            center_thresh = max_res[0]
            center_height = max_res[1]
            print("new center_thresh =", center_thresh)
            print("new center_height =", center_height)

            thresh_interval = int(thresh_interval / 10)
            if int(height_interval / 10) == 0:
                height_interval = 2
            else:
                height_interval = int(height_interval / 10)

            print("new thresh_interval = ", thresh_interval)
            print("new height_interval = ", height_interval)
            print("--- %s seconds ---" % (time.time() - global_start_time))


def find_params_together(images_directory, marked_directory, images_files, marked_files):
    global_start_time = time.time()
    center_thresh = 5000
    center_height = 150

    thresh_interval = 3000
    height_interval = 200

    p = Pool(processes=6)
    while thresh_interval != 0:
        if int(thresh_interval / 10) == 0:
            thresh_step = 1
        else:
            thresh_step = int(thresh_interval / 10)

        if int(height_interval / 10) == 0:
            height_step = int(1)
        else:
            height_step = int(height_interval / 10)

        print("thresh step = ", thresh_step)
        print("height step = ", height_step)

        thresh_start = center_thresh - int(thresh_interval / 2)
        thresh_end = center_thresh + int(thresh_interval / 2) + thresh_step
        heigth_start = center_height - int(height_interval / 2)
        height_end = center_height + int(height_interval / 2) + height_step

        images_counter = 0
        sum = 0
        for image in images_files:
            image_full_name = os.path.join(images_directory, image)
            truth_full_name = os.path.join(marked_directory, marked_files[images_counter])
            print("Current image is: ", image_full_name)
            images_counter += 1

            params = create_params(image_full_name, truth_full_name, 0,
                                   thresh_start=int(thresh_start),
                                   thresh_end=int(thresh_end),
                                   thresh_step=int(thresh_step),
                                   heigth_start=int(heigth_start),
                                   height_end=int(height_end),
                                   height_step=int(height_step))
            print(params)
            print(len(params))
            print('*********')

            res = p.map(runnableCalculate, params)
            res = np.asarray(res)

            if images_counter == 1:
                sum = res
            else:
                sum[:, 3] += res[:, 3]

            print("iteration with image: " + str(image) + " center: " + str(center_thresh) + " has passed")
            #print("sum = ", sum)

        with open('../results/params_results/' + 'overall' + '_' + "jac_" + str(thresh_start) + '_' + str(
                thresh_end) + '_and_'
                  + str(heigth_start) + "_" + str(height_end) + "_steps_" + str(thresh_step) + "_" + str(
            height_step) + ".txt", 'w') as outfile:
            for i in sum:
                outfile.write(np.array2string(i, formatter={'float': lambda x: '%0.8f' % x}) + '\n')

        max_res = max(sum, key=lambda item: item[3])
        center_thresh = max_res[0]
        center_height = max_res[1]
        print("new center_thresh =", center_thresh)
        print("new center_height =", center_height)

        thresh_interval = int(thresh_interval / 10)
        if int(height_interval / 10) == 0:
            height_interval = 2
        else:
            height_interval = int(height_interval / 10)

        print("new thresh_interval = ", thresh_interval)
        print("new height_interval = ", height_interval)

    print("--- %s seconds ---" % (time.time() - global_start_time))
    return

#MAIN FUNCTION
if __name__ == "__main__":
    images_directory = "../papka/AllRings/rings"
    marked_directory = "../papka/AllRings/marked"
    images_files = find_all_files_in_dir(images_directory)
    marked_files = find_all_files_in_dir(marked_directory)
    images_files.sort()
    marked_files.sort()
    print(images_files)
    print(marked_files)
    #find_params_separately(images_directory, marked_directory, images_files, marked_files)
    find_params_together(images_directory, marked_directory, images_files, marked_files)