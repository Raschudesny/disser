from core.find_rings import *


def find_all_files_in_dir(directory_name):
    files = []
    for (dirpath, dirnames, filenames) in walk(directory_name):
        files.extend(filenames)
        break
    files.sort()
    return files


if __name__ == "__main__":
    """
    images_directory = "../../papka/AllRings/rings"
    marked_directory = "../../papka/AllRings/marked"
    images_files = find_all_files_in_dir(images_directory)
    marked_files = find_all_files_in_dir(marked_directory)
    images_files.sort()
    marked_files.sort()
    print(images_files)
    print(marked_files)

    values = []
    count = 0
    for image in images_files:
        global_start_time = time.time()

        image_full_name = os.path.join(images_directory, image)
        truth_full_name = os.path.join(marked_directory, marked_files[count])
        print("Current image is: ", image_full_name)
        #4749 151
        #5435 221
        jac, rwm, pred = calculate(image_full_name, truth_full_name, info=0, thresh=5435, height=221, center_height=40,
                                   only_jaccard_metrics=False)
        #print("Jac=", jac)
        values.append(rwm)
        count += 1

    print(values)
    print(sum(values))
    """
    img_file = "../../papka/AllRings/rings/rings5.PNG"
    output_file = "../../output_img.png"
    img = cv.imread(img_file, 0)
    polar_img = cv.linearPolar(img, (img.shape[1] // 2, img.shape[0] // 2),
                               img.shape[1] // 2,
                               cv.WARP_FILL_OUTLIERS)
    cv.imwrite(output_file, polar_img)
