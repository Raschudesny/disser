from metrics.metrics_wrappers import *
from skimage import morphology
import os
import numpy as np



# read 3d image
def read_slices(input_folder, input_name, num_slices, start_index=0, step=1, mode=cv.IMREAD_GRAYSCALE):
    if num_slices < 0:
        return None
    # slice = io.imread(inputName.format(start))
    full_name = os.path.join(input_folder, input_name)
    image = cv.imread(full_name.format(start_index), mode)

    if image is None:
        return None

    im3d = np.zeros((num_slices,) + image.shape, dtype=image.dtype)
    im3d[0] = image
    for i in range(1, num_slices):
        # slice = io.imread(full_name.format(i))
        image = cv.imread(full_name.format(i * step + start_index), mode)
        im3d[i] = image
    return im3d


# write 3d image
def log_slices(im3d, output_folder, output_label, start_index=0, mode="grayscale"):
    create_path(output_folder)
    if len(im3d.shape) == 2 or (mode != "grayscale" and len(im3d.shape) == 3):
        fname = os.path.join(output_folder, output_label)
        cv.imwrite(fname+".png", im3d)
    elif len(im3d.shape) >= 3:
        fname = os.path.join(output_folder, output_label + "_{0:04}.png")
        for i in range(0, im3d.shape[0]):
            cv.imwrite(fname.format(i + start_index), im3d[i])


def create_path(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("\n\x1b[37m" + "Folder created:    " + output_folder + "\x1b[0m")


# pixel-wise metrics
def accuracy(im3d1, im3d2):
    return (im3d1 == im3d2).sum()/im3d1.size


# jaccard index
def jaccard_index(im3d1, im3d2):
    product = np.logical_and(im3d1, im3d2).sum()
    sum = np.logical_or(im3d1, im3d2).sum()
    return product / sum

def hm(g, im3d1, im3d2):
    product = np.logical_and(g, im3d1).sum()
    sum = np.logical_or(im3d1, im3d2).sum()
    return product / sum

"""
# region-wise metrics
def region_metrics(im3d_gt, im3d_seg, threshold=0.4, debug_folder=None):

    matches_for_DIOU = calc_matches_for_region_metrics(im3d_gt, im3d_seg, threshold, debug_folder=debug_folder)
    matches_for_GIOU = calc_matches_for_region_metrics(im3d_seg, im3d_gt, threshold, debug_folder=debug_folder)
    # print(matches_for_DIOU)
    # print(matches_for_GIOU)

    one2one, one2many, n_gt, n_seg = matches_for_DIOU
    many2one = matches_for_GIOU[1]
    #print(one2one, one2many, many2one)
    #print(n_seg, n_gt)
    weights = np.asarray((1.0, 0.75, 0.75))
    values = np.asarray((one2one, one2many, many2one))
    weighted_sum = (weights * values).sum()
    DR = weighted_sum / n_gt
    RA = weighted_sum / n_seg
    average = 2*DR*RA/(DR+RA)

    return DR, RA, average


def calc_matches_for_region_metrics(im3d_gt, im3d_seg, threshold, debug_folder=None):
    one2one, one2many = 0, 0
    im3d_gt_labeled, n_gt = morphology.label(im3d_gt, return_num=True)
    im3d_seg_labeled, n_seg = morphology.label(im3d_seg, return_num=True)

    for label in range(1, n_seg+1):
        # рассматриваем i-тую область связности и область ее пересечения со вторым изображением
        im3d_seg_region = im3d_seg_labeled == label
        im3d_regionsproduct = np.logical_and(im3d_gt_labeled, im3d_seg_region)

        # find corresponding region in im3d_gt
        partlabels = np.unique(im3d_gt_labeled * im3d_regionsproduct)     # the first element is always 0
        im3d_gt_region = np.zeros(im3d_seg.shape, dtype=np.uint8)
        for partlabel in partlabels[1:]:
            im3d_gt_region = np.logical_or(im3d_gt_region, im3d_gt_labeled == partlabel)
        #im3d_gt_region[np.isin(im3d_gt_labeled, partlabels[1:])]

        # calc index
        index = jaccard_index(im3d_seg_region, im3d_gt_region)
        #index = hm(im3d_seg_labeled, im3d_seg_region, im3d_gt_region)
        im3d_regionsproduct_labeled, k = morphology.label(im3d_regionsproduct, return_num=True)
        if index >= threshold:
            if k == 1:
                one2one += 1
            else:
                one2many += 1

        if debug_folder is not None:
            print("label", str(label))
            log_slices(255*np.uint8(im3d_seg_region), debug_folder, output_label=str(label) + " 0im3d_seg_region")
            log_slices(255*np.uint8(im3d_gt_region), debug_folder, output_label=str(label) + " 1im3d_gt_region")
            log_slices(255*np.uint8(im3d_regionsproduct), debug_folder, output_label=str(label) + " 2im3d_regionsproduct")
            print(index)
            print(one2one, one2many)

    return one2one, one2many, n_gt, n_seg



if __name__ == "__main__":
    num_slices = 1
    start_index = 0
    input_folder = "./"
    im3d_gt_name = "kek.png"
    im3d_seg_name = "lal.png"
    debug_folder = input_folder + "/debug"


    im3d_gt = read_slices(input_folder, im3d_gt_name, num_slices, start_index=start_index).astype(float) / 255
    im3d_seg = read_slices(input_folder, im3d_seg_name, num_slices, start_index=start_index).astype(float) / 255

    print("ACCURACY\n", accuracy(im3d_gt, im3d_seg), "\n")
    print("JACCARD INDEX\n", jaccard_index(im3d_gt, im3d_seg), "\n")
    print("REGION-WISE METRICS\n", region_metrics(im3d_gt, im3d_seg, debug_folder=debug_folder), "\n")
"""




# region-wise metrics
def region_metrics(im3d_gt, im3d_seg, threshold=0.1, debug_folder=None):

    matches_for_DIOU = calc_matches_for_region_metrics(im3d_gt, im3d_seg, threshold, debug_folder=debug_folder)
    matches_for_GIOU = calc_matches_for_region_metrics(im3d_seg, im3d_gt, threshold, debug_folder=debug_folder)
    print(matches_for_DIOU)
    print(matches_for_GIOU)

    one2one, one2many, n_gt, n_seg = matches_for_DIOU
    many2one = matches_for_GIOU[1]

    weights = np.asarray((1, 0.75, 0.75))
    values = np.asarray((one2one, one2many, many2one))
    weighted_sum = (weights * values).sum()
    DR = weighted_sum / n_gt
    RA = weighted_sum / n_seg
    average = 2*DR*RA/(DR+RA)

    return DR, RA, average


def calc_matches_for_region_metrics(im3d_gt, im3d_seg, threshold, debug_folder=None):
    one2one, one2many = 0, 0
    im3d_gt_labeled, n_gt = morphology.label(im3d_gt, return_num=True)
    im3d_seg_labeled, n_seg = morphology.label(im3d_seg, return_num=True)

    for label in range(1, n_seg+1):
        # рассматриваем i-тую область связности и область ее пересечения со вторым изображением
        im3d_seg_region = im3d_seg_labeled == label
        im3d_regionsproduct = np.logical_and(im3d_gt_labeled, im3d_seg_region)

        # find corresponding region in im3d_gt
        partlabels = np.unique(im3d_gt_labeled * im3d_regionsproduct)     # the first element is always 0
        im3d_gt_region = np.zeros(im3d_seg.shape, dtype=np.uint8)
        for partlabel in partlabels[1:]:
            im3d_gt_region = np.logical_or(im3d_gt_region, im3d_gt_labeled == partlabel)

        # calc index
        index = jaccard_index(im3d_seg_region, im3d_gt_region)
        im3d_regionsproduct_labeled, k = morphology.label(im3d_regionsproduct, return_num=True)
        if index >= threshold:
            if k == 1:
                one2one += 1
            else:
                one2many += 1

        if debug_folder is not None:
            print("label", str(label))
            log_slices(255*np.uint8(im3d_seg_region), debug_folder, output_label=str(label) + " 0im3d_seg_region")
            log_slices(255*np.uint8(im3d_gt_region), debug_folder, output_label=str(label) + " 1im3d_gt_region")
            log_slices(255*np.uint8(im3d_regionsproduct), debug_folder, output_label=str(label) + " 2im3d_regionsproduct")
            print(index)
            print(one2one, one2many)

    return one2one, one2many, n_gt, n_seg