from core.find_rings import read_image_slices

if __name__ == "__main__":
    images_dir = "../../papka/13sliceswithrings"
    truthPath = "../../papka/AllRings/marked3.png"

    read_image_slices(images_dir, '', 0, 1)