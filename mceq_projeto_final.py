import pandas as pd
from os import listdir, mkdir, system
from cv2 import imwrite
from functions.classifying import classify_to_data, open_img, ssim
from functions.distortions import gaussian_noise, blur, grayscale, negative, zoom


PATH_TO_DIR_VAL = "imagenette2/val/n01440764/"
PATH_TO_SAVE_VAL = "distorted-images/val/n01440764/"
PATH_TO_DIR_TRAIN = "imagenette2/train/n01440764/"
PATH_TO_SAVE_TRAIN = "distorted-images/train/n01440764/"


HEADERS = [
    "FILE_PATH",
    "FILE_PASS",
    "ORIG_CLASSES",
    "GAUSS_SSIM",
    "GAUSS_CLASSES",
    "BLUR_SSIM",
    "BLUR_CLASSES",
    "GRAYSCALE_SSIM",
    "GRAYSCALE_CLASSES",
    "NEGATIVE_SSIM",
    "NEGATIVE_CLASSES",
    "ZOOM_SSIM",
    "ZOOM_CLASSES",
]
df = pd.DataFrame(columns=HEADERS)


try:
    mkdir("distorted-images")
except FileExistsError:
    pass

try:
    mkdir("distorted-images/val/")
except FileExistsError:
    pass

try:
    mkdir(PATH_TO_SAVE_VAL)
except FileExistsError:
    pass

try:
    mkdir("distorted-images/train/")
except FileExistsError:
    pass

try:
    mkdir(PATH_TO_SAVE_TRAIN)
except FileExistsError:
    pass


files_val = listdir(PATH_TO_DIR_VAL)
files_train = listdir(PATH_TO_DIR_TRAIN)


n_passes = 5
n_files = len(files_val)
n_file = 0


for file in files_val:
    path = PATH_TO_DIR_VAL + file

    save = PATH_TO_SAVE_VAL + file.removesuffix(".JPEG")
    try:
        mkdir(save)
    except FileExistsError:
        pass

    n_file = n_file + 1

    orig_img = open_img(path=path)

    for passes in range(n_passes):
        file_pass = passes + 1

        system("cls")
        print(f"File ({n_file}/{n_files})")
        print(f"Pass ({file_pass}/{n_passes})")

        try:
            mkdir(f"{save}/pass_{file_pass}")
        except FileExistsError:
            pass

        classes_orig = classify_to_data(orig_img)

        after_gauss = gaussian_noise(orig_img, mean=10, std=10)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_gaussian.JPEG", after_gauss)

        ssim_gauss = ssim(orig_img, after_gauss)
        classes_gauss = classify_to_data(after_gauss)

        after_blur = blur(orig_img, ksize=(11, 11), sigmaX=0)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_blur.JPEG", after_blur)

        ssim_blur = ssim(orig_img, after_blur)
        classes_blur = classify_to_data(after_blur)

        after_grayscale = grayscale(orig_img)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_grayscale.JPEG", after_grayscale)

        ssim_grayscale = ssim(orig_img, after_grayscale)
        classes_grayscale = classify_to_data(after_grayscale)

        after_negative = negative(orig_img)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_negative.JPEG", after_negative)

        ssim_negative = ssim(orig_img, after_negative)
        classes_negative = classify_to_data(after_negative)

        after_zoom = zoom(orig_img, 1.5, coord=None)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_zoom.JPEG", after_zoom)

        ssim_zoom = ssim(orig_img, after_zoom)
        classes_zoom = classify_to_data(after_zoom)

        df.loc[-1] = [
            path.removeprefix("distorted-images/"),
            file_pass,
            classes_orig,
            ssim_gauss,
            classes_gauss,
            ssim_blur,
            classes_blur,
            ssim_grayscale,
            classes_grayscale,
            ssim_negative,
            classes_negative,
            ssim_zoom,
            classes_zoom,
        ]
        df.index = df.index + 1
        df = df.sort_index()

n_files = len(files_train)
n_file = 0

for file in files_train:
    path = PATH_TO_DIR_TRAIN + file

    save = PATH_TO_SAVE_TRAIN + file.removesuffix(".JPEG")
    try:
        mkdir(save)
    except FileExistsError:
        pass

    n_file = n_file + 1

    orig_img = open_img(path=path)

    for passes in range(n_passes):
        file_pass = passes + 1

        system("cls")
        print(f"File ({n_file}/{n_files})")
        print(f"Pass ({file_pass}/{n_passes})")

        try:
            mkdir(f"{save}/pass_{file_pass}")
        except FileExistsError:
            pass

        classes_orig = classify_to_data(orig_img)

        after_gauss = gaussian_noise(orig_img, mean=10, std=10)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_gaussian.JPEG", after_gauss)

        ssim_gauss = ssim(orig_img, after_gauss)
        classes_gauss = classify_to_data(after_gauss)

        after_blur = blur(orig_img, ksize=(11, 11), sigmaX=0)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_blur.JPEG", after_blur)

        ssim_blur = ssim(orig_img, after_blur)
        classes_blur = classify_to_data(after_blur)

        after_grayscale = grayscale(orig_img)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_grayscale.JPEG", after_grayscale)

        ssim_grayscale = ssim(orig_img, after_grayscale)
        classes_grayscale = classify_to_data(after_grayscale)

        after_negative = negative(orig_img)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_negative.JPEG", after_negative)

        ssim_negative = ssim(orig_img, after_negative)
        classes_negative = classify_to_data(after_negative)

        after_zoom = zoom(orig_img, 1.5, coord=None)
        imwrite(f"{save}/pass_{file_pass}/{file}_dist_zoom.JPEG", after_zoom)

        ssim_zoom = ssim(orig_img, after_zoom)
        classes_zoom = classify_to_data(after_zoom)

        df.loc[-1] = [
            path.removeprefix("distorted-images/"),
            file_pass,
            classes_orig,
            ssim_gauss,
            classes_gauss,
            ssim_blur,
            classes_blur,
            ssim_grayscale,
            classes_grayscale,
            ssim_negative,
            classes_negative,
            ssim_zoom,
            classes_zoom,
        ]
        df.index = df.index + 1
        df = df.sort_index()

df.to_csv("distorted-images/distorted-images-data.csv", sep=";")
