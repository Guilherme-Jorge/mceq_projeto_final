import pandas as pd
import threading
from os import listdir, mkdir, system
from cv2 import imwrite
from functions.classifying import classify_to_data, open_img, ssim
from functions.distortions import gaussian_noise, blur, grayscale, negative, zoom
from queue import Queue
import threading

PATH_TO_DIR_VAL = "imagenette2/val/n01440764/"
PATH_TO_SAVE_VAL = "distorted-images/val/n01440764/"
PATH_TO_DIR_TRAIN = "imagenette2/train/n01440764/"
PATH_TO_SAVE_TRAIN = "distorted-images/train/n01440764/"

HEADERS = [
    "FILE_PATH",
    "FILE_PASS",
    "ORIG_0_CLASS",
    "ORIG_0_CONFIDENCE",
    "ORIG_1_CLASS",
    "ORIG_1_CONFIDENCE",
    "ORIG_2_CLASS",
    "ORIG_2_CONFIDENCE",
    "ORIG_3_CLASS",
    "ORIG_3_CONFIDENCE",
    "ORIG_4_CLASS",
    "ORIG_4_CONFIDENCE",
    "GAUSS_SSIM",
    "GAUSS_0_CLASS",
    "GAUSS_0_CONFIDENCE",
    "GAUSS_1_CLASS",
    "GAUSS_1_CONFIDENCE",
    "GAUSS_2_CLASS",
    "GAUSS_2_CONFIDENCE",
    "GAUSS_3_CLASS",
    "GAUSS_3_CONFIDENCE",
    "GAUSS_4_CLASS",
    "GAUSS_4_CONFIDENCE",
    "BLUR_SSIM",
    "BLUR_0_CLASS",
    "BLUR_0_CONFIDENCE",
    "BLUR_1_CLASS",
    "BLUR_1_CONFIDENCE",
    "BLUR_2_CLASS",
    "BLUR_2_CONFIDENCE",
    "BLUR_3_CLASS",
    "BLUR_3_CONFIDENCE",
    "BLUR_4_CLASS",
    "BLUR_4_CONFIDENCE",
    "GRAYSCALE_SSIM",
    "GRAYSCALE_0_CLASS",
    "GRAYSCALE_0_CONFIDENCE",
    "GRAYSCALE_1_CLASS",
    "GRAYSCALE_1_CONFIDENCE",
    "GRAYSCALE_2_CLASS",
    "GRAYSCALE_2_CONFIDENCE",
    "GRAYSCALE_3_CLASS",
    "GRAYSCALE_3_CONFIDENCE",
    "GRAYSCALE_4_CLASS",
    "GRAYSCALE_4_CONFIDENCE",
    "NEGATIVE_SSIM",
    "NEGARIVE_0_CLASS",
    "NEGARIVE_0_CONFIDENCE",
    "NEGARIVE_1_CLASS",
    "NEGARIVE_1_CONFIDENCE",
    "NEGARIVE_2_CLASS",
    "NEGARIVE_2_CONFIDENCE",
    "NEGARIVE_3_CLASS",
    "NEGARIVE_3_CONFIDENCE",
    "NEGARIVE_4_CLASS",
    "NEGARIVE_4_CONFIDENCE",
    "ZOOM_SSIM",
    "ZOOM_0_CLASS",
    "ZOOM_0_CONFIDENCE",
    "ZOOM_1_CLASS",
    "ZOOM_1_CONFIDENCE",
    "ZOOM_2_CLASS",
    "ZOOM_2_CONFIDENCE",
    "ZOOM_3_CLASS",
    "ZOOM_3_CONFIDENCE",
    "ZOOM_4_CLASS",
    "ZOOM_4_CONFIDENCE",
]


# Create directories
def create_directories():
    directories = [
        "distorted-images",
        "distorted-images/val/",
        PATH_TO_SAVE_VAL,
        "distorted-images/train/",
        PATH_TO_SAVE_TRAIN,
    ]
    for directory in directories:
        try:
            mkdir(directory)
        except FileExistsError:
            pass


# Thread-safe DataFrame wrapper
class SafeDataFrame:
    def __init__(self, columns):
        self.df = pd.DataFrame(columns=columns)
        self.lock = threading.Lock()

    def append_row(self, row_data):
        with self.lock:
            self.df.loc[len(self.df)] = row_data

    def save_to_csv(self, path):
        with self.lock:
            self.df.to_csv(path, sep=";")


# Worker function for processing a single pass
def process_pass(file_path, save_path, file_name, pass_num, safe_df, progress_queue):
    orig_img = open_img(path=file_path)

    try:
        mkdir(f"{save_path}/pass_{pass_num}")
    except FileExistsError:
        pass

    classes_orig = classify_to_data(orig_img)

    # Process gaussian noise
    after_gauss = gaussian_noise(orig_img, mean=10, std=10)
    imwrite(f"{save_path}/pass_{pass_num}/{file_name}_dist_gaussian.JPEG", after_gauss)
    ssim_gauss = ssim(orig_img, after_gauss)
    classes_gauss = classify_to_data(after_gauss)

    # Process blur
    after_blur = blur(orig_img, ksize=(11, 11), sigmaX=0)
    imwrite(f"{save_path}/pass_{pass_num}/{file_name}_dist_blur.JPEG", after_blur)
    ssim_blur = ssim(orig_img, after_blur)
    classes_blur = classify_to_data(after_blur)

    # Process grayscale
    after_grayscale = grayscale(orig_img)
    imwrite(
        f"{save_path}/pass_{pass_num}/{file_name}_dist_grayscale.JPEG", after_grayscale
    )
    ssim_grayscale = ssim(orig_img, after_grayscale)
    classes_grayscale = classify_to_data(after_grayscale)

    # Process negative
    after_negative = negative(orig_img)
    imwrite(
        f"{save_path}/pass_{pass_num}/{file_name}_dist_negative.JPEG", after_negative
    )
    ssim_negative = ssim(orig_img, after_negative)
    classes_negative = classify_to_data(after_negative)

    # Process zoom
    after_zoom = zoom(orig_img, 1.5, coord=None)
    imwrite(f"{save_path}/pass_{pass_num}/{file_name}_dist_zoom.JPEG", after_zoom)
    ssim_zoom = ssim(orig_img, after_zoom)
    classes_zoom = classify_to_data(after_zoom)

    # Add results to DataFrame
    row_data = [
        file_path.removeprefix("distorted-images/"),
        pass_num,
        classes_orig[0][0],
        classes_orig[0][2],
        classes_orig[1][0],
        classes_orig[1][2],
        classes_orig[2][0],
        classes_orig[2][2],
        classes_orig[3][0],
        classes_orig[3][2],
        classes_orig[4][0],
        classes_orig[4][2],
        ssim_gauss,
        classes_gauss[0][0],
        classes_gauss[0][2],
        classes_gauss[1][0],
        classes_gauss[1][2],
        classes_gauss[2][0],
        classes_gauss[2][2],
        classes_gauss[3][0],
        classes_gauss[3][2],
        classes_gauss[4][0],
        classes_gauss[4][2],
        ssim_blur,
        classes_blur[0][0],
        classes_blur[0][2],
        classes_blur[1][0],
        classes_blur[1][2],
        classes_blur[2][0],
        classes_blur[2][2],
        classes_blur[3][0],
        classes_blur[3][2],
        classes_blur[4][0],
        classes_blur[4][2],
        ssim_grayscale,
        classes_grayscale[0][0],
        classes_grayscale[0][2],
        classes_grayscale[1][0],
        classes_grayscale[1][2],
        classes_grayscale[2][0],
        classes_grayscale[2][2],
        classes_grayscale[3][0],
        classes_grayscale[3][2],
        classes_grayscale[4][0],
        classes_grayscale[4][2],
        ssim_negative,
        classes_negative[0][0],
        classes_negative[0][2],
        classes_negative[1][0],
        classes_negative[1][2],
        classes_negative[2][0],
        classes_negative[2][2],
        classes_negative[3][0],
        classes_negative[3][2],
        classes_negative[4][0],
        classes_negative[4][2],
        ssim_zoom,
        classes_zoom[0][0],
        classes_zoom[0][2],
        classes_zoom[1][0],
        classes_zoom[1][2],
        classes_zoom[2][0],
        classes_zoom[2][2],
        classes_zoom[3][0],
        classes_zoom[3][2],
        classes_zoom[4][0],
        classes_zoom[4][2],
    ]
    safe_df.append_row(row_data)
    progress_queue.put(1)


def process_dataset(path_to_dir, path_to_save):
    files = listdir(path_to_dir)
    n_passes = 5
    n_files = len(files)

    safe_df = SafeDataFrame(HEADERS)
    progress_queue = Queue()
    total_tasks = n_files * n_passes
    completed_tasks = 0

    for file in files:
        file_path = path_to_dir + file
        save_path = path_to_save + file.removesuffix(".JPEG")

        try:
            mkdir(save_path)
        except FileExistsError:
            pass

        # Create and start threads for each pass
        threads = []
        for pass_num in range(1, n_passes + 1):
            thread = threading.Thread(
                target=process_pass,
                args=(file_path, save_path, file, pass_num, safe_df, progress_queue),
            )
            threads.append(thread)
            thread.start()

        # Wait for all passes to complete for this file
        for thread in threads:
            thread.join()

        # Update progress
        while not progress_queue.empty():
            progress_queue.get()
            completed_tasks += 1
            system("cls")
            print(f"Progress: {completed_tasks}/{total_tasks} tasks completed")

    return safe_df


def main():
    create_directories()

    # Process validation dataset
    print("Processing validation dataset...")
    val_df = process_dataset(PATH_TO_DIR_VAL, PATH_TO_SAVE_VAL)

    # Process training dataset
    print("Processing training dataset...")
    train_df = process_dataset(PATH_TO_DIR_TRAIN, PATH_TO_SAVE_TRAIN)

    # Combine and save results
    final_df = SafeDataFrame(HEADERS)
    final_df.df = pd.concat([val_df.df, train_df.df])
    final_df.save_to_csv("distorted-images/distorted-images-data.csv")


if __name__ == "__main__":
    main()
