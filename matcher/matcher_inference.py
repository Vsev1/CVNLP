import os
import glob
import cv2
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matcher_model import Matcher


def open_raster_image(image):
    # Open a raster image JP2 using rasterio and reshape to HWC format
    with rasterio.open(image, "r", driver='JP2OpenJPEG') as src:
        raster_image = src.read()
        # raster_meta = src.meta
    raster_image = reshape_as_image(raster_image)
    return raster_image


def show_raster_images(selected_images):
    # Display multiple raster images in a 2x3 grid.
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))

    for index, img in enumerate(selected_images):
        row = index // 3
        col = index % 3
        axes[row, col].imshow(img)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def show_matches(image0, image1, matcher, conf=0.8, accurate=False):
    # Run matching and visualize the results
    corresp = matcher.match(image0, image1, conf, accurate)
    keypoints_num = len(corresp['inliers'])
    inliers_num = sum(corresp['inliers'])[0]
    ratio = inliers_num / float(keypoints_num)

    print(f'key points: {keypoints_num}, inlier points: {inliers_num}, ratio: {ratio:.2f}')
    matcher.show_keypoints_matches(corresp)
    plt.show()


def main():
    # Get all raster JP2 files recursively from the data folder
    all_files_list = glob.glob(os.path.join("data", "**", "*_TCI.jp2"), recursive=True)
    # print(all_files_list)

    # Select specific images by index
    # images_ids = [0, 3, 12, 18, 21, 30]
    images_ids = [0, 35, 15, 18, 21, 49]
    # images_ids = [0, 35, 15]

    selected_images_paths = [all_files_list[i] for i in images_ids]

    # selected_images = []
    # for path in selected_images_paths:
    #     img = open_raster_image(path)
    #     height, width = img.shape[:2]
    #     scale = 1024 / max(height, width)
    #     resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    #     selected_images.append(resized_img)
    #
    # show_raster_images(selected_images)

    # Load images as numpy arrays
    model_images = [open_raster_image(path) for path in selected_images_paths]

    # Initialize matcher with target image size
    image_size = (1024, 1024)

    matcher = Matcher(image_size)

    # Show matches between the second and third selected images
    show_matches(model_images[1], model_images[2], matcher)


if __name__ == '__main__':
    main()
