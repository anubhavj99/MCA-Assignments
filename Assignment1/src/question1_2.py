import os
import pickle
from datetime import datetime
from math import ceil
from os import listdir
from os.path import isfile, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_all_files_in_dir(dir_name):
    return [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]


def get_dimensional_index(index, shape):
    dim = []
    for i in range(len(shape)):
        mult = 1
        for j in range(i + 1, len(shape)):
            mult *= shape[j]

        dim.append(int(index / mult))
        index = index % mult
    return dim


def calculate_blob_coord(log_image, img):
    blob_coord = []
    h, w = img.shape
    for _i in range(h - 1):
        i = _i + 1
        for _j in range(w - 1):
            j = _j + 1
            for _k in range(6):
                # get the 3D cube for the pixel
                log_filter_slice = log_image[_k:_k + 3, i - 1:i + 2, j - 1:j + 2]
                result = np.amax(log_filter_slice)
                # this is taken from the paper
                if result > 0.05:
                    z, x, y = get_dimensional_index(log_filter_slice.argmax(), log_filter_slice.shape)
                    z += _k
                    blob_coord.append((j + y - 1, i + x - 1, pow(np.e, z)))
    return blob_coord


def preprocess_and_get_log_blobs(img, increments, resize_percent=0.2, dev=False):
    # reduce the image to the resize percentage parameter
    img = cv2.resize(img, (int(img.shape[1] * resize_percent), int(img.shape[0] * resize_percent)))
    log_image = []
    for i in range(9):
        sigma_incremented = pow(increments, i)
        # get the size of grid. This 6 below is specified to be the most optimal.
        n = ceil(sigma_incremented * 6)
        # get the rid of the below specified size
        y, x = np.ogrid[-int(n / 2):int(n / 2) + 1, -int(n / 2):int(n / 2) + 1]
        # the formula of the implemented filter
        filter_log = (-(2 * pow(sigma_incremented, 2)) + (np.power(x, 2) + np.power(y, 2))) * np.exp(
            -1 * ((np.power(x, 2) + np.power(y, 2)) / (2.0 * pow(sigma_incremented, 2)))) * (
                             1 / (2 * np.pi * pow(sigma_incremented, 6)))

        log_filter_space = np.pad(cv2.filter2D(img, -1, filter_log), 1)
        # append the sigma increment space to the LoG filter
        log_image.append(log_filter_space)
    log_image = np.array(log_image)
    # calculate the blob coordinated
    blob_coord = np.array(calculate_blob_coord(log_image, img))

    # display the images with the blobs
    if dev:
        print(img.shape, log_image.shape)
        print("blob_coord.shape after removing redundant", blob_coord.shape)
        _, axis = plt.subplots()
        axis.imshow(img, interpolation='nearest', cmap="gray")
        for blob in blob_coord:
            x, y, r = blob
            c = plt.Circle((x, y), r * 1.414, fill=False)
            axis.add_patch(c)
        axis.plot()
        plt.show()

    return blob_coord


def start(save_dir, start_index, resize_percent, increment_parameter):
    all_images_files = ["HW-1/images/all_souls_000000.jpg",
                        "HW-1/images/ashmolean_000007.jpg"]  # get_all_files_in_dir("HW-1/images")
    all_image_value = {}
    batch_save_size = 5063
    if start_index > 0:
        all_images_files, start_index, all_image_value = pickle.load(
            open("{}/sift_{}.pkl".format(save_dir, start_index), "rb"))

    total_start_time = datetime.now()
    # iterate over all files
    for i, image_file in enumerate(all_images_files[start_index:]):
        start_time = datetime.now()
        # print("\ri:{} complete:{:.2f}% time left:{}".format(i, (i + 1) * 100.0 / len(all_images_files),
        #                                                     (5063 - i - 1) * (datetime.now() - total_start_time) / (
        #                                                                 i + 1)), image_file, end=" ")
        print("\ni:{} complete:{:.2f}%".format(i, (i + 1) * 100.0 / len(all_images_files)), image_file, end=" ")
        # get the image and convert to gray scale
        pre_img = Image.open(image_file).convert("L")
        # put the values between 0 and 1
        img = np.array(pre_img) / 255.0
        all_image_value[image_file] = preprocess_and_get_log_blobs(img=img, increments=increment_parameter,
                                                                   resize_percent=resize_percent, dev=True)
        if (i + 1) % batch_save_size == 0 or (i + 1) == len(all_images_files):
            print("saving at image number : {}".format(i + 1))
            pickle.dump((resize_percent, all_images_files, i + 1, all_image_value),
                        open("{}/sift_{}.pkl".format(save_dir, i + 1), "wb"))
            all_image_value = {}

        print("\ttotal_time_elapsed: {}".format(datetime.now() - start_time), end=" ")

    print("\ntime elapsed: {}".format(datetime.now() - total_start_time))


if __name__ == '__main__':
    res_per = 0.15
    k = 1.5
    start_index = 0
    if start_index == 0:
        os.system("rm -r save_Q2/{}".format(res_per * 100))
        os.system("mkdir save_Q2/{}".format(res_per * 100))
    start(save_dir="save_Q2/{}".format(res_per * 100), resize_percent=res_per, start_index=start_index,
          increment_parameter=k)
