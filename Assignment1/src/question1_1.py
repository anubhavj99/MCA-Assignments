import os
import pickle
from datetime import datetime
from os import listdir
from os.path import isfile, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def is_valid(x_max, y_max, point):
    if point[0] < 0 or point[0] >= x_max:
        return False
    if point[1] < 0 or point[1] >= y_max:
        return False
    return True


def get_neighbors(x_max, y_max, x, y, dist):
    points = []
    for xi in range(x - dist, x + dist + 1):
        if is_valid(x_max, y_max, (xi, y + dist)):
            points.append((xi, y + dist))
        if is_valid(x_max, y_max, (xi, y - dist)):
            points.append((xi, y - dist))

    for yi in range(y - dist + 1, y + dist):
        if is_valid(x_max, y_max, (x + dist, yi)):
            points.append((x + dist, yi))
        if is_valid(x_max, y_max, (x - dist, yi)):
            points.append((x - dist, yi))

    return points


def correlogram(photo, colors, distance_list, dev=False):
    x_max, y_max, _ = photo.shape
    color_set = [(color[0], color[1], color[2]) for color in colors]

    if dev:
        print("image shape:", photo.shape)
        print("unqiue colors:", colors.shape)
        print("Distances:", distance_list)

    color_prob_dist_map = {}
    for color in color_set:
        color_prob_dist_map[color] = []

    for k in distance_list:
        start_time_k = datetime.now()
        color_map = {}
        color_hist = {}
        for color in color_set:
            color_hist[color] = 0
            color_map[color] = 0

        for x in range(0, x_max):
            for y in range(0, y_max):
                color_i = photo[x][y]
                neighbour_pos = get_neighbors(x_max, y_max, x, y, k)
                color_hist[(color_i[0], color_i[1], color_i[2])] += 1
                for pos in neighbour_pos:
                    color_j = photo[pos[0]][pos[1]]

                    if (color_i[0], color_i[1], color_i[2]) == (color_j[0], color_j[1], color_j[2]):
                        color_map[(color_i[0], color_i[1], color_i[2])] += 1

        for color in color_set:
            color_prob_dist_map[color].append(float(color_map[color]) / (color_hist[color] * 8 * k))

        if dev:
            print("Done for k:{} in time:{}".format(k, datetime.now() - start_time_k))

    return {
        "color_prob_dist_map": color_prob_dist_map,
        "color_set": color_set
    }


def get_all_files_in_dir(dir_name):
    return [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]


def start_correlogram(img, dev=False, resize_percent=1.0):
    # if dev:
    #     img = "HW-1/images/all_souls_000000.jpg"  # get_all_files_in_dir("HW-1/images")[0]

    image = np.array(Image.open(img))
    if dev:
        plt.figure()
        plt.imshow(image)
        plt.xlabel("original")
        print(image.shape)

    image = cv2.resize(image, (int(image.shape[1] * resize_percent), int(image.shape[0] * resize_percent)))

    if dev:
        print(image.shape)
        plt.figure()
        plt.imshow(image)
        plt.xlabel("after resize: {}%".format(resize_percent * 100))
        plt.show()

    K = [i for i in range(1, 9, 2)]
    unique_color = np.unique(image.reshape((-1, 3)), axis=0)
    total_start_time = datetime.now()
    print("\tresizing to {}%".format(resize_percent * 100), end=" ")
    result = correlogram(image, unique_color, K, dev=dev)
    print("\ttime elapsed:", datetime.now() - total_start_time, end=" ")
    return result


def start(save_dir, start_beginning=True, start_index=0, res_per=1.0):
    all_images_files = get_all_files_in_dir("HW-1/images")
    all_image_value = {}
    batch_save_size = 500
    if not start_beginning:
        all_images_files, start_index, all_image_value = pickle.load(
            open("{}/correlogram_{}.pkl".format(save_dir, start_index), "rb"))

    total_start_time = datetime.now()
    for i, image_file in enumerate(all_images_files[start_index:]):
        print("\ni:{} complete:{:.2f}%".format(i, (i + 1) * 100.0 / len(all_images_files)), image_file, end=" ")

        all_image_value[image_file] = start_correlogram(image_file, resize_percent=res_per, dev=False)

        if (i + 1) % batch_save_size == 0 or (i + 1) == len(all_images_files):
            print("saving at image number : {}".format(i + 1))
            pickle.dump((all_images_files, i + 1, all_image_value),
                        open("{}/correlogram_{}.pkl".format(save_dir, i + 1), "wb"))
            all_image_value = {}

        print("\ttotal_time_elapsed: {}".format(datetime.now() - total_start_time), end=" ")

    print("time elapsed: {}".format(datetime.now() - total_start_time))


def get_val_of_formula(val1, val2):
    return abs(val1 - val2) / (1 + val1 + val2)


def compare_images():
    for res_per in [0.15]:
        img1, img2 = ["HW-1/images/all_souls_000000.jpg", "HW-1/images/all_souls_000001.jpg"]
        # img1 = Image.open("HW-1/images/all_souls_000013.jpg")
        # plt.imshow(img1)
        # plt.show()
        img1_val = start_correlogram(img1, resize_percent=res_per, dev=False)
        img2_val = start_correlogram(img2, resize_percent=res_per, dev=False)
        unique_colors = np.unique(img1_val["color_set"] + img2_val["color_set"], axis=0)
        # print(img1_val["color_prob_dist_map"])
        file = open("abc", "w")
        file.write(str(img1_val["color_prob_dist_map"]))
        file.close()
        print("unique_colors.shape", unique_colors.shape)
        # print(img1_val)
        dist = 0
        start_time_compare = datetime.now()
        K = [i for i in range(4)]
        for color_nd in unique_colors:
            color = (color_nd[0], color_nd[1], color_nd[2])
            # print(color)
            i1_comp = [0 for i in range(4)]
            i2_comp = [0 for i in range(4)]
            if color in img1_val["color_prob_dist_map"]:
                # print("present 1")
                i1_comp = img1_val["color_prob_dist_map"][color]
            if color in img2_val["color_prob_dist_map"]:
                # print("present 2")
                i2_comp = img2_val["color_prob_dist_map"][color]

            # print(i1_comp, i2_comp)
            for k in K:
                dist += get_val_of_formula(i1_comp[k], i2_comp[k])

        print("time taken to compare:", datetime.now() - start_time_compare)

        print("dist", dist / unique_colors.shape[0])


if __name__ == '__main__':
    compare_images()
    # res_per = 0.15
    # start_beginning = True
    # if start_beginning:
    #     os.system("rm -r save/{}".format(res_per * 100))
    #     os.system("mkdir save/{}".format(res_per * 100))
    # start(save_dir="save/{}".format(res_per * 100), res_per=res_per, start_beginning=start_beginning, start_index=0)
