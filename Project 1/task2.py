"""
Template Matching
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with template matching techniques, i.e., normalized cross correlation (NCC).

Please complete all the functions that are labelled with '# TODO'. When implementing those functions, comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'
and the functions you implement in 'task1.py' are of great help.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args

def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:
    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) / (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5
    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """

    img_delta = 0
    temp_delta = 0
    numerator = 0
    img_patch_mean = sum_m/(template_h * template_w)            
    temp_mean = temp_arr_sum/(template_h * template_w)    

    for x in range(len(op_im_arr)):
        img_delta += (img_patch_mean - op_im_arr[x]) ** 2
        temp_delta += (temp_mean - op_temp_arr[x]) ** 2
        numerator += (temp_mean - op_temp_arr[x]) * (img_patch_mean - op_im_arr[x])

    ncc_value = numerator/((img_delta * temp_delta)**0.5)

    return ncc_value

def match(img, template):
    """Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    """
    max_ncc_value = -9999
    x_val = 0
    y_val = 0

    # Dimensions of template (height,width)
    template_h = len(template)
    template_w = len(template[0])

    op_temp_arr = []
    temp_arr_sum = -999

    for h in range(len(img) - template_h):
        for w in range(len(img[0]) - template_w):
            op_im_arr = []
            sum_m = 0
            for k in range(template_h):
                for l in range(template_w):
                    op_im_arr.append(img[h+k][w+l])
                    sum_m += img[h+k][w+l]

                    #TODO: Improvise this check
                    if (len(op_temp_arr) < (template_h * template_w)) :
                        op_temp_arr.append(template[k][l])
                        temp_arr_sum += template[k][l]

            img_delta = 0
            temp_delta = 0
            numerator = 0
            img_patch_mean = sum_m/(template_h * template_w)            
            temp_mean = temp_arr_sum/(template_h * template_w)
    

            for x in range(len(op_im_arr)):
                img_delta += (img_patch_mean - op_im_arr[x]) ** 2
                temp_delta += (temp_mean - op_temp_arr[x]) ** 2
                numerator += (temp_mean - op_temp_arr[x]) * (img_patch_mean - op_im_arr[x])
            
            ncc_value = numerator/((img_delta * temp_delta)**0.5)
            if (max_ncc_value <= ncc_value):
                max_ncc_value = ncc_value
                x_val = w
                y_val = h

    return x_val,y_val,max_ncc_value


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = read_image(args.template_path)

    x, y, max_value = match(img, template)

    # The correct results are: x: 17, y: 129, max_value: 0.994
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":
    main()
