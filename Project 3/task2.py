"""
Denoise Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to denoise image using median filter.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are suggested to use utils.zero_pad.
"""


import utils
import numpy as np
import json

def median_filter(img):
    """
    Implement median filter on the given image.
    Steps:
    (1) Pad the image with zero to ensure that the output is of the same size as the input image.
    (2) Calculate the filtered image.
    Arg: Input image. 
    Return: Filtered image.
    """
    # TODO: implement this function.
    img_o = np.zeros((img.shape[0], img.shape[1]))
    img = utils.zero_pad(img,1,1)
    for y in range(img.shape[0]-2):
        for x in range(img.shape[1]-2):
            pix_list = img[y:y+3,x:x+3].flatten()
            pix_list.sort()

            list_len = len(pix_list)
            median = -999
            if(list_len % 2 == 0):
                # median -> ((n/2) + (n/2+1))/2
                median = ((pix_list[list_len//2]+pix_list[(list_len//2)-1]))//2
            else:
                median = pix_list[list_len//2]
            img_o[y][x] = median

    return np.array(img_o).astype(np.uint8)

def mse(img1, img2):
    """
    Calculate mean square error of two images.
    Arg: Two images to be compared.
    Return: Mean square error.
    """    
    # TODO: implement this function.
    return np.mean((img1-img2)**2)/(img1.shape[0] * img1.shape[1])
    

if __name__ == "__main__":
    img = utils.read_image('lenna-noise.png')
    gt = utils.read_image('lenna-denoise.png')

    result = median_filter(img)
    error = mse(gt, result)

    with open('results/task2.json', "w") as file:
        json.dump(error, file)
    utils.write_image(result,'results/task2_result.jpg')


