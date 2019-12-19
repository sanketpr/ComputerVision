"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    img1C = left_img
    img2C = right_img
    img1 = cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)          # queryImage
    img2 = cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)

    # find keypoints with Harris corner detection
    # kp1 = cv2.cornerHarris(img1,2,3,0.04)
    # kp1 = cv2.KeyPoint_convert(kp1)
    # kp2 = cv2.cornerHarris(img2,2,3,0.04)
    # kp2 = cv2.KeyPoint_convert(kp2)

    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)
    kp1 = sift.detect(img1,None)
    kp2 = sift.detect(img2,None)
    kp1, des1 = sift.compute(img1,kp1)
    kp2, des2 = sift.compute(img2,kp2)

    bf = cv2.BFMatcher()
    matches = np.asarray(bf.knnMatch(des1,des2, k=2))

    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])
    good = np.asarray(good)


    matches1 = np.zeros((len(good), 2), dtype=np.float32)
    matches2 = np.zeros((len(good), 2), dtype=np.float32)
    for i,points in enumerate(good[:,0]):
        matches1[i, :] = kp1[points.queryIdx].pt
        matches2[i, :] = kp2[points.trainIdx].pt
    H1,mask = cv2.findHomography(matches1,matches2)
    H2,mask = cv2.findHomography(matches2,matches1)


    width = img1C.shape[1] + img2C.shape[1]
    height = img1C.shape[0]

    wrap1 = cv2.warpPerspective(img1C, H1, (img1C.shape[1], height))
    wrap2 = cv2.warpPerspective(img2C, H2, (width, height))
    wrap2[:img1C.shape[0],:img1C.shape[1]] = img1C
    
    return wrap2

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg',result_image)


