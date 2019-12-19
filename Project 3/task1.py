"""
K-Means Segmentation Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to segment image using k-means clustering.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are allowed to add your own functions if needed.
You should design you algorithm as fast as possible. To avoid repetitve calculation, you are suggested to depict clustering based on statistic histogram [0,255]. 
You will be graded based on the total distortion, e.g., sum of distances, the less the better your clustering is.
"""


import utils
import numpy as np
import json
import time

MAX_INT_VAL = 9**99

def kmeans(img,k):
    """
    Implement kmeans clustering on the given image.
    Steps:
    (1) Random initialize the centers.
    (2) Calculate distances and update centers, stop when centers do not change.
    (3) Iterate all initializations and return the best result.
    Arg: Input image;
         Number of K. 
    Return: Clustering center values;
            Clustering labels of all pixels;
            Minimum summation of distance between each pixel and its center.  
    """

    img = np.array(img)
    upp_bound = 256


    histo_map = {}  # dict {key: value}-> {pix_intensity: pix_count}
    clust_labels = np.zeros((img.shape[0], img.shape[1]))

    for y in img:
        for x in y:
            if x in histo_map.keys():
                histo_map[x] = histo_map[x] + 1
            else:
                histo_map[x] = 1

    final_clust_map = {}
    final_clusters = []
    final_a_dist = MAX_INT_VAL
    final_b_dist = MAX_INT_VAL

    histo_map_keys = list(histo_map.keys())

    for pt1,k1 in enumerate(histo_map_keys):
        for pt2 in range(pt1+1, len(histo_map_keys)):
            k2 = histo_map_keys[pt2]
            mu = [int(k1),int(k2)]

            clust_map = {}  # {pix_intensity: cluster_label}
            for i in range (upp_bound):
                clust_map[i] = ''

            flag = True

            a = 0
            ai = 0
            a_dist = 0

            b = 0
            bi = 0
            b_dist = 0

            # K-means loop which ends when there are no changes in the centeroids
            while flag:
                new_mu = [0,0]
                for v in histo_map.keys():
                    d1 = np.linalg.norm(int(v)-int(mu[0]))
                    d2 = np.linalg.norm(int(v)-int(mu[1]))
                    if d1<d2:
                        clust_map[v] = 0
                        a = a + (v * histo_map[v])
                        ai = ai + histo_map[v]
                        a_dist = a_dist + (histo_map[v] * d1)

                    else:
                        clust_map[v] = 1
                        b = b + (v * histo_map[v])
                        bi = bi + histo_map[v]
                        b_dist = b_dist + (histo_map[v] * d2)

                if ai != 0:
                    new_mu[0] = a//ai # approximating value to nearest int
                if bi != 0:
                    new_mu[1] = b//bi # approximating value to nearest int

                if (set(new_mu) == set(mu)):
                    flag = False
                else:
                    mu = new_mu # Reassigning the centeroid

            # Updating values if we find a better pair of centroids
            if (a_dist+b_dist) < (final_a_dist + final_b_dist):
                final_clust_map = clust_map
                final_clusters = mu
                final_a_dist = a_dist
                final_b_dist = b_dist

    for j,y in enumerate(img):
        for i,x in enumerate(y):
                clust_labels[j][i] = final_clust_map[x]

    return final_clusters, clust_labels, int(final_a_dist+final_b_dist)


def visualize(centers,labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    image = np.zeros((img.shape[0], img.shape[1]))
    for j,y in enumerate(labels):
        for i,x in enumerate(y):
            image[j][i] = centers[int(x)]/255.0

    return image

     
if __name__ == "__main__":
    img = utils.read_image('lenna.png')
    k = 2

    start_time = time.time()
    centers, labels, sumdistance = kmeans(img,k)
    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('results/task1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers":centers, "distance":sumdistance, "time":running_time}))
    utils.write_image(result, 'results/task1_result.jpg')
