"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random

prev_points = []

def distance(p1,p2,a):
	return abs(((p2[1]-p1[1])*a[0]) - ((p2[0]-p1[0])*a[1]) + p2[0]*p1[1] - p2[1]*p1[0]) / ((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5

def get_random(input_points):
	points = random.sample(input_points,2)
	comp = [points[0]['name'], points[1]['name']]
	comp.sort()
	while (points[0]['name'] == points[1]['name'] and comp not in prev_points):
		points = random.choices(input_points,k=2)
		comp = [points[0]['name'], points[1]['name']]
		comp.sort()
	prev_points.append(comp)
	return points
def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """

    inlier_points_name = []
    outlier_points_name = []
    max_fit_count = -1
    avg_inlier_err = 999

    for i in range(k):
    	points = get_random(input_points)
    	tmp_inliers = []
    	inlier_error = 0
    	inlier_count = 0
    	tmp_outliers = []
    	for sample in input_points:
    		#returns a tuple of two randomly selected data points from distribution
    		dist = distance(points[0]['value'],points[1]['value'],sample['value'])

    		if dist <= t:
    			if (sample['name'] != points[0]['name'] and sample['name'] != points[1]['name']):
    				inlier_count = inlier_count + 1
    				inlier_error = inlier_error + dist
    			tmp_inliers.append(sample['name'])
    		else:
    			tmp_outliers.append(sample['name'])

    	if (inlier_count > 0 and inlier_count >= d and avg_inlier_err > (inlier_error/inlier_count)):
    		max_fit_count = len(tmp_inliers)
    		inlier_points_name = tmp_inliers
    		outlier_points_name = tmp_outliers
    		avg_inlier_err = (inlier_error/inlier_count)
    return inlier_points_name, outlier_points_name


if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]
    t = 0.5
    d = 3
    k = 100
    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)  # TODO
    assert len(inlier_points_name) + len(outlier_points_name) == 8  
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()


