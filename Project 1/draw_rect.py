import numpy as np
import cv2

img = cv2.imread('data/proj1-task2.jpg',cv2.IMREAD_COLOR)
cv2.rectangle(img,(129,17),(153,41),(0,0,255),1)
# template = np.array(img)[129:156,17:41]

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()