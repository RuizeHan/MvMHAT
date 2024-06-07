import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

size = (1366, 768)
input_dir = 'visual'
views = os.listdir(input_dir)
d = {}

for i, view in enumerate(views):
    path = os.path.join(input_dir, view)
    d[i] = [os.path.join(path, x) for x in os.listdir(path)]

frame = 1

while 1:
    img11 = cv2.imread(d[0][frame])
    img12 = cv2.imread(d[1][frame])
    img21 = cv2.imread(d[2][frame])
    if len(views) == 4:
        img22 = cv2.imread(d[3][frame])
    else:
        img22 = cv2.imread(d[2][frame])

    img1 = np.hstack((img11, img21))
    img2 = np.hstack((img12, img22))
    img = np.concatenate((img1, img2)) 

    cv2.imshow('show', cv2.resize(img, size))
    key = cv2.waitKey()
    if key & 255 == 27:  # ESC
        print("terminating")
        sys.exit(0)
    elif key & 255 == 115:
        frame += 1
    elif key & 255 == 97: # 'a'
        frame -= 1

    




