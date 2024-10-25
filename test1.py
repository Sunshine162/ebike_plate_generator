import os
import os.path as osp
import cv2
import numpy as np

img = cv2.imread('debug.jpg')
rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])
new_w = rows * sin + cols * cos
new_h = rows * cos + cols * sin
M[0, 2] += (new_w - cols) * 0.5
M[1, 2] += (new_h - rows) * 0.5
w = int(np.round(new_w))
h = int(np.round(new_h))
dst = cv2.warpAffine(img, M, (w, h))
cv2.imshow('rotation', dst)
cv2.waitKey(0)
