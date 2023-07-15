import cv2

import numpy as np
def preparation(ref_img,img):
    for i in range(1000):
        for j in range (1000):
            if not (ref_img[i][j]>20).any():
                img[i][j]=0
    return img
