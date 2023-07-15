import cv2
import numpy as np

img=cv2.imread("F:/eyes/DATA/BW1/BTT-L-2005101-BW1.jpg")
b = np.argwhere(img==0)
img = np.delete(img, b)

print("1")