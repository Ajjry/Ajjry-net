import numpy as np
import cv2
def denoising(rawimg):
    middlegray_img=cv2.blur(rawimg, (3,3))
    differ_img=np.maximum(rawimg.astype(np.int16)-middlegray_img.astype(np.int16),0)
    differ_img[differ_img<7000]=0
    rawimg=np.maximum(differ_img,rawimg)
    rawimg=rawimg.astype(np.uint16)
    return rawimg