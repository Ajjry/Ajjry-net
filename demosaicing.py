import numpy as np
import cv2
def demosaicing(splitimg):
    for i in range(3):
        splitimg[i]=splitimg[i].astype(np.uint16)
    green_fliter=np.array([
        [0, 0.25, 0],
        [0.25, 1, 0.25],
        [0, 0.25, 0],
    ])
    red_fliter=np.array([
        [0.25, 0.5, 0.25],
        [0.5, 1, 0.5],
        [0.25, 0.5, 0.25],
    ])
    blue_fliter=np.array([
        [0.25, 0.5, 0.25],
        [0.5, 1, 0.5],
        [0.25, 0.5, 0.25],
    ])
    splitimg[0]=cv2.filter2D(splitimg[0],-1,blue_fliter)
    splitimg[1]=cv2.filter2D(splitimg[1],-1,green_fliter)
    splitimg[2]=cv2.filter2D(splitimg[2],-1,red_fliter)
    demosaing_img=cv2.merge(splitimg)
    return demosaing_img