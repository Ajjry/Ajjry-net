import numpy as np
import cv2
import math
def GrayPixel(img,h,w):
    Npre = 0.01
    num = math.floor(Npre * h * w / 100)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src = np.log(img)
    srcMean = cv2.blur(src.astype(np.uint8), (3, 3))
    c1 = np.sqrt((src - srcMean) ** 2)  # std
    c1Mean = (c1[:, :, 0] + c1[:, :, 1] + c1[:, :, 2]) / 3
    GI = ((c1[:, :, 0] - c1Mean) ** 2 + (c1[:, :, 1] - c1Mean) ** 2 + (c1[:, :, 1] - c1Mean) ** 2) / (
                3 * (c1Mean ** 2)) # difference of r,g,b
    GI = cv2.blur((GI / imgGray), (7, 7))
    Gidx = GI[:]
    Gidx.sort()
    Thresh = Gidx[num]
    mask = np.zeros((h, w))
    mask[GI <= Thresh] = 1
    IllumR = (mask * img[:, :, 0]).sum()
    IllumG = (mask * img[:, :, 1]).sum()
    IllumB = (mask * img[:, :, 2]).sum()
    coff = IllumR + IllumG + IllumB
    Illum = [IllumR, IllumG, IllumB] / coff
    img[:, :, 0] = img[:, :, 0] / Illum[0]
    img[:, :, 1] = img[:, :, 1] / Illum[1]
    img[:, :, 2] = img[:, :, 2] / Illum[2]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    '''
    mask1 = np.expand_dims(mask, axis=2)
    mask1 = np.concatenate((mask1, mask1, mask1), axis=2)
    show=mask1*img
    '''
    return img