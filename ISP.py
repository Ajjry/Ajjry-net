import cv2
import numpy
import numpy as np
import grayworld
import rawpy
import colortrans
import demosaicing
import denoising
import greypixel
import chazhi
import gamma
import BM3D
#imread
img = rawpy.imread("G:/NIKON/DSC_0240.NEF")
rawimg=img.raw_image
#rawimg = np.maximum(rawimg.astype(np.int16) - 512, 0)
rawimg=rawimg.astype(np.uint16)
rawimg_shape=rawimg.shape
h=rawimg_shape[0]
w=rawimg_shape[1]

#denoising
#rawimg=denoising.denoising(rawimg)
#rawimg=BM3D.BM3D(rawimg)
#demosaicing

byer_img=chazhi.chazhi(rawimg,h,w)
demosaing_img=demosaicing.demosaicing(byer_img)


#denoising
demosaing_img=BM3D.BM3D(demosaing_img)
#white blance
balance_img=greypixel.GrayPixel(demosaing_img,h,w)

#colorspace trans
balance1_image=colortrans.Raw2sRGB(balance_img)
cv2.imwrite("G:/TEST/z/%d.jpg"%(int(i+1)),balance1_image)
#gama
image = gamma.gamma(balance1_image, 2.2)
cv2.imwrite("G:/TEST/z/%d.jpg"%(int(i+1)),balance1_image)
'''
    cv2.namedWindow("Lu",cv2.WINDOW_NORMAL)
    cv2.imshow("Lu",image)
    cv2.waitKey(0)
'''


