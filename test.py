import cv2
import numpy as np
import denoising
image=cv2.imread("C:/Users/admin/Desktop/3.png")
sRGB2XYZ = np.array([[0.4124564,0.3575761,0.1804375],
                         [0.2126729,0.7151522,0.0721750],
                         [0.0193339,0.1191920,0.9503041]],dtype=float)
XYZ2Cam= np.array([[9020,-2890,-715],[-4535,12436,2348],[-934,1919,7086]]
,dtype=float)/10000
sRGB2Cam =np.dot(XYZ2Cam,sRGB2XYZ)
sRGB2Cam1=np.expand_dims(np.sum(sRGB2Cam,1),axis=1)
sRGB2Cam=sRGB2Cam/np.concatenate((sRGB2Cam1,sRGB2Cam1,sRGB2Cam1),axis=1)
Cam2sRGB =  np.linalg.inv(sRGB2Cam)
image=(image-np.min(image))/(np.max(image)-np.min(image))
r = Cam2sRGB[0, 0] * image[:,:,0]+Cam2sRGB[0, 1] * image[:,:,1]+Cam2sRGB[0, 2] * image[:,:,2]
g = Cam2sRGB[1, 0] * image[:,:,0]+Cam2sRGB[1, 1] * image[:,:,1]+Cam2sRGB[1, 2] * image[:,:,2]
b = Cam2sRGB[2, 0] * image[:,:,0]+Cam2sRGB[2, 1] * image[:,:,1]+Cam2sRGB[2, 2] * image[:,:,2]
image=np.concatenate((np.expand_dims(r,2),np.expand_dims(g,2),np.expand_dims(b,2)),axis=2)
image=np.maximum(image,0)
image=np.minimum(image,1)
balance1_image=image*16383
balance1_image_shape=balance1_image.shape
h=balance1_image_shape[0]
w=balance1_image_shape[1]

balance1_image=denoising.denoising(balance1_image,h,w)
#balance1_image=balance1_image*0.1
cv2.namedWindow("Lu",cv2.WINDOW_NORMAL)
cv2.imshow("Lu",balance1_image)
cv2.waitKey(0)
