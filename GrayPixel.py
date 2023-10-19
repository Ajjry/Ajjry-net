import numpy as np
import cv2
import math
#def GrayPixel(img,h,w):
    # Npre = 0.01
    # num = math.floor(Npre*h*w/100)
    # imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # src = cv2.log(img,dst = None)
    # srcMean = cv2.blur(src,(3,3),borderType=cv2.Replicate)
    # c1 = cv2.sqrt((src-srcMean)**2)#std
    # c1Mean = (c1[:,:,0]+c1[:,:,1]+c1[:,:,2])/3
    # GI = ((c1[:,:,0]-c1Mean)**2+(c1[:,:,1]-c1Mean)**2+(c1[:,:,1]-c1Mean)**2) / (3*(c1Mean**2))#difference of r,g,b
    # GI = cv2.blur(GI/imgGray,(7,7),borderType=cv2.Replicate)
    # Gidx = GI[:]
    # Gidx.sort()
    # Thresh = Gidx[num]
    # mask = np.zeros((h,w))
    # mask[GI <= Thresh] = 1
    # IllumR = (mask*img[:,:,0]).sum()
    # IllumG = (mask*img[:,:,1]).sum()
    # IllumB = (mask*img[:,:,2]).sum()
    # coff = IllumR+IllumG+IllumB
    # Illum = [IllumR,IllumG,IllumB]
    # img[:,:,0] = img[:,:,0]/Illum[0]
    # img[:,:,1] = img[:,:,1]/Illum[1]
    # img[:,:,2] = img[:,:,2]/Illum[2]
    #return img
img = cv2.imread('C:/Users/admin/Desktop/IMG_8350.jpg')
Npre = 0.01
[h,w,d] = img.shape
num = math.floor(Npre*h*w/100)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
src = np.log(img)
# src = src.astype(np.float16)
# window = np.ones((3,3))
# srcMean = cv2.filter2D(src, -1, window, borderType=cv2.BORDER_REFLECT)
srcMean = cv2.blur(src.astype(np.uint8), (3,3))
c1 = np.sqrt((src-srcMean)**2)#std
c1Mean = (c1[:,:,0]+c1[:,:,1]+c1[:,:,2])/3
GI = ((c1[:,:,0]-c1Mean)**2+(c1[:,:,1]-c1Mean)**2+(c1[:,:,1]-c1Mean)**2) / (3*(c1Mean**2))#difference of r,g,b
GI = cv2.blur((GI/imgGray).astype(np.uint8),(7,7))
Gidx = GI[:]
Gidx.sort()
Thresh = Gidx[num]
mask = np.zeros((h,w))
mask[GI <= Thresh] = 1
IllumR = (mask*img[:,:,0]).sum()
IllumG = (mask*img[:,:,1]).sum()
IllumB = (mask*img[:,:,2]).sum()
coff = IllumR+IllumG+IllumB
Illum = [IllumR,IllumG,IllumB]/coff
print(np.max(img),np.min(img))
img=img.astype(np.float16)
Illum=Illum.astype(np.float16)
print(Illum)
img[:,:,0] = img[:,:,0]/Illum[0]
img[:,:,1] = img[:,:,1]/Illum[1]
img[:,:,2] = img[:,:,2]/Illum[2]
print(np.max(img),np.min(img))
img=img.astype(np.uint8)
cv2.namedWindow("Lu",cv2.WINDOW_NORMAL)
cv2.imshow("Lu",img)
cv2.waitKey(0)